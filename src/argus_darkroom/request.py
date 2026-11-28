"""The generate / pre-flight request. Policies are chosen by name only; a request never carries policy text."""

from __future__ import annotations

from typing import Annotated, Literal

from argus_cortex.moderation import THINKING_MODES
from argus_cortex.moderation import policies as policy_table
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

from argus_darkroom.models import MAX_POSE_PEOPLE, POSE_KEYPOINTS, PosePerson, SamplingParams

MAX_CLIP_SKIP = 12  # CLIP-L's layers
MIN_DENOISE = 0.05  # below this the output would be little more than the input
MAX_POSE_STRENGTH = 2.0  # ComfyUI's own ceiling is higher; past 2 a ControlNet only burns the image
# How the pose photo's frame maps onto the generation canvas (argus-puppet's RenderOptions.fit, less `stretch`, which
# distorts the skeleton).
PoseFit = Literal["contain", "cover", "skeleton"]
MAX_VERIFY_THRESHOLD = 10.0  # torso units: far past any two poses
MAX_POSE_FRAME = 65535  # the largest frame argus-puppet draws keypoints from


class RequestError(ValueError):
    """The request cannot be served as asked (HTTP 400)."""


class LoRASpec(BaseModel):
    """A LoRA as requested: by filename and weight, before the file is found and hashed."""

    model_config = ConfigDict(extra="forbid")

    name: str
    weight: float = 1.0
    trigger_words: list[str] = Field(default_factory=list)


class InitImage(BaseModel):
    """An image-to-image input. It gets the same image checks as an output before anything reaches ComfyUI, and it
    is never written to disk: it travels to ComfyUI inside the graph, which is deleted from ComfyUI's history after
    the run. darkroom does not store it; the record keeps its hashes and size.

    With a mask it is inpainting: only the mask's white region is repainted, and every other pixel of the output is
    the input's. The mask selects a region and carries no imagery, so it is not scanned; it travels and is forgotten
    like the input."""

    model_config = ConfigDict(extra="forbid")

    # PNG, JPEG or WebP bytes, base64. Checked when the request is prepared (HTTP 400): see ``images.load_input``.
    data_base64: str = Field(min_length=1)
    # How much of the input the sampler replaces: near 0 keeps it, 1 ignores it.
    denoise: float = Field(default=0.5, ge=MIN_DENOISE, le=1.0)
    # Inpainting: PNG, JPEG or WebP bytes, base64, the input's size once upright; white repaints, black keeps.
    # Checked when the request is prepared (HTTP 400): see ``images.load_mask``. None: plain image-to-image.
    mask_base64: str | None = Field(default=None, min_length=1)

    def __repr__(self) -> str:  # never the pixels, in a log or a traceback
        mask = "" if self.mask_base64 is None else f", mask=<{len(self.mask_base64)} base64 chars>"
        return f"InitImage(<{len(self.data_base64)} base64 chars>, denoise={self.denoise}{mask})"

    __str__ = __repr__


# One person's keypoints: 133 (x, y, score) points in the COCO-WholeBody order, as argus-puppet reports them.
PoseKeypoints = Annotated[
    list[tuple[FiniteFloat, FiniteFloat, FiniteFloat]], Field(min_length=POSE_KEYPOINTS, max_length=POSE_KEYPOINTS)
]


class PoseArea(BaseModel):
    """A rectangle of the generation, in its pixels: where a pose's skeleton is drawn."""

    model_config = ConfigDict(extra="forbid")

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class PoseInput(BaseModel):
    """A pose input: a body pose that steers the generation through a pose ControlNet. It is given one of two ways.

    **A photo** (``data_base64``): darkroom extracts the pose. The photo is an input image like any other. It gets
    the same image checks as an output before anything else sees it, and goes on only when it passes. Then
    argus-puppet reads the people's keypoints from it.

    **Keypoints** (``keypoints`` with their frame ``width`` x ``height``): the pose is given directly, for example a
    pose from an argus-puppet library or one a tool built. There is no photo, so there is nothing to scan: keypoints
    are numbers, not imagery.

    Either way argus-puppet draws the keypoints as an OpenPose-convention skeleton at the generation size, darkroom
    checks that what came back is a drawing, and that drawing is all ComfyUI gets. The output is scanned and decided
    as usual. A request can never send a control image: any picture would do as one, and nobody would have checked
    it. A photo is never written; the record keeps its hashes and the skeleton."""

    model_config = ConfigDict(extra="forbid")

    # A photo to take the pose from: PNG, JPEG or WebP bytes, base64. Checked when the request is prepared (HTTP
    # 400): see ``images.load_input``. Exactly one of this and ``keypoints``.
    data_base64: str | None = Field(default=None, min_length=1)
    # The pose itself: one person's 133 points, or a list of people (at most MAX_POSE_PEOPLE), in a frame of
    # ``width`` x ``height`` pixels. The shape of argus-puppet's bare pose, so a library entry's fields fit as they are.
    keypoints: (
        PoseKeypoints | Annotated[list[PoseKeypoints], Field(min_length=1, max_length=MAX_POSE_PEOPLE)] | None
    ) = None
    width: int | None = Field(default=None, ge=1, le=MAX_POSE_FRAME)
    height: int | None = Field(default=None, ge=1, le=MAX_POSE_FRAME)
    # With keypoints only: the sha256 of the image they came from, as the caller states it (a library entry's
    # ``source_sha256``). Recorded for tracing; darkroom never sees that image and cannot verify it.
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # The ControlNet file, as ComfyUI loads it (a path under ``controlnet/``). None: the server's default
    # (``ARGUS_DARKROOM_POSE_CONTROLNET``).
    controlnet: str | None = Field(default=None, min_length=1)
    # How hard the ControlNet holds the pose: around 0.6 to 1.0 holds it, lower lets the prompt override it.
    strength: float = Field(default=0.8, gt=0.0, le=MAX_POSE_STRENGTH)
    # Which parts of the skeleton are drawn. Some ControlNets are confused by face dots or were trained body-only.
    body: bool = True
    hands: bool = True
    face: bool = True
    # Draw only the primary subject (argus-puppet's choice in a photo; the first person of given keypoints) instead
    # of everyone.
    primary_only: bool = False
    fit: PoseFit = "contain"
    # Where in the generation the skeleton goes: it is fitted (by ``fit``) into this rectangle, and the rest of the
    # control image is black. For inpainting, the region being painted. None: the whole generation.
    area: PoseArea | None = None
    # After the image is released, have argus-puppet read its pose and compare it with the one asked for
    # (``record.pose_adherence``). A measurement only: it never changes the outcome. False skips it.
    verify: bool = True
    # The pose distance below which a person holds the pose. None: argus-puppet's default (0.25).
    verify_threshold: float | None = Field(default=None, ge=0.0, le=MAX_VERIFY_THRESHOLD)

    @model_validator(mode="after")
    def _one_source_and_something_drawn(self) -> PoseInput:
        if (self.data_base64 is None) == (self.keypoints is None):
            raise ValueError("a pose input is exactly one of a photo (data_base64) and keypoints")
        if self.keypoints is not None and (self.width is None or self.height is None):
            raise ValueError("keypoints need the width and height of the frame they are in")
        if self.data_base64 is not None and not (self.width is None and self.height is None):
            raise ValueError("width and height go with keypoints: a photo has its own size")
        if self.data_base64 is not None and self.source_sha256 is not None:
            raise ValueError("source_sha256 goes with keypoints: a photo is hashed by darkroom")
        if not (self.body or self.hands or self.face):
            raise ValueError("a pose input draws at least one of body, hands and face")
        return self

    def people(self) -> tuple[PosePerson, ...]:
        """The given keypoints as people, the first one primary. Empty for a photo."""
        if self.keypoints is None:
            return ()
        one = isinstance(self.keypoints[0][0], (int, float))  # a person is a list of points; a point of numbers
        listed = [self.keypoints] if one else self.keypoints
        return tuple(PosePerson(keypoints=person, primary=i == 0) for i, person in enumerate(listed))

    def __repr__(self) -> str:  # never the pixels, in a log or a traceback
        given = (
            f"{len(self.people())} people" if self.data_base64 is None else f"<{len(self.data_base64)} base64 chars>"
        )
        return f"PoseInput({given}, strength={self.strength}, fit={self.fit!r})"

    __str__ = __repr__


class GenerateRequest(BaseModel):
    # Unknown fields are refused, so a client sending ``policy_text`` gets an error, not silence.
    model_config = ConfigDict(extra="forbid")

    user_prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    expand: bool = False
    seed: int | None = None
    sampling: SamplingParams | None = None
    checkpoint: str | None = None
    vae: str | None = None
    loras: list[LoRASpec] = Field(default_factory=list)
    # None means the server default.
    enforcing_policy: str | None = None
    advisory_policies: list[str] = Field(default_factory=list)
    thinking: str | None = None
    nsfw_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    # Image-to-image. None: text-to-image.
    init_image: InitImage | None = None
    # Pose conditioning, from a photo or from keypoints (argus-puppet draws the skeleton). None: no ControlNet.
    pose: PoseInput | None = None

    @field_validator("user_prompt")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("the prompt is empty")
        return value

    @field_validator("thinking")
    @classmethod
    def _known_thinking(cls, value: str | None) -> str | None:
        if value is not None and value not in THINKING_MODES:
            raise ValueError(f"thinking must be one of {', '.join(THINKING_MODES)}")
        return value

    @field_validator("sampling")
    @classmethod
    def _clip_skip_in_range(cls, value: SamplingParams | None) -> SamplingParams | None:
        # Checked here, not on SamplingParams, so a stored record never stops loading. Past 12, CLIP-L's depth,
        # ComfyUI falls back to the last layer, which is the one that can be all NaN.
        if value is not None and not 1 <= value.clip_skip <= MAX_CLIP_SKIP:
            raise ValueError(f"clip_skip must be from 1 to {MAX_CLIP_SKIP}")
        return value

    @property
    def trigger_words(self) -> list[str]:
        return [word for lora in self.loras for word in lora.trigger_words]


def check_policies(enforcing: str, advisory: list[str], allow_nsfw: bool) -> list[str]:
    """The advisory policies, deduplicated, once the selection is known to be valid.

    Raises :class:`RequestError` for an unknown name, or an NSFW policy enforcing without ``--allow-nsfw``.
    NSFW policies can still run as advisory for comparison.
    """
    for name in [enforcing, *advisory]:
        try:
            policy_table.policy_text(name)
        except ValueError as exc:
            raise RequestError(str(exc)) from exc
    if policy_table.mode(enforcing) == "nsfw" and not allow_nsfw:
        raise RequestError(f"{enforcing!r} is an NSFW policy: this server was not started with --allow-nsfw")
    return [name for name in dict.fromkeys(advisory) if name != enforcing]
