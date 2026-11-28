"""The darkroom wire contract: one :class:`GenerationRecord` per attempt.

A record is kept for every attempt, refused and deleted ones included. It is the
only thing left of a deleted image, so it holds every parameter needed to
regenerate it exactly, and never a pixel, thumbnail or preview of one.

``ModelRef``, ``LoRARef`` and ``SamplingParams`` copy argus-proof's field names
exactly (darkroom does not import proof), so lifting them into cortex later is a
rename.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from argus_cortex.moderation import GuardResult
from argus_cortex.wire import make_versioned_base, render_schema
from pydantic import BaseModel, Field, FiniteFloat

# 1.0: initial contract.
RECORD_VERSION = "1.0"
SUPPORTED_RECORD_MAJORS: tuple[str, ...] = ("1",)
WIRE_TITLE = "argus-darkroom wire contract"

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", description="lowercase hex SHA256 digest")]

Outcome = Literal["refused", "passed", "review", "deleted", "omitted", "failed"]
Mode = Literal["sfw", "nsfw"]
CellState = Literal["safe", "flagged", "illegal", "unchecked"]
# How the minor-exposed rule places exposure: anywhere in the frame, or by box overlap (ARGUS_DARKROOM_MINOR_EXPOSED).
MinorExposedRule = Literal["frame", "overlap"]
# The keypoint layout argus-puppet reports: COCO-WholeBody, 133 points (body, feet, face, hands).
POSE_LAYOUT = "coco_wholebody_133"
POSE_KEYPOINTS = 133
# The most people of a pose photo that are drawn and recorded: argus-puppet's own default cap per image.
MAX_POSE_PEOPLE = 20

_Versioned = make_versioned_base(
    "record_version", RECORD_VERSION, SUPPORTED_RECORD_MAJORS, label="record_version", class_name="_Versioned"
)


class ModelRef(BaseModel):
    """A model file pinned by content: the name the engine loads and its sha256."""

    name: str
    sha256: Sha256


class LoRARef(ModelRef):
    weight: float = 1.0
    # The words the request supplied for this LoRA (already in final_prompt), kept so a record can refill a form.
    trigger_words: list[str] = Field(default_factory=list)


class SamplingParams(BaseModel):
    sampler: str
    scheduler: str
    steps: int = Field(gt=0)
    cfg: float
    # Flux's distilled guidance (the workflow's ``$guidance`` slot, a FluxGuidance node). None: not used.
    guidance: float | None = Field(default=None, ge=0)
    clip_skip: int = 1
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Expansion(BaseModel):
    """How Ollama expanded the prompt: enough to know which system prompt was used."""

    model: str
    system_prompt_sha256: Sha256


class MaskRef(BaseModel):
    """The identity of an inpainting mask, never its pixels. The mask selects the region of the input that is
    repainted; it carries no imagery, so it is not scanned.

    ``sha256`` is of the mask as ComfyUI got it: turned upright, cut into black and white at 128 and re-encoded as a
    1-bit PNG with no metadata. ``source_sha256`` is of the bytes the caller sent.
    """

    sha256: Sha256
    source_sha256: Sha256
    # The share of the input's pixels that are repainted, before the template grows the mask's edge.
    masked_fraction: float = Field(ge=0.0, le=1.0)


class InitImageRef(BaseModel):
    """The identity of an image-to-image input, never its pixels: the caller keeps its own input.

    The hashes and size are of the image as darkroom judged it and sent it to ComfyUI: decoded, turned upright, made
    RGB and re-encoded as a PNG with no metadata. ``source_sha256`` is of the bytes the caller sent.
    """

    sha256: Sha256
    phash: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    source_sha256: Sha256
    denoise: float
    # Inpainting: the mask, by its hashes. null for plain image-to-image.
    mask: MaskRef | None = None


class PosePerson(BaseModel):
    """One person of the pose skeleton: argus-puppet's whole-body keypoints, never pixels."""

    # 133 (x, y, score) points in the pose photo's pixel frame (as judged: upright), in the COCO-WholeBody order.
    keypoints: list[tuple[FiniteFloat, FiniteFloat, FiniteFloat]] = Field(
        min_length=POSE_KEYPOINTS, max_length=POSE_KEYPOINTS
    )
    # Whether argus-puppet named this person the photo's primary subject.
    primary: bool = False


class PoseEstimatorRef(BaseModel):
    """What read the keypoints: argus-puppet's version and its two pinned models, as it reported them."""

    package_version: str | None = None
    # puppet's wire version (``puppet_version``) of the answer the keypoints came in.
    wire_version: str | None = None
    detector: ModelRef | None = None
    pose: ModelRef | None = None


class ControlImageRef(BaseModel):
    """The identity of the skeleton drawing ComfyUI got: the PNG's sha256 (re-encoded with no metadata) and size."""

    sha256: Sha256
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class PoseRef(BaseModel):
    """A pose input: where the pose came from, the skeleton, and what was drawn from it. Never a pixel.

    ``source`` is ``photo`` when darkroom took the pose from a photo it checked. ``sha256``, ``phash``, ``width`` and
    ``height`` are then of the photo as darkroom judged it and sent it to argus-puppet (decoded, turned upright, made
    RGB and re-encoded as a PNG with no metadata), ``source_sha256`` is of the bytes the caller sent, and ``people``
    and ``estimator`` are what argus-puppet read from it.

    ``source`` is ``keypoints`` when the request gave the pose directly. There was no photo: ``sha256`` and ``phash``
    are null, ``width`` and ``height`` are the frame the request stated, ``people`` are the request's keypoints,
    ``estimator`` is null, and ``source_sha256`` is the image the caller says they came from, if it said (never seen
    or verified by darkroom).

    Only the skeleton drawn from ``people`` reaches ComfyUI (``control``). For a photo, ``people``, ``estimator`` and
    ``control`` are filled once it passed its scan and argus-puppet answered; on a record refused or failed before
    that they are empty and null.
    """

    source: Literal["photo", "keypoints"] = "photo"
    sha256: Sha256 | None = None
    phash: str | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    source_sha256: Sha256 | None = None
    controlnet: ModelRef
    strength: float
    # What was drawn, and how the pose's frame was fitted to the generation's.
    body: bool = True
    hands: bool = True
    face: bool = True
    primary_only: bool = False
    fit: str = "contain"
    # The rectangle of the generation the skeleton was fitted into (x, y, width, height); null: all of it.
    area: tuple[int, int, int, int] | None = None
    layout: str = POSE_LAYOUT
    # The people that were drawn: everyone (at most MAX_POSE_PEOPLE), or the primary subject alone.
    people: list[PosePerson] = Field(default_factory=list)
    estimator: PoseEstimatorRef | None = None
    control: ControlImageRef | None = None


class PoseAdherenceMatch(BaseModel):
    """One drawn person and the person in the output holding their pose, if any (argus-puppet's ``AdherenceMatch``)."""

    target: int = Field(ge=0)  # index into request.pose.people
    output: int | None = None  # index among the people argus-puppet found in the output; null when unmatched
    distance: float | None = None  # pose distance as posed, in torso units; left stays left
    mirrored_distance: float | None = None  # the same against the output person's mirror image
    close: bool = False  # distance is below the threshold: this person holds the pose
    mirrored: bool = False  # not close as posed, but the mirror image is: the figure came out flipped


class PoseAdherenceRecord(BaseModel):
    """Whether the released image holds the pose it was asked for, as argus-puppet's ``POST /adherence`` judged it.

    A measurement, never a verdict: it changes no outcome and is made only for an image that was already released
    (passed or review). It compares pose shape (the 17 body joints, hip-centred and torso-scaled), not where a figure
    stands in the frame, its size, hands or face. ``adheres`` is null, with ``error`` saying why, when the check
    could not run; the image is released either way.
    """

    adheres: bool | None = None
    threshold: float | None = None
    # The largest and the mean matched distance; null with no match.
    distance: float | None = None
    mean_distance: float | None = None
    target_people: int = 0
    output_people: int = 0
    comparable: int = 0  # drawn people whose pose could be compared
    matched: int = 0
    close: int = 0
    matches: list[PoseAdherenceMatch] = Field(default_factory=list)
    # What read the output's pose.
    estimator: PoseEstimatorRef | None = None
    error: str | None = None


class RequestRecord(BaseModel):
    """What was asked for. ``final_prompt`` is the exact string ComfyUI receives."""

    # Derived from ``guard.enforcing_policy``; stored so records can be filtered by it.
    mode: Mode
    user_prompt: str
    expanded_prompt: str | None = None
    final_prompt: str
    negative_prompt: str = ""
    expansion: Expansion | None = None
    seed: int | None = None
    # The next five are null on a record refused in pre-flight: nothing was resolved or hashed.
    sampling: SamplingParams | None = None
    checkpoint: ModelRef | None = None
    vae: ModelRef | None = None
    loras: list[LoRARef] = Field(default_factory=list)
    workflow_sha256: Sha256 | None = None
    # The template's name (its file stem, e.g. ``sdxl+lora1``), chosen by checkpoint family and LoRA count.
    workflow: str | None = None
    engine: Literal["comfyui"] = "comfyui"
    engine_version: str | None = None
    # Image-to-image: the input's identity and the denoise it was rendered with. null for text-to-image.
    init_image: InitImageRef | None = None
    # Pose conditioning: the photo's identity, the ControlNet and the skeleton. null without a pose input.
    pose: PoseRef | None = None


class Cell(BaseModel):
    """One policy's verdict in one context: a state, and the SingGuard answer or the error behind it."""

    state: CellState
    result: GuardResult | None = None
    error: str | None = None


class ScanCells(BaseModel):
    """One policy's two image scans: alone, and alongside the prompt that made the image."""

    isolated: Cell
    with_prompt: Cell


class NudeNetBox(BaseModel):
    """One NudeNet detection: a class, its score and where it is. A box, never pixels."""

    label: str
    score: float
    bbox: tuple[float, float, float, float]  # x, y, w, h in pixels


class NudeNetRecord(BaseModel):
    """What NudeNet said, kept for every rendered image. *exposed_score* is null when it did not answer (or is not
    configured), and the labels are then empty: under an NSFW policy that blurs the image, under SFW it omits it."""

    exposed_score: float | None
    labels: dict[str, float] = Field(default_factory=dict)
    # Every detection NudeNet reported (all score >= 0.2), so an ``overlap`` minor-exposed decision can be recomputed
    # from the record. Empty when it did not answer, and in records from before boxes were kept.
    detections: list[NudeNetBox] = Field(default_factory=list)
    # The user's ``nsfw_threshold`` (blur). The SFW gate is a fixed 0.2 and ignores it.
    threshold: float
    # Whether safelight blurs the image until the user reveals it. True under an NSFW policy when NudeNet scores it at
    # or above the threshold, or did not answer. Always false under an SFW policy.
    blur: bool
    # The NudeNet detector that scored the image, as its server reports it (cortex's ``NudeNetClient.info()``):
    # "640m", "320n", or null when NudeNet did not answer, the server did not say (the upstream 320n image has no
    # /health), or names another model. 640m and 320n score the same image differently, so the scores (and the
    # threshold they were judged against) are comparable only within one detector. Null in older records.
    detector: str | None = None
    # The weights file the server named (e.g. "640m.onnx"), or null.
    detector_model: str | None = None


class FaceAge(BaseModel):
    """A MiVOLO age estimate for one detected face. A tool's estimate, never a stated age."""

    age_estimate_years: float
    bbox: tuple[float, float, float, float]  # x, y, w, h in pixels, as curator reports it


class PersonAge(BaseModel):
    """One person (body) box from curator's person detector, with its MiVOLO estimate when it has one.

    ``face_index`` is the face in :attr:`AgeRecord.faces` this person was matched to, or ``None`` for a body with
    no face. A matched person's estimate may be ``None``: the face carries it. An unmatched person's estimate is
    also in :attr:`AgeRecord.faces`, so ``faces`` alone still holds every estimate.
    """

    bbox: tuple[float, float, float, float]  # x, y, w, h in pixels, as curator reports it
    age_estimate_years: float | None = None
    face_index: int | None = None


class AgeRecord(BaseModel):
    source: Literal["mivolo"] = "mivolo"
    # Every estimate: each face, then each person with no face (a faceless minor is still a minor).
    faces: list[FaceAge] = Field(default_factory=list)
    # Every person box, matched or not. Empty when the estimator found none or does not report them.
    people: list[PersonAge] = Field(default_factory=list)
    minor_exposed: bool = False
    # Which rule decided minor_exposed. Records from before the setting existed used "frame".
    minor_exposed_rule: MinorExposedRule = "frame"


class InputScanRecord(BaseModel):
    """An input image's own scan (the image-to-image input, or a pose photo): the same checks an output gets, before
    anything reaches ComfyUI or argus-puppet.

    The input goes on only when ``outcome`` is ``passed``; anything else refuses the attempt. ``outcome`` and
    ``reason`` are what the outcome table said of the input, as if it were an output.
    """

    scans: dict[str, ScanCells] = Field(default_factory=dict)
    nudenet: NudeNetRecord
    # null when the age check is unavailable
    age: AgeRecord | None = None
    outcome: Outcome
    reason: str | None = None


class GuardRecord(BaseModel):
    enforcing_policy: str
    advisory_policies: list[str] = Field(default_factory=list)
    thinking: str
    # Pre-flight is text-only: one cell per policy, framed prompt and negative prompt together.
    preflight: dict[str, Cell] = Field(default_factory=dict)
    scans: dict[str, ScanCells] = Field(default_factory=dict)
    nudenet: NudeNetRecord | None = None
    # null when the age check is unavailable
    age: AgeRecord | None = None
    # The image-to-image input's scan. null for text-to-image, and when pre-flight refused before the input was
    # scanned.
    input: InputScanRecord | None = None
    # The pose photo's scan, the same checks again. null without a pose input, for a pose given as keypoints (there is
    # no photo to scan), and when pre-flight refused before the photo was scanned.
    pose_input: InputScanRecord | None = None


class ImageRecord(BaseModel):
    """The identity of the output. The hashes are kept even when the pixels are not."""

    sha256: Sha256
    phash: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    # Set only when the image is passed or review (the only outcomes that write it): the file name under the
    # output root.
    path: str | None = None


class DepthRecord(BaseModel):
    """A depth map derived from the record's stored image (``POST /depth/{id}``), never from a new prompt.

    The PNG is 8-bit greyscale at the image's size, near bright (255) and far dark (0), stretched min to max per image,
    with no metadata. ``sha256`` is of the stored PNG's bytes. ``workflow`` names darkroom's built-in depth graph.
    """

    model: str
    model_license: str
    sha256: Sha256
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    workflow: str
    engine_version: str | None = None


class GenerationRecord(_Versioned):  # type: ignore[misc, valid-type]
    id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    darkroom_version: str
    outcome: Outcome
    # Rule codes, "minor-exposed", "unchecked", "sfw-exposed", "sfw-unchecked", or the generation error. A refusal of
    # an image-to-image input is "input image: " and the input's own reason; of a pose photo, "pose image: " and its
    # reason. A pose that could not be read or drawn is a failure: "pose: " and why.
    outcome_reason: str | None = None
    # True for illegal and minor-exposed deletions; false for everything else. Lets non-critical
    # records be pruned first when retention is decided.
    critical: bool = False
    request: RequestRecord
    guard: GuardRecord
    image: ImageRecord | None = None
    # Set when the user removed their own output (``DELETE /record/{id}``): the file is gone, the record stays.
    removed_at: datetime | None = None
    # A depth map of the stored image, when one was asked for. null otherwise, and again once the image is removed.
    depth: DepthRecord | None = None
    # Pose requests only: whether the released image holds the pose it was asked for. null without a pose, when the
    # request turned the check off (pose.verify), and for an image that was not released (it is sent nowhere).
    pose_adherence: PoseAdherenceRecord | None = None


def render_wire_schema() -> str:
    """The canonical committed-schema string (``schema`` writes it, ``schema --check`` compares)."""
    return render_schema([GenerationRecord], title=WIRE_TITLE)
