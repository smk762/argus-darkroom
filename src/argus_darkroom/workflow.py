"""The ComfyUI workflow template: loaded, checked, and rendered for one run.

A template is a ComfyUI **API-format** graph, ``{node_id: {"class_type": str, "inputs": {...}}}``, with
placeholder strings where a run's values go. The placeholders are argus-proof's, so a proof template
works here once its ``SaveImage`` is swapped for ``SaveImageWebsocket``:

* scalars: ``$base_checkpoint`` ``$vae`` ``$positive`` ``$negative`` ``$seed`` ``$steps`` ``$cfg``
  ``$guidance`` ``$sampler`` ``$scheduler`` ``$width`` ``$height`` ``$clip_skip`` (sent as ComfyUI's negative
  ``stop_at_clip_layer``, never above -2: see :func:`clip_layer`);
* LoRAs: ``$lora`` / ``$lora_weight`` for the first, ``$lora_2`` / ``$lora_2_weight`` for the second, and so on;
* image-to-image: ``$init_image`` (the checked input, as base64, the ``image`` input of the one
  ``DarkroomLoadImageBase64`` node) and ``$denoise``. A template has both or neither.

Only a value that is exactly a placeholder is replaced, so a prompt is never interpolated into other text and
ints and floats stay typed.

**One template or a set.** ``ARGUS_DARKROOM_WORKFLOW`` names a file, used for every checkpoint, or a directory of
``<family>[+lora<N>][+img2img].json`` files (:func:`load_workflows`). A run then gets the template for its
checkpoint's family (the checkpoint's top-level folder under ``checkpoints/``), its number of LoRAs and whether it has
an input image, or a :class:`WorkflowError` when there is none: never a template made for another family, LoRA count or
mode.

**Nothing is written before the verdict.** The one output node is ``SaveImageWebsocket``, which sends the
image over the websocket and writes nothing. A template with any other node that writes to disk is refused
**at load**, not when it runs (see :func:`load_workflow`).

**Every input image is checked.** The only way an image gets into a run is ``$init_image``, which darkroom fills
with an input it has scanned. A node that loads an image from ComfyUI's own disk (``LoadImage`` and the like) would
bring in one nobody checked, so it is refused at load too.
"""

from __future__ import annotations

import copy
import hashlib
import json
import posixpath
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from argus_darkroom.models import LoRARef, SamplingParams

SAVE_NODE = "SaveImageWebsocket"
# The image-to-image loader (``comfyui/darkroom_load_image_base64.py``): base64 in, IMAGE out, no disk.
LOADER_NODE = "DarkroomLoadImageBase64"
INIT_IMAGE = "$init_image"
DENOISE = "$denoise"

# Any node whose class name says it saves, previews, exports or writes is refused. That covers ComfyUI's own
# (SaveImage, PreviewImage, SaveAnimatedWEBP, SaveLatent, SaveVideo, ...) and custom nodes that follow the usual
# naming. ComfyUI is also asked at run time whether any other node is an output node
# (``ComfyUIClient.output_classes``), for the ones the name does not give away.
_WRITING_WORDS = ("save", "preview", "export", "write")

_LORA = re.compile(r"^\$lora(?:_(\d+))?(_weight)?$")
# A file in a workflow directory: ``sdxl.json``, ``sdxl+lora1.json``, ``flux-klein.json``.
_SET_MEMBER = re.compile(r"^(?P<family>[^+/]+?)(?:\+lora(?P<loras>[1-9]\d*))?(?P<img2img>\+img2img)?$")


class WorkflowError(ValueError):
    """The workflow template cannot be used."""


def writes_to_disk(class_type: str) -> bool:
    """Whether *class_type* is, or by its name looks like, a node that writes a file."""
    if class_type == SAVE_NODE:
        return False
    lowered = class_type.casefold()
    return any(word in lowered for word in _WRITING_WORDS)


# A node whose class name says it loads an image, a latent or video frames from ComfyUI's disk (LoadImage,
# LoadImageMask, LoadLatent, LoadVideo, VHS_LoadVideo, ...) brings in pixels darkroom never checked.
_LOADING_WORDS = ("loadimage", "loadlatent", "loadvideo")


def loads_unchecked_image(class_type: str) -> bool:
    """Whether *class_type* is, or by its name looks like, a node that loads an image darkroom did not check."""
    lowered = class_type.casefold().replace("_", "")
    return class_type != LOADER_NODE and any(word in lowered for word in _LOADING_WORDS)


@dataclass(frozen=True)
class Workflow:
    template: dict
    sha256: str
    save_node_id: str
    # The file's stem, recorded with every run that uses it.
    name: str = "workflow"
    # The checkpoint family it is for. None: every checkpoint (a single-file ARGUS_DARKROOM_WORKFLOW).
    family: str | None = None
    lora_slots: int = 0
    # The scalar placeholders it fills, so a client can tell which request values it uses.
    slots: frozenset[str] = frozenset()
    # Image-to-image: it renders from an input image ($init_image, $denoise).
    img2img: bool = False

    @property
    def class_types(self) -> set[str]:
        return {node["class_type"] for node in self.template.values()}

    def describe(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "loras": self.lora_slots,
            "img2img": self.img2img,
            "slots": sorted(self.slots),
        }


def checkpoint_family(checkpoint: str) -> str | None:
    """The checkpoint's family: its top-level folder under ``checkpoints/``. None for a file at the top.

    The name is normalised first, as the models root resolves it, so ``sdxl/../flux/f`` is in ``flux``, not ``sdxl``.
    """
    head, sep, _ = posixpath.normpath(checkpoint).partition("/")
    return head if sep and head else None


@dataclass(frozen=True)
class WorkflowSet:
    """The loaded templates. :meth:`select` picks the one for a run."""

    workflows: tuple[Workflow, ...]

    def select(self, checkpoint: str, loras: int, img2img: bool = False) -> Workflow:
        """The template for *checkpoint* with *loras* LoRAs, image-to-image or not. Raises :class:`WorkflowError`
        when none fits."""
        family = checkpoint_family(checkpoint)
        for workflow in self.workflows:
            if workflow.family in (None, family) and workflow.lora_slots == loras and workflow.img2img == img2img:
                return workflow
        if family is None and all(w.family is not None for w in self.workflows):
            raise WorkflowError(
                f"checkpoint {checkpoint!r} is not in a family folder (checkpoints/<family>/...), so no workflow "
                "can be chosen for it"
            )
        counts = self.lora_counts(checkpoint, img2img)
        kind = "image-to-image workflow" if img2img else "workflow"
        offered = f"it renders with {', '.join(map(str, counts))} LoRA(s)" if counts else f"it has no {kind}"
        what = f"the {family!r} family" if family else "this checkpoint"
        raise WorkflowError(f"no {kind} renders {what} with {loras} LoRA(s): {offered}")

    def lora_counts(self, checkpoint: str, img2img: bool = False) -> list[int]:
        """The LoRA counts *checkpoint* can be rendered with, image-to-image or not. Empty when no workflow fits."""
        family = checkpoint_family(checkpoint)
        return sorted({w.lora_slots for w in self.workflows if w.family in (None, family) and w.img2img == img2img})

    @property
    def names(self) -> list[str]:
        return [w.name for w in self.workflows]


@dataclass(frozen=True)
class RenderParams:
    positive: str
    negative: str
    seed: int
    sampling: SamplingParams
    checkpoint: str
    vae: str | None = None
    loras: tuple[LoRARef, ...] = ()
    # Image-to-image: the checked input as base64, and the denoise. None for text-to-image.
    init_image: str | None = None
    denoise: float | None = None

    def __repr__(self) -> str:  # never the input's pixels, in a log or a traceback
        init = None if self.init_image is None else f"<{len(self.init_image)} base64 chars>"
        return f"RenderParams(seed={self.seed}, checkpoint={self.checkpoint!r}, init_image={init})"


def template_sha256(template: dict) -> str:
    """The sha256 of the template in canonical form, so formatting changes do not change the identity."""
    return hashlib.sha256(json.dumps(template, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_workflow(
    template: object, source: str = "workflow", name: str = "workflow", family: str | None = None
) -> Workflow:
    """Check *template* and wrap it. Raises :class:`WorkflowError`; see :func:`load_workflow`."""
    if not isinstance(template, dict) or not template:
        raise WorkflowError(f"{source} is not a non-empty ComfyUI API-format graph")
    save_ids: list[str] = []
    loader_ids: list[str] = []
    for node_id, node in template.items():
        if not isinstance(node, dict) or not isinstance(node.get("class_type"), str):
            raise WorkflowError(f"{source}: node {node_id!r} has no class_type")
        if not isinstance(node.get("inputs", {}), dict):
            raise WorkflowError(f"{source}: node {node_id!r} has no inputs object")
        class_type = node["class_type"]
        if writes_to_disk(class_type):
            raise WorkflowError(
                f"{source}: node {node_id!r} ({class_type}) writes to disk. Nothing may be written before the "
                f"verdict; use {SAVE_NODE}"
            )
        if loads_unchecked_image(class_type):
            raise WorkflowError(
                f"{source}: node {node_id!r} ({class_type}) loads an image darkroom never checked. An input image "
                f"reaches a run only through {LOADER_NODE} and {INIT_IMAGE}"
            )
        if class_type == SAVE_NODE:
            save_ids.append(str(node_id))
        if class_type == LOADER_NODE:
            loader_ids.append(str(node_id))
    if len(save_ids) != 1:
        raise WorkflowError(f"{source} needs exactly one {SAVE_NODE} node (found {len(save_ids)})")

    placeholders = {v for node in template.values() for v in node.get("inputs", {}).values() if isinstance(v, str)}
    if "$positive" not in placeholders:
        # A checked prompt that never reaches the sampler would make every check meaningless.
        raise WorkflowError(f"{source} has no $positive slot, so the checked prompt would not be what is rendered")
    # $negative is optional (Flux ignores one): a run with a negative prompt is refused by a template without it.
    names: set[int] = set()
    weights: set[int] = set()
    for value in placeholders:
        if match := _LORA.match(value):
            (weights if match.group(2) else names).add(int(match.group(1) or 1))
    lora_slots = len(names)
    if names != weights or names != set(range(1, lora_slots + 1)):
        raise WorkflowError(
            f"{source}: LoRA slots must be $lora, $lora_2, ... with no gaps, each with its _weight slot"
        )
    img2img = _check_input_slots(template, source, placeholders, loader_ids)
    return Workflow(
        template=template,
        sha256=template_sha256(template),
        save_node_id=save_ids[0],
        name=name,
        family=family,
        lora_slots=lora_slots,
        slots=frozenset(placeholders & _SCALAR_VALUES.keys()),
        img2img=img2img,
    )


def _check_input_slots(template: dict, source: str, placeholders: set[str], loader_ids: list[str]) -> bool:
    """Whether the template is image-to-image, once its input slots are known to be sound.

    An image-to-image template has exactly one :data:`LOADER_NODE`, whose ``image`` is ``$init_image``, and that is
    the only ``$init_image`` in it; and it has a ``$denoise``. A text-to-image template has neither slot and no
    loader (a loader with a fixed image would bring in one nobody checked).
    """
    uses = [
        (node_id, key)
        for node_id, node in template.items()
        for key, value in node.get("inputs", {}).items()
        if value == INIT_IMAGE
    ]
    if INIT_IMAGE not in placeholders:
        if loader_ids:
            raise WorkflowError(f"{source}: a {LOADER_NODE} node needs {INIT_IMAGE} as its image")
        if DENOISE in placeholders:
            raise WorkflowError(f"{source} has a {DENOISE} slot but no {INIT_IMAGE}: it is not image-to-image")
        return False
    if len(loader_ids) != 1:
        raise WorkflowError(
            f"{source}: an image-to-image template needs exactly one {LOADER_NODE} (found {len(loader_ids)})"
        )
    if uses != [(loader_ids[0], "image")]:
        raise WorkflowError(f"{source}: {INIT_IMAGE} must appear once, as the image of the {LOADER_NODE} node")
    if DENOISE not in placeholders:
        raise WorkflowError(f"{source} is image-to-image but has no {DENOISE} slot")
    return True


def load_workflow(path: str | Path, family: str | None = None) -> Workflow:
    """Load a template and refuse it if it could write anything before the verdict.

    Refused: any node that writes to disk (``SaveImage``, ``PreviewImage``, other ``Save*`` / ``*Preview*`` /
    ``*Export*`` / ``*Write*`` nodes), any node that loads an image other than :data:`LOADER_NODE`, no
    :data:`SAVE_NODE` or more than one, a template with no ``$positive`` slot, LoRA slots with a gap or a name without
    its weight, and image-to-image slots that are not exactly one loader with ``$init_image`` and a ``$denoise``.
    """
    path = Path(path)
    try:
        template = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkflowError(f"cannot read workflow {path}: {exc}") from exc
    return parse_workflow(template, str(path), name=path.stem, family=family)


def load_workflows(path: str | Path) -> WorkflowSet:
    """A single template for every checkpoint (*path* is a file), or the set in a directory.

    In a directory each ``*.json`` must be named ``<family>.json`` (no LoRA slots) or ``<family>+lora<N>.json``
    (exactly N), with ``+img2img`` at the end for an image-to-image template (and only for one), and every one must
    load: a misnamed or broken template refuses the whole set, not just itself.
    Other files are ignored, and so are hidden ones (``._flux.json``, which macOS leaves on a mounted volume).
    """
    path = Path(path)
    if not path.is_dir():
        return WorkflowSet((load_workflow(path),))
    found = []
    for file in sorted(path.glob("*.json")):
        if file.name.startswith("."):
            continue
        match = _SET_MEMBER.match(file.stem)
        if match is None:
            raise WorkflowError(
                f"{file}: a workflow in a directory is named <family>.json or <family>+lora<N>.json, with +img2img "
                "at the end for an image-to-image one"
            )
        workflow = load_workflow(file, family=match.group("family"))
        expected = int(match.group("loras") or 0)
        if workflow.lora_slots != expected:
            raise WorkflowError(f"{file} has {workflow.lora_slots} LoRA slot(s) but its name says {expected}")
        named_img2img = match.group("img2img") is not None
        if workflow.img2img != named_img2img:
            has = f"has {INIT_IMAGE}" if workflow.img2img else f"has no {INIT_IMAGE}"
            says = "says image-to-image (+img2img)" if named_img2img else "does not say +img2img"
            raise WorkflowError(f"{file} {has} but its name {says}")
        found.append(workflow)
    if not found:
        raise WorkflowError(f"{path} holds no workflow (*.json)")
    return WorkflowSet(tuple(sorted(found, key=lambda w: (w.family or "", w.img2img, w.lora_slots))))


def clip_layer(clip_skip: int) -> int:
    """ComfyUI's ``stop_at_clip_layer`` for *clip_skip*, read as A1111 reads it for SDXL: 1 and 2 are both -2.

    SDXL conditions on the penultimate text-encoder layer, which is ComfyUI's own default. -1 would run the last
    layer, which SDXL never uses, and some checkpoints ship it as all NaN: the render then comes back blank.
    """
    return -max(clip_skip, 2)


# Every scalar placeholder and where a run's value for it comes from: the one list both a template's ``slots`` and
# :func:`render` go by.
_SCALAR_VALUES: dict[str, Callable[[RenderParams], object]] = {
    "$base_checkpoint": lambda p: p.checkpoint,
    "$vae": lambda p: p.vae,
    "$positive": lambda p: p.positive,
    "$negative": lambda p: p.negative,
    "$seed": lambda p: p.seed,
    "$steps": lambda p: p.sampling.steps,
    "$cfg": lambda p: p.sampling.cfg,
    "$guidance": lambda p: p.sampling.guidance,
    "$sampler": lambda p: p.sampling.sampler,
    "$scheduler": lambda p: p.sampling.scheduler,
    "$width": lambda p: p.sampling.width,
    "$height": lambda p: p.sampling.height,
    "$clip_skip": lambda p: clip_layer(p.sampling.clip_skip),
    INIT_IMAGE: lambda p: p.init_image,
    DENOISE: lambda p: p.denoise,
}


def _scalars(params: RenderParams) -> dict[str, object]:
    return {slot: value(params) for slot, value in _SCALAR_VALUES.items()}


def render(workflow: Workflow, params: RenderParams) -> dict:
    """The concrete graph for *params*.

    Raises :class:`WorkflowError` when a slot has no value, when the template's LoRA and VAE slots do not
    line up one-for-one with what was asked for, or when the run sets a value the template would ignore (a
    negative prompt, ``guidance``, a ``clip_skip`` above 2), so nothing is silently dropped while the record
    claims it was used.
    """
    scalars = _scalars(params)
    loras = {i: lora for i, lora in enumerate(params.loras, start=1)}
    seen: set[str] = set()
    seen_names: set[int] = set()
    seen_weights: set[int] = set()

    def resolve(value: object) -> object:
        if not isinstance(value, str) or not value.startswith("$"):
            return value
        if value in scalars:
            if scalars[value] is None:
                raise WorkflowError(f"the workflow uses {value} but this run has no value for it")
            seen.add(value)
            return scalars[value]
        match = _LORA.match(value)
        if match:
            index, is_weight = int(match.group(1) or 1), bool(match.group(2))
            lora = loras.get(index)
            if lora is None:
                raise WorkflowError(f"the workflow uses {value} but this run has {len(loras)} LoRA(s)")
            (seen_weights if is_weight else seen_names).add(index)
            return lora.weight if is_weight else lora.name
        return value  # an unknown "$..." string is a literal

    graph = copy.deepcopy(workflow.template)
    for node in graph.values():
        inputs = node.get("inputs", {})
        for key, value in inputs.items():
            inputs[key] = resolve(value)

    unused = []
    for index in loras:
        stem = "$lora" if index == 1 else f"$lora_{index}"
        if index not in seen_names:
            unused.append(stem)
        if index not in seen_weights:
            unused.append(f"{stem}_weight")
    if unused:
        raise WorkflowError(
            f"the workflow has no slot for {', '.join(unused)}: every LoRA needs a name and a weight slot"
        )
    if params.init_image is not None and INIT_IMAGE not in seen:
        raise WorkflowError(f"this run has an input image but the workflow has no {INIT_IMAGE} slot")
    if params.denoise is not None and DENOISE not in seen:
        raise WorkflowError(f"this run sets denoise but the workflow has no {DENOISE} slot")
    if params.vae is not None and "$vae" not in seen:
        raise WorkflowError("this run sets a VAE but the workflow has no $vae slot")
    if params.negative.strip() and "$negative" not in seen:
        raise WorkflowError("this run has a negative prompt but the workflow has no $negative slot")
    if params.sampling.guidance is not None and "$guidance" not in seen:
        raise WorkflowError("this run sets guidance but the workflow has no $guidance slot")
    if clip_layer(params.sampling.clip_skip) != clip_layer(1) and "$clip_skip" not in seen:
        raise WorkflowError("this run sets clip_skip above 2 but the workflow has no $clip_skip slot")
    return graph
