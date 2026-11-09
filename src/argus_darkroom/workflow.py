"""The ComfyUI workflow template: loaded, checked, and rendered for one run.

A template is a ComfyUI **API-format** graph, ``{node_id: {"class_type": str, "inputs": {...}}}``, with
placeholder strings where a run's values go. The placeholders are argus-proof's, so a proof template
works here once its ``SaveImage`` is swapped for ``SaveImageWebsocket``:

* scalars: ``$base_checkpoint`` ``$vae`` ``$positive`` ``$negative`` ``$seed`` ``$steps`` ``$cfg``
  ``$guidance`` ``$sampler`` ``$scheduler`` ``$width`` ``$height`` ``$clip_skip`` (sent as ComfyUI's negative
  ``stop_at_clip_layer``, never above -2: see :func:`clip_layer`);
* LoRAs: ``$lora`` / ``$lora_weight`` for the first, ``$lora_2`` / ``$lora_2_weight`` for the second, and so on.

Only a value that is exactly a placeholder is replaced, so a prompt is never interpolated into other text and
ints and floats stay typed.

**One template or a set.** ``ARGUS_DARKROOM_WORKFLOW`` names a file, used for every checkpoint, or a directory of
``<family>.json`` and ``<family>+lora<N>.json`` files (:func:`load_workflows`). A run then gets the template for its
checkpoint's family (the checkpoint's top-level folder under ``checkpoints/``) and its number of LoRAs, or a
:class:`WorkflowError` when there is none: never a template made for another family or LoRA count.

**Nothing is written before the verdict.** The one output node is ``SaveImageWebsocket``, which sends the
image over the websocket and writes nothing. A template with any other node that writes to disk is refused
**at load**, not when it runs (see :func:`load_workflow`).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from argus_darkroom.models import LoRARef, SamplingParams

SAVE_NODE = "SaveImageWebsocket"

# Any node whose class name says it saves, previews, exports or writes is refused. That covers ComfyUI's own
# (SaveImage, PreviewImage, SaveAnimatedWEBP, SaveLatent, SaveVideo, ...) and custom nodes that follow the usual
# naming. ComfyUI is also asked at run time whether any other node is an output node
# (``ComfyUIClient.output_classes``), for the ones the name does not give away.
_WRITING_WORDS = ("save", "preview", "export", "write")

_LORA = re.compile(r"^\$lora(?:_(\d+))?(_weight)?$")
# A file in a workflow directory: ``sdxl.json``, ``sdxl+lora1.json``, ``flux-klein.json``.
_SET_MEMBER = re.compile(r"^(?P<family>[^+/]+?)(?:\+lora(?P<loras>[1-9]\d*))?$")
SCALARS = (
    "$base_checkpoint", "$vae", "$positive", "$negative", "$seed", "$steps", "$cfg", "$guidance",
    "$sampler", "$scheduler", "$width", "$height", "$clip_skip",
)  # fmt: skip


class WorkflowError(ValueError):
    """The workflow template cannot be used."""


def writes_to_disk(class_type: str) -> bool:
    """Whether *class_type* is, or by its name looks like, a node that writes a file."""
    if class_type == SAVE_NODE:
        return False
    lowered = class_type.casefold()
    return any(word in lowered for word in _WRITING_WORDS)


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

    @property
    def class_types(self) -> set[str]:
        return {node["class_type"] for node in self.template.values()}

    def describe(self) -> dict:
        return {"name": self.name, "family": self.family, "loras": self.lora_slots, "slots": sorted(self.slots)}


def checkpoint_family(checkpoint: str) -> str | None:
    """The checkpoint's family: its top-level folder under ``checkpoints/``. None for a file at the top."""
    head, sep, _ = checkpoint.partition("/")
    return head if sep and head else None


@dataclass(frozen=True)
class WorkflowSet:
    """The loaded templates. :meth:`select` picks the one for a run."""

    workflows: tuple[Workflow, ...]

    def select(self, checkpoint: str, loras: int) -> Workflow:
        """The template for *checkpoint* with *loras* LoRAs. Raises :class:`WorkflowError` when none fits."""
        family = checkpoint_family(checkpoint)
        for workflow in self.workflows:
            if workflow.family in (None, family) and workflow.lora_slots == loras:
                return workflow
        if family is None and all(w.family is not None for w in self.workflows):
            raise WorkflowError(
                f"checkpoint {checkpoint!r} is not in a family folder (checkpoints/<family>/...), so no workflow "
                "can be chosen for it"
            )
        counts = self.lora_counts(checkpoint)
        offered = f"it renders with {', '.join(map(str, counts))} LoRA(s)" if counts else "it has no workflow"
        what = f"the {family!r} family" if family else "this checkpoint"
        raise WorkflowError(f"no workflow renders {what} with {loras} LoRA(s): {offered}")

    def lora_counts(self, checkpoint: str) -> list[int]:
        """The LoRA counts *checkpoint* can be rendered with. Empty when no workflow fits it."""
        family = checkpoint_family(checkpoint)
        return sorted({w.lora_slots for w in self.workflows if w.family in (None, family)})

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
        if class_type == SAVE_NODE:
            save_ids.append(str(node_id))
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
    return Workflow(
        template=template,
        sha256=template_sha256(template),
        save_node_id=save_ids[0],
        name=name,
        family=family,
        lora_slots=lora_slots,
        slots=frozenset(placeholders & set(SCALARS)),
    )


def load_workflow(path: str | Path, family: str | None = None) -> Workflow:
    """Load a template and refuse it if it could write anything before the verdict.

    Refused: any node that writes to disk (``SaveImage``, ``PreviewImage``, other ``Save*`` / ``*Preview*`` /
    ``*Export*`` / ``*Write*`` nodes), no :data:`SAVE_NODE` or more than one, a template with no ``$positive``
    slot, and LoRA slots with a gap or a name without its weight.
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
    (exactly N), and every one must load: a misnamed or broken template refuses the whole set, not just itself.
    Other files are ignored.
    """
    path = Path(path)
    if not path.is_dir():
        return WorkflowSet((load_workflow(path),))
    found = []
    for file in sorted(path.glob("*.json")):
        match = _SET_MEMBER.match(file.stem)
        if match is None:
            raise WorkflowError(f"{file}: a workflow in a directory is named <family>.json or <family>+lora<N>.json")
        workflow = load_workflow(file, family=match.group("family"))
        expected = int(match.group("loras") or 0)
        if workflow.lora_slots != expected:
            raise WorkflowError(f"{file} has {workflow.lora_slots} LoRA slot(s) but its name says {expected}")
        found.append(workflow)
    if not found:
        raise WorkflowError(f"{path} holds no workflow (*.json)")
    return WorkflowSet(tuple(sorted(found, key=lambda w: (w.family or "", w.lora_slots))))


def clip_layer(clip_skip: int) -> int:
    """ComfyUI's ``stop_at_clip_layer`` for *clip_skip*, read as A1111 reads it for SDXL: 1 and 2 are both -2.

    SDXL conditions on the penultimate text-encoder layer, which is ComfyUI's own default. -1 would run the last
    layer, which SDXL never uses, and some checkpoints ship it as all NaN: the render then comes back blank.
    """
    return -max(clip_skip, 2)


def _scalars(params: RenderParams) -> dict[str, object]:
    s = params.sampling
    return {
        "$base_checkpoint": params.checkpoint,
        "$vae": params.vae,
        "$positive": params.positive,
        "$negative": params.negative,
        "$seed": params.seed,
        "$steps": s.steps,
        "$cfg": s.cfg,
        "$guidance": s.guidance,
        "$sampler": s.sampler,
        "$scheduler": s.scheduler,
        "$width": s.width,
        "$height": s.height,
        "$clip_skip": clip_layer(s.clip_skip),
    }


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
    if params.vae is not None and "$vae" not in seen:
        raise WorkflowError("this run sets a VAE but the workflow has no $vae slot")
    if params.negative.strip() and "$negative" not in seen:
        raise WorkflowError("this run has a negative prompt but the workflow has no $negative slot")
    if params.sampling.guidance is not None and "$guidance" not in seen:
        raise WorkflowError("this run sets guidance but the workflow has no $guidance slot")
    if clip_layer(params.sampling.clip_skip) != clip_layer(1) and "$clip_skip" not in seen:
        raise WorkflowError("this run sets clip_skip above 2 but the workflow has no $clip_skip slot")
    return graph
