"""The ComfyUI workflow template: loaded, checked, and rendered for one run.

A template is a ComfyUI **API-format** graph, ``{node_id: {"class_type": str, "inputs": {...}}}``, with
placeholder strings where a run's values go. The placeholders are argus-proof's, so a proof template
works here once its ``SaveImage`` is swapped for ``SaveImageWebsocket``:

* scalars: ``$base_checkpoint`` ``$vae`` ``$positive`` ``$negative`` ``$seed`` ``$steps`` ``$cfg``
  ``$sampler`` ``$scheduler`` ``$width`` ``$height`` ``$clip_skip`` (sent as ComfyUI's negative
  ``stop_at_clip_layer``);
* LoRAs: ``$lora`` / ``$lora_weight`` for the first, ``$lora_2`` / ``$lora_2_weight`` for the second, and so on.

Only a value that is exactly a placeholder is replaced, so a prompt is never interpolated into other text and
ints and floats stay typed.

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

    @property
    def class_types(self) -> set[str]:
        return {node["class_type"] for node in self.template.values()}


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


def parse_workflow(template: object, source: str = "workflow") -> Workflow:
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
    for needed in ("$positive", "$negative"):
        if needed not in placeholders:
            # A checked prompt that never reaches the sampler would make every check meaningless.
            raise WorkflowError(f"{source} has no {needed} slot, so the checked prompt would not be what is rendered")
    return Workflow(template=template, sha256=template_sha256(template), save_node_id=save_ids[0])


def load_workflow(path: str | Path) -> Workflow:
    """Load a template and refuse it if it could write anything before the verdict.

    Refused: any node that writes to disk (``SaveImage``, ``PreviewImage``, other ``Save*`` / ``*Preview*`` /
    ``*Export*`` / ``*Write*`` nodes), no :data:`SAVE_NODE` or more than one, and a template with no
    ``$positive`` or ``$negative`` slot.
    """
    path = Path(path)
    try:
        template = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkflowError(f"cannot read workflow {path}: {exc}") from exc
    return parse_workflow(template, str(path))


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
        "$sampler": s.sampler,
        "$scheduler": s.scheduler,
        "$width": s.width,
        "$height": s.height,
        "$clip_skip": -s.clip_skip,  # CLIPSetLastLayer wants a negative index
    }


def render(workflow: Workflow, params: RenderParams) -> dict:
    """The concrete graph for *params*.

    Raises :class:`WorkflowError` when a slot has no value, or when the template's LoRA and VAE slots do not
    line up one-for-one with what was asked for, so a LoRA, its weight or a VAE cannot be silently dropped
    while the record claims it was used.
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
    return graph
