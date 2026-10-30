from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from argus_darkroom.models import LoRARef, SamplingParams
from argus_darkroom.workflow import RenderParams, WorkflowError, load_workflow, parse_workflow, render, writes_to_disk

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "workflows"
SHA = "a" * 64
SAMPLING = SamplingParams(sampler="euler", scheduler="normal", steps=20, cfg=7.0, clip_skip=2, width=512, height=768)


def params(**kw) -> RenderParams:
    base = {"positive": "a cat", "negative": "blurry", "seed": 5, "sampling": SAMPLING, "checkpoint": "m.safetensors"}
    return RenderParams(**(base | kw))


def graph(**extra: dict) -> dict:
    g = json.loads((EXAMPLES / "sdxl.json").read_text())
    g.update(extra)
    return g


@pytest.mark.parametrize("name", ["sdxl.json", "sdxl_lora.json"])
def test_the_shipped_examples_load(name) -> None:
    wf = load_workflow(EXAMPLES / name)
    assert wf.save_node_id == "9"
    assert len(wf.sha256) == 64


# --- refused at load -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "class_type",
    [
        "SaveImage",
        "PreviewImage",
        "SaveAnimatedWEBP",
        "SaveLatent",
        "SaveVideo",
        "ImageSaveToDisk",
        "MyImageWriter",
        "ExportMesh",
    ],
)
def test_a_disk_writing_node_is_refused_at_load(tmp_path, class_type) -> None:
    template = graph(**{"20": {"class_type": class_type, "inputs": {"images": ["8", 0]}}})
    path = tmp_path / "w.json"
    path.write_text(json.dumps(template))
    with pytest.raises(WorkflowError, match="writes to disk"):
        load_workflow(path)


def test_replacing_the_save_node_with_save_image_is_refused() -> None:
    template = graph()
    template["9"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "x", "images": ["8", 0]}}
    with pytest.raises(WorkflowError, match="writes to disk"):
        parse_workflow(template)


def test_the_websocket_node_itself_is_not_disk_writing() -> None:
    assert not writes_to_disk("SaveImageWebsocket")
    assert not writes_to_disk("KSampler")


def test_exactly_one_websocket_save_node_is_needed() -> None:
    none = graph()
    del none["9"]
    with pytest.raises(WorkflowError, match="exactly one"):
        parse_workflow(none)
    two = graph(**{"12": {"class_type": "SaveImageWebsocket", "inputs": {"images": ["8", 0]}}})
    with pytest.raises(WorkflowError, match="exactly one"):
        parse_workflow(two)


@pytest.mark.parametrize("slot", ["$positive", "$negative"])
def test_a_prompt_slot_is_required(slot) -> None:
    template = graph()
    for node in template.values():
        for key, value in node["inputs"].items():
            if value == slot:
                node["inputs"][key] = "hard coded"
    with pytest.raises(WorkflowError, match="checked prompt"):
        parse_workflow(template)


@pytest.mark.parametrize("content", ["not json", "[]", "{}", '{"1": {"inputs": {}}}', '{"1": "x"}'])
def test_malformed_templates_are_refused(tmp_path, content) -> None:
    path = tmp_path / "w.json"
    path.write_text(content)
    with pytest.raises(WorkflowError):
        load_workflow(path)


def test_a_missing_file_is_refused(tmp_path) -> None:
    with pytest.raises(WorkflowError, match="cannot read"):
        load_workflow(tmp_path / "nope.json")


def test_sha_ignores_key_order_and_formatting() -> None:
    a = parse_workflow(graph())
    b = parse_workflow(dict(reversed(list(graph().items()))))
    assert a.sha256 == b.sha256
    changed = graph()
    changed["3"]["inputs"]["denoise"] = 0.9
    assert parse_workflow(changed).sha256 != a.sha256


# --- render --------------------------------------------------------------------------------------------


def test_render_fills_typed_values() -> None:
    out = render(parse_workflow(graph()), params())
    assert out["3"]["inputs"] | {"model": 0} == {
        "seed": 5, "steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
        "model": 0, "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0],
    }  # fmt: skip
    assert out["6"]["inputs"]["text"] == "a cat"
    assert out["7"]["inputs"]["text"] == "blurry"
    assert out["5"]["inputs"] == {"width": 512, "height": 768, "batch_size": 1}
    assert out["11"]["inputs"]["stop_at_clip_layer"] == -2
    assert out["4"]["inputs"]["ckpt_name"] == "m.safetensors"


def test_render_does_not_change_the_template() -> None:
    wf = parse_workflow(graph())
    before = copy.deepcopy(wf.template)
    render(wf, params())
    assert wf.template == before


def test_a_prompt_that_looks_like_a_placeholder_is_text_not_a_slot() -> None:
    out = render(parse_workflow(graph()), params(positive="$seed", negative="$positive $lora"))
    assert out["6"]["inputs"]["text"] == "$seed"
    assert out["7"]["inputs"]["text"] == "$positive $lora"
    assert out["3"]["inputs"]["seed"] == 5


def test_loras_fill_name_and_weight() -> None:
    wf = load_workflow(EXAMPLES / "sdxl_lora.json")
    out = render(wf, params(loras=(LoRARef(name="l.safetensors", sha256=SHA, weight=0.7),)))
    assert out["10"]["inputs"]["lora_name"] == "l.safetensors"
    assert out["10"]["inputs"]["strength_model"] == out["10"]["inputs"]["strength_clip"] == 0.7


def test_a_lora_cannot_be_silently_dropped() -> None:
    lora = LoRARef(name="l", sha256=SHA)
    with pytest.raises(WorkflowError, match=r"no slot for \$lora, \$lora_weight"):
        render(parse_workflow(graph()), params(loras=(lora,)))
    with pytest.raises(WorkflowError, match=r"no slot for \$lora_2"):
        render(load_workflow(EXAMPLES / "sdxl_lora.json"), params(loras=(lora, lora)))


def test_a_slot_with_no_lora_is_refused() -> None:
    with pytest.raises(WorkflowError, match="0 LoRA"):
        render(load_workflow(EXAMPLES / "sdxl_lora.json"), params())


def test_a_vae_the_template_ignores_is_refused() -> None:
    with pytest.raises(WorkflowError, match="no \\$vae slot"):
        render(parse_workflow(graph()), params(vae="v.safetensors"))


def test_a_vae_slot_with_no_vae_is_refused() -> None:
    template = graph()
    template["8"]["inputs"]["vae"] = "$vae"
    with pytest.raises(WorkflowError, match="no value"):
        render(parse_workflow(template), params())
    assert render(parse_workflow(template), params(vae="v.safetensors"))["8"]["inputs"]["vae"] == "v.safetensors"
