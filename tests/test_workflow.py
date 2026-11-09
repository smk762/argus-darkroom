from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from argus_darkroom.models import LoRARef, SamplingParams
from argus_darkroom.workflow import (
    RenderParams,
    WorkflowError,
    checkpoint_family,
    load_workflow,
    load_workflows,
    parse_workflow,
    render,
    writes_to_disk,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "workflows"
SHA = "a" * 64
SDXL_SHA = "02e08f842f7a66e0e1be2ebc506ab8339f3b39df639beb86fc04dc240f3ef38d"
SDXL_LORA_SHA = "63365ba8795f7cc69b5ebb10d3d6d548abb7623ef1ce392f61af70dabe2b7e0b"
SAMPLING = SamplingParams(sampler="euler", scheduler="normal", steps=20, cfg=7.0, clip_skip=2, width=512, height=768)


def params(**kw) -> RenderParams:
    base = {"positive": "a cat", "negative": "blurry", "seed": 5, "sampling": SAMPLING, "checkpoint": "m.safetensors"}
    return RenderParams(**(base | kw))


def graph(**extra: dict) -> dict:
    g = json.loads((EXAMPLES / "sdxl.json").read_text())
    g.update(extra)
    return g


@pytest.mark.parametrize("name", sorted(p.name for p in EXAMPLES.glob("*.json")))
def test_the_shipped_examples_load(name) -> None:
    wf = load_workflow(EXAMPLES / name)
    assert wf.save_node_id == "9"
    assert len(wf.sha256) == 64


def test_the_shipped_examples_load_as_a_set_keyed_by_family_and_lora_count() -> None:
    found = {(w.name, w.family, w.lora_slots) for w in load_workflows(EXAMPLES).workflows}
    assert found == {
        ("sdxl", "sdxl", 0), ("sdxl+lora1", "sdxl", 1), ("sdxl+lora2", "sdxl", 2),
        ("flux", "flux", 0), ("flux+lora1", "flux", 1), ("flux+lora2", "flux", 2),
        ("flux-klein", "flux-klein", 0), ("sd35", "sd35", 0),
    }  # fmt: skip


def test_the_sdxl_examples_keep_their_identity() -> None:
    # Records name these hashes: changing a shipped template changes what an old record replays.
    assert load_workflow(EXAMPLES / "sdxl.json").sha256 == SDXL_SHA
    assert load_workflow(EXAMPLES / "sdxl+lora1.json").sha256 == SDXL_LORA_SHA


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


def without(slot: str) -> dict:
    template = graph()
    for node in template.values():
        for key, value in node["inputs"].items():
            if value == slot:
                node["inputs"][key] = "hard coded"
    return template


def test_the_positive_slot_is_required() -> None:
    with pytest.raises(WorkflowError, match="checked prompt"):
        parse_workflow(without("$positive"))


def test_without_a_negative_slot_a_negative_prompt_is_refused() -> None:
    wf = parse_workflow(without("$negative"))
    assert render(wf, params(negative=""))["7"]["inputs"]["text"] == "hard coded"
    assert render(wf, params(negative="  "))  # blank is no negative prompt
    with pytest.raises(WorkflowError, match="no \\$negative slot"):
        render(wf, params(negative="blurry"))


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


@pytest.mark.parametrize(("clip_skip", "layer"), [(1, -2), (2, -2), (3, -3), (12, -12)])
def test_clip_skip_never_reaches_the_last_text_encoder_layer(clip_skip, layer) -> None:
    sampling = SAMPLING.model_copy(update={"clip_skip": clip_skip})
    assert render(parse_workflow(graph()), params(sampling=sampling))["11"]["inputs"]["stop_at_clip_layer"] == layer


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
    wf = load_workflow(EXAMPLES / "sdxl+lora1.json")
    out = render(wf, params(loras=(LoRARef(name="l.safetensors", sha256=SHA, weight=0.7),)))
    assert out["10"]["inputs"]["lora_name"] == "l.safetensors"
    assert out["10"]["inputs"]["strength_model"] == out["10"]["inputs"]["strength_clip"] == 0.7


def test_a_lora_cannot_be_silently_dropped() -> None:
    lora = LoRARef(name="l", sha256=SHA)
    with pytest.raises(WorkflowError, match=r"no slot for \$lora, \$lora_weight"):
        render(parse_workflow(graph()), params(loras=(lora,)))
    with pytest.raises(WorkflowError, match=r"no slot for \$lora_2"):
        render(load_workflow(EXAMPLES / "sdxl+lora1.json"), params(loras=(lora, lora)))


def test_a_slot_with_no_lora_is_refused() -> None:
    with pytest.raises(WorkflowError, match="0 LoRA"):
        render(load_workflow(EXAMPLES / "sdxl+lora1.json"), params())


def test_a_vae_the_template_ignores_is_refused() -> None:
    with pytest.raises(WorkflowError, match="no \\$vae slot"):
        render(parse_workflow(graph()), params(vae="v.safetensors"))


def test_a_vae_slot_with_no_vae_is_refused() -> None:
    template = graph()
    template["8"]["inputs"]["vae"] = "$vae"
    with pytest.raises(WorkflowError, match="no value"):
        render(parse_workflow(template), params())
    assert render(parse_workflow(template), params(vae="v.safetensors"))["8"]["inputs"]["vae"] == "v.safetensors"


# --- LoRA slots ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inputs",
    [
        {"lora_name": "$lora_2", "strength_model": "$lora_2_weight"},  # no first LoRA
        {"lora_name": "$lora"},  # no weight
        {"strength_model": "$lora_weight"},  # a weight with no name
    ],
)
def test_lora_slots_with_a_gap_or_a_missing_half_are_refused(inputs) -> None:
    with pytest.raises(WorkflowError, match="LoRA slots"):
        parse_workflow(graph(**{"20": {"class_type": "LoraLoader", "inputs": inputs}}))


def test_a_template_knows_its_lora_slots_and_scalar_slots() -> None:
    wf = load_workflow(EXAMPLES / "sdxl+lora2.json")
    assert wf.lora_slots == 2
    assert wf.describe() == {
        "name": "sdxl+lora2",
        "family": None,
        "loras": 2,
        "slots": [
            "$base_checkpoint", "$cfg", "$clip_skip", "$height", "$negative", "$positive", "$sampler", "$scheduler",
            "$seed", "$steps", "$width",
        ],
    }  # fmt: skip
    assert {"$guidance", "$positive"} <= load_workflow(EXAMPLES / "flux.json").slots
    assert "$negative" not in load_workflow(EXAMPLES / "flux.json").slots


# --- a set of workflows --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("checkpoint", "family"),
    [
        ("sdxl/a.safetensors", "sdxl"),
        ("sdxl/sub/a.safetensors", "sdxl"),
        ("flux-klein/k.safetensors", "flux-klein"),
        ("a.safetensors", None),
        ("/a.safetensors", None),
    ],
)
def test_the_family_is_the_top_level_folder(checkpoint, family) -> None:
    assert checkpoint_family(checkpoint) == family


@pytest.mark.parametrize(
    ("checkpoint", "loras", "name"),
    [
        ("sdxl/a.safetensors", 0, "sdxl"),
        ("sdxl/deep/a.safetensors", 2, "sdxl+lora2"),
        ("flux/f.safetensors", 1, "flux+lora1"),
        ("flux-klein/k.safetensors", 0, "flux-klein"),
        ("sd35/s.safetensors", 0, "sd35"),
    ],
)
def test_select_picks_by_family_and_lora_count(checkpoint, loras, name) -> None:
    assert load_workflows(EXAMPLES).select(checkpoint, loras).name == name


@pytest.mark.parametrize(
    ("checkpoint", "loras", "message"),
    [
        ("sdxl/a.safetensors", 3, "'sdxl' family with 3 LoRA.*renders with 0, 1, 2"),
        ("sd35/s.safetensors", 1, "'sd35' family with 1 LoRA.*renders with 0 LoRA"),
        ("illustrious/a.safetensors", 0, "'illustrious' family.*no workflow"),
        ("a.safetensors", 0, "not in a family folder"),
    ],
)
def test_no_substitute_when_nothing_fits(checkpoint, loras, message) -> None:
    with pytest.raises(WorkflowError, match=message):
        load_workflows(EXAMPLES).select(checkpoint, loras)


def test_a_single_file_serves_every_checkpoint_with_its_lora_count() -> None:
    one = load_workflows(EXAMPLES / "sdxl+lora1.json")
    assert one.names == ["sdxl+lora1"]
    assert one.select("a.safetensors", 1).family is None
    assert one.select("flux/f.safetensors", 1).name == "sdxl+lora1"  # as before: the operator chose one template
    assert one.lora_counts("x.safetensors") == [1]
    with pytest.raises(WorkflowError, match="0 LoRA"):
        one.select("a.safetensors", 0)


def write_set(tmp_path, **files: dict) -> object:
    for name, template in files.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(template))
    return tmp_path


@pytest.mark.parametrize("name", ["sdxl+lora", "sdxl+lora0", "sdxl+foo", "+lora1", "sdxl+lora1+lora2"])
def test_a_misnamed_member_refuses_the_set(tmp_path, name) -> None:
    with pytest.raises(WorkflowError, match="named <family>.json"):
        load_workflows(write_set(tmp_path, sdxl=graph(), **{name: graph()}))


def test_a_name_that_disagrees_with_the_slots_refuses_the_set(tmp_path) -> None:
    lora = json.loads((EXAMPLES / "sdxl+lora1.json").read_text())
    with pytest.raises(WorkflowError, match="1 LoRA slot.*says 0"):
        load_workflows(write_set(tmp_path, sdxl=lora))
    with pytest.raises(WorkflowError, match="0 LoRA slot.*says 2"):
        load_workflows(write_set(tmp_path, **{"flux+lora2": graph()}))


def test_one_broken_member_refuses_the_whole_set(tmp_path) -> None:
    writes = graph(**{"20": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}}})
    with pytest.raises(WorkflowError, match="writes to disk"):
        load_workflows(write_set(tmp_path, sdxl=graph(), flux=writes))


def test_an_empty_directory_is_refused_and_other_files_are_ignored(tmp_path) -> None:
    with pytest.raises(WorkflowError, match="no workflow"):
        load_workflows(tmp_path)
    (tmp_path / "README.md").write_text("notes")
    assert load_workflows(write_set(tmp_path, sdxl=graph())).names == ["sdxl"]


# --- values a template has no slot for -----------------------------------------------------------------


def flux_params(**kw) -> RenderParams:
    sampling = SAMPLING.model_copy(update={"guidance": 3.5, "cfg": 1.0, "clip_skip": 1})
    return params(**({"negative": "", "sampling": sampling} | kw))


def test_flux_fills_guidance_and_takes_no_negative_prompt() -> None:
    out = render(load_workflow(EXAMPLES / "flux.json"), flux_params())
    assert out["14"] == {"class_type": "FluxGuidance", "inputs": {"conditioning": ["6", 0], "guidance": 3.5}}
    assert out["6"]["inputs"]["text"] == "a cat"
    with pytest.raises(WorkflowError, match="no \\$negative slot"):
        render(load_workflow(EXAMPLES / "flux.json"), flux_params(negative="blurry"))


def test_a_guidance_slot_needs_a_value() -> None:
    sampling = SAMPLING.model_copy(update={"clip_skip": 1})
    with pytest.raises(WorkflowError, match="\\$guidance but this run has no value"):
        render(load_workflow(EXAMPLES / "flux.json"), flux_params(sampling=sampling))


def test_guidance_without_a_slot_is_refused() -> None:
    with pytest.raises(WorkflowError, match="no \\$guidance slot"):
        render(parse_workflow(graph()), params(sampling=SAMPLING.model_copy(update={"guidance": 3.5})))


@pytest.mark.parametrize(("clip_skip", "allowed"), [(1, True), (2, True), (3, False)])
def test_clip_skip_without_a_slot_is_refused_only_when_it_would_change_something(clip_skip, allowed) -> None:
    sampling = SAMPLING.model_copy(update={"guidance": 3.5, "clip_skip": clip_skip})
    run = lambda: render(load_workflow(EXAMPLES / "flux.json"), flux_params(sampling=sampling))  # noqa: E731
    if allowed:
        run()
    else:
        with pytest.raises(WorkflowError, match="no \\$clip_skip slot"):
            run()
