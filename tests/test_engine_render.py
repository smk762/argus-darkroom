"""Engine.render: models resolved and hashed, the checked prompt rendered, nothing written."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fake_comfyui import FakeComfyUI, event

from argus_darkroom.comfyui import ComfyUIClient, GenerationError
from argus_darkroom.config import Settings
from argus_darkroom.engine import Engine
from argus_darkroom.guard import Guard
from argus_darkroom.hashing import sha256_cached
from argus_darkroom.models import SamplingParams
from argus_darkroom.request import GenerateRequest, RequestError
from argus_darkroom.workflow import WorkflowError

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "workflows"
SAMPLING = SamplingParams(sampler="euler", scheduler="normal", steps=4, cfg=5.0, width=64, height=64)


@pytest.fixture
def models(tmp_path) -> Path:
    root = tmp_path / "models"
    for kind, name, content in [
        ("checkpoints", "base.safetensors", b"checkpoint bytes"),
        ("checkpoints", "sub/other.safetensors", b"other"),
        ("loras", "style.safetensors", b"lora bytes"),
        ("vae", "v.safetensors", b"vae bytes"),
    ]:
        (root / kind / name).parent.mkdir(parents=True, exist_ok=True)
        (root / kind / name).write_bytes(content)
    (tmp_path / "secret.safetensors").write_bytes(b"outside")
    return root


def make(fake: FakeComfyUI, tmp_path, models, workflow="sdxl_lora.json", **kw) -> Engine:
    settings = Settings(
        singguard_url="http://127.0.0.1:1",
        output_root=tmp_path / "out",
        models_root=models,
        workflow=EXAMPLES / workflow,
        nudenet_url="http://nudenet.invalid",
        **kw,
    )
    return Engine(settings, guard=Guard("http://127.0.0.1:1"), comfyui=ComfyUIClient(fake.url, ws_url=fake.ws_url))


def request(**kw) -> GenerateRequest:
    base = {
        "user_prompt": "a cat",
        "negative_prompt": "blurry",
        "seed": 11,
        "sampling": SAMPLING,
        "checkpoint": "base.safetensors",
        "loras": [{"name": "style.safetensors", "weight": 0.6, "trigger_words": ["sks style"]}],
    }
    return GenerateRequest(**(base | kw))


def test_render_returns_the_image_in_memory_and_a_record_that_reproduces_it(comfyui, tmp_path, models) -> None:
    eng = make(comfyui, tmp_path, models)
    out = eng.render(request())
    assert out.image == comfyui.png
    rec = out.request
    assert rec.mode == "sfw"
    assert rec.user_prompt == "a cat"
    assert rec.final_prompt == "a cat, sks style"
    assert rec.negative_prompt == "blurry"
    assert rec.seed == 11
    assert rec.sampling == SAMPLING
    assert (rec.checkpoint.name, rec.checkpoint.sha256) == (
        "base.safetensors",
        hashlib.sha256(b"checkpoint bytes").hexdigest(),
    )
    assert [(lora.name, lora.weight, lora.sha256) for lora in rec.loras] == [
        ("style.safetensors", 0.6, hashlib.sha256(b"lora bytes").hexdigest())
    ]
    assert rec.vae is None
    assert rec.workflow_sha256 == eng.workflow.sha256
    assert rec.engine == "comfyui" and rec.engine_version == "0.3.test"


def test_comfyui_receives_exactly_the_prompt_that_was_checked(comfyui, tmp_path, models) -> None:
    eng = make(comfyui, tmp_path, models)
    req = request(user_prompt="a cat")
    out = eng.render(req, expanded_prompt="a cat on a rug")
    graph = comfyui.graphs[0]
    assert graph["6"]["inputs"]["text"] == eng.final_prompt(req, "a cat on a rug") == out.request.final_prompt
    assert out.request.expanded_prompt == "a cat on a rug"
    assert graph["7"]["inputs"]["text"] == "blurry"


def test_replaying_a_records_parameters_sends_the_same_graph(comfyui, tmp_path, models) -> None:
    eng = make(comfyui, tmp_path, models)
    first = eng.render(request(seed=None))
    assert isinstance(first.request.seed, int)
    again = request(seed=first.request.seed)
    eng.render(again)
    assert comfyui.graphs[0] == comfyui.graphs[1]


def test_models_are_hashed_once_and_found_in_subfolders_and_with_a_vae(comfyui, tmp_path, models) -> None:
    eng = make(comfyui, tmp_path, models)
    template = json.loads((EXAMPLES / "sdxl_lora.json").read_text())
    template["8"]["inputs"]["vae"] = "$vae"
    path = tmp_path / "vae.json"
    path.write_text(json.dumps(template))
    eng.workflow = type(eng.workflow)(template, eng.workflow.sha256, eng.workflow.save_node_id)
    out = eng.render(request(checkpoint="sub/other.safetensors", vae="v.safetensors"))
    assert out.request.vae.sha256 == sha256_cached(models / "vae" / "v.safetensors")
    assert comfyui.graphs[0]["8"]["inputs"]["vae"] == "v.safetensors"
    assert comfyui.graphs[0]["4"]["inputs"]["ckpt_name"] == "sub/other.safetensors"


@pytest.mark.parametrize(
    "bad",
    [
        {"checkpoint": "missing.safetensors"},
        {"checkpoint": "../secret.safetensors"},
        {"checkpoint": "../../secret.safetensors"},
        {"checkpoint": "/etc/passwd"},
        {"checkpoint": "sub"},
        {"loras": [{"name": "nope.safetensors"}]},
        {"vae": "missing.safetensors"},
    ],
)
def test_a_missing_or_escaping_model_fails_before_comfyui_is_asked_anything(comfyui, tmp_path, models, bad) -> None:
    with pytest.raises(RequestError, match="not found"):
        make(comfyui, tmp_path, models).render(request(**bad))
    assert comfyui.calls == []


@pytest.mark.parametrize("bad", [{"sampling": None}, {"checkpoint": None}])
def test_sampling_and_checkpoint_are_required(comfyui, tmp_path, models, bad) -> None:
    with pytest.raises(RequestError, match="sampling parameters and a checkpoint"):
        make(comfyui, tmp_path, models).render(request(**bad))
    assert comfyui.calls == []


def test_no_models_root_is_a_request_error(comfyui, tmp_path) -> None:
    with pytest.raises(RequestError, match="MODELS_ROOT"):
        make(comfyui, tmp_path, None).render(request())


def test_no_workflow_is_a_request_error(comfyui, tmp_path, models) -> None:
    eng = make(comfyui, tmp_path, models)
    eng.workflow = None
    with pytest.raises(RequestError, match="no workflow"):
        eng.render(request())


def test_sfw_without_nudenet_never_reaches_comfyui(comfyui, tmp_path, models) -> None:
    eng = make(comfyui, tmp_path, models)
    eng.settings = Settings(singguard_url="http://127.0.0.1:1", output_root=tmp_path / "out", models_root=models)
    with pytest.raises(RequestError, match="NudeNet"):
        eng.render(request())
    assert comfyui.calls == []


def test_a_workflow_that_writes_is_refused_when_the_engine_loads_it(comfyui, tmp_path, models) -> None:
    template = json.loads((EXAMPLES / "sdxl.json").read_text())
    template["20"] = {"class_type": "PreviewImage", "inputs": {"images": ["8", 0]}}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(template))
    settings = Settings(singguard_url="http://127.0.0.1:1", output_root=tmp_path / "out", workflow=path)
    with pytest.raises(WorkflowError, match="writes to disk"):
        Engine(settings)
    assert comfyui.calls == []


def test_comfyui_is_asked_which_nodes_are_outputs_and_a_surprise_is_refused(tmp_path, models) -> None:
    # "KSampler" is not named like a writer, but this ComfyUI says it is an output node.
    with FakeComfyUI(output_nodes=frozenset({"SaveImageWebsocket", "KSampler"})) as fake:
        with pytest.raises(GenerationError, match=r"output node\(s\) other than SaveImageWebsocket: KSampler"):
            make(fake, tmp_path, models).render(request())
        assert ("POST", "/prompt") not in fake.calls


def test_an_unknown_node_is_refused_before_a_run(tmp_path, models) -> None:
    with FakeComfyUI(unknown_nodes=frozenset({"LoraLoader"})) as fake:
        with pytest.raises(GenerationError, match="does not know the node 'LoraLoader'"):
            make(fake, tmp_path, models).render(request())
        assert ("POST", "/prompt") not in fake.calls


def test_the_output_check_runs_once_per_workflow(comfyui, tmp_path, models) -> None:
    eng = make(comfyui, tmp_path, models)
    eng.render(request())
    eng.render(request())
    assert sum(path.startswith("/object_info/KSampler") for _, path in comfyui.calls) == 1


def test_rendering_leaves_nothing_on_disk_or_in_comfyui_history(comfyui, tmp_path, models) -> None:
    eng = make(comfyui, tmp_path, models)
    before = sorted(p.name for p in tmp_path.rglob("*") if models not in p.parents and p != models)
    eng.render(request())
    assert comfyui.history == {}
    assert sorted(p.name for p in tmp_path.rglob("*") if models not in p.parents and p != models) == before
    assert list((tmp_path / "out").iterdir()) == []


def test_a_failed_generation_leaves_no_history(tmp_path, models) -> None:
    def script(fake, pid, graph):
        return [event("execution_error", prompt_id=pid, node_type="KSampler", exception_message="boom")]

    with FakeComfyUI(script) as fake:
        with pytest.raises(GenerationError, match="boom"):
            make(fake, tmp_path, models).render(request())
        assert fake.history == {}
