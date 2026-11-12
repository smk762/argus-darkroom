"""The HTTP server, over Starlette's TestClient, against the same fakes as the engine tests."""

from __future__ import annotations

import base64
import io
import json
import threading
import zipfile
from dataclasses import replace

import pytest
from argus_cortex.moderation import policies as policy_table
from fake_comfyui import FakeComfyUI, event, grey_png
from fakes import FakeFaces, safe_answer, unsafe_answer
from PIL import Image
from rig import EXAMPLES, ILLEGAL_A, SAMPLING, Rig, answers, is_image_scan
from starlette.testclient import TestClient

from argus_darkroom.comfyui import ComfyUIClient
from argus_darkroom.config import ConfigError, Settings
from argus_darkroom.modelfiles import ModelLibrary, pick_default
from argus_darkroom.prompt import frame
from argus_darkroom.server import create_app
from argus_darkroom.server.app import POLICY_SUMMARIES

BODY = {
    "user_prompt": "a cat",
    "negative_prompt": "blurry",
    "seed": 3,
    "sampling": SAMPLING.model_dump(),
    "checkpoint": "base.safetensors",
    "loras": [{"name": "style.safetensors", "trigger_words": ["sks"]}],
}
EVIL = {"Origin": "https://evil.example"}


def client(rig: Rig, **settings) -> TestClient:
    rig.settings = replace(rig.settings, **settings)
    return TestClient(create_app(rig.engine()))


def stream_events(c: TestClient, body: dict | None = None) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    with c.stream("POST", "/generate/stream", json=BODY | (body or {})) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        kind = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                kind = line.removeprefix("event: ")
            elif line.startswith("data: "):
                events.append((kind, json.loads(line.removeprefix("data: "))))
    return events


def generate(c: TestClient, body: dict | None = None) -> dict:
    """Run one generation and return the record from the ``complete`` event."""
    events = stream_events(c, body)
    assert events[-1][0] == "complete"
    return events[-1][1]["record"]


# --- start up and info ---------------------------------------------------------------------------------


def test_an_empty_singguard_url_stops_the_server_starting(monkeypatch) -> None:
    monkeypatch.delenv("ARGUS_DARKROOM_SINGGUARD_URL", raising=False)
    with pytest.raises(ConfigError, match="unguarded"):
        create_app()


def test_health(rig) -> None:
    body = client(rig).get("/health").json()
    # configured and answering; an unconfigured Ollama is simply not up. `age` is whether the age check can run.
    assert body["backends"] == {"comfyui": True, "singguard": True, "nudenet": True, "ollama": False, "age": False}
    assert body["default_policy"] == "default" and body["thinking"] == "cascade" and body["generating"] is False
    assert body["nsfw_allowed"] is False and body["devmode"] is False and body["sfw_available"] is True
    assert body["read_only"] is False
    assert body["policies"] == list(policy_table.POLICIES)
    assert body["age_unavailable"]
    assert "127.0.0.1" not in json.dumps(body)  # whether each answers, never where


def test_health_age_is_up_when_curator_faces_answers(rig) -> None:
    with FakeFaces() as fake:
        body = client(rig, age_url=fake.url).get("/health").json()
    assert body["backends"]["age"] is True
    assert body["age_unavailable"] is None


def test_health_says_what_is_down(rig) -> None:
    body = client(rig, nudenet_url="http://127.0.0.1:1", comfyui_url="http://127.0.0.1:1").get("/health").json()
    assert body["backends"]["nudenet"] is False and body["backends"]["comfyui"] is False
    assert body["backends"]["singguard"] is True


def test_policies_mark_nsfw_as_not_enforceable_without_the_flag(rig) -> None:
    body = client(rig).get("/policies").json()
    by_name = {p["name"]: p for p in body["policies"]}
    assert body["default"] == "default"
    assert by_name["default"]["mode"] == "sfw" and by_name["default"]["enforceable"] is True
    assert by_name["adult-granular"]["enforceable"] is False
    assert by_name["adult-granular"]["illegal_rules"] == [
        {"code": "A", "name": "Minors"},
        {"code": "H", "name": "Non-Consensual Sexual Content"},
        {"code": "I", "name": "Sexual Violence"},
    ]
    assert all(p["summary"] for p in body["policies"])
    assert set(policy_table.POLICIES) <= set(POLICY_SUMMARIES)
    assert "legal adult entertainment" in POLICY_SUMMARIES["adult-cinematic"]
    assert "rule G covers how-to" in POLICY_SUMMARIES["adult-cinematic"]
    assert by_name["default"]["illegal_rules"] == [] and by_name["default"]["text"] is None
    assert "Minors" in by_name["adult-granular"]["text"]


def test_policies_with_the_flag(rig) -> None:
    body = client(rig, allow_nsfw=True, policy="adult-granular").get("/policies").json()
    assert body["default"] == "adult-granular"
    assert all(p["enforceable"] for p in body["policies"])


def test_models(rig) -> None:
    assert client(rig).get("/models").json() == {
        # One workflow file (not a directory) serves every checkpoint, family or not.
        "checkpoints": [
            {
                "name": "base.safetensors",
                "family": None,
                "lora_counts": [1],
                "img2img_lora_counts": [],
                "inpaint_lora_counts": [],
            }
        ],
        "default_checkpoint": None,  # nothing under sdxl/ in the rig
        "loras": [{"name": "style.safetensors"}],
        "vaes": [],
        "workflows": [
            {
                "name": "sdxl+lora1",
                "family": None,
                "loras": 1,
                "mode": "txt2img",
                "img2img": False,
                "slots": [
                    "$base_checkpoint",
                    "$cfg",
                    "$clip_skip",
                    "$height",
                    "$negative",
                    "$positive",
                    "$sampler",
                    "$scheduler",
                    "$seed",
                    "$steps",
                    "$width",
                ],
            }
        ],
        "samplers": ["euler", "dpmpp_2m"],  # what ComfyUI's KSampler accepts
        "schedulers": ["normal", "karras"],
    }


def test_models_and_health_report_the_workflow_set(rig) -> None:
    for name in ["sdxl/a.safetensors", "flux/f.safetensors", "illustrious/i.safetensors"]:
        path = rig.root / "models" / "checkpoints" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    c = client(rig, workflow=EXAMPLES)
    body = c.get("/models").json()
    per_mode = ("img2img_lora_counts", "inpaint_lora_counts")
    assert [{k: v for k, v in c.items() if k not in per_mode} for c in body["checkpoints"]] == [
        {"name": "base.safetensors", "family": None, "lora_counts": []},  # not in a family folder: cannot render
        {"name": "flux/f.safetensors", "family": "flux", "lora_counts": [0, 1, 2]},
        {"name": "illustrious/i.safetensors", "family": "illustrious", "lora_counts": []},
        {"name": "sdxl/a.safetensors", "family": "sdxl", "lora_counts": [0, 1, 2]},
    ]
    assert [c["img2img_lora_counts"] for c in body["checkpoints"]] == [[], [], [], [0, 1]]
    assert [c["inpaint_lora_counts"] for c in body["checkpoints"]] == [[], [], [], [0, 1]]
    by_name = {w["name"]: w for w in body["workflows"]}
    assert set(by_name) == {
        "sdxl",
        "sdxl+lora1",
        "sdxl+lora2",
        "sdxl+img2img",
        "sdxl+lora1+img2img",
        "sdxl+inpaint",
        "sdxl+lora1+inpaint",
        "flux",
        "flux+lora1",
        "flux+lora2",
        "flux-klein",
        "sd35",
    }
    assert by_name["flux"]["family"] == "flux" and "$guidance" in by_name["flux"]["slots"]
    assert "$negative" not in by_name["flux"]["slots"]
    assert {n: (w["mode"], w["img2img"]) for n, w in by_name.items() if w["mode"] != "txt2img"} == {
        "sdxl+img2img": ("img2img", True),
        "sdxl+lora1+img2img": ("img2img", True),
        "sdxl+inpaint": ("inpaint", False),
        "sdxl+lora1+inpaint": ("inpaint", False),
    }
    assert sorted(c.get("/health").json()["workflows"]) == sorted(by_name)

    r = c.post(
        "/generate/stream",
        json=BODY | {"checkpoint": "sdxl/a.safetensors", "loras": [{"name": "style.safetensors"}] * 3},
    )
    assert r.status_code == 400 and "'sdxl' family with 3 LoRA" in r.json()["detail"]
    assert not c.app.state.busy.locked()


def test_models_names_the_configured_default(rig) -> None:
    assert client(rig, default_checkpoint="base.safetensors").get("/models").json()["default_checkpoint"] == (
        "base.safetensors"
    )


@pytest.mark.parametrize(
    ("preferred", "expected"),
    [
        ("sdxl/b.safetensors", "sdxl/b.safetensors"),  # an exact name wins
        ("sdxl", "sdxl/a.safetensors"),  # a folder gives its first model
        ("sdxl/", "sdxl/a.safetensors"),
        ("sd", None),  # a folder, not a prefix of one
        ("missing", None),
        ("", None),
        ("/", None),
    ],
)
def test_pick_default(preferred: str, expected: str | None) -> None:
    names = ["flux/a.safetensors", "sdxl/a.safetensors", "sdxl/b.safetensors"]
    assert pick_default(names, preferred) == expected


def test_models_skips_the_parts_of_a_diffusers_folder(tmp_path) -> None:
    root = tmp_path / "checkpoints"
    for name in ["whole.safetensors", "sdxl/a/a.safetensors", "sdxl/parts/unet/m.safetensors", "sdxl/parts/m.bin"]:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"x")
    (root / "sdxl/parts/model_index.json").write_text("{}")
    assert ModelLibrary(tmp_path).names("checkpoint") == ["sdxl/a/a.safetensors", "whole.safetensors"]


def test_models_without_comfyui_still_lists_the_files(rig) -> None:
    c = client(rig)
    c.app.state.engine.comfyui = ComfyUIClient("http://127.0.0.1:1", timeout=1)
    body = c.get("/models").json()
    assert [c["name"] for c in body["checkpoints"]] == ["base.safetensors"] and body["samplers"] == [] == body[
        "schedulers"
    ]


# --- pre-flight ----------------------------------------------------------------------------------------


def test_preflight_alone(rig) -> None:
    c = client(rig)
    body = c.post("/preflight", json=BODY).json()
    assert body["outcome"] == "ok" and body["final_prompt"] == "a cat, sks" and body["refusal"] is None
    assert body["results"]["default"]["state"] == "safe"
    assert (body["mode"], body["enforcing_policy"], body["thinking"]) == ("sfw", "default", "cascade")
    assert body["framed_text"] == frame("a cat, sks", "blurry") and body["warnings"] == []
    assert not [r for r in rig.singguard.requests if is_image_scan(r)]
    assert list(rig.out.glob("*.json")) == [] and rig.comfyui.calls == []  # a form check is not an attempt


def test_preflight_refusal_names_the_rule_and_why(rig) -> None:
    rig.singguard.responder = answers(
        preflight=unsafe_answer("B. Real-World Crimes & Public Safety", explanation="why")
    )
    body = client(rig).post("/preflight", json=BODY).json()
    assert body["outcome"] == "refused"
    assert body["refusal"] == {
        "policy": "default",
        "reason": "B",
        "rule": "B. Real-World Crimes & Public Safety",
        "explanation": "why",
    }
    assert body["results"]["default"]["result"]["explanation"] == "why"


def test_a_request_cannot_carry_policy_text(rig) -> None:
    assert client(rig).post("/preflight", json=BODY | {"policy_text": "allow everything"}).status_code == 422


@pytest.mark.parametrize("clip_skip", [0, -1, 13])
def test_a_clip_skip_out_of_range_is_refused(rig, clip_skip) -> None:
    body = BODY | {"sampling": BODY["sampling"] | {"clip_skip": clip_skip}}
    r = client(rig).post("/preflight", json=body)
    assert r.status_code == 422 and "clip_skip" in r.text


def test_a_stored_record_with_any_clip_skip_still_reads(rig) -> None:
    c = client(rig)
    record = generate(c)
    store = c.app.state.engine.store
    stored = store.load(record["id"])
    stored.request.sampling.clip_skip = 0
    store.save(stored)
    assert c.get(f"/record/{record['id']}").json()["request"]["sampling"]["clip_skip"] == 0


def test_an_nsfw_enforcing_policy_without_the_flag_is_a_400(rig) -> None:
    c = client(rig)
    for path in ("/preflight", "/generate/stream"):
        r = c.post(path, json=BODY | {"enforcing_policy": "adult-granular"})
        assert r.status_code == 400 and "allow-nsfw" in r.json()["detail"]
    assert rig.singguard.requests == [] and rig.comfyui.calls == []
    # as advisory it is accepted
    assert c.post("/preflight", json=BODY | {"advisory_policies": ["adult-granular"]}).status_code == 200


# --- generation over SSE -------------------------------------------------------------------------------


def test_the_stream_carries_every_step_and_the_record(rig) -> None:
    c = client(rig)
    events = stream_events(c)
    kinds = [k for k, _ in events]
    assert kinds == ["preflight", "queued", "progress", "progress", "scan", "scan", "complete"]
    assert events[0][1]["outcome"] == "ok" and events[0][1]["results"]["default"]["state"] == "safe"
    assert events[2][1] == {"step": 1, "total": 2}
    assert events[-3][1] == {
        "policy": "default",
        "context": "isolated",
        "cell": {"state": "safe", "result": events[-3][1]["cell"]["result"], "error": None},
    }
    record = events[-1][1]["record"]
    assert record["outcome"] == "passed"
    assert c.get(f"/record/{record['id']}").json() == record
    image = c.get(f"/image/{record['id']}")
    assert image.status_code == 200 and image.content == rig.comfyui.png
    assert image.headers["content-type"] == "image/png" and "no-store" in image.headers["cache-control"]


def test_a_refusal_streams_preflight_and_complete_only(rig) -> None:
    rig.singguard.responder = answers(preflight=unsafe_answer("B. Real-World Crimes & Public Safety"))
    events = stream_events(client(rig))
    assert [k for k, _ in events] == ["preflight", "complete"]
    assert events[-1][1]["record"]["outcome"] == "refused"
    assert events[0][1]["refusal"]["reason"] == "B"
    assert rig.comfyui.calls == []


def test_a_request_error_is_a_400_before_the_stream_opens_and_frees_the_lock(rig) -> None:
    c = client(rig)
    assert c.post("/generate/stream", json=BODY | {"checkpoint": "gone.safetensors"}).status_code == 400
    assert c.post("/generate/stream", json=BODY | {"loras": []}).status_code == 400  # the template has a LoRA slot
    assert generate(c)["outcome"] == "passed"  # the lock was released each time


def test_sfw_generation_without_nudenet_is_a_400(tmp_path, singguard, comfyui, nudenet, monkeypatch) -> None:
    rig = Rig(tmp_path, singguard, comfyui, nudenet, nudenet_url="")
    c = TestClient(create_app(rig.engine()))
    r = c.post("/generate/stream", json=BODY)
    assert r.status_code == 400 and "NudeNet" in r.json()["detail"]
    assert c.get("/health").json()["sfw_available"] is False


def test_a_second_generation_gets_a_409(rig) -> None:
    c = client(rig)
    c.app.state.busy.acquire()
    try:
        r = c.post("/generate/stream", json=BODY)
        assert r.status_code == 409
        assert c.get("/health").json()["generating"] is True
    finally:
        c.app.state.busy.release()
    assert generate(c)["outcome"] == "passed"


def test_one_at_a_time_for_real(tmp_path, singguard, nudenet) -> None:
    release = threading.Event()

    def slow(fake, pid, graph):
        release.wait(10)
        return [
            event("executing", node="9", prompt_id=pid),
            b"\x00\x00\x00\x01\x00\x00\x00\x02" + fake.png,
            event("executing", node=None, prompt_id=pid),
        ]

    with FakeComfyUI(slow) as comfyui:
        rig = Rig(tmp_path, singguard, comfyui, nudenet)
        c = TestClient(create_app(rig.engine()))
        first: list = []
        t = threading.Thread(target=lambda: first.append(stream_events(c)))
        t.start()
        for _ in range(100):
            if c.get("/health").json()["generating"]:
                break
            threading.Event().wait(0.05)
        assert c.post("/generate/stream", json=BODY).status_code == 409
        release.set()
        t.join(10)
        assert first[0][-1][1]["record"]["outcome"] == "passed"
        assert generate(c)["outcome"] == "passed"


def test_an_unexpected_failure_is_an_error_event_and_frees_the_lock(rig, monkeypatch) -> None:
    c = client(rig)

    def boom(*a, **k):
        raise RuntimeError("secret internals")

    monkeypatch.setattr(c.app.state.engine.scanner, "scan", boom)
    events = stream_events(c)
    assert events[-1][0] == "error" and "secret internals" not in json.dumps(events[-1][1])
    assert not c.app.state.busy.locked()
    assert list(rig.out.glob("*.png")) == []  # nothing released


# --- images and records --------------------------------------------------------------------------------


def test_a_deleted_record_has_no_image_and_404_names_the_record(rig) -> None:
    c = client(rig, allow_nsfw=True, policy="adult-granular")
    rig.singguard.responder = answers(isolated=ILLEGAL_A)
    record = generate(c)
    assert record["outcome"] == "deleted"
    r = c.get(f"/image/{record['id']}")
    assert r.status_code == 404
    assert r.json() == {"detail": "this record has no stored image", "record_id": record["id"], "outcome": "deleted"}
    assert c.get(f"/record/{record['id']}").json()["image"]["sha256"]  # the hashes are still there


@pytest.mark.parametrize("outcome", ["omitted", "refused", "failed"])
def test_other_outcomes_without_an_image_answer_404(rig, outcome) -> None:
    c = client(rig)
    if outcome == "omitted":
        rig.nudenet.labels = {"FEMALE_BREAST_EXPOSED": 0.9}
    elif outcome == "refused":
        rig.singguard.responder = answers(preflight=unsafe_answer("B. Real-World Crimes & Public Safety"))
    else:
        rig.comfyui.reject_with = {"error": "bad", "node_errors": {"3": {}}}
    record = generate(c)
    assert record["outcome"] == outcome
    r = c.get(f"/image/{record['id']}")
    assert r.status_code == 404 and r.json()["record_id"] == record["id"]


@pytest.mark.parametrize("bad", ["nope", "../etc/passwd", "..%2f..%2fetc%2fpasswd", "a" * 32, "A" * 32, "%00"])
def test_ids_only_no_paths(rig, bad) -> None:
    c = client(rig)
    assert c.get(f"/image/{bad}").status_code == 404
    assert c.get(f"/record/{bad}").status_code == 404
    assert c.delete(f"/record/{bad}").status_code == 404


def test_records_are_paged_and_filtered(rig) -> None:
    c = client(rig)
    ids = [generate(c, {"seed": n})["id"] for n in range(3)]
    rig.singguard.responder = answers(preflight=unsafe_answer("B. Real-World Crimes & Public Safety"))
    refused = generate(c)["id"]
    page = c.get("/records").json()
    assert (page["total"], page["page"], page["page_size"]) == (4, 1, 50)
    assert page["records"][0]["id"] == refused  # newest first
    assert [r["id"] for r in c.get("/records", params={"outcome": "passed"}).json()["records"]] == ids[::-1]
    two = c.get("/records", params={"page_size": 2, "page": 2}).json()
    assert two["total"] == 4 and [r["id"] for r in two["records"]] == [r["id"] for r in page["records"][2:]]
    assert c.get("/records", params={"mode": "sfw"}).json()["total"] == 4
    assert c.get("/records", params={"mode": "nsfw"}).json()["total"] == 0
    assert c.get("/records", params={"mode": "both"}).status_code == 422
    assert c.get("/records", params={"page": 0}).status_code == 422
    assert c.get("/records", params={"outcome": "nonsense"}).status_code == 422
    assert c.get("/records", params={"page_size": 10_000}).status_code == 422


def test_the_user_can_remove_their_own_output_and_the_record_stays(rig) -> None:
    c = client(rig)
    record = generate(c)
    r = c.delete(f"/record/{record['id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["removed_at"] and body["image"]["path"] is None and body["image"]["sha256"] == record["image"]["sha256"]
    assert not (rig.out / f"{record['id']}.png").exists()
    assert (rig.out / f"{record['id']}.json").exists()
    assert c.get(f"/image/{record['id']}").status_code == 404
    assert c.delete(f"/record/{record['id']}").json() == body  # again: nothing more to do


def test_there_is_nothing_to_remove_from_a_refused_record(rig) -> None:
    rig.singguard.responder = answers(preflight=unsafe_answer("B. Real-World Crimes & Public Safety"))
    c = client(rig)
    assert c.delete(f"/record/{generate(c)['id']}").status_code == 409


# --- archive -------------------------------------------------------------------------------------------


def test_the_archive_holds_every_record_and_only_the_stored_images(rig) -> None:
    c = client(rig, allow_nsfw=True, policy="adult-granular")
    passed = generate(c)
    rig.singguard.responder = answers(isolated=ILLEGAL_A)
    deleted = generate(c)
    r = c.get("/archive")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert "attachment" in r.headers["content-disposition"]
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert sorted(names) == sorted(
        [f"records/{passed['id']}.json", f"images/{passed['id']}.png", f"records/{deleted['id']}.json"]
    )
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert json.loads(zf.read(f"records/{deleted['id']}.json"))["outcome"] == "deleted"
    assert zf.read(f"images/{passed['id']}.png") == rig.comfyui.png
    assert zf.testzip() is None


def test_the_archive_is_always_available_and_read_only(rig) -> None:
    c = client(rig)  # no devmode
    assert c.get("/archive").status_code == 200
    assert c.post("/archive").status_code == 405


# --- purge ---------------------------------------------------------------------------------------------


def test_purge_does_not_exist_without_devmode(rig) -> None:
    c = client(rig)
    assert c.post("/purge", json={"confirm": "purge"}).status_code == 404
    assert c.post("/purge").status_code == 404
    assert c.get("/health").json()["devmode"] is False


def test_purge_needs_the_confirm_body(rig) -> None:
    c = client(rig, devmode=True)
    generate(c)
    for kwargs in ({}, {"json": {}}, {"json": {"confirm": "yes"}}, {"content": b"not json"}, {"json": ["purge"]}):
        assert c.post("/purge", **kwargs).status_code == 400
    assert c.get("/records").json()["total"] == 1


def test_purge_during_a_generation_is_a_409(rig) -> None:
    c = client(rig, devmode=True)
    generate(c)
    c.app.state.busy.acquire()
    try:
        assert c.post("/purge", json={"confirm": "purge"}).status_code == 409
    finally:
        c.app.state.busy.release()
    assert c.get("/records").json()["total"] == 1


def test_purging_one_record_does_not_exist_without_devmode(rig) -> None:
    c = client(rig)
    record = generate(c)
    assert c.post(f"/purge/{record['id']}", json={"confirm": "purge"}).status_code == 404
    assert c.get("/records").json()["total"] == 1


def test_purging_one_record_needs_the_confirm_body(rig) -> None:
    c = client(rig, devmode=True)
    record = generate(c)
    for kwargs in ({}, {"json": {}}, {"json": {"confirm": "yes"}}, {"content": b"not json"}):
        assert c.post(f"/purge/{record['id']}", **kwargs).status_code == 400
    assert c.get("/records").json()["total"] == 1


def test_purging_one_record_during_a_generation_is_a_409(rig) -> None:
    c = client(rig, devmode=True)
    record = generate(c)
    c.app.state.busy.acquire()
    try:
        assert c.post(f"/purge/{record['id']}", json={"confirm": "purge"}).status_code == 409
    finally:
        c.app.state.busy.release()
    assert c.get("/records").json()["total"] == 1


def test_purging_one_record_removes_it_and_its_image_and_nothing_else(rig) -> None:
    c = client(rig, devmode=True)
    gone = generate(c, {"seed": 1})["id"]
    kept = generate(c, {"seed": 2})["id"]
    r = c.post(f"/purge/{gone}", json={"confirm": "purge"})
    assert r.status_code == 200 and r.json() == {"records": 1, "images": 1}
    assert c.get(f"/record/{gone}").status_code == 404
    assert c.get(f"/image/{gone}").status_code == 404
    assert [record["id"] for record in c.get("/records").json()["records"]] == [kept]
    assert c.get(f"/image/{kept}").status_code == 200
    assert c.get("/health").json()["generating"] is False


def test_purging_one_record_with_no_image_counts_none(rig) -> None:
    c = client(rig, devmode=True)
    rig.singguard.responder = answers(preflight=unsafe_answer("B. Real-World Crimes & Public Safety"))
    refused = generate(c)
    assert refused["outcome"] == "refused"
    assert c.post(f"/purge/{refused['id']}", json={"confirm": "purge"}).json() == {"records": 1, "images": 0}


def test_purging_one_record_removes_a_record_that_no_longer_reads(rig) -> None:
    c = client(rig, devmode=True)
    record = generate(c)
    store = c.app.state.engine.store
    store.record_path(record["id"]).write_text("not a record", encoding="utf-8")
    assert c.post(f"/purge/{record['id']}", json={"confirm": "purge"}).json() == {"records": 1, "images": 1}
    assert not store.record_path(record["id"]).exists() and not store.image_path(record["id"]).exists()


def test_purging_one_record_removes_what_a_crash_left(rig) -> None:
    c = client(rig, devmode=True)
    record = generate(c)
    store = c.app.state.engine.store
    tmp = store.record_path(record["id"]).with_suffix(".json.tmp")
    tmp.write_text("{}", encoding="utf-8")
    assert c.post(f"/purge/{record['id']}", json={"confirm": "purge"}).status_code == 200
    assert not tmp.exists()


def test_a_failed_image_removal_keeps_the_record_that_finds_it(rig, monkeypatch) -> None:
    c = client(rig, devmode=True)
    record = generate(c)
    store = c.app.state.engine.store

    def refuse(record_id: str) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(store, "remove_image", refuse)
    with pytest.raises(OSError, match="read-only"):
        store.delete(record["id"])
    assert store.record_path(record["id"]).is_file()


def test_purging_an_unknown_or_malformed_record_is_a_404_and_releases_the_lock(rig) -> None:
    c = client(rig, devmode=True)
    for record_id in ("0" * 32, "not-an-id"):
        assert c.post(f"/purge/{record_id}", json={"confirm": "purge"}).status_code == 404
    assert c.get("/health").json()["generating"] is False


@pytest.mark.parametrize("route", ["/purge", "/purge/{id}"])
def test_a_removal_in_flight_cannot_bring_back_a_purged_record(rig, monkeypatch, route) -> None:
    c = client(rig, devmode=True)
    record = generate(c)
    store = c.app.state.engine.store
    load, loaded, resume = store.load, threading.Event(), threading.Event()

    def paused_load(record_id: str):
        found = load(record_id)
        loaded.set()
        resume.wait(5)
        return found

    monkeypatch.setattr(store, "load", paused_load)
    removal = threading.Thread(target=c.delete, args=(f"/record/{record['id']}",))
    purge = threading.Thread(
        target=c.post, args=(route.format(id=record["id"]),), kwargs={"json": {"confirm": "purge"}}
    )
    removal.start()
    assert loaded.wait(5)
    purge.start()
    purge.join(0.3)  # long enough for an unguarded purge to finish while the removal holds its stale copy
    resume.set()
    removal.join(5)
    purge.join(5)
    assert not removal.is_alive() and not purge.is_alive()
    assert not store.exists(record["id"])


def test_a_removal_is_not_refused_during_a_generation(rig) -> None:
    c = client(rig)
    record = generate(c)
    c.app.state.busy.acquire()
    try:
        assert c.delete(f"/record/{record['id']}").status_code == 200
    finally:
        c.app.state.busy.release()


def test_purge_leaves_a_clean_slate_and_an_archive_taken_before_holds_everything(rig) -> None:
    c = client(rig, devmode=True)
    ids = {generate(c, {"seed": n})["id"] for n in range(2)}
    rig.singguard.responder = answers(preflight=unsafe_answer("B. Real-World Crimes & Public Safety"))
    ids.add(generate(c)["id"])
    # a prompt darkroom sent that was never deleted, and one that is not darkroom's
    rig.comfyui.history["leftover"] = {"prompt": [1, "leftover", {}, {"client_id": "argus-darkroom-abc"}, []]}
    rig.comfyui.history["other"] = {"prompt": [2, "other", {}, {"client_id": "someone-else"}, []]}
    archive = c.get("/archive").content
    assert {
        n.split("/")[1].removesuffix(".json")
        for n in zipfile.ZipFile(io.BytesIO(archive)).namelist()
        if n.startswith("records/")
    } == ids

    r = c.post("/purge", json={"confirm": "purge"})
    assert r.status_code == 200
    assert r.json() == {"records": 3, "images": 2, "comfyui_history": 1}
    assert list(rig.out.iterdir()) == []
    assert c.get("/records").json()["total"] == 0
    assert set(rig.comfyui.history) == {"other"}  # ours cleared, theirs untouched
    assert c.get("/health").json()["generating"] is False


def test_purge_still_clears_local_data_when_comfyui_is_down(rig) -> None:
    c = client(rig, devmode=True)
    generate(c)
    c.app.state.engine.comfyui = type(c.app.state.engine.comfyui)("http://127.0.0.1:1", http_timeout=1)
    r = c.post("/purge", json={"confirm": "purge"})
    assert r.json() == {"records": 1, "images": 1, "comfyui_history": None}
    assert list(rig.out.iterdir()) == []


# --- who may write -------------------------------------------------------------------------------------


def test_a_cross_site_write_is_refused_and_a_cross_site_read_is_not_granted(rig) -> None:
    c = client(rig)
    r = c.post("/preflight", json=BODY, headers=EVIL)
    assert r.status_code == 403 and "cross-site" in r.json()["detail"]
    assert rig.singguard.requests == []
    read = c.get("/records", headers=EVIL)
    assert read.status_code == 200 and "access-control-allow-origin" not in read.headers
    assert c.delete("/record/" + "0" * 32, headers=EVIL).status_code == 403


def test_read_only_is_a_boolean_on_health_and_comes_from_the_environment(rig, monkeypatch) -> None:
    assert client(rig, read_only=True).get("/health").json()["read_only"] is True
    monkeypatch.setenv("ARGUS_DARKROOM_SINGGUARD_URL", "http://guard:9150")
    monkeypatch.delenv("ARGUS_DARKROOM_READ_ONLY", raising=False)
    assert Settings.from_env().read_only is False
    monkeypatch.setenv("ARGUS_DARKROOM_READ_ONLY", "1")
    assert Settings.from_env().read_only is True


def test_read_only_refuses_every_write_before_it_reaches_a_backend(rig) -> None:
    c = client(rig, read_only=True, devmode=True)
    origin = {"Origin": "http://localhost:3000"}  # a trusted origin: read-only outranks it
    calls = [
        lambda h: c.post("/preflight", json=BODY, headers=h),
        lambda h: c.post("/generate/stream", json=BODY, headers=h),
        lambda h: c.delete("/record/" + "0" * 32, headers=h),
        lambda h: c.post("/purge", json={"confirm": "purge"}, headers=h),
        lambda h: c.post("/purge/" + "0" * 32, json={"confirm": "purge"}, headers=h),
    ]
    for call in calls:
        for headers in ({}, origin):
            r = call(headers)
            assert r.status_code == 403 and "read-only" in r.json()["detail"]
    assert rig.singguard.requests == []
    # CORS stays outermost, so a browser can read the refusal.
    assert c.post("/preflight", json=BODY, headers=origin).headers["access-control-allow-origin"] == origin["Origin"]


def test_read_only_keeps_records_and_images_readable(rig) -> None:
    record = generate(client(rig))
    c = client(rig, read_only=True)
    assert c.delete("/record/" + record["id"]).status_code == 403
    assert c.get("/image/" + record["id"]).status_code == 200
    assert c.get("/record/" + record["id"]).json()["removed_at"] is None
    assert c.get("/records").status_code == 200 and c.get("/policies").status_code == 200
    assert c.get("/archive").status_code == 200


def test_the_configured_origin_may_read_and_write(rig) -> None:
    c = client(rig, cors_origins=("https://safelight.example",))
    origin = {"Origin": "https://safelight.example"}
    assert c.post("/preflight", json=BODY, headers=origin).status_code == 200
    assert c.get("/health", headers=origin).headers["access-control-allow-origin"] == "https://safelight.example"
    assert c.post("/preflight", json=BODY, headers={"Origin": "http://localhost:3000"}).status_code == 403


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
@pytest.mark.parametrize("port", [3000, 3097])
def test_by_default_only_safelights_origins_are_trusted(rig, host, port) -> None:
    c = client(rig)
    origin = {"Origin": f"http://{host}:{port}"}
    assert c.post("/preflight", json=BODY, headers=origin).status_code == 200
    assert c.get("/health", headers=origin).headers["access-control-allow-origin"] == origin["Origin"]
    assert c.post("/preflight", json=BODY, headers={"Origin": f"http://{host}:3098"}).status_code == 403


def test_a_wildcard_grants_reads_never_writes(rig) -> None:
    c = client(rig, cors_origins=("*",))
    assert c.get("/health", headers=EVIL).headers["access-control-allow-origin"] == "*"
    assert c.post("/preflight", json=BODY, headers=EVIL).status_code == 403
    named = client(rig, cors_origins=("*", "https://safelight.example"))
    assert named.post("/preflight", json=BODY, headers={"Origin": "https://safelight.example"}).status_code == 200


def test_requests_without_an_origin_are_not_browser_writes(rig) -> None:
    assert client(rig).post("/preflight", json=BODY).status_code == 200  # curl, the CLI, server to server


def test_grey_png_helper_is_a_real_png() -> None:
    assert grey_png().startswith(b"\x89PNG") and safe_answer()["safe"]


# --- image-to-image ------------------------------------------------------------------------------------


def test_the_stream_carries_the_input_scan_before_the_render(rig) -> None:
    c = client(rig, workflow=EXAMPLES / "sdxl+lora1+img2img.json")
    events = stream_events(c, {"init_image": {"data_base64": base64.b64encode(grey_png(32)).decode()}})
    kinds = [k for k, _ in events]
    assert kinds == [
        "preflight", "input_scan", "input_scan", "queued", "progress", "progress", "scan", "scan", "complete",
    ]  # fmt: skip
    assert events[1][1]["policy"] == "default" and events[1][1]["context"] == "isolated"
    assert events[2][1]["context"] == "with_prompt" and events[2][1]["cell"]["state"] == "safe"
    record = events[-1][1]["record"]
    assert record["outcome"] == "passed" and record["guard"]["input"]["outcome"] == "passed"
    assert record["request"]["init_image"]["denoise"] == 0.5
    assert c.get(f"/record/{record['id']}").json() == record


def test_a_refused_input_streams_no_render(rig) -> None:
    rig.nudenet.labels = {"FEMALE_BREAST_EXPOSED": 0.9}
    c = client(rig, workflow=EXAMPLES / "sdxl+lora1+img2img.json")
    events = stream_events(c, {"init_image": {"data_base64": base64.b64encode(grey_png(32)).decode()}})
    assert [k for k, _ in events] == ["preflight", "input_scan", "input_scan", "complete"]
    record = events[-1][1]["record"]
    assert (record["outcome"], record["outcome_reason"]) == ("refused", "input image: sfw-exposed")
    assert c.get(f"/image/{record['id']}").status_code == 404
    assert rig.comfyui.calls == []


@pytest.mark.parametrize(
    ("init_image", "status"),
    [
        ({"data_base64": "not base64!"}, 400),
        ({"data_base64": "aGVsbG8="}, 400),  # base64 of text, not a picture
        ({"data_base64": "aGVsbG8=", "denoise": 0}, 422),  # a field out of range, like nsfw_threshold
        ({"data_base64": "aGVsbG8=", "mask": "x"}, 422),
    ],
)
def test_a_bad_input_is_refused_before_the_stream_opens(rig, init_image, status) -> None:
    c = client(rig, workflow=EXAMPLES / "sdxl+lora1+img2img.json")
    assert c.post("/generate/stream", json=BODY | {"init_image": init_image}).status_code == status
    assert not c.app.state.busy.locked()
    assert rig.singguard.requests == [] and rig.comfyui.calls == []


def test_an_input_with_no_image_to_image_template_is_a_400(rig) -> None:
    c = client(rig)
    r = c.post("/generate/stream", json=BODY | {"init_image": {"data_base64": base64.b64encode(grey_png()).decode()}})
    assert r.status_code == 400 and "no image-to-image workflow" in r.json()["detail"]
    assert generate(c)["outcome"] == "passed"  # text-to-image is unaffected, and the lock was released


# --- inpainting ----------------------------------------------------------------------------------------


def black_and_white(size: int = 32, painted: bool = True) -> bytes:
    image = Image.new("L", (size, size), 0)
    if painted:
        image.paste(255, (0, 0, size // 2, size))
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def inpaint_init(mask: bytes | None = None, size: int = 32) -> dict:
    return {
        "data_base64": base64.b64encode(grey_png(size)).decode(),
        "mask_base64": base64.b64encode(mask if mask is not None else black_and_white(size)).decode(),
        "denoise": 0.9,
    }


def test_an_inpaint_stream_scans_only_the_input_and_records_the_mask(rig) -> None:
    c = client(rig, workflow=EXAMPLES / "sdxl+lora1+inpaint.json")
    events = stream_events(c, {"init_image": inpaint_init()})
    assert [k for k, _ in events] == [
        "preflight", "input_scan", "input_scan", "queued", "progress", "progress", "scan", "scan", "complete",
    ]  # fmt: skip
    record = events[-1][1]["record"]
    assert record["outcome"] == "passed" and record["request"]["workflow"] == "sdxl+lora1+inpaint"
    mask = record["request"]["init_image"]["mask"]
    assert mask["masked_fraction"] == 0.5 and len(mask["sha256"]) == 64 and len(mask["source_sha256"]) == 64
    assert c.get(f"/record/{record['id']}").json() == record


@pytest.mark.parametrize(
    ("mask", "detail"),
    [
        (black_and_white(16), "the mask is 16x16 but the init image is 32x32"),
        (b"not a picture", "the mask cannot be decoded"),
        (black_and_white(32, painted=False), "masks nothing"),
    ],
)
def test_a_bad_mask_is_a_400_before_the_stream_opens(rig, mask, detail) -> None:
    c = client(rig, workflow=EXAMPLES / "sdxl+lora1+inpaint.json")
    r = c.post("/generate/stream", json=BODY | {"init_image": inpaint_init(mask)})
    assert r.status_code == 400 and detail in r.json()["detail"]
    assert not c.app.state.busy.locked()
    assert rig.singguard.requests == [] and rig.comfyui.calls == []


def test_a_mask_with_no_inpaint_template_is_a_400(rig) -> None:
    c = client(rig, workflow=EXAMPLES / "sdxl+lora1+img2img.json")
    r = c.post("/generate/stream", json=BODY | {"init_image": inpaint_init()})
    assert r.status_code == 400 and "no inpainting workflow" in r.json()["detail"]
    assert not c.app.state.busy.locked()
