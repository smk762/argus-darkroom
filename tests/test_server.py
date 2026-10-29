"""The HTTP server, over Starlette's TestClient, against the same fakes as the engine tests."""

from __future__ import annotations

import io
import json
import threading
import zipfile
from dataclasses import replace

import pytest
from fake_comfyui import FakeComfyUI, event, grey_png
from fakes import safe_answer, unsafe_answer
from rig import ILLEGAL_A, SAMPLING, Rig, answers, is_image_scan
from starlette.testclient import TestClient

from argus_darkroom.comfyui import ComfyUIClient
from argus_darkroom.config import ConfigError
from argus_darkroom.prompt import frame
from argus_darkroom.server import create_app

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
    assert body["policies"] == ["default", "adult", "adult-granular", "adult-cautious"]
    assert body["age_unavailable"]
    assert "127.0.0.1" not in json.dumps(body)  # whether each answers, never where


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
        {"code": "I", "name": "Sexual Violence & Extreme Content"},
    ]
    assert all(p["summary"] for p in body["policies"])
    assert by_name["default"]["illegal_rules"] == [] and by_name["default"]["text"] is None
    assert "Minors" in by_name["adult-granular"]["text"]


def test_policies_with_the_flag(rig) -> None:
    body = client(rig, allow_nsfw=True, policy="adult-granular").get("/policies").json()
    assert body["default"] == "adult-granular"
    assert all(p["enforceable"] for p in body["policies"])


def test_models(rig) -> None:
    assert client(rig).get("/models").json() == {
        "checkpoints": [{"name": "base.safetensors"}],
        "loras": [{"name": "style.safetensors"}],
        "vaes": [],
        "samplers": ["euler", "dpmpp_2m"],  # what ComfyUI's KSampler accepts
        "schedulers": ["normal", "karras"],
    }


def test_models_without_comfyui_still_lists_the_files(rig) -> None:
    c = client(rig)
    c.app.state.engine.comfyui = ComfyUIClient("http://127.0.0.1:1", timeout=1)
    body = c.get("/models").json()
    assert body["checkpoints"] == [{"name": "base.safetensors"}] and body["samplers"] == [] == body["schedulers"]


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


def test_the_configured_origin_may_read_and_write(rig) -> None:
    c = client(rig, cors_origins=("https://safelight.example",))
    origin = {"Origin": "https://safelight.example"}
    assert c.post("/preflight", json=BODY, headers=origin).status_code == 200
    assert c.get("/health", headers=origin).headers["access-control-allow-origin"] == "https://safelight.example"
    assert c.post("/preflight", json=BODY, headers={"Origin": "http://localhost:3000"}).status_code == 403


def test_by_default_only_safelights_dev_origin_is_trusted(rig) -> None:
    c = client(rig)
    assert c.post("/preflight", json=BODY, headers={"Origin": "http://localhost:3000"}).status_code == 200


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
