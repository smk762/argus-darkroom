"""Depth maps: ``POST /depth/{id}`` and ``GET /depth/{id}``, over the same fakes as the server tests, and the vendored
node's checks (``comfyui/darkroom_depth.py``), which run without torch or ComfyUI."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fake_comfyui import is_depth
from fakes import unsafe_answer
from PIL import Image
from rig import ILLEGAL_A, answers
from test_server import BODY, EVIL, client, generate

from argus_darkroom import depth
from argus_darkroom.models import render_wire_schema

NODE = Path(__file__).resolve().parent.parent / "comfyui" / "darkroom_depth.py"
SCHEMA = Path(__file__).resolve().parent.parent / "schema" / "darkroom-wire.schema.json"


def depth_runs(rig) -> int:
    return sum(1 for g in rig.comfyui.graphs if is_depth(g))


def make_depth(c, rig, record_id: str) -> None:
    """Make the depth map, and check it is there, so a test of its removal cannot pass on one never made."""
    assert c.post(f"/depth/{record_id}").status_code == 200
    assert (rig.out / f"{record_id}.depth.png").is_file()
    assert c.get(f"/depth/{record_id}").status_code == 200


def test_post_makes_the_depth_map_and_get_serves_it(rig) -> None:
    c = client(rig)
    record = generate(c)
    r = c.post(f"/depth/{record['id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "depth-anything-v2-small" and body["model_license"] == "Apache-2.0"
    assert (body["width"], body["height"]) == (64, 64)  # the record image's size, not the model's 32x32
    assert body["workflow"] == "depth-anything-v2-small" and body["engine_version"] == "0.3.test"
    assert body["created_at"]
    got = c.get(f"/depth/{record['id']}")
    assert got.status_code == 200 and got.headers["content-type"] == "image/png"
    assert got.headers["cache-control"] == "private, no-store"
    assert hashlib.sha256(got.content).hexdigest() == body["sha256"]
    with Image.open(io.BytesIO(got.content)) as img:
        assert img.mode == "L" and img.size == (64, 64) and not img.info.get("dpi") and not getattr(img, "text", {})
        pixels = np.asarray(img)
    assert pixels.min() == 0 and pixels.max() == 255  # min to max
    assert pixels[-1].mean() > pixels[0].mean()  # the near (bright) bottom stays bright
    assert c.get(f"/record/{record['id']}").json()["depth"] == body
    assert c.get("/records").json()["records"][0]["depth"] == body
    assert (rig.out / f"{record['id']}.depth.png").is_file()
    assert rig.comfyui.history == {}  # the graph, and the image in it, is gone from ComfyUI


def test_the_depth_graph_sends_the_stored_image_through_the_base64_loader(rig) -> None:
    c = client(rig)
    record = generate(c)
    c.post(f"/depth/{record['id']}")
    graph = next(g for g in rig.comfyui.graphs if is_depth(g))
    classes = sorted(n["class_type"] for n in graph.values())
    assert classes == ["DarkroomDepthAnything", "DarkroomLoadImageBase64", "SaveImageWebsocket"]
    assert graph["2"]["inputs"]["model"] == "Depth-Anything-V2-Small-hf"

    assert base64.b64decode(graph["1"]["inputs"]["image"]) == rig.comfyui.png


def test_a_second_post_returns_the_same_map_without_recomputing(rig) -> None:
    c = client(rig)
    record = generate(c)
    first = c.post(f"/depth/{record['id']}").json()
    assert depth_runs(rig) == 1
    second = c.post(f"/depth/{record['id']}")
    assert second.status_code == 200 and second.json() == first
    assert depth_runs(rig) == 1


def test_get_without_a_depth_map_is_a_404(rig) -> None:
    c = client(rig)
    record = generate(c)
    r = c.get(f"/depth/{record['id']}")
    assert r.status_code == 404 and r.json()["detail"] == "this record has no depth map"
    assert r.json()["record_id"] == record["id"]
    assert c.get(f"/record/{record['id']}").json()["depth"] is None


@pytest.mark.parametrize("bad", ["0" * 32, "nope", "A" * 32, "..%2f..%2fetc%2fpasswd"])
def test_an_unknown_record_is_a_404(rig, bad) -> None:
    c = client(rig)
    for r in (c.post(f"/depth/{bad}"), c.get(f"/depth/{bad}")):
        assert r.status_code == 404 and r.json()["detail"]
    assert depth_runs(rig) == 0


@pytest.mark.parametrize("outcome", ["deleted", "omitted", "refused", "failed", "removed"])
def test_a_record_with_no_stored_image_is_a_409(rig, outcome) -> None:
    c = client(rig, allow_nsfw=True, policy="adult-granular") if outcome == "deleted" else client(rig)
    if outcome == "deleted":
        rig.singguard.responder = answers(isolated=ILLEGAL_A)
    elif outcome == "omitted":
        rig.nudenet.labels = {"FEMALE_BREAST_EXPOSED": 0.9}
    elif outcome == "refused":
        rig.singguard.responder = answers(preflight=unsafe_answer("B. Real-World Crimes & Public Safety"))
    elif outcome == "failed":
        rig.comfyui.reject_with = {"error": "bad", "node_errors": {"3": {}}}
    record = generate(c)
    rig.comfyui.reject_with = None
    if outcome == "removed":
        assert c.delete(f"/record/{record['id']}").status_code == 200
    else:
        assert record["outcome"] == outcome
    r = c.post(f"/depth/{record['id']}")
    assert r.status_code == 409 and r.json()["detail"] == "this record has no stored image"
    assert depth_runs(rig) == 0


def test_a_stored_image_that_does_not_match_its_record_is_a_409(rig) -> None:
    c = client(rig)
    record = generate(c)
    (rig.out / f"{record['id']}.png").write_bytes(b"something else")
    r = c.post(f"/depth/{record['id']}")
    assert r.status_code == 409 and "does not match" in r.json()["detail"]
    assert depth_runs(rig) == 0


def test_a_depth_run_during_a_generation_is_a_409(rig) -> None:
    c = client(rig)
    record = generate(c)
    c.app.state.busy.acquire()
    try:
        r = c.post(f"/depth/{record['id']}")
        assert r.status_code == 409 and r.json()["detail"] == "a generation is already running"
    finally:
        c.app.state.busy.release()
    assert depth_runs(rig) == 0
    assert c.post(f"/depth/{record['id']}").status_code == 200
    assert not c.app.state.busy.locked()


def test_a_depth_run_holds_the_generation_lock(rig) -> None:
    c = client(rig)
    record = generate(c)
    started, release = threading.Event(), threading.Event()
    script = rig.comfyui.script

    def slow(fake, pid, graph):
        started.set()
        release.wait(10)
        return script(fake, pid, graph)

    rig.comfyui.script = slow
    t = threading.Thread(target=c.post, args=(f"/depth/{record['id']}",))
    t.start()
    assert started.wait(5)
    assert c.get("/health").json()["generating"] is True

    assert c.post("/generate/stream", json=BODY).status_code == 409
    release.set()
    t.join(10)
    assert not c.app.state.busy.locked()


def test_a_comfyui_failure_is_a_502_and_stores_nothing(rig) -> None:
    c = client(rig)
    record = generate(c)
    rig.comfyui.reject_with = {"error": "no such node", "node_errors": {"2": {}}}
    r = c.post(f"/depth/{record['id']}")
    assert r.status_code == 502 and "could not be made" in r.json()["detail"]
    assert not (rig.out / f"{record['id']}.depth.png").exists()
    assert c.get(f"/record/{record['id']}").json()["depth"] is None
    assert not c.app.state.busy.locked()


def test_a_depth_node_that_is_an_output_node_is_refused(rig) -> None:
    c = client(rig)
    record = generate(c)
    rig.comfyui.output_nodes = frozenset({"SaveImageWebsocket", "DarkroomDepthAnything"})
    assert c.post(f"/depth/{record['id']}").status_code == 502
    assert depth_runs(rig) == 0


def test_no_guard_is_asked_about_a_depth_map(rig) -> None:
    c = client(rig)
    record = generate(c)
    guard_calls, nudenet_calls = len(rig.singguard.requests), rig.nudenet.calls
    assert c.post(f"/depth/{record['id']}").status_code == 200
    assert (len(rig.singguard.requests), rig.nudenet.calls) == (guard_calls, nudenet_calls)


def test_read_only_and_cross_site_refuse_a_depth_run(rig) -> None:
    record = generate(client(rig))
    c = client(rig, read_only=True)
    r = c.post(f"/depth/{record['id']}")
    assert r.status_code == 403 and "read-only" in r.json()["detail"]
    assert c.post(f"/depth/{record['id']}", headers={"Origin": "http://localhost:3000"}).status_code == 403
    assert c.get(f"/depth/{record['id']}").status_code == 404  # reads stay open
    c = client(rig, read_only=False)
    assert c.post(f"/depth/{record['id']}", headers=EVIL).status_code == 403
    assert depth_runs(rig) == 0


def test_removing_the_image_removes_the_depth_map(rig) -> None:
    c = client(rig)
    record = generate(c)
    make_depth(c, rig, record["id"])
    removed = c.delete(f"/record/{record['id']}").json()
    assert removed["depth"] is None
    assert not (rig.out / f"{record['id']}.depth.png").exists()
    assert c.get(f"/depth/{record['id']}").status_code == 404


def test_purging_one_record_removes_its_depth_map(rig) -> None:
    c = client(rig, devmode=True)
    record = generate(c)
    make_depth(c, rig, record["id"])
    r = c.post(f"/purge/{record['id']}", json={"confirm": "purge"})
    assert r.json() == {"records": 1, "images": 1}
    assert list(rig.out.iterdir()) == []


def test_purge_removes_every_depth_map(rig) -> None:
    c = client(rig, devmode=True)
    record = generate(c)
    make_depth(c, rig, record["id"])
    r = c.post("/purge", json={"confirm": "purge"})
    assert r.json() == {"records": 1, "images": 1, "comfyui_history": 0}
    assert list(rig.out.iterdir()) == []


def test_the_archive_holds_the_depth_maps(rig) -> None:
    c = client(rig)
    with_depth, without = generate(c, {"seed": 1}), generate(c, {"seed": 2})
    c.post(f"/depth/{with_depth['id']}")
    zf = zipfile.ZipFile(io.BytesIO(c.get("/archive").content))
    assert sorted(zf.namelist()) == sorted(
        [
            f"records/{with_depth['id']}.json",
            f"images/{with_depth['id']}.png",
            f"depth/{with_depth['id']}.png",
            f"records/{without['id']}.json",
            f"images/{without['id']}.png",
        ]
    )
    assert zf.read(f"depth/{with_depth['id']}.png") == c.get(f"/depth/{with_depth['id']}").content
    assert json.loads(zf.read(f"records/{with_depth['id']}.json"))["depth"]["sha256"]


def test_the_depth_workflow_is_never_offered_for_generation(rig) -> None:
    c = client(rig)
    names = [w["name"] for w in c.get("/models").json()["workflows"]]
    assert depth.WORKFLOW_NAME not in names
    assert depth.WORKFLOW_NAME not in c.get("/health").json()["workflows"]


def test_the_wire_schema_has_the_depth_field() -> None:
    assert SCHEMA.read_text(encoding="utf-8") == render_wire_schema()
    assert "DepthRecord" in render_wire_schema()


# --- to_depth_png ----------------------------------------------------------------------------------------


def png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def test_to_depth_png_resizes_stretches_and_keeps_no_metadata() -> None:
    src = Image.new("RGB", (4, 2), (100, 100, 100))
    src.putpixel((3, 1), (150, 150, 150))
    found = depth.to_depth_png(png(src), 8, 4)
    with Image.open(io.BytesIO(found.png)) as img:
        assert img.mode == "L" and img.size == (8, 4) and img.info == {}
        assert img.getextrema() == (0, 255)
    assert found.sha256 == hashlib.sha256(found.png).hexdigest()


def test_a_flat_depth_map_is_all_far() -> None:
    found = depth.to_depth_png(png(Image.new("L", (3, 3), 77)), 3, 3)
    with Image.open(io.BytesIO(found.png)) as img:
        assert img.getextrema() == (0, 0)


# --- the vendored node -------------------------------------------------------------------------------------


def load_node():
    spec = importlib.util.spec_from_file_location("darkroom_depth", NODE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_folder(root: Path, name: str, hidden: int = 384, model_type: str = "depth_anything") -> Path:
    path = root / name
    path.mkdir(parents=True)
    config = {"model_type": model_type, "backbone_config": {"model_type": "dinov2", "hidden_size": hidden}}
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return path


def test_the_node_offers_only_the_small_model_and_is_not_an_output() -> None:
    node = load_node()
    cls = node.NODE_CLASS_MAPPINGS["DarkroomDepthAnything"]
    assert cls.RETURN_TYPES == ("IMAGE",) and not getattr(cls, "OUTPUT_NODE", False)
    assert cls.INPUT_TYPES() == {"required": {"image": ("IMAGE",), "model": (["Depth-Anything-V2-Small-hf"],)}}
    assert node.SMALL == depth.MODEL_DIR


def test_the_node_loads_the_small_model_folder(tmp_path) -> None:
    node = load_node()
    path = model_folder(tmp_path, "Depth-Anything-V2-Small-hf")
    assert node.check_model(str(tmp_path), "Depth-Anything-V2-Small-hf") == str(path)


@pytest.mark.parametrize("name", ["Depth-Anything-V2-Large-hf", "Depth-Anything-V2-Base-hf", "../Small", ""])
def test_the_node_refuses_any_other_model_name(tmp_path, name) -> None:
    with pytest.raises(ValueError, match="CC-BY-NC"):
        load_node().check_model(str(tmp_path), name)


@pytest.mark.parametrize(("hidden", "model_type"), [(1024, "depth_anything"), (768, "depth_anything"), (384, "dpt")])
def test_the_node_refuses_a_folder_that_is_not_the_small_model(tmp_path, hidden, model_type) -> None:
    model_folder(tmp_path, "Depth-Anything-V2-Small-hf", hidden, model_type)
    with pytest.raises(ValueError, match="not Depth Anything V2 Small"):
        load_node().check_model(str(tmp_path), "Depth-Anything-V2-Small-hf")


def test_the_node_says_how_to_download_a_missing_model(tmp_path) -> None:
    with pytest.raises(ValueError, match="huggingface-cli download depth-anything/Depth-Anything-V2-Small-hf"):
        load_node().check_model(str(tmp_path), "Depth-Anything-V2-Small-hf")


def test_the_node_normalises_min_to_max() -> None:
    node = load_node()
    out = node.normalise(np.array([[2.0, 4.0], [6.0, 10.0]]))
    assert out.dtype == np.float32 and out.min() == 0.0 and out.max() == 1.0 and out[0, 1] == pytest.approx(0.25)
    assert not node.normalise(np.full((2, 2), 3.0)).any()
    assert not node.normalise(np.array([[np.nan, 1.0]])).any()
