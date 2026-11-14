"""Image-to-image: the input image gets the same checks as an output, before anything reaches ComfyUI.

The input is a harmless generated picture (flat colours). Every bad verdict on it comes from a fake answer keyed on
the input's bytes, never from its content.
"""

from __future__ import annotations

import base64
import hashlib
import io
import tempfile
from pathlib import Path

import pytest
from fake_comfyui import FakeComfyUI
from fakes import FakeNudeNet, FakeSingGuard, policy_of, safe_answer, unsafe_answer
from PIL import Image
from pydantic import ValidationError
from rig import EXAMPLES, EXPOSED, ILLEGAL_A, Rig, is_image_scan, nsfw

from argus_darkroom.images import MAX_INPUT_BYTES, load_input
from argus_darkroom.models import FaceAge, GenerationRecord
from argus_darkroom.prompt import frame
from argus_darkroom.request import GenerateRequest, RequestError

FLAG_B = unsafe_answer("B. Real-World Crimes & Public Safety")


def picture(size: tuple[int, int] = (48, 32), fmt: str = "PNG", colour=(40, 90, 200), **save) -> bytes:
    """A flat-colour picture with one stripe: nothing in it but colour."""
    image = Image.new("RGB", size, colour)
    for x in range(size[0]):
        image.putpixel((x, 0), (250, 250, 250))
    buf = io.BytesIO()
    image.save(buf, fmt, **save)
    return buf.getvalue()


UPLOAD = picture()
CANONICAL = load_input(base64.b64encode(UPLOAD).decode()).png  # what is judged and what ComfyUI gets
CANONICAL_B64 = base64.b64encode(CANONICAL).decode()


def init(data: bytes = UPLOAD, denoise: float = 0.4) -> dict:
    return {"data_base64": base64.b64encode(data).decode(), "denoise": denoise}


@pytest.fixture
def rig(tmp_path, singguard, comfyui, nudenet, monkeypatch) -> Rig:
    """The usual rig, with the image-to-image template for every checkpoint."""
    r = Rig(tmp_path, singguard, comfyui, nudenet, workflow=EXAMPLES / "sdxl+lora1+img2img.json")
    monkeypatch.setattr(tempfile, "tempdir", str(r.systmp))
    return r


def is_input(body: dict) -> bool:
    return body.get("image_base64") == CANONICAL_B64


def input_answers(isolated=None, with_prompt=None):
    """A SingGuard responder that answers the input image's scans as given (per policy or for all) and everything
    else safe. Keyed on the input's bytes: the verdict is canned, not read from the picture."""

    def pick(spec, body):
        if spec is None:
            return safe_answer()
        return spec.get(policy_of(body), safe_answer()) if "safe" not in spec else spec

    def respond(body: dict):
        if not is_input(body):
            return 200, safe_answer()
        return 200, pick(isolated if body["input"] == "" else with_prompt, body)

    return respond


def holds_the_input(rig: Rig) -> bool:
    """Whether any file under the tmp dir (models excluded) holds the input: as uploaded, as judged, or as base64."""
    needles = [UPLOAD, CANONICAL, base64.b64encode(UPLOAD), CANONICAL_B64.encode()]
    for path in rig.root.rglob("*"):
        if path.is_file() and "models" not in path.relative_to(rig.root).parts:
            data = path.read_bytes()
            if any(n in data for n in needles):
                return True
    return False


class FakeAge:
    def __init__(self, *ages: float) -> None:
        self.ages = ages

    def estimate(self, data: bytes) -> list[FaceAge]:
        return [FaceAge(age_estimate_years=a, bbox=(0, 0, 10, 10)) for a in self.ages]


# --- the input passes ------------------------------------------------------------------------------------


def test_a_passed_input_reaches_comfyui_in_the_graph_and_the_output_is_judged_as_usual(rig) -> None:
    record = rig.generate(rig.engine(), init_image=init(denoise=0.35))
    assert record.outcome == "passed" and record.outcome_reason is None

    (graph,) = rig.comfyui.graphs
    loader = next(n for n in graph.values() if n["class_type"] == "DarkroomLoadImageBase64")
    assert loader["inputs"]["image"] == CANONICAL_B64  # the very bytes that were scanned
    assert graph["3"]["inputs"]["denoise"] == 0.35
    assert graph["3"]["inputs"]["latent_image"] == ["14", 0]
    assert graph["13"]["inputs"] | {"image": None} == {
        "upscale_method": "lanczos", "width": 64, "height": 64, "crop": "center", "image": None,
    }  # fmt: skip
    assert rig.comfyui.history == {}  # the graph, and the input in it, is deleted after the run

    ref = record.request.init_image
    assert (ref.width, ref.height, ref.denoise) == (48, 32, 0.35)
    assert ref.sha256 == hashlib.sha256(CANONICAL).hexdigest()
    assert ref.source_sha256 == hashlib.sha256(UPLOAD).hexdigest()
    assert ref.phash
    assert record.request.workflow == "sdxl+lora1+img2img"
    assert record.guard.input.outcome == "passed" and record.guard.input.reason is None
    assert set(record.guard.input.scans) == {"default"}
    assert record.guard.input.nudenet.exposed_score is None or record.guard.input.nudenet.exposed_score < 0.2

    assert (rig.out / record.image.path).read_bytes() == rig.comfyui.png  # the output, released as usual
    assert sorted(p.name for p in rig.out.iterdir()) == [f"{record.id}.json", f"{record.id}.png"]
    assert not holds_the_input(rig)  # darkroom never stores the input: the record has its hashes only
    assert rig.engine().store.load(record.id) == record
    probe = rig.systmp / "probe"
    probe.write_text(CANONICAL_B64)
    assert holds_the_input(rig)  # the no-bytes checks are only worth something if this one can see a file
    probe.unlink()


def test_the_input_is_scanned_isolated_and_with_the_prompt_under_every_policy_before_the_render(rig) -> None:
    nsfw(rig)
    rig.generate(rig.engine(age=FakeAge(30)), init_image=init(), advisory_policies=["default"])
    images = [b for b in rig.singguard.requests if is_image_scan(b)]
    inputs = [b for b in images if is_input(b)]
    assert [(policy_of(b), b["input"] == "") for b in inputs] == [
        ("adult-granular", True),
        ("adult-granular", False),
        ("default", True),
        ("default", False),
    ]
    assert {b["input"] for b in inputs} == {"", frame("a cat, sks", "blurry")}  # no age estimate sent, ever
    assert images[:4] == inputs  # the input first, then the output
    kinds = [k for k, _ in rig.events]
    assert kinds[: kinds.index("queued")] == ["preflight", *["input_scan"] * 4]
    assert kinds.count("scan") == 4
    assert rig.events[1][1]["policy"] == "adult-granular" and rig.events[1][1]["context"] == "isolated"
    assert rig.events[1][1]["cell"]["state"] == "safe"


def test_an_nsfw_input_with_exposure_passes_and_is_judged_like_an_nsfw_output(rig) -> None:
    nsfw(rig)
    rig.nudenet.labels = EXPOSED
    record = rig.generate(rig.engine(age=FakeAge(30)), init_image=init())
    assert record.guard.input.outcome == "passed" and record.guard.input.nudenet.blur is True
    assert record.outcome == "passed" and record.guard.nudenet.blur is True


# --- the input is refused: nothing reaches ComfyUI, nothing is written ---------------------------------


def assert_refused_with_no_bytes(rig: Rig, record: GenerationRecord, before: list[str], reason: str) -> None:
    assert record.outcome == "refused"
    assert record.outcome_reason == f"input image: {reason}"
    assert record.image is None
    assert rig.comfyui.calls == [] and rig.comfyui.graphs == []  # not even asked about its nodes
    assert rig.comfyui.history == {}
    assert rig.files() == sorted([*before, str(Path("out") / f"{record.id}.json")])  # the record, and only it
    assert not holds_the_input(rig)
    assert record.guard.input is not None and record.guard.input.outcome != "passed"
    assert record.request.init_image is not None and record.request.init_image.sha256  # named by its hashes only
    kinds = [k for k, _ in rig.events]
    assert kinds[0] == "preflight" and kinds[-1] == "complete" and "queued" not in kinds
    assert "input_scan" in kinds and "scan" not in kinds


def test_an_illegal_input_is_refused_and_critical(rig) -> None:
    nsfw(rig)
    rig.singguard.responder = input_answers(isolated=ILLEGAL_A)
    before = rig.files()
    record = rig.generate(rig.engine(), init_image=init())
    assert_refused_with_no_bytes(rig, record, before, "adult-granular:A")
    assert record.critical is True and record.guard.input.outcome == "deleted"


def test_an_input_flagged_under_sfw_is_refused_not_sent_to_review(rig) -> None:
    rig.singguard.responder = input_answers(with_prompt=FLAG_B)
    before = rig.files()
    record = rig.generate(rig.engine(), init_image=init())
    assert_refused_with_no_bytes(rig, record, before, "default:B")
    assert record.critical is False and record.guard.input.outcome == "review"


def test_an_input_flagged_under_an_advisory_policy_is_refused_too(rig) -> None:
    # An output flagged by an advisory policy goes to review; an input that would go to review does not go on.
    nsfw(rig)
    rig.singguard.responder = input_answers(isolated={"default": FLAG_B})
    before = rig.files()
    record = rig.generate(rig.engine(), init_image=init(), advisory_policies=["default"])
    assert_refused_with_no_bytes(rig, record, before, "default:B")


def test_an_exposed_input_under_sfw_is_refused(rig) -> None:
    rig.nudenet.labels = EXPOSED
    before = rig.files()
    record = rig.generate(rig.engine(), init_image=init())
    assert_refused_with_no_bytes(rig, record, before, "sfw-exposed")
    assert record.critical is False and record.guard.input.outcome == "omitted"
    assert record.guard.input.nudenet.exposed_score == 0.9


def test_a_minor_exposed_input_is_refused_and_critical(rig) -> None:
    nsfw(rig)
    rig.nudenet.labels = EXPOSED
    before = rig.files()
    record = rig.generate(rig.engine(age=FakeAge(31, 16.5)), init_image=init(), nsfw_threshold=1.0)
    assert_refused_with_no_bytes(rig, record, before, "minor-exposed")
    assert record.critical is True
    assert record.guard.input.age.minor_exposed and record.guard.input.age.source == "mivolo"


def test_an_input_singguard_cannot_check_is_refused(rig) -> None:
    rig.singguard.responder = lambda body: (500, {}) if is_input(body) else (200, safe_answer())
    before = rig.files()
    record = rig.generate(rig.engine(), init_image=init())
    assert_refused_with_no_bytes(rig, record, before, "unchecked")
    assert record.critical is False
    assert record.guard.input.scans["default"].isolated.state == "unchecked"


def test_an_input_nudenet_cannot_check_under_sfw_is_refused(rig) -> None:
    rig.nudenet.status = 500
    before = rig.files()
    record = rig.generate(rig.engine(), init_image=init())
    assert_refused_with_no_bytes(rig, record, before, "sfw-unchecked")
    assert record.guard.input.nudenet.exposed_score is None


def test_singguard_down_for_the_input_refuses(tmp_path, comfyui, nudenet, monkeypatch) -> None:
    def respond(body):
        if is_image_scan(body):
            raise ConnectionError  # the handler dies: the client sees a failed request
        return 200, safe_answer()

    with FakeSingGuard(respond) as singguard:
        rig = Rig(tmp_path, singguard, comfyui, nudenet, workflow=EXAMPLES / "sdxl+lora1+img2img.json")
        monkeypatch.setattr(tempfile, "tempdir", str(rig.systmp))
        before = rig.files()
        record = rig.generate(rig.engine(), init_image=init())
        assert_refused_with_no_bytes(rig, record, before, "unchecked")


def test_pre_flight_refuses_first_and_the_input_is_never_scanned(rig) -> None:
    rig.singguard.responder = lambda body: (200, safe_answer() if is_image_scan(body) else FLAG_B)
    record = rig.generate(rig.engine(), init_image=init())
    assert (record.outcome, record.outcome_reason) == ("refused", "default:B")
    assert not [b for b in rig.singguard.requests if is_image_scan(b)]
    assert rig.nudenet.calls == 0
    assert rig.comfyui.calls == []
    assert record.guard.input is None
    assert record.request.init_image.sha256 == hashlib.sha256(CANONICAL).hexdigest()
    assert [k for k, _ in rig.events] == ["preflight", "complete"]


def test_engine_render_will_not_send_an_input_that_was_not_scanned(rig) -> None:
    eng = rig.engine()
    with pytest.raises(RequestError, match="scan passed"):
        eng.render(rig.request(init_image=init()))
    assert rig.comfyui.calls == []


def test_a_comfyui_error_that_echoes_the_input_does_not_put_it_in_the_record(rig) -> None:
    rig.comfyui.reject_with = {"error": "invalid prompt", "node_errors": {"12": {"inputs": {"image": CANONICAL_B64}}}}
    record = rig.generate(rig.engine(), init_image=init())
    assert record.outcome == "failed" and "base64 removed" in record.outcome_reason
    assert CANONICAL_B64[:100] not in record.model_dump_json()
    assert not holds_the_input(rig)


# --- the request ------------------------------------------------------------------------------------------


def test_text_to_image_is_unchanged_beside_an_image_to_image_template(tmp_path, singguard, comfyui, nudenet) -> None:
    rig = Rig(tmp_path, singguard, comfyui, nudenet)  # the plain sdxl+lora1 template
    record = rig.generate(rig.engine())
    assert record.outcome == "passed" and record.request.init_image is None and record.guard.input is None
    assert "input_scan" not in [k for k, _ in rig.events]


def test_an_input_with_no_image_to_image_template_is_a_request_error(tmp_path, singguard, comfyui, nudenet) -> None:
    rig = Rig(tmp_path, singguard, comfyui, nudenet)
    with pytest.raises(RequestError, match="no image-to-image workflow"):
        rig.generate(rig.engine(), init_image=init())
    assert singguard.requests == [] and comfyui.calls == []


def test_a_text_to_image_request_on_an_image_to_image_template_is_a_request_error(rig) -> None:
    with pytest.raises(RequestError, match="no workflow renders this checkpoint"):
        rig.generate(rig.engine())
    assert rig.singguard.requests == []


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


@pytest.mark.parametrize(
    ("data_base64", "message"),
    [
        ("not base64!", "not valid base64"),
        (b64(b"plain text, not a picture"), "cannot be decoded"),
        (b64(picture(fmt="GIF")), "cannot be decoded"),
        (b64(picture(size=(4097, 2))), "4097x2"),
        (b64(picture(size=(2, 4100))), "2x4100"),
        ("A" * (4 * (MAX_INPUT_BYTES // 3) + 8), "over 16 MiB"),
    ],
)
def test_an_input_that_cannot_be_used_is_a_request_error_and_touches_nothing(rig, data_base64, message) -> None:
    with pytest.raises(RequestError, match=message):
        rig.generate(rig.engine(), init_image={"data_base64": data_base64})
    assert rig.singguard.requests == [] and rig.comfyui.calls == [] and rig.nudenet.calls == 0
    assert list(rig.out.iterdir()) == []


def test_an_input_just_over_the_size_limit_once_decoded_is_refused(monkeypatch) -> None:
    import argus_darkroom.images as images

    monkeypatch.setattr(images, "MAX_INPUT_BYTES", len(UPLOAD) - 1)
    with pytest.raises(images.UnreadableImage, match="over"):
        images.load_input(b64(UPLOAD))


@pytest.mark.parametrize("fmt", ["JPEG", "WEBP"])
def test_jpeg_and_webp_inputs_are_judged_and_sent_as_one_png(rig, fmt) -> None:
    record = rig.generate(rig.engine(), init_image=init(picture(fmt=fmt)))
    assert record.outcome == "passed"
    sent = base64.b64decode(rig.comfyui.graphs[0]["12"]["inputs"]["image"])
    assert sent.startswith(b"\x89PNG") and hashlib.sha256(sent).hexdigest() == record.request.init_image.sha256
    scanned = {base64.b64decode(b["image_base64"]) for b in rig.singguard.requests if is_image_scan(b)}
    assert sent in scanned


def test_a_one_colour_input_is_allowed_unlike_a_one_colour_render(rig) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 120, 10)).save(buf, "PNG")
    assert rig.generate(rig.engine(), init_image=init(buf.getvalue())).outcome == "passed"


def test_the_input_is_turned_upright_and_loses_alpha_and_metadata() -> None:
    image = Image.new("RGBA", (30, 10), (200, 10, 10, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 on display
    buf = io.BytesIO()
    image.save(buf, "PNG", exif=exif)
    loaded = load_input(b64(buf.getvalue()))
    with Image.open(io.BytesIO(loaded.png)) as out:
        assert out.mode == "RGB" and out.size == (10, 30)
        assert "exif" not in out.info
    assert (loaded.record.width, loaded.record.height) == (10, 30)


@pytest.mark.parametrize("denoise", [0.0, 0.04, 1.01, -1])
def test_denoise_out_of_range_is_refused(rig, denoise) -> None:
    with pytest.raises(ValidationError, match="denoise"):
        rig.request(init_image={"data_base64": b64(UPLOAD), "denoise": denoise})


def test_the_default_denoise_is_a_half_and_unknown_fields_are_refused(rig) -> None:
    assert rig.request(init_image={"data_base64": b64(UPLOAD)}).init_image.denoise == 0.5
    with pytest.raises(ValidationError, match="mask"):
        rig.request(init_image={"data_base64": b64(UPLOAD), "mask": "x"})


def test_the_input_never_shows_in_a_repr(rig) -> None:
    request = GenerateRequest(user_prompt="a cat", init_image={"data_base64": b64(UPLOAD)})
    assert b64(UPLOAD)[:40] not in repr(request) and b64(UPLOAD)[:40] not in str(request)
    prepared = rig.engine().prepare(rig.request(init_image=init()))
    assert CANONICAL_B64[:40] not in repr(prepared) and repr(CANONICAL)[:40] not in repr(prepared)
    params = rig.engine()._params(rig.request(init_image=init()), prepared, "a cat")
    assert CANONICAL_B64[:40] not in repr(params)


def test_nudenet_sees_the_input_and_then_the_output(tmp_path, singguard, monkeypatch) -> None:
    with FakeNudeNet() as nudenet, FakeComfyUI() as comfyui:
        rig = Rig(tmp_path, singguard, comfyui, nudenet, workflow=EXAMPLES / "sdxl+lora1+img2img.json")
        monkeypatch.setattr(tempfile, "tempdir", str(rig.systmp))
        rig.generate(rig.engine(), init_image=init())
        assert nudenet.calls == 2


def test_a_piece_of_base64_cut_short_at_the_end_of_an_error_is_removed() -> None:
    from argus_darkroom.engine import _scrub

    cut = 'ComfyUI rejected the workflow: {"12": {"inputs": {"image": "' + CANONICAL_B64[:60]
    assert CANONICAL_B64[:60] not in _scrub(cut)
    assert _scrub("ComfyUI failed in node KSampler: torch.OutOfMemoryError") == (
        "ComfyUI failed in node KSampler: torch.OutOfMemoryError"
    )
