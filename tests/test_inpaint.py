"""Inpainting: an image-to-image input with a mask. The input is judged exactly as for image-to-image; the mask is a
region selector with no imagery, so it is never scanned, never written, and named in the record by its hashes only.

Every picture here is a harmless generated one (flat colours and stripes). Bad verdicts come from fake answers.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tempfile

import pytest
from fakes import policy_of, safe_answer
from PIL import Image
from pydantic import ValidationError
from rig import EXAMPLES, EXPOSED, ILLEGAL_A, Rig, is_image_scan, nsfw
from test_img2img import CANONICAL, CANONICAL_B64, UPLOAD, assert_refused_with_no_bytes, b64, input_answers, picture

from argus_darkroom.images import MAX_INPUT_BYTES, UnreadableImage, load_mask
from argus_darkroom.request import GenerateRequest, RequestError

SIZE = (48, 32)  # the input picture's size


def mask_png(size: tuple[int, int] = SIZE, box=(0, 0, 24, 16), fill: int = 255, mode: str = "L", fmt="PNG", **save):
    """Black, with *box* painted *fill*: the top-left quarter by default."""
    image = Image.new(mode, size, 0)
    image.paste(fill if mode in ("L", "1") else (fill,) * len(mode), box)
    buf = io.BytesIO()
    image.save(buf, fmt, **save)
    return buf.getvalue()


MASK = mask_png()
MASK_CANONICAL = load_mask(b64(MASK), SIZE).png
MASK_CANONICAL_B64 = base64.b64encode(MASK_CANONICAL).decode()


def init(mask: bytes | None = MASK, data: bytes = UPLOAD, denoise: float = 0.9) -> dict:
    body = {"data_base64": b64(data), "denoise": denoise}
    if mask is not None:
        body["mask_base64"] = b64(mask)
    return body


@pytest.fixture
def rig(tmp_path, singguard, comfyui, nudenet, monkeypatch) -> Rig:
    """The usual rig, with the inpainting template for every checkpoint."""
    r = Rig(tmp_path, singguard, comfyui, nudenet, workflow=EXAMPLES / "sdxl+lora1+inpaint.json")
    monkeypatch.setattr(tempfile, "tempdir", str(r.systmp))
    return r


def mask_needles() -> list[bytes]:
    return [MASK, MASK_CANONICAL, base64.b64encode(MASK), MASK_CANONICAL_B64.encode()]


def holds_the_mask(rig: Rig) -> bool:
    for path in rig.root.rglob("*"):
        if path.is_file() and "models" not in path.relative_to(rig.root).parts:
            data = path.read_bytes()
            if any(n in data for n in mask_needles()):
                return True
    return False


def mask_went_to_a_guard(rig: Rig) -> bool:
    sent = [json.dumps(b).encode() for b in rig.singguard.requests] + list(rig.nudenet.bodies)
    return any(n in body for body in sent for n in mask_needles())


# --- the input passes ------------------------------------------------------------------------------------


def test_an_inpaint_run_sends_the_input_and_the_mask_and_scans_only_the_input(rig) -> None:
    record = rig.generate(rig.engine(), init_image=init(denoise=0.95))
    assert record.outcome == "passed" and record.outcome_reason is None
    assert record.request.workflow == "sdxl+lora1+inpaint"

    (graph,) = rig.comfyui.graphs
    by_class = {n["class_type"]: n for n in graph.values()}
    assert by_class["DarkroomLoadImageBase64"]["inputs"]["image"] == CANONICAL_B64  # the very bytes scanned
    assert by_class["DarkroomLoadMaskBase64"]["inputs"]["mask"] == MASK_CANONICAL_B64
    assert by_class["KSampler"]["inputs"]["denoise"] == 0.95
    assert by_class["ImageCompositeMasked"]["inputs"]["destination"] == ["13", 0]
    assert rig.comfyui.history == {}  # the graph, with the input and the mask in it, is deleted after the run

    ref = record.request.init_image
    assert ref.sha256 == hashlib.sha256(CANONICAL).hexdigest() and ref.denoise == 0.95
    assert ref.mask.sha256 == hashlib.sha256(MASK_CANONICAL).hexdigest()
    assert ref.mask.source_sha256 == hashlib.sha256(MASK).hexdigest()
    assert ref.mask.masked_fraction == 0.25

    # The input is scanned once per cell, exactly as for image-to-image; the mask goes to no guard.
    inputs = [b for b in rig.singguard.requests if b.get("image_base64") == CANONICAL_B64]
    assert [(policy_of(b), b["input"] == "") for b in inputs] == [("default", True), ("default", False)]
    assert len([b for b in rig.singguard.requests if is_image_scan(b)]) == 4  # input twice, output twice
    assert rig.nudenet.calls == 2  # input, output
    assert not mask_went_to_a_guard(rig)
    assert any(CANONICAL in body for body in rig.nudenet.bodies)  # the check above can see what NudeNet got
    assert record.guard.input.outcome == "passed"

    assert (rig.out / record.image.path).read_bytes() == rig.comfyui.png  # the output, released as usual
    assert sorted(p.name for p in rig.out.iterdir()) == [f"{record.id}.json", f"{record.id}.png"]
    assert not holds_the_mask(rig)
    assert rig.engine().store.load(record.id) == record
    probe = rig.systmp / "probe"
    probe.write_text(MASK_CANONICAL_B64)
    assert holds_the_mask(rig)  # the no-bytes check is only worth something if it can see a file
    probe.unlink()


def test_the_output_is_judged_as_usual(rig) -> None:
    nsfw(rig)
    rig.singguard.responder = lambda body: (
        200,
        ILLEGAL_A if is_image_scan(body) and body["image_base64"] != CANONICAL_B64 else safe_answer(),
    )
    record = rig.generate(rig.engine(), init_image=init())
    assert record.guard.input.outcome == "passed"
    assert (record.outcome, record.critical, record.image.path) == ("deleted", True, None)
    assert not rig.holds_the_image() and not holds_the_mask(rig)


# --- the input is refused: the mask changes nothing --------------------------------------------------------


def test_a_refused_input_with_a_mask_reaches_nothing_and_leaves_no_bytes(rig) -> None:
    nsfw(rig)
    rig.singguard.responder = input_answers(isolated=ILLEGAL_A)
    before = rig.files()
    record = rig.generate(rig.engine(), init_image=init())
    assert_refused_with_no_bytes(rig, record, before, "adult-granular:A")
    assert record.critical is True
    assert record.request.init_image.mask.sha256 == hashlib.sha256(MASK_CANONICAL).hexdigest()
    assert not holds_the_mask(rig) and not mask_went_to_a_guard(rig)


def test_an_exposed_input_under_sfw_with_a_mask_is_refused(rig) -> None:
    rig.nudenet.labels = EXPOSED
    before = rig.files()
    record = rig.generate(rig.engine(), init_image=init())
    assert_refused_with_no_bytes(rig, record, before, "sfw-exposed")
    assert not holds_the_mask(rig)


def test_pre_flight_refuses_first_and_the_record_names_the_mask(rig) -> None:
    rig.singguard.responder = lambda body: (200, safe_answer() if is_image_scan(body) else ILLEGAL_A)
    record = rig.generate(rig.engine(), init_image=init())
    assert record.outcome == "refused" and record.guard.input is None
    assert rig.comfyui.calls == [] and rig.nudenet.calls == 0
    assert record.request.init_image.mask.masked_fraction == 0.25
    assert not holds_the_mask(rig)


# --- the mask -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mask_base64", "message"),
    [
        (b64(mask_png(size=(32, 48))), "the mask is 32x48 but the init image is 48x32"),
        (b64(mask_png(size=(49, 32))), "the mask is 49x32"),
        ("not base64!", "the mask is not valid base64"),
        (b64(b"plain text, not a mask"), "the mask cannot be decoded"),
        (b64(mask_png(fmt="GIF")), "the mask cannot be decoded"),
        (b64(mask_png(fill=0)), "masks nothing"),
        (b64(mask_png(fill=127)), "masks nothing"),  # grey below the cut keeps
        (b64(mask_png(size=(4097, 2))), "4097x2"),
        ("A" * (4 * (MAX_INPUT_BYTES // 3) + 8), "the mask is over 16 MiB"),
    ],
)
def test_a_mask_that_cannot_be_used_is_a_request_error_and_touches_nothing(rig, mask_base64, message) -> None:
    with pytest.raises(RequestError, match=message):
        rig.generate(rig.engine(), init_image=init(mask=None) | {"mask_base64": mask_base64})
    assert rig.singguard.requests == [] and rig.comfyui.calls == [] and rig.nudenet.calls == 0
    assert list(rig.out.iterdir()) == []


def test_an_empty_mask_string_is_refused_by_the_model(rig) -> None:
    with pytest.raises(ValidationError, match="mask_base64"):
        rig.request(init_image=init(mask=None) | {"mask_base64": ""})


def test_the_mask_is_cut_at_128_into_a_one_bit_png() -> None:
    grey = Image.new("L", SIZE, 127)
    grey.paste(128, (0, 0, 12, 32))  # a quarter at the cut, the rest just under it
    buf = io.BytesIO()
    grey.save(buf, "PNG")
    mask = load_mask(b64(buf.getvalue()), SIZE)
    assert mask.masked_fraction == 0.25 and (mask.width, mask.height) == SIZE
    with Image.open(io.BytesIO(mask.png)) as out:
        assert out.mode == "1" and not out.info.get("exif")
        assert out.convert("L").getextrema() == (0, 255)
        assert out.convert("L").getpixel((11, 31)) == 255 and out.convert("L").getpixel((12, 0)) == 0


@pytest.mark.parametrize(
    "data",
    [
        mask_png(mode="RGB"),
        mask_png(mode="RGBA"),
        mask_png(mode="1"),
        mask_png(fmt="JPEG", quality=95),
        mask_png(fmt="WEBP", lossless=True),
    ],
)
def test_masks_in_other_modes_and_formats_come_out_the_same(data) -> None:
    mask = load_mask(b64(data), SIZE)
    assert mask.png == MASK_CANONICAL and mask.masked_fraction == 0.25


def test_the_mask_is_turned_upright_like_the_image() -> None:
    # Stored 32x48 with "rotate 90 on display": upright it is 48x32, the input's size.
    sideways = Image.new("L", (32, 48), 0)
    sideways.paste(255, (0, 0, 32, 12))
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    sideways.save(buf, "PNG", exif=exif)
    mask = load_mask(b64(buf.getvalue()), SIZE)
    assert (mask.width, mask.height) == SIZE and mask.masked_fraction == 0.25
    with pytest.raises(UnreadableImage, match="32x48"):
        load_mask(b64(mask_png(size=(32, 48))), SIZE)


def test_a_full_mask_is_allowed() -> None:
    assert load_mask(b64(mask_png(box=(0, 0, *SIZE))), SIZE).masked_fraction == 1.0


def test_the_mask_is_checked_against_the_upright_input(rig) -> None:
    # The input stored sideways (32x48, rotate 90): upright it is 48x32, and so must the mask be.
    sideways = Image.new("RGB", (32, 48), (40, 90, 200))
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    sideways.save(buf, "PNG", exif=exif)
    assert rig.generate(rig.engine(), init_image=init(data=buf.getvalue())).outcome == "passed"
    with pytest.raises(RequestError, match="the mask is 32x48 but the init image is 48x32"):
        rig.generate(rig.engine(), init_image=init(mask=mask_png(size=(32, 48)), data=buf.getvalue()))


# --- templates by mode ------------------------------------------------------------------------------------


def test_a_mask_with_no_inpaint_template_is_a_request_error(tmp_path, singguard, comfyui, nudenet) -> None:
    rig = Rig(tmp_path, singguard, comfyui, nudenet, workflow=EXAMPLES / "sdxl+lora1+img2img.json")
    with pytest.raises(RequestError, match="no inpainting workflow"):
        rig.generate(rig.engine(), init_image=init())
    assert singguard.requests == [] and comfyui.calls == []
    assert rig.generate(rig.engine(), init_image=init(mask=None)).outcome == "passed"  # image-to-image still works


def test_an_input_without_a_mask_does_not_use_an_inpaint_template(rig) -> None:
    with pytest.raises(RequestError, match="no image-to-image workflow"):
        rig.generate(rig.engine(), init_image=init(mask=None))
    with pytest.raises(RequestError, match="no workflow renders"):
        rig.generate(rig.engine())
    assert rig.singguard.requests == []


def test_a_directory_picks_the_inpaint_template_for_a_mask(tmp_path, singguard, comfyui, nudenet) -> None:
    rig = Rig(tmp_path, singguard, comfyui, nudenet, workflow=EXAMPLES)
    family = rig.root / "models" / "checkpoints" / "sdxl"
    family.mkdir()
    (family / "a.safetensors").write_bytes(b"a")
    eng = rig.engine()
    for body, name in [
        ({}, "sdxl+lora1"),
        ({"init_image": init(mask=None)}, "sdxl+lora1+img2img"),
        ({"init_image": init()}, "sdxl+lora1+inpaint"),
    ]:
        assert eng.prepare(rig.request(checkpoint="sdxl/a.safetensors", **body)).workflow.name == name


# --- never in a repr -----------------------------------------------------------------------------------------


def test_the_mask_never_shows_in_a_repr(rig) -> None:
    request = GenerateRequest(user_prompt="a cat", init_image=init())
    for text in (repr(request), str(request), repr(request.init_image), str(request.init_image)):
        assert b64(MASK)[:40] not in text
    assert "mask=<" in repr(request.init_image)
    prepared = rig.engine().prepare(rig.request(init_image=init()))
    assert MASK_CANONICAL_B64[:40] not in repr(prepared) and repr(MASK_CANONICAL)[:40] not in repr(prepared)
    params = rig.engine()._params(rig.request(init_image=init()), prepared, "a cat")
    assert MASK_CANONICAL_B64[:40] not in repr(params)
    assert params.init_mask == MASK_CANONICAL_B64


def test_a_picture_helper_sanity_check() -> None:
    assert Image.open(io.BytesIO(picture())).size == SIZE
