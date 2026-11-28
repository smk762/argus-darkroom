"""Pose input: a photo's body pose steers the generation through a pose ControlNet.

The photo gets the same checks as an output before it reaches anything else. Then argus-puppet (a fake here) reads
its keypoints and draws the skeleton, and the drawing is all ComfyUI gets.

The photo is a harmless generated picture (flat colours). Every bad verdict on it comes from a fake answer keyed on
its bytes, never from its content, and the "skeleton" is a few coloured squares.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import tempfile
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from fakes import FakePuppet, policy_of, safe_answer, skeleton_png, unsafe_answer
from fakes import person as fake_person
from PIL import Image
from pydantic import ValidationError
from rig import EXAMPLES, EXPOSED, ILLEGAL_A, Rig, is_image_scan, nsfw
from test_gpu import Recording, kinds
from test_server import BODY, client, stream_events

from argus_darkroom.images import MAX_CONTROL_COLOURS, UnreadableImage, load_control, load_input
from argus_darkroom.models import MAX_POSE_PEOPLE, GenerationRecord
from argus_darkroom.puppet import PoseUnavailable, PuppetClient
from argus_darkroom.request import GenerateRequest, PoseInput, RequestError
from argus_darkroom.workflow import WorkflowError, load_workflow, parse_workflow

FLAG_B = unsafe_answer("B. Real-World Crimes & Public Safety")
CANVAS = (64, 64)  # the rig's sampling size


def picture(size: tuple[int, int] = (40, 56), fmt: str = "PNG", colour=(200, 120, 40)) -> bytes:
    """A flat-colour picture with one stripe: nothing in it but colour."""
    image = Image.new("RGB", size, colour)
    for y in range(size[1]):
        image.putpixel((0, y), (250, 250, 250))
    buf = io.BytesIO()
    image.save(buf, fmt)
    return buf.getvalue()


def noise_png(size: tuple[int, int] = CANVAS) -> bytes:
    """What a photograph looks like to the colour check: thousands of colours. Seeded noise, no picture."""
    rng = random.Random(7)
    image = Image.frombytes("RGB", size, bytes(rng.randrange(256) for _ in range(size[0] * size[1] * 3)))
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


UPLOAD = picture()
CANONICAL = load_input(base64.b64encode(UPLOAD).decode()).png  # what is judged and what argus-puppet gets
CANONICAL_B64 = base64.b64encode(CANONICAL).decode()
PHOTO_NEEDLES = [UPLOAD, CANONICAL, base64.b64encode(UPLOAD), CANONICAL_B64.encode()]


def pose(data: bytes = UPLOAD, **kw) -> dict:
    return {"data_base64": base64.b64encode(data).decode(), **kw}


@pytest.fixture
def puppet() -> Iterator[FakePuppet]:
    with FakePuppet() as fake:
        yield fake


@pytest.fixture
def rig(tmp_path, singguard, comfyui, nudenet, puppet, monkeypatch) -> Rig:
    """The usual rig, with the pose template for every checkpoint, argus-puppet and a default ControlNet."""
    r = Rig(
        tmp_path,
        singguard,
        comfyui,
        nudenet,
        workflow=EXAMPLES / "sdxl+lora1+pose.json",
        puppet_url=puppet.url,
        pose_controlnet="openpose.safetensors",
    )
    r.puppet = puppet
    monkeypatch.setattr(tempfile, "tempdir", str(r.systmp))
    return r


def is_photo(body: dict) -> bool:
    return body.get("image_base64") == CANONICAL_B64


def photo_answers(isolated=None, with_prompt=None):
    """A SingGuard responder that answers the pose photo's scans as given and everything else safe. Keyed on the
    photo's bytes: the verdict is canned, not read from the picture."""

    def pick(spec, body):
        if spec is None:
            return safe_answer()
        return spec.get(policy_of(body), safe_answer()) if "safe" not in spec else spec

    def respond(body: dict):
        if not is_photo(body):
            return 200, safe_answer()
        return 200, pick(isolated if body["input"] == "" else with_prompt, body)

    return respond


def holds_the_photo(rig: Rig) -> bool:
    """Whether any file under the tmp dir (models excluded) holds the photo: as uploaded, as judged, or as base64."""
    for path in rig.root.rglob("*"):
        if path.is_file() and "models" not in path.relative_to(rig.root).parts:
            data = path.read_bytes()
            if any(n in data for n in PHOTO_NEEDLES):
                return True
    return False


def comfyui_saw_the_photo(rig: Rig) -> bool:
    sent = json.dumps(rig.comfyui.graphs).encode()
    return any(n in sent for n in (CANONICAL_B64.encode(), base64.b64encode(UPLOAD)))


# --- the photo passes: its skeleton, and only that, reaches ComfyUI ----------------------------------------------


def test_a_passed_photo_is_read_and_drawn_by_puppet_and_only_the_drawing_reaches_comfyui(rig) -> None:
    record = rig.generate(rig.engine(), pose=pose(strength=0.7))
    assert record.outcome == "passed" and record.outcome_reason is None

    # puppet: the judged photo, then its keypoints (numbers only) drawn at the generation's size
    assert rig.puppet.calls == ["POST /pose", "POST /render"]
    assert rig.puppet.posed == [CANONICAL]
    (drawn,) = rig.puppet.renders
    assert drawn["pose"] == {"keypoints": [fake_person(primary=True)["keypoints"]], "width": 40, "height": 56}
    assert drawn["options"] == {
        "body": True, "hands": True, "face": True, "width": 64, "height": 64, "fit": "contain",
    }  # fmt: skip
    assert not any(n in json.dumps(drawn).encode() for n in PHOTO_NEEDLES)

    control = load_control(skeleton_png(CANVAS), CANVAS)
    (graph,) = rig.comfyui.graphs
    loader = next(n for n in graph.values() if n["class_type"] == "DarkroomLoadImageBase64")
    assert loader["inputs"]["image"] == control.base64  # the drawing, never the photo
    net = next(n for n in graph.values() if n["class_type"] == "ControlNetLoader")
    assert net["inputs"]["control_net_name"] == "openpose.safetensors"
    apply = next(n for n in graph.values() if n["class_type"] == "ControlNetApplyAdvanced")
    assert apply["inputs"]["strength"] == 0.7
    assert not comfyui_saw_the_photo(rig)
    assert rig.comfyui.history == {}

    ref = record.request.pose
    assert (ref.width, ref.height, ref.strength, ref.fit) == (40, 56, 0.7, "contain")
    assert ref.sha256 == hashlib.sha256(CANONICAL).hexdigest()
    assert ref.source_sha256 == hashlib.sha256(UPLOAD).hexdigest()
    assert ref.controlnet.name == "openpose.safetensors"
    assert ref.controlnet.sha256 == hashlib.sha256(b"openpose.safetensors").hexdigest()
    # the record keeps the skeleton: every keypoint that was drawn, what read them, and the drawing's identity
    assert [p.model_dump(mode="json") for p in ref.people] == [
        {"keypoints": fake_person(primary=True)["keypoints"], "primary": True}
    ]
    assert ref.layout == "coco_wholebody_133"
    assert ref.estimator.package_version == "0.4.0" and ref.estimator.wire_version == "1.3"
    assert (ref.estimator.detector.name, ref.estimator.pose.name) == ("yolox_m", "dwpose-l-384x288")
    assert ref.estimator.pose.sha256 == "b" * 64
    assert ref.control.model_dump() == {"sha256": control.sha256, "width": 64, "height": 64}
    assert record.request.workflow == "sdxl+lora1+pose" and record.request.init_image is None
    assert record.guard.pose_input.outcome == "passed" and record.guard.input is None

    assert (rig.out / record.image.path).read_bytes() == rig.comfyui.png
    assert sorted(p.name for p in rig.out.iterdir()) == [f"{record.id}.json", f"{record.id}.png"]
    assert not holds_the_photo(rig)  # darkroom never stores the photo: the record has its hashes only
    assert rig.engine().store.load(record.id) == record
    probe = rig.systmp / "probe"
    probe.write_text(CANONICAL_B64)
    assert holds_the_photo(rig)  # the no-bytes checks are only worth something if this one can see a file
    probe.unlink()


def test_the_photo_is_scanned_like_an_output_before_puppet_sees_it(rig) -> None:
    nsfw(rig)
    seen_by_guard_first: list[int] = []
    rig.puppet.on_call = lambda path: seen_by_guard_first.append(sum(is_photo(b) for b in rig.singguard.requests))
    rig.generate(rig.engine(), pose=pose(), advisory_policies=["default"])
    photos = [b for b in rig.singguard.requests if is_photo(b)]
    assert [(policy_of(b), b["input"] == "") for b in photos] == [
        ("adult-granular", True),
        ("adult-granular", False),
        ("default", True),
        ("default", False),
    ]
    assert seen_by_guard_first == [4, 4]  # both puppet calls came after all four of the photo's scans
    assert CANONICAL in rig.nudenet.bodies[0]  # NudeNet scores the photo first, then the output
    kinds_seen = [k for k, _ in rig.events]
    assert kinds_seen[: kinds_seen.index("queued")] == ["preflight", *["pose_scan"] * 4, "pose"]
    told = dict(rig.events)["pose"]
    assert told["people"] == 1 and told["control"]["width"] == 64 and len(told["control"]["sha256"]) == 64
    # the drawing itself is not sent to a guard: only the photo and the output are judged
    control_b64 = load_control(skeleton_png(CANVAS), CANVAS).base64
    assert not any(b.get("image_base64") == control_b64 for b in rig.singguard.requests)


def test_puppet_is_called_before_the_gpu_lease_is_asked_for(rig) -> None:
    coordinator = Recording()
    leases_at_call: list[int] = []
    rig.puppet.on_call = lambda path: leases_at_call.append(len(coordinator.calls))
    record = rig.generate(rig.engine(gpu_coordinator=coordinator), pose=pose())
    assert record.outcome == "passed"
    assert leases_at_call == [0, 0]  # puppet leases its own GPU: darkroom holds none across the calls
    assert kinds(coordinator) == [("generate", "comfyui")] and not coordinator.nested


def test_the_callers_priority_reaches_puppets_pose_call(rig) -> None:
    rig.engine().generate(rig.request(pose=pose()), priority="batch")
    assert rig.puppet.headers[0]["x-argus-priority"] == "batch"
    assert rig.puppet.headers[0]["content-type"] == "image/png"


def test_the_request_chooses_what_is_drawn(rig) -> None:
    rig.puppet.people = [fake_person(10.0), fake_person(20.0, primary=True)]
    record = rig.generate(
        rig.engine(), pose=pose(hands=False, face=False, fit="skeleton", primary_only=True, strength=1.5)
    )
    (drawn,) = rig.puppet.renders
    assert drawn["pose"]["keypoints"] == [fake_person(20.0)["keypoints"]]  # the primary subject alone
    assert drawn["options"] | {"width": 0, "height": 0} == {
        "body": True, "hands": False, "face": False, "fit": "skeleton", "width": 0, "height": 0,
    }  # fmt: skip
    ref = record.request.pose
    assert (ref.hands, ref.face, ref.fit, ref.primary_only, ref.strength) == (False, False, "skeleton", True, 1.5)
    assert len(ref.people) == 1 and ref.people[0].primary


def test_everyone_is_drawn_by_default_up_to_the_cap(rig) -> None:
    rig.puppet.people = [fake_person(float(i), primary=i == 0) for i in range(MAX_POSE_PEOPLE + 3)]
    record = rig.generate(rig.engine(), pose=pose())
    assert len(rig.puppet.renders[0]["pose"]["keypoints"]) == MAX_POSE_PEOPLE
    assert len(record.request.pose.people) == MAX_POSE_PEOPLE


def test_the_request_can_name_its_controlnet(rig) -> None:
    other = rig.root / "models" / "controlnet" / "sdxl" / "union.safetensors"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"union")
    record = rig.generate(rig.engine(), pose=pose(controlnet="sdxl/union.safetensors"))
    assert record.request.pose.controlnet.model_dump() == {
        "name": "sdxl/union.safetensors",
        "sha256": hashlib.sha256(b"union").hexdigest(),
    }
    net = next(n for n in rig.comfyui.graphs[0].values() if n["class_type"] == "ControlNetLoader")
    assert net["inputs"]["control_net_name"] == "sdxl/union.safetensors"


# --- the photo is refused: nothing reaches puppet or ComfyUI, nothing is written -------------------------------


def assert_refused_with_no_bytes(rig: Rig, record: GenerationRecord, before: list[str], reason: str) -> None:
    assert record.outcome == "refused"
    assert record.outcome_reason == f"pose image: {reason}"
    assert record.image is None
    assert rig.puppet.calls == []  # a photo that did not pass is never sent on, not even to puppet
    assert rig.comfyui.calls == [] and rig.comfyui.graphs == []
    assert rig.files() == sorted([*before, str(Path("out") / f"{record.id}.json")])  # the record, and only it
    assert not holds_the_photo(rig)
    assert record.guard.pose_input is not None and record.guard.pose_input.outcome != "passed"
    ref = record.request.pose
    assert ref is not None and ref.sha256 and ref.people == [] and ref.control is None and ref.estimator is None
    kinds_seen = [k for k, _ in rig.events]
    assert "queued" not in kinds_seen and "pose" not in kinds_seen and kinds_seen[-1] == "complete"


def test_an_illegal_photo_is_refused_and_critical(rig) -> None:
    nsfw(rig)
    rig.singguard.responder = photo_answers(isolated=ILLEGAL_A)
    before = rig.files()
    record = rig.generate(rig.engine(), pose=pose())
    assert_refused_with_no_bytes(rig, record, before, "adult-granular:A")
    assert record.critical is True and record.guard.pose_input.outcome == "deleted"


def test_a_photo_flagged_under_sfw_is_refused_not_sent_to_review(rig) -> None:
    rig.singguard.responder = photo_answers(with_prompt=FLAG_B)
    before = rig.files()
    record = rig.generate(rig.engine(), pose=pose())
    assert_refused_with_no_bytes(rig, record, before, "default:B")
    assert record.critical is False and record.guard.pose_input.outcome == "review"


def test_an_exposed_photo_under_sfw_is_refused(rig) -> None:
    rig.nudenet.labels = EXPOSED
    before = rig.files()
    record = rig.generate(rig.engine(), pose=pose())
    assert_refused_with_no_bytes(rig, record, before, "sfw-exposed")


def test_a_photo_singguard_cannot_check_is_refused(rig) -> None:
    rig.singguard.responder = lambda body: (500, {}) if is_photo(body) else (200, safe_answer())
    before = rig.files()
    record = rig.generate(rig.engine(), pose=pose())
    assert_refused_with_no_bytes(rig, record, before, "unchecked")


def test_a_text_refusal_comes_first_and_the_photo_is_never_scanned(rig) -> None:
    rig.singguard.responder = lambda body: (200, safe_answer() if is_image_scan(body) else FLAG_B)
    record = rig.generate(rig.engine(), pose=pose())
    assert record.outcome == "refused" and record.outcome_reason == "default:B"
    assert not any(is_image_scan(b) for b in rig.singguard.requests)
    assert rig.puppet.calls == [] and rig.comfyui.graphs == []
    assert record.guard.pose_input is None
    assert record.request.pose.sha256 == hashlib.sha256(CANONICAL).hexdigest()  # named by its hashes only
    assert not holds_the_photo(rig)


def test_engine_render_will_not_draw_from_a_photo_that_was_not_scanned(rig) -> None:
    engine = rig.engine()
    request = rig.request(pose=pose())
    with pytest.raises(RequestError, match="only after its photo's scan passed"):
        engine.render(request)
    assert rig.comfyui.graphs == [] and rig.puppet.calls == []


# --- the photo passed, and the pose could not be read or drawn: a failed record, nothing rendered ----------------


def assert_failed_before_comfyui(rig: Rig, record: GenerationRecord, reason: str) -> None:
    assert record.outcome == "failed" and record.critical is False
    assert record.outcome_reason.startswith("pose: ") and reason in record.outcome_reason
    assert record.image is None and rig.comfyui.graphs == [] and rig.comfyui.calls == []
    assert record.guard.pose_input.outcome == "passed"
    assert record.request.pose.control is None
    assert not holds_the_photo(rig)
    assert sorted(p.name for p in rig.out.iterdir()) == [f"{record.id}.json"]


def test_a_photo_with_nobody_in_it_fails_and_nothing_is_rendered(rig) -> None:
    rig.puppet.people = []
    record = rig.generate(rig.engine(), pose=pose())
    assert_failed_before_comfyui(rig, record, "found nobody in the pose image")
    assert rig.puppet.calls == ["POST /pose"]


def test_a_drawing_that_is_a_photograph_never_reaches_comfyui(rig) -> None:
    rig.puppet.drawing = noise_png()  # a puppet that answers with a picture: nothing checked it
    record = rig.generate(rig.engine(), pose=pose())
    assert_failed_before_comfyui(rig, record, f"more than {MAX_CONTROL_COLOURS} colours")
    assert base64.b64encode(rig.puppet.drawing) not in (rig.out / f"{record.id}.json").read_bytes()


@pytest.mark.parametrize(
    ("drawing", "reason"),
    [
        (skeleton_png((32, 64)), "is 32x64 but the generation is 64x64"),
        (skeleton_png(CANVAS, dots=0), "is blank"),
        (b"not a png", "cannot be decoded as a PNG"),
        (picture(CANVAS, "JPEG"), "cannot be decoded as a PNG"),
    ],
)
def test_a_drawing_that_is_not_a_skeleton_of_the_generation_size_fails(rig, drawing, reason) -> None:
    rig.puppet.drawing = drawing
    assert_failed_before_comfyui(rig, rig.generate(rig.engine(), pose=pose()), reason)


@pytest.mark.parametrize(("route", "status"), [("pose", 503), ("pose", 500), ("render", 413), ("render", 503)])
def test_a_puppet_error_fails_the_run_and_never_echoes_its_answer(rig, route, status) -> None:
    setattr(rig.puppet, f"{route}_status", status)
    record = rig.generate(rig.engine(), pose=pose())
    assert_failed_before_comfyui(rig, record, f"argus-puppet answered HTTP {status} to /{route}")
    assert "detail" not in record.outcome_reason


def test_an_unreachable_puppet_fails_the_run_without_naming_its_url(rig) -> None:
    rig.settings = replace(rig.settings, puppet_url="http://127.0.0.1:1")
    record = rig.generate(rig.engine(), pose=pose())
    assert_failed_before_comfyui(rig, record, "argus-puppet could not be reached for /pose")
    assert "127.0.0.1" not in record.outcome_reason


@pytest.mark.parametrize(
    "answer",
    [
        ["not", "an", "object"],
        {"width": 40, "height": 56, "people": [{"keypoints": [[1, 2, 0.5]] * 17}]},  # not 133 points
        {"width": 40, "height": 56, "people": [{"keypoints": [[1, 2, "NaN"]] * 133}]},
        {"width": 40, "height": 56, "layout": "coco-17", "people": [fake_person()]},
    ],
)
def test_an_answer_that_is_not_a_pose_result_fails(rig, answer) -> None:
    rig.puppet.pose_answer = answer
    record = rig.generate(rig.engine(), pose=pose())
    assert_failed_before_comfyui(rig, record, "argus-puppet")
    assert rig.puppet.calls == ["POST /pose"]


def test_keypoints_from_another_frame_are_refused(rig) -> None:
    rig.puppet.frame = (80, 112)  # puppet says it read an image of another size: its keypoints would not fit
    record = rig.generate(rig.engine(), pose=pose())
    assert_failed_before_comfyui(rig, record, "read an image of 80x112, not the 40x56 it was sent")


def test_a_redirect_from_puppet_is_not_followed(rig) -> None:
    with FakePuppet() as elsewhere:
        rig.puppet.redirect_to = elsewhere.url + "/pose"
        record = rig.generate(rig.engine(), pose=pose())
        assert elsewhere.calls == []  # the photo is sent to the configured puppet and nowhere else
    assert_failed_before_comfyui(rig, record, "a redirect is not followed")


# --- requests that cannot be served: HTTP 400 before anything happens -------------------------------------------


def refused_before_anything(rig: Rig, match: str, **kw) -> None:
    with pytest.raises(RequestError, match=match):
        rig.engine().begin(rig.request(**kw))
    assert rig.singguard.requests == [] and rig.puppet.calls == [] and rig.comfyui.graphs == []
    assert list(rig.out.glob("*")) == []


def test_a_pose_input_without_puppet_is_a_request_error(rig) -> None:
    rig.settings = replace(rig.settings, puppet_url="")
    refused_before_anything(rig, "needs argus-puppet", pose=pose())


def test_a_pose_input_without_a_controlnet_is_a_request_error(rig) -> None:
    rig.settings = replace(rig.settings, pose_controlnet="")
    refused_before_anything(rig, "needs a ControlNet", pose=pose())


@pytest.mark.parametrize("name", ["missing.safetensors", "../checkpoints/base.safetensors"])
def test_a_controlnet_that_is_not_under_the_models_root_is_a_request_error(rig, name) -> None:
    refused_before_anything(rig, "was not found under the models root", pose=pose(controlnet=name))


def test_a_pose_input_with_no_pose_template_is_a_request_error(rig) -> None:
    rig.settings = replace(rig.settings, workflow=EXAMPLES / "sdxl+lora1.json")
    refused_before_anything(rig, "no pose workflow renders this checkpoint", pose=pose())


def test_a_pose_input_and_an_init_image_together_are_a_request_error(rig) -> None:
    init = {"data_base64": base64.b64encode(picture()).decode()}
    refused_before_anything(rig, "cannot be combined with an init image", pose=pose(), init_image=init)


@pytest.mark.parametrize(
    ("data", "reason"),
    [(b"not an image", "the pose image cannot be decoded"), (picture((5000, 8)), "the pose image is 5000x8")],
)
def test_a_photo_that_cannot_be_used_is_a_request_error(rig, data, reason) -> None:
    refused_before_anything(rig, reason, pose=pose(data))


def test_a_request_without_a_pose_does_not_use_the_pose_template(rig) -> None:
    with pytest.raises(RequestError, match="no workflow renders this checkpoint"):
        rig.engine().begin(rig.request())  # the rig's one template is a pose template


@pytest.mark.parametrize(
    "extra",
    [
        {"control_image": "aGVsbG8="},  # a caller's own control image would be a picture nobody checked
        {"keypoints": [[0, 0, 1]] * 133},  # and its own keypoints a pose from no checked photo
        {"image_base64": "aGVsbG8="},
    ],
)
def test_a_request_cannot_carry_its_own_control_image_or_keypoints(extra) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GenerateRequest(**BODY | {"pose": pose() | extra})


@pytest.mark.parametrize(
    "bad",
    [
        {"strength": 0},
        {"strength": 2.5},
        {"fit": "stretch"},
        {"body": False, "hands": False, "face": False},
        {"data_base64": ""},
        {"controlnet": ""},
    ],
)
def test_pose_options_are_validated(bad) -> None:
    with pytest.raises(ValidationError):
        PoseInput(**pose() | bad)


def test_the_photo_never_shows_in_a_repr(rig) -> None:
    request = rig.request(pose=pose())
    prepared = rig.engine().prepare(request)
    for text in (repr(request), str(request.pose), repr(prepared)):
        assert CANONICAL_B64 not in text and base64.b64encode(UPLOAD).decode() not in text
    assert "base64 chars" in repr(request.pose)


# --- the control image check and the client, on their own -----------------------------------------------------


def test_load_control_takes_a_drawing_and_refuses_a_picture() -> None:
    control = load_control(skeleton_png((96, 64)), (96, 64))
    assert (control.width, control.height) == (96, 64) and "base64" not in repr(control)
    assert hashlib.sha256(control.png).hexdigest() == control.sha256
    with pytest.raises(UnreadableImage, match="not a skeleton drawing"):
        load_control(noise_png((96, 64)), (96, 64))


def test_load_control_drops_metadata() -> None:
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text("comment", "something a drawing should not carry")
    buf = io.BytesIO()
    Image.open(io.BytesIO(skeleton_png(CANVAS))).save(buf, "PNG", pnginfo=info)
    assert b"something a drawing" in buf.getvalue()
    assert b"something a drawing" not in load_control(buf.getvalue(), CANVAS).png


@pytest.mark.parametrize("url", ["puppet:8220", "ftp://puppet", "", "http://"])
def test_the_client_refuses_a_url_that_is_not_http(url) -> None:
    with pytest.raises(ValueError, match="not an absolute http"):
        PuppetClient(url)


def test_the_client_raises_pose_unavailable_for_an_unreachable_host() -> None:
    with pytest.raises(PoseUnavailable, match="could not be reached for /pose"):
        PuppetClient("http://127.0.0.1:1", timeout=1, deadline=2).read(CANONICAL, (40, 56))


# --- templates ---------------------------------------------------------------------------------------------


def pose_template() -> dict:
    return json.loads((EXAMPLES / "sdxl+pose.json").read_text())


@pytest.mark.parametrize("name", ["sdxl+pose.json", "sdxl+lora1+pose.json"])
def test_the_pose_examples_are_pose_templates(name) -> None:
    wf = load_workflow(EXAMPLES / name)
    assert wf.mode == "pose" and not wf.img2img
    assert {"$pose_image", "$controlnet", "$pose_strength"} <= wf.slots
    assert wf.describe()["mode"] == "pose"


@pytest.mark.parametrize("slot", ["$pose_image", "$controlnet", "$pose_strength"])
def test_a_pose_template_needs_all_three_slots(slot) -> None:
    template = pose_template()
    for node in template.values():
        for key, value in node["inputs"].items():
            if value == slot:
                node["inputs"][key] = "fixed"
    with pytest.raises(WorkflowError):
        parse_workflow(template)


def test_a_pose_template_has_one_loader_and_its_image_is_the_skeleton() -> None:
    template = pose_template()
    template["90"] = {"class_type": "DarkroomLoadImageBase64", "inputs": {"image": "aGVsbG8="}}
    with pytest.raises(WorkflowError, match="exactly one DarkroomLoadImageBase64"):
        parse_workflow(template)  # a second loader with a fixed image would bring in one nobody checked
    template = pose_template()
    template["90"] = {"class_type": "ImageScale", "inputs": {"image": "$pose_image"}}
    with pytest.raises(WorkflowError, match=r"\$pose_image must appear once"):
        parse_workflow(template)


@pytest.mark.parametrize("slot", ["$init_image", "$denoise", "$init_mask"])
def test_a_pose_template_takes_no_image_to_image_slot(slot) -> None:
    template = pose_template()
    template["90"] = {"class_type": "Anything", "inputs": {"x": slot}}
    with pytest.raises(WorkflowError, match="pose conditioning is text-to-image"):
        parse_workflow(template)


def test_a_pose_template_still_refuses_a_node_that_loads_from_disk() -> None:
    template = pose_template()
    template["90"] = {"class_type": "LoadImage", "inputs": {"image": "pose.png"}}
    with pytest.raises(WorkflowError, match="loads an image darkroom never checked"):
        parse_workflow(template)


def test_a_pose_template_in_a_set_must_be_named_for_it(tmp_path) -> None:
    from argus_darkroom.workflow import load_workflows

    (tmp_path / "sdxl.json").write_text(json.dumps(pose_template()))
    with pytest.raises(WorkflowError, match=r"has \$pose_image but its name says neither"):
        load_workflows(tmp_path)
    (tmp_path / "sdxl.json").unlink()
    (tmp_path / "sdxl+pose.json").write_text((EXAMPLES / "sdxl.json").read_text())
    with pytest.raises(WorkflowError, match=r"has no \$init_image but its name says pose"):
        load_workflows(tmp_path)


# --- the server ----------------------------------------------------------------------------------------------


def test_the_stream_reports_the_photos_scan_and_the_pose(rig) -> None:
    events = stream_events(client(rig), {"pose": pose()})
    kinds_seen = [k for k, _ in events]
    assert kinds_seen[:4] == ["preflight", "pose_scan", "pose_scan", "pose"]
    record = events[-1][1]["record"]
    assert record["outcome"] == "passed" and len(record["request"]["pose"]["people"][0]["keypoints"]) == 133
    assert CANONICAL_B64 not in json.dumps(events)  # no event carries the photo


def test_a_pose_request_that_cannot_be_served_is_a_400(rig) -> None:
    response = client(rig, puppet_url="").post("/generate/stream", json=BODY | {"pose": pose()})
    assert response.status_code == 400 and "needs argus-puppet" in response.json()["detail"]


def test_health_and_models_report_pose_support(rig) -> None:
    c = client(rig)
    health = c.get("/health").json()
    assert health["pose_available"] is True and health["backends"]["puppet"] is True
    assert health["checks"]["puppet"]["ok"] is True and health["checks"]["puppet"]["required"] is False
    assert health["status"] == "ok"
    models = c.get("/models").json()
    assert models["controlnets"] == [{"name": "openpose.safetensors"}]
    assert models["default_pose_controlnet"] == "openpose.safetensors"
    assert models["checkpoints"][0]["pose_lora_counts"] == [1]


def test_a_puppet_that_is_down_is_reported_and_never_degrades_health(rig) -> None:
    health = client(rig, puppet_url="http://127.0.0.1:1").get("/health").json()
    assert health["pose_available"] is True and health["backends"]["puppet"] is False
    assert health["checks"]["puppet"]["ok"] is False and health["status"] == "ok"


def test_without_puppet_health_says_pose_is_unavailable(rig) -> None:
    health = client(rig, puppet_url="").get("/health").json()
    assert health["pose_available"] is False and "puppet" not in health["checks"]
