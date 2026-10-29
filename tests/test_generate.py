"""The whole flow against fakes: pre-flight, render, scan, decide, release or destroy.

An illegal verdict comes from a canned answer, never from image content: the fake image is a grey square.
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import pytest
from fake_comfyui import FakeComfyUI, event
from fakes import FakeNudeNet, FakeSingGuard, policy_of, safe_answer, unsafe_answer

from argus_darkroom.age import AgeUnavailable, CuratorAgeEstimator
from argus_darkroom.comfyui import ComfyUIClient
from argus_darkroom.config import Settings
from argus_darkroom.engine import Engine
from argus_darkroom.guard import Guard
from argus_darkroom.models import FaceAge, GenerationRecord, SamplingParams
from argus_darkroom.prompt import frame
from argus_darkroom.request import GenerateRequest, RequestError

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "workflows"
SAMPLING = SamplingParams(sampler="euler", scheduler="normal", steps=4, cfg=5.0, width=64, height=64)
EXPOSED = {"FEMALE_BREAST_EXPOSED": 0.9}
ILLEGAL_A = unsafe_answer("A. Minors")


class Rig:
    """Fake SingGuard, ComfyUI and NudeNet wired to an engine, in a tmp dir."""

    def __init__(
        self, tmp_path: Path, singguard: FakeSingGuard, comfyui: FakeComfyUI, nudenet: FakeNudeNet, **kw
    ) -> None:
        self.root = tmp_path
        self.out = tmp_path / "out"
        self.systmp = tmp_path / "systmp"
        self.systmp.mkdir()
        models = tmp_path / "models"
        for kind, name in [("checkpoints", "base.safetensors"), ("loras", "style.safetensors")]:
            (models / kind).mkdir(parents=True, exist_ok=True)
            (models / kind / name).write_bytes(name.encode())
        self.singguard, self.comfyui, self.nudenet = singguard, comfyui, nudenet
        settings = Settings(
            singguard_url=singguard.url,
            nudenet_url=kw.pop("nudenet_url", nudenet.url),
            output_root=self.out,
            models_root=models,
            workflow=EXAMPLES / "sdxl_lora.json",
            **kw,
        )
        self.settings = settings
        self.events: list[tuple[str, dict]] = []

    def engine(self, **kw) -> Engine:
        return Engine(
            self.settings,
            guard=Guard(self.settings.singguard_url),
            comfyui=ComfyUIClient(self.comfyui.url, ws_url=self.comfyui.ws_url, timeout=5),
            **kw,
        )

    def request(self, **kw) -> GenerateRequest:
        base = {
            "user_prompt": "a cat",
            "negative_prompt": "blurry",
            "seed": 3,
            "sampling": SAMPLING,
            "checkpoint": "base.safetensors",
            "loras": [{"name": "style.safetensors", "trigger_words": ["sks"]}],
        }
        return GenerateRequest(**(base | kw))

    def generate(self, engine: Engine, **kw) -> GenerationRecord:
        return engine.generate(self.request(**kw), lambda kind, data: self.events.append((kind, data)))

    def files(self) -> list[str]:
        """Every file under the output root and the (redirected) system temp dir."""
        return sorted(str(p.relative_to(self.root)) for base in (self.out, self.systmp) for p in base.rglob("*"))

    def holds_the_image(self) -> bool:
        """Whether the PNG, or any base64 of it, is in any file under the tmp dir (models excluded)."""
        png = self.comfyui.png
        for path in self.root.rglob("*"):
            if path.is_file() and "models" not in path.relative_to(self.root).parts:
                data = path.read_bytes()
                if png in data or base64.b64encode(png) in data:
                    return True
        return False


@pytest.fixture
def rig(tmp_path, singguard, comfyui, nudenet, monkeypatch) -> Rig:
    r = Rig(tmp_path, singguard, comfyui, nudenet)
    monkeypatch.setattr(tempfile, "tempdir", str(r.systmp))
    return r


def nsfw(rig: Rig) -> None:
    rig.settings = Settings(**{**rig.settings.__dict__, "allow_nsfw": True, "policy": "adult-granular"})


def is_image_scan(body: dict) -> bool:
    return "image_base64" in body


def answers(preflight=None, isolated=None, with_prompt=None):
    """A SingGuard responder: an answer (or per-policy dict of answers) per context; default safe."""

    def pick(spec, body):
        if spec is None:
            return safe_answer()
        return spec.get(policy_of(body), safe_answer()) if isinstance(spec, dict) and "safe" not in spec else spec

    def respond(body: dict):
        if not is_image_scan(body):
            return 200, pick(preflight, body)
        return 200, pick(isolated if body["input"] == "" else with_prompt, body)

    return respond


# --- released --------------------------------------------------------------------------------------------


def test_a_passed_image_is_written_served_and_recorded(rig) -> None:
    record = rig.generate(rig.engine())
    assert record.outcome == "passed" and record.outcome_reason is None and not record.critical
    assert record.image.path == f"{record.id}.png"
    assert (rig.out / record.image.path).read_bytes() == rig.comfyui.png
    assert record.image.width == record.image.height == 64
    assert len(record.image.sha256) == 64 and record.image.phash
    assert record.image.blur is False
    assert sorted(p.name for p in rig.out.iterdir()) == [f"{record.id}.json", f"{record.id}.png"]
    assert rig.engine().store.load(record.id) == record
    assert rig.comfyui.history == {}
    assert rig.holds_the_image()  # the no-bytes checks below are only worth something if this one can see a file


def test_every_cell_is_scanned_isolated_and_with_the_prompt(rig) -> None:
    nsfw(rig)
    record = rig.generate(rig.engine(), advisory_policies=["default"])
    images = [b for b in rig.singguard.requests if is_image_scan(b)]
    assert [(policy_of(b), b["input"] == "") for b in images] == [
        ("adult-granular", True),
        ("adult-granular", False),
        ("default", True),
        ("default", False),
    ]
    framed = frame("a cat, sks", "blurry")
    assert {b["input"] for b in images} == {"", framed}  # nothing but the framed prompt: no age, no extras
    assert set(record.guard.scans) == {"adult-granular", "default"}
    assert all(c.isolated.state == c.with_prompt.state == "safe" for c in record.guard.scans.values())
    assert [k for k, _ in rig.events].count("scan") == 4
    assert [k for k, _ in rig.events][0] == "preflight" and rig.events[-1][0] == "complete"
    assert ("progress", {"step": 1, "of": 2}) in rig.events


def test_a_disagreement_is_review_and_shows_in_the_record(rig) -> None:
    flag = unsafe_answer("B. Real-World Crimes & Public Safety")
    rig.singguard.responder = answers(isolated=safe_answer(), with_prompt=flag)
    record = rig.generate(rig.engine())
    assert record.outcome == "review" and record.outcome_reason == "default:B"
    scans = record.guard.scans["default"]
    assert (scans.isolated.state, scans.with_prompt.state) == ("safe", "flagged")
    assert (rig.out / record.image.path).exists()  # review is served, behind its shield


# --- destroyed: nothing written anywhere ---------------------------------------------------------------


def assert_no_bytes(rig: Rig, record: GenerationRecord, before: list[str]) -> None:
    assert record.image is not None and record.image.path is None
    assert rig.files() == sorted([*before, str(Path("out") / f"{record.id}.json")])  # the record, and only it
    assert not rig.holds_the_image()
    assert rig.comfyui.history == {}


@pytest.mark.parametrize("where", ["isolated", "with_prompt"])
def test_an_illegal_cell_deletes_and_leaves_no_bytes(rig, where) -> None:
    nsfw(rig)  # `default` has no illegal rules: its A is all sexual content, so it can only send to review
    rig.singguard.responder = answers(**{where: ILLEGAL_A})
    before = rig.files()
    record = rig.generate(rig.engine())
    assert (record.outcome, record.critical, record.outcome_reason) == ("deleted", True, "adult-granular:A")
    assert_no_bytes(rig, record, before)
    # the record still holds everything needed to regenerate it
    assert record.request.seed == 3 and record.request.checkpoint.sha256 and record.request.loras[0].sha256
    assert record.request.workflow_sha256 and record.request.final_prompt == "a cat, sks"
    assert len(record.image.sha256) == 64


def test_under_default_the_same_rule_only_sends_to_review(rig) -> None:
    rig.singguard.responder = answers(with_prompt=ILLEGAL_A)
    record = rig.generate(rig.engine())
    assert (record.outcome, record.critical) == ("review", False)


def test_an_illegal_verdict_under_an_advisory_policy_deletes(rig) -> None:
    nsfw(rig)
    rig.singguard.responder = answers(isolated={"adult": ILLEGAL_A})
    before = rig.files()
    record = rig.generate(rig.engine(), advisory_policies=["adult"])
    assert (record.outcome, record.critical, record.outcome_reason) == ("deleted", True, "adult:A")
    assert_no_bytes(rig, record, before)


def test_the_unchecked_enforcing_cell_deletes_but_is_not_critical(rig) -> None:
    rig.singguard.responder = lambda body: (500, {}) if is_image_scan(body) else (200, safe_answer())
    before = rig.files()
    record = rig.generate(rig.engine())
    assert (record.outcome, record.outcome_reason, record.critical) == ("deleted", "unchecked", False)
    assert_no_bytes(rig, record, before)


def test_singguard_going_down_after_preflight_serves_nothing(rig) -> None:
    calls = []

    def respond(body):
        calls.append(body)
        if is_image_scan(body):
            raise ConnectionError  # the handler dies: the client sees a failed request
        return 200, safe_answer()

    rig.singguard.responder = respond
    before = rig.files()
    record = rig.generate(rig.engine())
    assert record.outcome == "deleted"
    assert_no_bytes(rig, record, before)


def test_an_advisory_cell_that_fails_changes_nothing(rig) -> None:
    nsfw(rig)
    rig.singguard.responder = lambda body: (
        (500, {}) if is_image_scan(body) and policy_of(body) == "default" else (200, safe_answer())
    )
    record = rig.generate(rig.engine(), advisory_policies=["default"])
    assert record.outcome == "passed"
    assert record.guard.scans["default"].isolated.state == "unchecked"


# --- SFW and NudeNet ------------------------------------------------------------------------------------


def test_sfw_exposed_is_omitted_not_written_and_not_critical(rig) -> None:
    rig.nudenet.labels = EXPOSED
    before = rig.files()
    record = rig.generate(rig.engine())
    assert (record.outcome, record.outcome_reason, record.critical) == ("omitted", "sfw-exposed", False)
    assert record.guard.nudenet.exposed_score == 0.9
    assert_no_bytes(rig, record, before)


def test_sfw_exposed_score_exactly_at_the_floor_is_omitted(rig) -> None:
    rig.nudenet.labels = {"FEMALE_BREAST_EXPOSED": 0.2}
    assert rig.generate(rig.engine()).outcome == "omitted"


def test_sfw_below_the_reporting_floor_is_served(rig) -> None:
    rig.nudenet.labels = {"FEMALE_BREAST_EXPOSED": 0.19}  # NudeNet never reports under 0.2: no detection
    assert rig.generate(rig.engine()).outcome == "passed"


def test_sfw_covered_parts_do_not_count(rig) -> None:
    rig.nudenet.labels = {"FEMALE_BREAST_COVERED": 0.95, "FACE_FEMALE": 0.9}
    assert rig.generate(rig.engine()).outcome == "passed"


def test_sfw_nudenet_timeout_is_omitted(rig) -> None:
    rig.nudenet.status = 500
    before = rig.files()
    record = rig.generate(rig.engine())
    assert (record.outcome, record.outcome_reason, record.critical) == ("omitted", "sfw-unchecked", False)
    assert record.guard.nudenet is None
    assert_no_bytes(rig, record, before)


def test_sfw_exposed_and_illegal_is_deleted_and_critical(rig) -> None:
    rig.nudenet.labels = EXPOSED
    rig.singguard.responder = answers(with_prompt=ILLEGAL_A)
    before = rig.files()
    record = rig.generate(rig.engine(), enforcing_policy="default", advisory_policies=["adult"])
    assert (record.outcome, record.critical) == ("deleted", True)
    assert_no_bytes(rig, record, before)


def test_sfw_generation_without_nudenet_is_a_request_error_and_touches_nothing(
    tmp_path, singguard, comfyui, nudenet
) -> None:
    rig = Rig(tmp_path, singguard, comfyui, nudenet, nudenet_url="")
    with pytest.raises(RequestError, match="NudeNet"):
        rig.generate(rig.engine())
    assert singguard.requests == [] and comfyui.calls == []
    assert list(rig.out.iterdir()) == []


@pytest.mark.parametrize(
    ("labels", "threshold", "blur"),
    [({}, 0.2, False), (EXPOSED, 0.2, True), (EXPOSED, 0.95, False), (EXPOSED, 0.9, True)],
)
def test_nsfw_nudenet_blurs_by_the_threshold_and_never_omits(rig, labels, threshold, blur) -> None:
    nsfw(rig)
    rig.nudenet.labels = labels
    record = rig.generate(rig.engine(), nsfw_threshold=threshold)
    assert record.outcome == "passed"
    assert record.image.blur is blur
    assert record.guard.nudenet.blur is blur and record.guard.nudenet.threshold == threshold
    assert (rig.out / record.image.path).exists()


def test_nsfw_with_no_nudenet_answer_blurs(rig) -> None:
    nsfw(rig)
    rig.nudenet.status = 500
    record = rig.generate(rig.engine())
    assert record.outcome == "passed" and record.image.blur is True and record.guard.nudenet is None


# --- refused / failed ----------------------------------------------------------------------------------


def test_a_refusal_never_reaches_comfyui_and_is_recorded(rig) -> None:
    rig.singguard.responder = answers(preflight=unsafe_answer("B. Real-World Crimes & Public Safety"))
    record = rig.generate(rig.engine())
    assert (record.outcome, record.outcome_reason, record.image) == ("refused", "default:B", None)
    assert rig.comfyui.calls == []
    assert not [b for b in rig.singguard.requests if is_image_scan(b)]
    assert [k for k, _ in rig.events] == ["preflight", "complete"]
    assert sorted(p.suffix for p in rig.out.iterdir()) == [".json"]


def test_a_model_that_is_missing_fails_before_pre_flight(rig) -> None:
    with pytest.raises(RequestError, match="not found"):
        rig.generate(rig.engine(), checkpoint="gone.safetensors")
    assert rig.singguard.requests == [] and rig.comfyui.calls == []


def test_a_comfyui_failure_is_a_failed_record_with_no_image(tmp_path, singguard, nudenet, monkeypatch) -> None:
    def script(fake, pid, graph):
        return [event("execution_error", prompt_id=pid, node_type="KSampler", exception_message="out of memory")]

    with FakeComfyUI(script) as comfyui:
        rig = Rig(tmp_path, singguard, comfyui, nudenet)
        record = rig.generate(rig.engine())
        assert (record.outcome, record.critical) == ("failed", False)
        assert "out of memory" in record.outcome_reason
        assert record.image is None and record.guard.scans == {}
        assert record.request.seed == 3 and record.request.checkpoint.sha256
        assert comfyui.history == {}
        assert [p.suffix for p in rig.out.iterdir()] == [".json"]
        assert not [b for b in singguard.requests if is_image_scan(b)]


def test_an_undecodable_image_is_a_failed_record(tmp_path, singguard, nudenet) -> None:
    with FakeComfyUI() as comfyui:
        comfyui.png = b"\x89PNG\r\n\x1a\n" + b"not really a png"
        rig = Rig(tmp_path, singguard, comfyui, nudenet)
        record = rig.generate(rig.engine())
        assert record.outcome == "failed" and "cannot be decoded" in record.outcome_reason
        assert [p.suffix for p in rig.out.iterdir()] == [".json"]


def test_if_the_record_cannot_be_saved_the_released_image_is_removed(rig, monkeypatch) -> None:
    eng = rig.engine()

    def boom(record):
        raise OSError("disk full")

    monkeypatch.setattr(eng.store, "save", boom)
    with pytest.raises(OSError):
        rig.generate(eng)
    assert list(rig.out.iterdir()) == []


# --- age -----------------------------------------------------------------------------------------------


class FakeAge:
    def __init__(self, *ages: float, error: Exception | None = None) -> None:
        self.ages, self.error = ages, error

    def estimate(self, data: bytes) -> list[FaceAge]:
        if self.error:
            raise self.error
        return [FaceAge(age_estimate_years=a, bbox=(0, 0, 10, 10)) for a in self.ages]


def test_a_minor_estimate_with_exposure_deletes_and_is_critical(rig) -> None:
    nsfw(rig)
    rig.nudenet.labels = EXPOSED
    before = rig.files()
    record = rig.generate(rig.engine(age=FakeAge(31, 16.5)))
    assert (record.outcome, record.outcome_reason, record.critical) == ("deleted", "minor-exposed", True)
    assert record.guard.age.minor_exposed and record.guard.age.source == "mivolo"
    assert [f.age_estimate_years for f in record.guard.age.faces] == [31, 16.5]
    assert_no_bytes(rig, record, before)


def test_the_minor_rule_ignores_the_users_nudenet_threshold(rig) -> None:
    nsfw(rig)
    rig.nudenet.labels = {"FEMALE_BREAST_EXPOSED": 0.3}
    assert rig.generate(rig.engine(age=FakeAge(12)), nsfw_threshold=1.0).outcome == "deleted"


def test_a_minor_estimate_without_exposure_is_recorded_not_deleted(rig) -> None:
    nsfw(rig)
    record = rig.generate(rig.engine(age=FakeAge(12)))
    assert record.outcome == "passed"
    assert record.guard.age.faces[0].age_estimate_years == 12 and not record.guard.age.minor_exposed


def test_an_adult_estimate_with_exposure_is_served(rig) -> None:
    nsfw(rig)
    rig.nudenet.labels = EXPOSED
    assert rig.generate(rig.engine(age=FakeAge(18, 40))).outcome == "passed"


@pytest.mark.parametrize(
    "estimator", [None, FakeAge(error=AgeUnavailable("no weights")), FakeAge(error=RuntimeError("boom"))]
)
def test_without_the_age_check_the_record_says_null_and_the_scan_carries_on(rig, estimator) -> None:
    nsfw(rig)
    engine = rig.engine(age=estimator)
    record = rig.generate(engine)
    assert record.guard.age is None
    assert record.outcome == "passed"


# --- curator's detect_faces, faked ----------------------------------------------------------------------


def fake_detect(*ages: float | None):
    """A stand-in for ``argus_curator.faces.detect_faces``: fills ``ImageResult.faces`` as curator does."""
    from argus_curator.models import FaceAttributes, FaceDetection

    calls = []

    def detect(results, items, cfg, **kw):
        calls.append((results, items, cfg))
        for age in ages:
            attrs = FaceAttributes(age_years=age, age_source="mivolo") if age is not None else None
            results[0].faces.append(FaceDetection(bbox=[1, 2, 30, 40], det_score=0.9, attributes=attrs))

    detect.calls = calls
    return detect


def test_the_age_check_runs_curators_pipeline_from_memory() -> None:
    detect = fake_detect(30.0, 15.2)
    faces = CuratorAgeEstimator(detect).estimate(b"png bytes")
    assert [(f.age_estimate_years, f.bbox) for f in faces] == [(30.0, (1, 2, 30, 40)), (15.2, (1, 2, 30, 40))]
    ((results, items, cfg),) = detect.calls
    assert items == [("image", "", b"png bytes")]  # bytes in, no path to a file
    assert cfg.enabled and cfg.mivolo


def test_no_face_is_an_empty_list_not_unavailable() -> None:
    assert CuratorAgeEstimator(fake_detect()).estimate(b"x") == []


def test_a_face_with_no_estimate_makes_the_check_unavailable_for_the_image() -> None:
    with pytest.raises(AgeUnavailable, match="no age"):
        CuratorAgeEstimator(fake_detect(30.0, None)).estimate(b"x")


def test_a_failing_detector_is_unavailable() -> None:
    def detect(*a, **k):
        raise RuntimeError("onnx exploded")

    with pytest.raises(AgeUnavailable, match="onnx exploded"):
        CuratorAgeEstimator(detect).estimate(b"x")


def test_age_status_says_why_it_is_unavailable(monkeypatch, tmp_path) -> None:
    from argus_curator import faces, mivolo

    from argus_darkroom.age import age_status

    monkeypatch.setattr(faces, "faces_available", lambda: False)
    assert "InsightFace" in age_status().reason
    monkeypatch.setattr(faces, "faces_available", lambda: True)
    monkeypatch.setattr(mivolo, "model_path", lambda: tmp_path / "missing.onnx")
    assert "mivolo-convert" in age_status().reason
    (tmp_path / "m.onnx").write_bytes(b"x")
    monkeypatch.setattr(mivolo, "model_path", lambda: tmp_path / "m.onnx")
    assert age_status().available
