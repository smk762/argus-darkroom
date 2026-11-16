"""The NudeNet detector is recorded with its score (#54): 640m and 320n score the same image differently.

The detector is metadata. Nothing here may change a verdict: a server that cannot say which model it runs, or whose
``/health`` fails, still scores the image exactly as before.
"""

from __future__ import annotations

import tempfile

import pytest
from fakes import FakeNudeNet
from fastapi.testclient import TestClient
from rig import EXAMPLES, EXPOSED, Rig, nsfw
from test_img2img import init

from argus_darkroom.server.app import create_app


def test_a_640m_server_is_recorded_with_the_score(rig) -> None:
    record = rig.generate(rig.engine())
    assert record.outcome == "passed"
    assert record.guard.nudenet.detector == "640m"
    assert record.guard.nudenet.detector_model == "640m.onnx"
    assert record.guard.nudenet.exposed_score == 0.0
    assert rig.nudenet.health_calls == 1  # asked once, after the score


def test_the_upstream_320n_image_has_no_health_route_and_records_no_detector(rig) -> None:
    rig.nudenet.model = None  # 404 on /health, as the upstream image answers
    nsfw(rig)
    rig.nudenet.labels = EXPOSED
    record = rig.generate(rig.engine(), enforcing_policy="adult-granular")
    assert record.guard.nudenet.detector is None and record.guard.nudenet.detector_model is None
    # The score and the blur decision are exactly what they were without the detector.
    assert record.guard.nudenet.exposed_score == 0.9 and record.guard.nudenet.blur is True
    assert record.outcome == "passed"


def test_a_failing_health_route_costs_only_the_detector(rig) -> None:
    rig.nudenet.health_status = 500
    record = rig.generate(rig.engine())
    assert record.outcome == "passed"
    assert record.guard.nudenet.exposed_score == 0.0
    assert record.guard.nudenet.detector is None


def test_another_model_is_kept_by_name_with_no_detector(rig) -> None:
    rig.nudenet.model = "custom-detector.onnx"
    record = rig.generate(rig.engine())
    assert record.guard.nudenet.detector is None
    assert record.guard.nudenet.detector_model == "custom-detector.onnx"


def test_no_score_means_the_detector_is_not_asked(tmp_path, singguard, comfyui, monkeypatch) -> None:
    with FakeNudeNet(status=500) as broken:
        rig = Rig(tmp_path, singguard, comfyui, broken)
        monkeypatch.setattr(tempfile, "tempdir", str(rig.systmp))
        nsfw(rig)
        record = rig.generate(rig.engine(), enforcing_policy="adult-granular")
    assert record.guard.nudenet.exposed_score is None and record.guard.nudenet.blur is True  # fails closed, as before
    assert record.guard.nudenet.detector is None
    assert broken.health_calls == 0


def test_an_input_image_records_its_detector_too(tmp_path, singguard, comfyui, nudenet, monkeypatch) -> None:
    rig = Rig(tmp_path, singguard, comfyui, nudenet, workflow=EXAMPLES / "sdxl+lora1+img2img.json")
    monkeypatch.setattr(tempfile, "tempdir", str(rig.systmp))
    record = rig.generate(rig.engine(), init_image=init())
    assert record.outcome == "passed"
    assert record.guard.input.nudenet.detector == "640m"
    assert record.guard.nudenet.detector == "640m"


@pytest.mark.parametrize(("model", "expected"), [("640m.onnx", "640m"), ("320n.onnx", "320n"), (None, None)])
def test_health_reports_the_detector(rig, model, expected) -> None:
    rig.nudenet.model = model
    assert TestClient(create_app(rig.engine())).get("/health").json()["nudenet_detector"] == expected


def test_health_reports_no_detector_without_nudenet(rig) -> None:
    from dataclasses import replace

    rig.settings = replace(rig.settings, nudenet_url="")
    assert TestClient(create_app(rig.engine())).get("/health").json()["nudenet_detector"] is None
