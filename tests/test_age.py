"""Age check over curator's POST /faces, and the in-process extra fallback."""

from __future__ import annotations

import pytest
from fakes import FakeFaces

from argus_darkroom.age import (
    AgeUnavailable,
    HttpAgeEstimator,
    default_estimator,
    remote_age_status,
)
from argus_darkroom.config import Settings


def test_http_estimator_posts_the_buffer_and_returns_faces() -> None:
    with FakeFaces(faces=[{"age_estimate_years": 34.2, "bbox": [1, 2, 3, 4]}]) as fake:
        faces = HttpAgeEstimator(fake.url).estimate(b"png-bytes")
    assert [(f.age_estimate_years, f.bbox) for f in faces] == [(34.2, (1.0, 2.0, 3.0, 4.0))]
    assert fake.posts == [b"png-bytes"]
    assert fake.paths == ["/faces?body=true"]


def test_http_estimator_empty_when_there_is_no_face() -> None:
    with FakeFaces(faces=[]) as fake:
        assert HttpAgeEstimator(fake.url).estimate(b"x") == []


def test_http_estimator_503_is_unavailable() -> None:
    with FakeFaces(status=503) as fake, pytest.raises(AgeUnavailable, match="HTTP 503"):
        HttpAgeEstimator(fake.url).estimate(b"x")


def test_http_estimator_malformed_face_is_unavailable() -> None:
    with FakeFaces(faces=[{"bbox": [1, 2, 3, 4]}]) as fake, pytest.raises(AgeUnavailable, match="without an estimate"):
        HttpAgeEstimator(fake.url).estimate(b"x")


def test_http_estimator_includes_unmatched_people() -> None:
    with FakeFaces(
        faces=[{"age_estimate_years": 34.2, "bbox": [1, 2, 3, 4]}],
        people=[
            {"age_estimate_years": 34.2, "bbox": [0, 0, 10, 20], "face_index": 0},
            {"age_estimate_years": 15.0, "bbox": [40, 0, 10, 20], "face_index": None},
            {"age_estimate_years": None, "bbox": [80, 0, 10, 20], "face_index": None},
        ],
        people_check="ok",
    ) as fake:
        faces = HttpAgeEstimator(fake.url).estimate(b"x")
    assert [f.age_estimate_years for f in faces] == [34.2, 15.0]


def test_http_estimator_people_check_failed_is_unavailable() -> None:
    with (
        FakeFaces(people_check="detection_failed", people_error="onnx exploded") as fake,
        pytest.raises(AgeUnavailable, match="detection_failed"),
    ):
        HttpAgeEstimator(fake.url).estimate(b"x")


def test_http_estimator_unknown_people_check_is_unavailable() -> None:
    with FakeFaces(people_check="mystery") as fake, pytest.raises(AgeUnavailable, match="mystery"):
        HttpAgeEstimator(fake.url).estimate(b"x")


def test_remote_status_reads_faces_on_health() -> None:
    with FakeFaces() as fake:
        assert remote_age_status(fake.url).available
    with FakeFaces(health={"status": "ok", "faces": False, "mivolo": False}) as fake:
        status = remote_age_status(fake.url)
        assert not status.available
        assert "unavailable" in (status.reason or "")
    assert not remote_age_status("http://127.0.0.1:1").available


def test_default_estimator_prefers_the_url() -> None:
    with FakeFaces(faces=[{"age_estimate_years": 21, "bbox": [0, 0, 1, 1]}]) as fake:
        est, status = default_estimator(fake.url)
        assert status.available
        assert est is not None
        assert est.estimate(b"x")[0].age_estimate_years == 21.0


def test_default_estimator_keeps_the_http_client_when_health_is_down() -> None:
    est, status = default_estimator("http://127.0.0.1:1")
    assert not status.available
    assert isinstance(est, HttpAgeEstimator)


def test_from_env_reads_age_url(monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_DARKROOM_SINGGUARD_URL", "http://guard:9150")
    monkeypatch.setenv("ARGUS_DARKROOM_AGE_URL", "http://curator:8101")
    assert Settings.from_env().age_url == "http://curator:8101"
