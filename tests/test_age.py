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
from argus_darkroom.config import ConfigError, Settings


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
        ],
        people_check="ok",
    ) as fake:
        faces = HttpAgeEstimator(fake.url).estimate(b"x")
    assert [f.age_estimate_years for f in faces] == [34.2, 15.0]


def test_http_estimator_unmatched_person_with_no_estimate_is_unavailable() -> None:
    # Fail closed: a person with no estimate could be a minor, so the image has no age check, not a clean one.
    with (
        FakeFaces(
            faces=[{"age_estimate_years": 34.2, "bbox": [1, 2, 3, 4]}],
            people=[{"age_estimate_years": None, "bbox": [80, 0, 10, 20], "face_index": None}],
            people_check="ok",
        ) as fake,
        pytest.raises(AgeUnavailable, match="person without an estimate"),
    ):
        HttpAgeEstimator(fake.url).estimate(b"x")


def test_http_estimator_matched_person_with_no_body_estimate_is_skipped() -> None:
    # A matched person is the face's own body; the face carries the estimate.
    with FakeFaces(
        faces=[{"age_estimate_years": 34.2, "bbox": [1, 2, 3, 4]}],
        people=[{"age_estimate_years": None, "bbox": [0, 0, 10, 20], "face_index": 0}],
        people_check="ok",
    ) as fake:
        faces = HttpAgeEstimator(fake.url).estimate(b"x")
    assert [f.age_estimate_years for f in faces] == [34.2]


def test_http_estimator_keeps_every_person_box_and_its_face() -> None:
    with FakeFaces(
        faces=[{"age_estimate_years": 34.2, "bbox": [1, 2, 3, 4]}],
        people=[
            {"age_estimate_years": None, "bbox": [0, 0, 10, 20], "face_index": 0},
            {"age_estimate_years": 15.0, "bbox": [40, 0, 10, 20], "face_index": None},
        ],
        people_check="ok",
    ) as fake:
        reading = HttpAgeEstimator(fake.url).read(b"x")
    # faces means what it always has: the frame rule reads only it.
    assert [f.age_estimate_years for f in reading.faces] == [34.2, 15.0]
    assert [(p.bbox, p.age_estimate_years, p.face_index) for p in reading.people] == [
        ((0.0, 0.0, 10.0, 20.0), None, 0),
        ((40.0, 0.0, 10.0, 20.0), 15.0, None),
    ]


@pytest.mark.parametrize(
    "person",
    [
        {"age_estimate_years": None, "bbox": [0, 0, 10], "face_index": 0},
        {"age_estimate_years": None, "bbox": [0, 0, 10, 20], "face_index": 5},
        {"age_estimate_years": None, "bbox": [0, 0, 10, 20], "face_index": True},
    ],
)
def test_http_estimator_drops_a_matched_person_it_cannot_place(person) -> None:
    # Not unavailable: the face still carries the estimate. The overlap rule then sees a face with no body.
    with FakeFaces(
        faces=[{"age_estimate_years": 34.2, "bbox": [1, 2, 3, 4]}], people=[person], people_check="ok"
    ) as fake:
        reading = HttpAgeEstimator(fake.url).read(b"x")
    assert [f.age_estimate_years for f in reading.faces] == [34.2]
    assert reading.people == []


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


def test_minor_exposed_defaults_to_frame(monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_DARKROOM_SINGGUARD_URL", "http://guard:9150")
    monkeypatch.delenv("ARGUS_DARKROOM_MINOR_EXPOSED", raising=False)
    assert Settings.from_env().minor_exposed == "frame"
    monkeypatch.setenv("ARGUS_DARKROOM_MINOR_EXPOSED", "overlap")
    assert Settings.from_env().minor_exposed == "overlap"


@pytest.mark.parametrize("value", ["frames", "Overlap", "off", "1"])
def test_an_unknown_minor_exposed_rule_refuses_to_start(monkeypatch, value) -> None:
    monkeypatch.setenv("ARGUS_DARKROOM_SINGGUARD_URL", "http://guard:9150")
    monkeypatch.setenv("ARGUS_DARKROOM_MINOR_EXPOSED", value)
    with pytest.raises(ConfigError, match="MINOR_EXPOSED"):
        Settings.from_env()


def test_from_env_reads_age_url(monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_DARKROOM_SINGGUARD_URL", "http://guard:9150")
    monkeypatch.setenv("ARGUS_DARKROOM_AGE_URL", "http://curator:8101")
    assert Settings.from_env().age_url == "http://curator:8101"


@pytest.mark.parametrize("years", [float("nan"), float("inf"), True, -1])
def test_http_estimator_rejects_a_non_estimate(years) -> None:
    face = {"age_estimate_years": years, "bbox": [1, 2, 3, 4]}
    with FakeFaces(faces=[face]) as fake, pytest.raises(AgeUnavailable, match="without an estimate"):
        HttpAgeEstimator(fake.url).estimate(b"x")


@pytest.mark.parametrize("years", [float("nan"), float("inf"), True, -1])
def test_http_estimator_rejects_a_non_estimate_for_an_unmatched_person(years) -> None:
    person = {"age_estimate_years": years, "bbox": [0, 0, 10, 20], "face_index": None}
    with (
        FakeFaces(people=[person], people_check="ok") as fake,
        pytest.raises(AgeUnavailable, match="person without an estimate"),
    ):
        HttpAgeEstimator(fake.url).estimate(b"x")


def test_remote_status_server_error_is_unavailable_whatever_the_body() -> None:
    with FakeFaces(health_status=503) as fake:
        status = remote_age_status(fake.url)
    assert not status.available
    assert "503" in (status.reason or "")
