"""The optional age check: MiVOLO estimates, over HTTP or in process.

Prefer ``ARGUS_DARKROOM_AGE_URL`` (curator's ``POST /faces?body=true``). The
image stays in memory; nothing is written. The ``[age]`` extra (curator's
InsightFace and MiVOLO in this process) is the fallback when no URL is set.

An estimate is a tool's reading of a face, never a stated age, and is named as
such everywhere it appears. It is never sent to SingGuard as context.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import structlog

from argus_darkroom.models import FaceAge

logger = structlog.get_logger()

PROBE_TIMEOUT = 2.0
CHECK_TIMEOUT = 30.0
_PEOPLE_CHECK_FAILED = frozenset({"detection_failed", "body_ages_failed"})


class AgeUnavailable(RuntimeError):
    """The age check cannot run for this image."""


class AgeEstimator(Protocol):
    def estimate(self, data: bytes) -> list[FaceAge]:
        """One estimate per detected face (none when there is no face). Raises :class:`AgeUnavailable`."""
        ...


@dataclass(frozen=True)
class AgeStatus:
    available: bool
    reason: str | None = None  # why not, for /health


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def remote_age_status(base_url: str) -> AgeStatus:
    """Whether curator's ``POST /faces`` can run: ``GET /health`` reports ``faces`` or ``mivolo``."""
    url = base_url.rstrip("/") + "/health"
    try:
        with _opener().open(url, timeout=PROBE_TIMEOUT) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        # Any HTTP answer means the server is up; still need faces/mivolo in the body.
        raw = exc.read() if exc.fp is not None else b""
        if exc.code >= 500 and not raw:
            return AgeStatus(False, f"AGE_URL /health returned HTTP {exc.code}")
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return AgeStatus(False, "AGE_URL did not answer")
    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return AgeStatus(False, "AGE_URL /health was not curator faces")
    if not isinstance(body, dict):
        return AgeStatus(False, "AGE_URL /health was not curator faces")
    if body.get("faces") is True or body.get("mivolo") is True:
        return AgeStatus(True)
    if "faces" in body or "mivolo" in body:
        return AgeStatus(False, "curator reports faces unavailable")
    return AgeStatus(False, "AGE_URL /health was not curator faces")


class HttpAgeEstimator:
    """MiVOLO estimates from curator's ``POST /faces?body=true``. The buffer is posted; nothing is written here."""

    def __init__(self, base_url: str, timeout: float = CHECK_TIMEOUT) -> None:
        self.url = base_url.rstrip("/") + "/faces?body=true"
        self.timeout = timeout

    def estimate(self, data: bytes) -> list[FaceAge]:
        req = urllib.request.Request(
            self.url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        try:
            with _opener().open(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            raise AgeUnavailable(f"age check returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            raise AgeUnavailable(f"age check failed: {exc}") from exc
        return _faces_from_payload(payload)


def _face_from_item(item: object, kind: str) -> FaceAge:
    if not isinstance(item, dict):
        raise AgeUnavailable(f"age check returned a malformed {kind}")
    years = item.get("age_estimate_years")
    bbox = item.get("bbox")
    if isinstance(years, bool) or not isinstance(years, (int, float)):
        raise AgeUnavailable(f"age check returned a {kind} without an estimate")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise AgeUnavailable(f"age check returned a {kind} without an estimate")
    try:
        box = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError) as exc:
        raise AgeUnavailable("age check returned a malformed bbox") from exc
    return FaceAge(age_estimate_years=float(years), bbox=box)


def _faces_from_payload(payload: object) -> list[FaceAge]:
    """Faces, plus unmatched people that still have an estimate.

    Curator 0.10 ``POST /faces?body=true`` returns body-only ages in ``people``.
    Those must feed ``minor_exposed``: a faceless minor is still a minor.
    ``people_check`` other than ``ok`` means an empty list is not "none found".
    Matched people duplicate a face and are skipped. A person with no estimate
    is skipped so partial body evidence cannot fail the face list.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("faces"), list):
        raise AgeUnavailable("age check returned no faces list")
    check = payload.get("people_check")
    if check in _PEOPLE_CHECK_FAILED:
        detail = payload.get("people_error") or check
        raise AgeUnavailable(f"people check {check}: {detail}")
    if check is not None and check != "ok":
        raise AgeUnavailable(f"people check {check!r}")
    faces = [_face_from_item(item, "face") for item in payload["faces"]]
    people = payload.get("people")
    if people is None:
        return faces
    if not isinstance(people, list):
        raise AgeUnavailable("age check returned a malformed people list")
    for item in people:
        if not isinstance(item, dict):
            raise AgeUnavailable("age check returned a malformed person")
        if item.get("face_index") is not None:
            continue
        if item.get("age_estimate_years") is None:
            continue
        faces.append(_face_from_item(item, "person"))
    return faces


class CuratorAgeEstimator:
    """Estimates ages with curator's InsightFace detector and MiVOLO, in this process.

    *detect_faces* is curator's function, injectable so tests can fake it.
    """

    def __init__(self, detect_faces: Callable | None = None) -> None:
        try:
            from argus_curator.models import FaceConfig, ImageResult
        except ImportError as exc:  # pragma: no cover - needs the extra's absence
            raise AgeUnavailable("argus-curator is not installed (pip install argus-darkroom[age])") from exc
        if detect_faces is None:
            from argus_curator.faces import detect_faces
        self._detect = detect_faces
        self._config = FaceConfig(enabled=True, mivolo=True, mivolo_body=True)
        self._result = ImageResult

    def estimate(self, data: bytes) -> list[FaceAge]:
        result = self._result(rel_path="image", abs_path="", passed=True)
        try:
            # strict: a failed detector must not read as "no faces" (empty list, a pass).
            self._detect([result], [("image", "", data)], self._config, strict=True)
        except Exception as exc:
            raise AgeUnavailable(f"face detection failed: {exc}") from exc
        check = getattr(result, "people_check", None)
        if check in _PEOPLE_CHECK_FAILED:
            detail = getattr(result, "people_error", None) or check
            raise AgeUnavailable(f"people check {check}: {detail}")
        faces: list[FaceAge] = []
        for face in result.faces:
            attrs = face.attributes
            years = attrs.age_years if attrs is not None and attrs.age_source in (None, "mivolo") else None
            if years is None:
                # A face with no estimate cannot be waved through: partial evidence would look like more
                # confidence than there is, so the whole check is unavailable for this image.
                raise AgeUnavailable("MiVOLO gave no age for a detected face")
            x, y, w, h = face.bbox
            faces.append(FaceAge(age_estimate_years=float(years), bbox=(x, y, w, h)))
        for person in getattr(result, "people", None) or []:
            if getattr(person, "face_index", None) is not None:
                continue
            body_years = getattr(person, "age_years", None)
            if body_years is None:
                continue
            x, y, w, h = person.bbox
            faces.append(FaceAge(age_estimate_years=float(body_years), bbox=(x, y, w, h)))
        return faces


def age_status() -> AgeStatus:
    """Whether :class:`CuratorAgeEstimator` can run: the extra is installed, and so are InsightFace and the weights."""
    try:
        from argus_curator.faces import faces_available
        from argus_curator.mivolo import model_path
    except ImportError:
        return AgeStatus(False, "argus-curator is not installed (pip install argus-darkroom[age])")
    if not faces_available():
        return AgeStatus(False, "InsightFace is not installed (pip install argus-darkroom[age])")
    path = model_path()
    if not path.is_file():
        return AgeStatus(False, f"the MiVOLO model is missing at {path} (run `argus-curator mivolo-convert`)")
    return AgeStatus(True)


def default_estimator(age_url: str = "") -> tuple[AgeEstimator | None, AgeStatus]:
    """Prefer curator over HTTP; fall back to the in-process extra when no URL is set.

    When a URL is set, the HTTP client is always returned. A failed boot probe
    must not disable the check for the process lifetime: ``/health`` re-probes,
    and ``estimate`` maps a down curator to :class:`AgeUnavailable` per image.
    """
    if age_url.strip():
        status = remote_age_status(age_url)
        if not status.available:
            logger.info("age.unavailable", reason=status.reason)
        return HttpAgeEstimator(age_url), status
    status = age_status()
    if not status.available:
        logger.info("age.unavailable", reason=status.reason)
        return None, status
    return CuratorAgeEstimator(), status
