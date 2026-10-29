"""The optional age check: MiVOLO age estimates through curator's face pipeline, in process.

Behind the ``[age]`` extra (``argus-curator[faces]``) and curator's MiVOLO ONNX file (``argus-curator
mivolo-convert``). When either is missing the check is skipped, the record says ``age: null`` and ``/health``
says why. The image is judged from memory: curator's ``detect_faces`` takes ``(rel, abs, bytes)`` items, so
nothing is written to disk.

An estimate is a tool's reading of a face, never a stated age, and is named as such everywhere it appears.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import structlog

from argus_darkroom.models import FaceAge

logger = structlog.get_logger()


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


class CuratorAgeEstimator:
    """Estimates ages with curator's InsightFace detector and MiVOLO.

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
        self._config = FaceConfig(enabled=True, mivolo=True)
        self._result = ImageResult

    def estimate(self, data: bytes) -> list[FaceAge]:
        result = self._result(rel_path="image", abs_path="", passed=True)
        try:
            self._detect([result], [("image", "", data)], self._config)
        except Exception as exc:
            raise AgeUnavailable(f"face detection failed: {exc}") from exc
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


def default_estimator() -> tuple[AgeEstimator | None, AgeStatus]:
    status = age_status()
    if not status.available:
        logger.info("age.unavailable", reason=status.reason)
        return None, status
    return CuratorAgeEstimator(), status
