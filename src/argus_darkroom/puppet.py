"""argus-puppet over HTTP: the keypoints of a pose photo, and the skeleton drawn from them.

A pose input is a photo. Once it has passed its own scan (never before), darkroom asks argus-puppet two things:

* ``POST /pose``: every person's whole-body keypoints in the photo. Puppet writes nothing for this route.
* ``POST /render``: those keypoints drawn as an OpenPose-convention control image at the generation's size. Only
  numbers go in, so the drawing cannot carry any of the photo's appearance.

Darkroom draws no skeleton itself and loads no pose model: both are puppet's. The URL comes from the settings only
(``ARGUS_DARKROOM_PUPPET_URL``, then the suite-wide ``ARGUS_PUPPET_URL``), that one host is the only one called, and
redirects are not followed. Nothing here is written to disk, and no error carries a piece of an answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urlsplit

from argus_cortex.gpu_wire import PriorityClass
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from argus_darkroom.models import MAX_POSE_PEOPLE, POSE_LAYOUT, ModelRef, PoseEstimatorRef, PosePerson

READ_TIMEOUT = 60.0  # per read: a cold puppet loads two models first
DEADLINE = 180.0  # the whole call, a wait for puppet's GPU lease included
MAX_POSE_REPLY = 8 * 1024 * 1024  # a PoseResult of 20 people is about 100 KiB
MAX_CONTROL_REPLY = 32 * 1024 * 1024  # a skeleton PNG is mostly black: far under this at any size darkroom renders


class PoseUnavailable(RuntimeError):
    """argus-puppet could not read or draw the pose. The message never holds a piece of its answer."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _Estimator(BaseModel):
    model_config = ConfigDict(extra="ignore")

    package_version: str | None = None
    detector: _Model | None = None
    pose: _Model | None = None


class _PoseAnswer(BaseModel):
    """The part of puppet's ``PoseResult`` darkroom reads. Everything else in it is ignored."""

    model_config = ConfigDict(extra="ignore")

    puppet_version: str | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    layout: str = POSE_LAYOUT
    people: list[PosePerson] = Field(default_factory=list)
    estimator: _Estimator | None = None


@dataclass(frozen=True)
class PoseRead:
    """What argus-puppet found in a pose photo: every person (largest first, as it lists them) and what read them."""

    people: tuple[PosePerson, ...]
    estimator: PoseEstimatorRef

    def drawn(self, primary_only: bool) -> tuple[PosePerson, ...]:
        """The people a request draws: the primary subject alone, or everyone up to :data:`MAX_POSE_PEOPLE`."""
        if primary_only:
            return tuple(p for p in self.people if p.primary)[:1] or self.people[:1]
        return self.people[:MAX_POSE_PEOPLE]


class PuppetClient:
    def __init__(self, base_url: str, timeout: float = READ_TIMEOUT, deadline: float = DEADLINE) -> None:
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("the argus-puppet URL is not an absolute http(s) URL")
        self.base_url = base_url.rstrip("/")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
        # Puppet is an internal service: its one host may be private. No other host is ever called.
        self._allow = (f"{host}:{port}",)
        self.timeout = timeout
        self.deadline = deadline

    def _post(self, path: str, content: bytes, content_type: str, max_bytes: int, headers: dict[str, str]) -> bytes:
        from argus_cortex.net import FetchFailed, UrlRefused, fetch

        try:
            reply = fetch(
                self.base_url + path,
                method="POST",
                content=content,
                headers={"Content-Type": content_type, **headers},
                allow_hosts=self._allow,
                timeout=self.timeout,
                deadline=self.deadline,
                max_bytes=max_bytes,
                max_redirects=0,
            ).raise_for_status()
        except FetchFailed as exc:
            busy = " (busy, or its model or GPU is not ready)" if exc.status == 503 else ""
            raise PoseUnavailable(f"argus-puppet answered HTTP {exc.status} to {path}{busy}") from exc
        except (UrlRefused, OSError) as exc:
            # The reason names the host at most; never a body.
            raise PoseUnavailable(f"argus-puppet could not be reached for {path}: {type(exc).__name__}") from exc
        if not reply.ok:
            raise PoseUnavailable(f"argus-puppet answered HTTP {reply.status} to {path}; a redirect is not followed")
        return reply.body

    def read(self, png: bytes, size: tuple[int, int], priority: PriorityClass | None = None) -> PoseRead:
        """The people in *png* (a checked pose photo of *size*). Raises :class:`PoseUnavailable`, also when the answer
        is not about an image of *size*: its keypoints would then be in another frame."""
        headers = {"X-Argus-Priority": priority} if priority else {}
        body = self._post("/pose?name=pose.png", png, "image/png", MAX_POSE_REPLY, headers)
        try:
            answer = _PoseAnswer.model_validate(json.loads(body))
        except (ValueError, ValidationError) as exc:
            raise PoseUnavailable("argus-puppet's answer to /pose is not a pose result") from exc
        if answer.layout != POSE_LAYOUT:
            raise PoseUnavailable(f"argus-puppet's keypoints are not in the {POSE_LAYOUT} layout")
        if (answer.width, answer.height) != size:
            raise PoseUnavailable(
                f"argus-puppet read an image of {answer.width}x{answer.height}, not the {size[0]}x{size[1]} it was sent"
            )
        found = answer.estimator or _Estimator()
        return PoseRead(
            people=tuple(answer.people),
            estimator=PoseEstimatorRef(
                package_version=found.package_version,
                wire_version=answer.puppet_version,
                detector=ModelRef(name=found.detector.name, sha256=found.detector.sha256) if found.detector else None,
                pose=ModelRef(name=found.pose.name, sha256=found.pose.sha256) if found.pose else None,
            ),
        )

    def draw(
        self,
        people: tuple[PosePerson, ...],
        frame: tuple[int, int],
        canvas: tuple[int, int],
        *,
        body: bool = True,
        hands: bool = True,
        face: bool = True,
        fit: str = "contain",
    ) -> bytes:
        """*people* (keypoints in a photo of *frame*) drawn as an OpenPose-convention PNG of *canvas*. Only numbers
        are sent. The bytes that come back are not trusted to be a drawing: see ``images.load_control``. Raises
        :class:`PoseUnavailable`."""
        request = {
            "pose": {"keypoints": [person.keypoints for person in people], "width": frame[0], "height": frame[1]},
            "options": {
                "body": body,
                "hands": hands,
                "face": face,
                "width": canvas[0],
                "height": canvas[1],
                "fit": fit,
            },
        }
        return self._post("/render", json.dumps(request).encode(), "application/json", MAX_CONTROL_REPLY, {})
