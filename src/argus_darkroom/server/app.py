"""The darkroom FastAPI server.

========================  =====================================================================================
``GET /health``           cortex's ``ServiceHealth`` (status, checks) and darkroom's own fields: which backends
                          answer, the policy list, ``nsfw_allowed``, ``devmode``, ``read_only``; ``?checks=0`` probes none
``GET /policies``         names, the enforcing default, each policy's illegal rules and wording
``GET /models``           checkpoints (with family and LoRA counts), LoRAs, VAEs, ControlNets, workflows, defaults
``POST /preflight``       the text check alone, so a form can show a verdict before Generate
``POST /generate/stream`` SSE: ``preflight``, ``input_scan``, ``pose_scan``, ``pose``, ``queued``, ``progress``,
                          ``scan``, ``complete``.
                          ``X-Argus-Priority`` picks the GPU lease class (``interactive`` by default)
``GET /records``          paged, filterable by outcome; ``GET /record/{id}``
``GET /image/{id}``       ids only, no path input; 404 with the record id when there is no stored image
``DELETE /record/{id}``   the user removing their own output: the file goes (its depth map too), the record stays
``POST /depth/{id}``      a depth map of the record's stored image (idempotent; shares the generation lock)
``GET /depth/{id}``       that depth map, as an 8-bit greyscale PNG; 404 with the record id when there is none
``GET /archive``          a streamed zip of every record, stored image and depth map (read-only, always on)
``POST /purge``           everything, from a clean slate; exists only with ``ARGUS_DARKROOM_DEVMODE=1``
``POST /purge/{id}``      one record and its image; exists only with ``ARGUS_DARKROOM_DEVMODE=1``
========================  =====================================================================================

With no SingGuard URL the server still starts, as :func:`unguarded_app`: ``/health``, the record and image reads and
``DELETE /record/{id}`` work, and everything else answers 503 naming the variables to set.

Reads are open to the configured CORS origins; writes are refused from any other origin (``WriteGuard`` and
``cross_site_refuse``, because CORS alone is not a write boundary). The server binds to localhost by default.
With ``ARGUS_DARKROOM_READ_ONLY=1`` every write is refused from every origin, ``POST /preflight`` included.

One generation runs at a time and a second gets 409, unless the GPU coordinator is the broker (argus-iris): then up to
:data:`~argus_darkroom.gpu.MAX_IN_FLIGHT` run at once, each waiting for its own lease, and iris queues them by
priority. A purge always waits for none to be running.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import io
import json
import re
import threading
import urllib.error
import urllib.request
import zipfile
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import structlog
from argus_cortex.gpu import aregistered
from argus_cortex.gpu_wire import PriorityClass
from argus_cortex.moderation import NudeNetClient
from argus_cortex.moderation import policies as policy_table
from argus_cortex.server import WriteGuard, constant_refuse, cross_site_refuse
from argus_cortex.service import DependencyCheck, install_health
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from argus_darkroom import __version__, gpu
from argus_darkroom.age import AgeStatus, HttpAgeEstimator, remote_age_status
from argus_darkroom.comfyui import GenerationError
from argus_darkroom.config import GUARD_VARIABLES, Settings, Unguarded
from argus_darkroom.engine import Engine
from argus_darkroom.modelfiles import pick_default
from argus_darkroom.models import RECORD_VERSION, DepthRecord, GenerationRecord, Outcome
from argus_darkroom.request import GenerateRequest, RequestError
from argus_darkroom.store import DEPTH_SUFFIX, RecordStore, valid_id
from argus_darkroom.workflow import checkpoint_family

logger = structlog.get_logger()

PROBE_TIMEOUT = 2.0
# A health check is one probe (and NudeNet's detector, asked alongside): given room past the probe's own timeout, so
# the check reports what the probe found and is not cut off first.
HEALTH_CHECK_TIMEOUT = 10.0
MAX_PAGE = 200


def probe(url: str) -> bool:
    """Whether something answers at *url*. Any HTTP answer counts (a 404 is a server that is up)."""
    if not url:
        return False
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=PROBE_TIMEOUT):
            pass
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return True


class _Probes:
    """What the health checks last found, for the fields darkroom's ``/health`` has always carried."""

    def __init__(self, age: AgeStatus) -> None:
        self.up = dict.fromkeys(("comfyui", "singguard", "nudenet", "ollama", "puppet"), False)
        self.age = age
        self.detector: str | None = None

    def done(self, name: str, up: bool) -> bool:
        self.up[name] = up
        return up


def nudenet_detector(client: NudeNetClient | None) -> str | None:
    """The NudeNet detector the server reports (``640m``, ``320n``), or ``None``. Never raises: it is only reported."""
    if client is None:
        return None
    try:
        return client.info().detector
    except Exception:  # an unreachable server is already in "backends"
        return None


GUARD_UNCONFIGURED = "SingGuard is not configured"
READ_ONLY_REFUSAL = "argus-darkroom is read-only: generation, pre-flight and record changes are disabled"
POLICY_SUMMARIES = {
    "default": "SingGuard's own rules, which treat sexual content as a risk. The SFW policy: with no illegal rules, "
    "it can only send an image to review.",
    "adult": "Allows adult content. Rule A, illegal sexual content, is illegal.",
    "adult-granular": "Allows adult content. Minors (A), non-consensual content (H) and sexual violence (I) are "
    "separate illegal rules.",
    "adult-cautious": "Like adult, but unclear age or unclear consent counts as rule A.",
    "adult-cinematic": "Like adult: legal adult entertainment is Safe. Depicted crime and violence are Safe with no "
    "sign of fiction needed; rule G covers how-to and real targeting, and B to F sit one letter up from adult. "
    "Rule A is illegal.",
}
_RULE_HEADING = re.compile(r"^### ([A-Z])\. (.+)$", re.MULTILINE)


def rule_names(policy: str) -> dict[str, str]:
    """Rule letter to the name the policy gives it, read from the policy text (``{}`` for `default`)."""
    text = policy_table.policy_text(policy) or ""
    return {code: name.strip() for code, name in _RULE_HEADING.findall(text)}


def sse(kind: str, data: object) -> bytes:
    return f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode()


class RunGate:
    """Generations and depth runs share it (:meth:`enter`); a purge takes it alone (:meth:`acquire`).

    Up to *limit* shared runs at once, and none while a purge holds it. ``acquire``, ``release`` and ``locked`` keep the
    shape of the lock this replaced.
    """

    def __init__(self, limit: int = 1) -> None:
        self.limit = max(1, limit)
        self._cond = threading.Condition()
        self._running = 0
        self._exclusive = False
        self._waiting_exclusive = 0  # blocked in acquire: no new shared run starts, so the purge cannot starve

    @property
    def running(self) -> int:
        return self._running

    def enter(self) -> bool:
        """Start a shared run, or ``False`` at once when the limit is reached or a purge is running or waiting."""
        with self._cond:
            if self._exclusive or self._waiting_exclusive or self._running >= self.limit:
                return False
            self._running += 1
            return True

    def leave(self) -> None:
        with self._cond:
            self._running -= 1
            self._cond.notify_all()

    def acquire(self, blocking: bool = True) -> bool:
        """Take it alone: no shared run and no other purge."""
        with self._cond:
            if not blocking:
                if self._exclusive or self._running:
                    return False
            else:
                self._waiting_exclusive += 1
                try:
                    while self._exclusive or self._running:
                        self._cond.wait()
                finally:
                    self._waiting_exclusive -= 1
            self._exclusive = True
            return True

    def release(self) -> None:
        with self._cond:
            self._exclusive = False
            self._cond.notify_all()

    def locked(self) -> bool:
        return self._exclusive or self._running > 0


def priority_of(x_argus_priority: str | None = Header(None)) -> PriorityClass:
    """The GPU lease class from ``X-Argus-Priority``: mint's sample runs send ``batch``, a person's form nothing."""
    try:
        return gpu.parse_priority(x_argus_priority)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class _Sink(io.RawIOBase):
    """A write-only stream that hands what was written to the zip back out as chunks."""

    def __init__(self) -> None:
        self.buffer = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, data: bytes) -> int:  # type: ignore[override]
        self.buffer += data
        return len(data)

    def take(self) -> bytes:
        chunk, self.buffer = bytes(self.buffer), bytearray()
        return chunk


def _guard_writes(app: FastAPI, cors_origins: tuple[str, ...], read_only: bool) -> None:
    """CORS and the write guard, the same on the guarded app and the unguarded one."""
    wildcard = "*" in cors_origins
    named = [o for o in cors_origins if o != "*"]
    # A wildcard grants anonymous reads, never a cross-site write: naming an origin is the way to get those.
    app.add_middleware(WriteGuard, refuse=cross_site_refuse(named))
    if read_only:
        # Method-based, so it covers every write route, current and future, and POST /preflight, which writes nothing
        # but spends a SingGuard call. Added before CORS so that CORS stays outermost and a 403 keeps its headers.
        app.add_middleware(WriteGuard, refuse=constant_refuse(READ_ONLY_REFUSAL))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if wildcard else named,
        allow_credentials=not wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _serve_records(app: FastAPI, store: RecordStore, record_lock: threading.Lock) -> Callable[[str], GenerationRecord]:
    """The routes that only read what was already judged and released, or take it away: records, stored images, depth
    maps, the archive, and a user removing their own output.

    They are given the record store and nothing else: no guard, no engine and no backend client, so they work the same
    with or without a SingGuard (:func:`unguarded_app`) and cannot generate, check or release anything. A route that
    needs more than the store does not belong here. Returns the record lookup, for the routes that share it.
    """

    def record_or_404(record_id: str) -> GenerationRecord:
        record = store.load(record_id) if valid_id(record_id) else None
        if record is None:
            raise HTTPException(404, "no such record")
        return record

    # -- records and images ------------------------------------------------------------------------------
    @app.get("/records")
    def records(
        outcome: Outcome | None = None,
        mode: Literal["sfw", "nsfw"] | None = None,
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=MAX_PAGE),
    ) -> dict:
        found, total = store.page(outcome, mode, (page - 1) * page_size, page_size)
        return {
            "records": [r.model_dump(mode="json") for r in found],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    @app.get("/record/{record_id}")
    def record(record_id: str) -> GenerationRecord:
        return record_or_404(record_id)

    @app.get("/image/{record_id}")
    def image(record_id: str):
        found = record_or_404(record_id)
        path = store.image_path(record_id)
        if found.image is None or found.image.path is None or not path.is_file():
            # A deleted, omitted, refused, failed or user-removed record has no image to serve.
            return JSONResponse(
                {"detail": "this record has no stored image", "record_id": record_id, "outcome": found.outcome},
                status_code=404,
            )
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, no-store"})

    @app.delete("/record/{record_id}")
    def remove(record_id: str) -> GenerationRecord:
        """The user removing their own output: the file is deleted, the record stays, marked as removed."""
        from datetime import UTC, datetime

        with record_lock:
            found = record_or_404(record_id)
            if found.removed_at is not None:
                return found
            if found.image is None or found.image.path is None:
                raise HTTPException(409, "this record has no stored image to remove")
            store.remove_image(record_id)  # and its depth map
            found.image.path = None
            found.depth = None
            found.removed_at = datetime.now(UTC)
            store.save(found)
            return found

    @app.get("/depth/{record_id}")
    def depth_image(record_id: str):
        found = record_or_404(record_id)
        path = store.depth_path(record_id)
        if found.depth is None or not path.is_file():
            return JSONResponse(
                {"detail": "this record has no depth map", "record_id": record_id, "outcome": found.outcome},
                status_code=404,
            )
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, no-store"})

    @app.get("/archive")
    def archive() -> StreamingResponse:
        """Every record (JSON), every stored image and every depth map (``depth/<id>.png``), as one zip. Deleted and
        omitted images were never written, so they are not in it."""

        def stream() -> Iterator[bytes]:
            sink = _Sink()
            with zipfile.ZipFile(sink, "w", zipfile.ZIP_DEFLATED) as zf:
                # nosemgrep: handler-scans-all - export-all: streaming every record is the operation
                for path in sorted(store.root.glob("*.json")):
                    if not valid_id(path.stem):
                        continue
                    zf.write(path, f"records/{path.name}")
                    yield sink.take()
                    image_path = store.root / f"{path.stem}.png"
                    if image_path.is_file():
                        zf.write(image_path, f"images/{image_path.name}")
                        yield sink.take()
                    depth_path = store.root / f"{path.stem}{DEPTH_SUFFIX}"
                    if depth_path.is_file():
                        zf.write(depth_path, f"depth/{path.stem}.png")
                        yield sink.take()
            yield sink.take()

        return StreamingResponse(
            stream(),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="darkroom-archive.zip"'},
        )

    return record_or_404


# Every method a client can send: the unguarded app's refusal answers all of them.
_ALL_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]


def unguarded_app(cfg: Unguarded) -> FastAPI:
    """The server with no SingGuard configured: up, and closed.

    It is not the guarded app with a switch turned off. No engine, guard, ComfyUI, Ollama, NudeNet or age client is
    built here, and *cfg* holds none of their URLs, so no request can reach one. It serves ``/health`` (``degraded``,
    with a failed required ``singguard`` check that names the variables to set), and what :func:`_serve_records` reads
    from the record store. Every other path and method, including any route added to :func:`create_app` later, falls
    through to one handler that answers 503 with the same reason. Generation, pre-flight, depth maps, models, policies
    and purge are all in that set.
    """
    # Once, at startup: variable names only, never a value (a URL can carry a credential).
    logger.error("guard.not_configured", detail=cfg.reason, serving="health and stored records only; all else is 503")
    app = FastAPI(title="Argus Darkroom", description="Guarded image generation.", version=__version__)
    app.state.engine = None
    app.state.unguarded = cfg
    _guard_writes(app, cfg.cors_origins, cfg.read_only)

    def health_fields() -> dict:
        # Every field the guarded /health has, so a client that reads one finds it: nothing answers, nothing can run.
        return {
            "backends": dict.fromkeys(("comfyui", "singguard", "nudenet", "ollama", "puppet", "age"), False),
            "age_unavailable": GUARD_UNCONFIGURED,
            "minor_exposed": cfg.minor_exposed,
            "policies": list(policy_table.POLICIES),
            "default_policy": cfg.policy,
            "thinking": cfg.thinking,
            "nsfw_allowed": cfg.allow_nsfw,
            "sfw_available": False,
            "pose_available": False,
            "nudenet_detector": None,
            "expansion_enabled": False,
            "workflow_loaded": False,
            "workflows": [],
            "devmode": cfg.devmode,
            "demo": False,
            "generating": False,
            "gpu": {"coordinator": "none", "in_flight": 0, "limit": 0},
            "guard_configured": False,
            "guard_unconfigured": cfg.reason,
        }

    detail = f"{GUARD_UNCONFIGURED}: {cfg.problem}. Set {' or '.join(GUARD_VARIABLES)}"
    install_health(
        app,
        service="argus-darkroom",
        version=__version__,
        wire={"record_version": RECORD_VERSION},
        # Required and always failing, so the status is `degraded` (200, like an unreachable SingGuard): the
        # container stays up and healthy enough to be asked why.
        checks={"singguard": lambda: DependencyCheck(ok=False, detail=detail)},
        read_only=cfg.read_only,
        extra=health_fields,
        check_ttl_s=0,
    )
    _serve_records(app, RecordStore(cfg.output_root), threading.Lock())

    @app.api_route("/{path:path}", methods=_ALL_METHODS, include_in_schema=False)
    def refuse(path: str) -> None:
        raise HTTPException(503, cfg.reason)

    return app


def create_app(engine: Engine | None = None, settings: Settings | None = None) -> FastAPI:
    """The app. With neither argument it reads ``ARGUS_DARKROOM_*`` from the environment.

    Darkroom never runs unguarded. With no usable SingGuard URL (unset, ``off`` or malformed) there is no engine, and
    what is returned is :func:`unguarded_app`: it stays up, says why on ``/health`` and refuses everything but reading
    what was already released. Any other configuration that cannot run raises
    :class:`~argus_darkroom.config.ConfigError`.
    """
    if engine is None:
        found = settings or Settings.for_server()
        if isinstance(found, Unguarded):
            return unguarded_app(found)
        engine = Engine(found)
    cfg = engine.settings

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # With the broker, darkroom is a tenant from startup, not from its first lease: the broker gets the manifest's
        # kind defaults before any work arrives, and a goodbye at shutdown. Never blocks or fails startup; a no-op for
        # the local coordinators.
        async with aregistered(engine.gpu):
            yield

    app = FastAPI(
        title="Argus Darkroom", description="Guarded image generation.", version=__version__, lifespan=lifespan
    )
    app.state.engine = engine
    # Entered for the whole of each generation or depth run, taken alone by a purge. One run at a time, unless the GPU
    # broker queues them: then each waits for its own lease and the gate only bounds how many may wait.
    busy = RunGate(gpu.MAX_IN_FLIGHT if gpu.queues(engine.gpu) else 1)
    app.state.busy = busy
    # Held while a stored record changes, so a removal that read a record cannot save it back over a purge. Separate from
    # busy, which would refuse every removal during a generation.
    record_lock = threading.Lock()

    _guard_writes(app, cfg.cors_origins, cfg.read_only)
    record_or_404 = _serve_records(app, engine.store, record_lock)

    def busy_detail() -> str:
        if busy.limit == 1:
            return "a generation or depth map is already running"
        return f"{busy.limit} generations or depth maps are already queued or running"

    # -- read-only info ----------------------------------------------------------------------------------
    # GET /health: cortex's ServiceHealth, with every field darkroom served before it beside the contract's.
    #
    # Each backend is a dependency check, probed on every request (no cache: a client reads `backends` to decide
    # what it may send, so the answer is as fresh as it was before the contract). `backends` holds what the checks
    # last found; `GET /health?checks=0` probes nothing, so it repeats the last full answer (all false before one).
    # Only a configured backend is a check: an unset NudeNet or Ollama is false in `backends` and absent from `checks`.
    remote_age = engine.scanner.age if isinstance(engine.scanner.age, HttpAgeEstimator) else None
    last = _Probes(age=engine.age_status)

    def check_comfyui() -> bool:
        return last.done("comfyui", probe(cfg.comfyui_url.rstrip("/") + "/system_stats"))

    def check_singguard() -> bool:
        return last.done("singguard", probe(cfg.singguard_url))

    def check_nudenet() -> bool:
        # The detector is asked alongside, as it was: reported, never used to decide.
        with ThreadPoolExecutor(1) as pool:
            detector = pool.submit(nudenet_detector, engine.scanner.nudenet)
            up = last.done("nudenet", probe(cfg.nudenet_url))
            last.detector = detector.result()
        return up

    def check_ollama() -> bool:
        return last.done("ollama", probe(cfg.ollama_url))

    def check_puppet() -> bool:
        return last.done("puppet", probe(cfg.puppet_url.rstrip("/") + "/health?checks=0"))

    def check_age(base_url: str) -> DependencyCheck:
        # Re-probe the estimator the engine actually uses; the boot status is stale.
        last.age = remote_age_status(base_url)
        return DependencyCheck(ok=last.age.available, detail=last.age.reason)

    checks: dict[str, Callable[[], bool | DependencyCheck]] = {"comfyui": check_comfyui, "singguard": check_singguard}
    if cfg.nudenet_url:
        checks["nudenet"] = check_nudenet
    if cfg.ollama_url:
        checks["ollama"] = check_ollama
    if cfg.puppet_url:
        checks["puppet"] = check_puppet
    if remote_age is not None:
        checks["age"] = functools.partial(check_age, remote_age.base_url)

    def health_fields() -> dict:
        return {
            # Whether each is configured and answers, never where: the URL can carry a credential. `age` is whether
            # the age check can run; `age_unavailable` says why not.
            "backends": {**last.up, "age": last.age.available},
            "age_unavailable": last.age.reason,
            # How the minor-exposed rule places exposure: "frame" or "overlap" (ARGUS_DARKROOM_MINOR_EXPOSED).
            "minor_exposed": cfg.minor_exposed,
            "policies": list(policy_table.POLICIES),
            "default_policy": cfg.policy,
            "thinking": cfg.thinking,
            "nsfw_allowed": cfg.allow_nsfw,
            "sfw_available": cfg.sfw_available,
            # Whether a request may carry a pose input: argus-puppet is configured (`backends.puppet`: it answers).
            "pose_available": cfg.pose_available,
            # The NudeNet detector the server reports ("640m", "320n"), or null when it does not say or is unreachable.
            "nudenet_detector": last.detector,
            "expansion_enabled": cfg.expansion_enabled,
            "workflow_loaded": engine.workflows is not None,
            "workflows": engine.workflows.names if engine.workflows else [],
            "devmode": cfg.devmode,
            # True only for demo/run.py's fake backends: nothing it "generates" touches a GPU.
            "demo": cfg.demo,
            "generating": busy.locked(),
            # The coordinator (none, lease or broker); how many runs are in flight, and how many may be.
            "gpu": {"coordinator": engine.gpu.name, "in_flight": busy.running, "limit": busy.limit},
            # False only on the app that has no SingGuard URL and refuses everything (unguarded_app).
            "guard_configured": True,
        }

    install_health(
        app,
        service="argus-darkroom",
        version=__version__,
        wire={"record_version": RECORD_VERSION},
        checks=checks,
        # Expansion and pose input are optional: a request that asks for one without its backend is refused or fails,
        # and nothing else changes.
        informational={"ollama", "puppet"} & checks.keys(),
        read_only=cfg.read_only,
        extra=health_fields,
        check_ttl_s=0,
        check_timeout_s=HEALTH_CHECK_TIMEOUT,
    )

    @app.get("/policies")
    def policies() -> dict:
        return {
            "default": cfg.policy,
            "policies": [
                {
                    "name": name,
                    "mode": policy_table.mode(name),
                    "summary": POLICY_SUMMARIES.get(name, ""),
                    "illegal_rules": [
                        {"code": code, "name": rule_names(name).get(code, "")}
                        for code in sorted(policy_table.illegal_rules(name))
                    ],
                    # Whether this policy can enforce here. NSFW policies run as advisory only unless the server was
                    # started with --allow-nsfw, and an SFW policy needs NudeNet to gate the output.
                    "enforceable": cfg.allow_nsfw if policy_table.mode(name) == "nsfw" else cfg.sfw_available,
                    # SingGuard's own rules for `default`: nothing is sent, so there is no wording to show.
                    "text": policy_table.policy_text(name),
                }
                for name in policy_table.POLICIES
            ],
        }

    @app.get("/models")
    def models() -> dict:
        workflows = engine.workflows
        try:
            checkpoints = engine.models.names("checkpoint")
            controlnets = engine.models.names("controlnet")
            found = {
                # lora_counts: the LoRA counts a workflow renders this checkpoint with. Empty: it cannot be rendered.
                # img2img_lora_counts: the same for image-to-image. Empty: it takes no input image.
                # inpaint_lora_counts: the same for inpainting (an input image with a mask). Empty: it takes no mask.
                # pose_lora_counts: the same with a pose input (a pose ControlNet). Empty: it takes no pose.
                # img2img_pose_lora_counts, inpaint_pose_lora_counts: an input image (and a mask) with a pose.
                "checkpoints": [
                    {
                        "name": n,
                        "family": checkpoint_family(n),
                        "lora_counts": workflows.lora_counts(n) if workflows else [],
                        "img2img_lora_counts": workflows.lora_counts(n, "img2img") if workflows else [],
                        "inpaint_lora_counts": workflows.lora_counts(n, "inpaint") if workflows else [],
                        "pose_lora_counts": workflows.lora_counts(n, "pose") if workflows else [],
                        "img2img_pose_lora_counts": workflows.lora_counts(n, "img2img+pose") if workflows else [],
                        "inpaint_pose_lora_counts": workflows.lora_counts(n, "inpaint+pose") if workflows else [],
                    }
                    for n in checkpoints
                ],
                "default_checkpoint": pick_default(checkpoints, cfg.default_checkpoint),
                "loras": [{"name": n} for n in engine.models.names("lora")],
                "vaes": [{"name": n} for n in engine.models.names("vae")],
                # ControlNet files (the models root's controlnet/ folder), for a pose input; the one used when a
                # request names none, or null when ARGUS_DARKROOM_POSE_CONTROLNET is unset or matches nothing.
                "controlnets": [{"name": n} for n in controlnets],
                "default_pose_controlnet": cfg.pose_controlnet if cfg.pose_controlnet in controlnets else None,
                "workflows": [w.describe() for w in workflows.workflows] if workflows else [],
            }
        except RequestError as exc:
            raise HTTPException(400, str(exc)) from exc
        # What ComfyUI accepts, so the form offers only values that run (empty when ComfyUI does not say).
        return found | engine.comfyui.sampling_choices()

    # -- pre-flight and generation -----------------------------------------------------------------------
    @app.post("/preflight")
    async def preflight(request: GenerateRequest) -> dict:
        try:
            result = await run_in_threadpool(engine.preflight, request)
        except RequestError as exc:
            raise HTTPException(400, str(exc)) from exc
        # Expansion is not run here: it is not deterministic, so a verdict on one expansion would say nothing about
        # the one a generation makes. The generation judges its own final prompt, expansion included.
        return engine.preflight_view(request, result) | {"expansion_applied": False}

    @app.post("/generate/stream")
    async def generate_stream(
        request: GenerateRequest, priority: PriorityClass = Depends(priority_of)
    ) -> StreamingResponse:
        if not busy.enter():
            raise HTTPException(409, busy_detail())
        try:
            run = await run_in_threadpool(engine.begin, request, priority)
        except RequestError as exc:
            busy.leave()
            raise HTTPException(400, str(exc)) from exc
        except BaseException:
            busy.leave()
            raise

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, object] | None] = asyncio.Queue()

        def put(item: tuple[str, object] | None) -> None:
            with contextlib.suppress(RuntimeError):  # the loop is gone: the server is shutting down
                loop.call_soon_threadsafe(queue.put_nowait, item)

        def work() -> None:
            try:
                run(lambda kind, data: put((kind, data)))
            except Exception:
                logger.exception("generate.failed")
                put(("error", {"detail": "the generation failed unexpectedly; no image was released"}))
            finally:
                busy.leave()  # the run goes on to its record even if the client has hung up
                put(None)

        threading.Thread(target=work, name="darkroom-generate", daemon=True).start()

        async def stream():
            while (item := await queue.get()) is not None:
                yield sse(*item)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    # -- depth maps --------------------------------------------------------------------------------------
    def stored_image(found: GenerationRecord) -> bytes:
        """The record's stored image, checked against the record's hash. 409 when it has none."""
        path = engine.store.image_path(found.id)
        if found.image is None or found.image.path is None or found.removed_at is not None or not path.is_file():
            raise HTTPException(409, "this record has no stored image")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != found.image.sha256:
            raise HTTPException(409, "the stored image does not match its record")
        return data

    @app.post("/depth/{record_id}")
    async def make_depth(record_id: str, priority: PriorityClass = Depends(priority_of)) -> DepthRecord:
        """A depth map of the record's stored image, computed once and kept beside it.

        No pre-flight and no scan: the image was already judged and released, and nothing new (no prompt, no imagery)
        goes in. See :mod:`argus_darkroom.depth`. A write: ``WriteGuard`` and read-only cover it like every POST.
        """
        found = record_or_404(record_id)
        if found.depth is not None and engine.store.depth_path(record_id).is_file():
            return found.depth  # already made: never recomputed
        image = await run_in_threadpool(stored_image, found)  # a file read: off the event loop
        if not busy.enter():
            raise HTTPException(409, busy_detail())

        def compute() -> DepthRecord:
            try:
                depth_map, depth_record = engine.depth(found, image, priority)
            except GenerationError as exc:
                logger.warning("depth.failed", record_id=record_id, error=str(exc))
                raise HTTPException(502, f"the depth map could not be made: {exc}") from exc
            with record_lock:
                # Read again: the image may have been removed, or the record purged, while ComfyUI ran.
                current = record_or_404(record_id)
                if current.depth is not None and engine.store.depth_path(record_id).is_file():
                    return current.depth
                if current.image is None or current.image.path is None or current.removed_at is not None:
                    raise HTTPException(409, "this record has no stored image")
                engine.store.write_depth(record_id, depth_map.png)
                current.depth = depth_record
                try:
                    engine.store.save(current)
                except Exception:
                    engine.store.depth_path(record_id).unlink(missing_ok=True)
                    raise
                return depth_record

        try:
            return await run_in_threadpool(compute)
        finally:
            busy.leave()

    # -- purge (devmode only) ----------------------------------------------------------------------------
    if cfg.devmode:

        async def require_confirm(request: Request) -> None:
            try:
                body = await request.json()
            except ValueError:
                body = None
            if body != {"confirm": "purge"}:
                raise HTTPException(400, 'the body must be {"confirm": "purge"}')

        @app.post("/purge/{record_id}")
        async def purge_record(record_id: str, request: Request) -> dict:
            """Delete one record and its stored image, which ``DELETE /record/{id}`` keeps. Records are the audit trail,
            so this is a dev tool like ``/purge``."""
            await require_confirm(request)
            if not busy.acquire(blocking=False):
                raise HTTPException(409, "a generation is running")

            def purge_one() -> dict:
                # In a worker thread: waiting on record_lock must not block the event loop.
                with record_lock:
                    # Not parsed: a record that no longer reads is one this must still be able to remove.
                    if not engine.store.exists(record_id):
                        raise HTTPException(404, "no such record")
                    return {"records": 1, "images": engine.store.delete(record_id)}

            try:
                return await run_in_threadpool(purge_one)
            finally:
                busy.release()

        @app.post("/purge")
        async def purge(request: Request) -> dict:
            """Delete every record and stored image, and clear darkroom's prompts from ComfyUI's history."""
            await require_confirm(request)
            if not busy.acquire(blocking=False):
                raise HTTPException(409, "a generation is running")

            def purge_files() -> dict[str, int]:
                # In a worker thread: waiting on record_lock must not block the event loop.
                removed = {"records": 0, "images": 0}
                with record_lock:
                    # nosemgrep: handler-scans-all - purge-all: visiting every record is the operation
                    for path in engine.store.root.iterdir():
                        if path.name.endswith(DEPTH_SUFFIX) and valid_id(path.name.removesuffix(DEPTH_SUFFIX)):
                            path.unlink(missing_ok=True)  # derived from an image; counted with neither
                        elif valid_id(path.stem) and path.suffix in (".json", ".png"):
                            path.unlink(missing_ok=True)
                            removed["records" if path.suffix == ".json" else "images"] += 1
                        elif path.name.endswith((".json.tmp", ".png.tmp")):
                            path.unlink(missing_ok=True)
                return removed

            try:
                removed = await run_in_threadpool(purge_files)
                history: int | None = 0
                # The configured ComfyUI and every one the GPU broker sent work to.
                for comfyui in engine.comfyui_clients():
                    try:
                        cleared = await run_in_threadpool(comfyui.purge_history)
                    except GenerationError as exc:
                        logger.warning("purge.history_failed", error=str(exc))
                        history = None  # a ComfyUI could not be reached: said so, not hidden
                    else:
                        history = None if history is None else history + cleared
                return removed | {"comfyui_history": history}
            finally:
                busy.release()

    return app
