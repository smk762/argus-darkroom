"""The darkroom FastAPI server.

========================  =====================================================================================
``GET /health``           version, which backends answer, the policy list, ``nsfw_allowed``, ``devmode``
``GET /policies``         names, the enforcing default, each policy's illegal rules and wording
``GET /models``           checkpoints, LoRAs and VAEs found under the models root
``POST /preflight``       the text check alone, so a form can show a verdict before Generate
``POST /generate/stream`` SSE: ``preflight``, ``queued``, ``progress``, ``scan``, ``complete``; one at a time
``GET /records``          paged, filterable by outcome; ``GET /record/{id}``
``GET /image/{id}``       ids only, no path input; 404 with the record id when there is no stored image
``DELETE /record/{id}``   the user removing their own output: the file goes, the record stays
``GET /archive``          a streamed zip of every record and every stored image (read-only, always on)
``POST /purge``           everything, from a clean slate; exists only with ``ARGUS_DARKROOM_DEVMODE=1``
========================  =====================================================================================

Reads are open to the configured CORS origins; writes are refused from any other origin (``WriteGuard`` and
``cross_site_refuse``, because CORS alone is not a write boundary). The server binds to localhost by default.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import threading
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import structlog
from argus_cortex.moderation import policies as policy_table
from argus_cortex.server import WriteGuard, cross_site_refuse
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from argus_darkroom import __version__
from argus_darkroom.comfyui import GenerationError
from argus_darkroom.config import Settings
from argus_darkroom.engine import Engine
from argus_darkroom.models import GenerationRecord, Outcome
from argus_darkroom.request import GenerateRequest, RequestError
from argus_darkroom.store import valid_id

logger = structlog.get_logger()

PROBE_TIMEOUT = 2.0
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


def sse(kind: str, data: object) -> bytes:
    return f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode()


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


def create_app(engine: Engine | None = None, settings: Settings | None = None) -> FastAPI:
    """The app. With neither argument it reads ``ARGUS_DARKROOM_*`` from the environment, and a configuration
    that cannot run (an empty ``SINGGUARD_URL``) raises :class:`~argus_darkroom.config.ConfigError`: darkroom
    never runs unguarded, so the server does not start."""
    engine = engine or Engine(settings or Settings.from_env())
    cfg = engine.settings
    app = FastAPI(title="Argus Darkroom", description="Guarded image generation.", version=__version__)
    app.state.engine = engine
    busy = threading.Lock()  # held for the whole of one generation (or a purge)
    app.state.busy = busy

    # -- CORS and the write guard ------------------------------------------------------------------------
    wildcard = "*" in cfg.cors_origins
    named = [o for o in cfg.cors_origins if o != "*"]
    # A wildcard grants anonymous reads, never a cross-site write: naming an origin is the way to get those.
    app.add_middleware(WriteGuard, refuse=cross_site_refuse(named))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if wildcard else named,
        allow_credentials=not wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def record_or_404(record_id: str) -> GenerationRecord:
        record = engine.store.load(record_id) if valid_id(record_id) else None
        if record is None:
            raise HTTPException(404, "no such record")
        return record

    # -- read-only info ----------------------------------------------------------------------------------
    @app.get("/health")
    def health() -> dict:
        urls = {
            "comfyui": cfg.comfyui_url.rstrip("/") + "/system_stats",
            "singguard": cfg.singguard_url,
            "nudenet": cfg.nudenet_url,
            "ollama": cfg.ollama_url,
        }
        with ThreadPoolExecutor(len(urls)) as pool:
            up = dict(zip(urls, pool.map(probe, urls.values()), strict=True))
        return {
            "version": __version__,
            # Whether each answers, never where: the URL can carry a credential.
            "backends": {name: {"configured": bool(url), "up": up[name]} for name, url in urls.items()},
            "policies": list(policy_table.POLICIES),
            "enforcing_policy": cfg.policy,
            "nsfw_allowed": cfg.allow_nsfw,
            "sfw_available": cfg.sfw_available,
            "expansion_enabled": cfg.expansion_enabled,
            "workflow_loaded": engine.workflow is not None,
            "age": {"available": engine.age_status.available, "reason": engine.age_status.reason},
            "devmode": cfg.devmode,
            "busy": busy.locked(),
        }

    @app.get("/policies")
    def policies() -> dict:
        return {
            "default": cfg.policy,
            "policies": [
                {
                    "name": name,
                    "mode": policy_table.mode(name),
                    "illegal_rules": sorted(policy_table.illegal_rules(name)),
                    # NSFW policies run as advisory only unless the server was started with --allow-nsfw.
                    "advisory_only": policy_table.mode(name) == "nsfw" and not cfg.allow_nsfw,
                    # SingGuard's own rules for `default`: nothing is sent, so there is no wording to show.
                    "text": policy_table.policy_text(name),
                }
                for name in policy_table.POLICIES
            ],
        }

    @app.get("/models")
    def models() -> dict:
        try:
            return {
                "checkpoints": engine.models.names("checkpoint"),
                "loras": engine.models.names("lora"),
                "vaes": engine.models.names("vae"),
            }
        except RequestError as exc:
            raise HTTPException(400, str(exc)) from exc

    # -- pre-flight and generation -----------------------------------------------------------------------
    @app.post("/preflight")
    async def preflight(request: GenerateRequest) -> dict:
        try:
            result = await run_in_threadpool(engine.preflight, request)
        except RequestError as exc:
            raise HTTPException(400, str(exc)) from exc
        # Expansion is not run here: it is not deterministic, so a verdict on one expansion would say nothing about
        # the one a generation makes. The generation judges its own final prompt, expansion included.
        return result.payload() | {"final_prompt": engine.final_prompt(request), "expansion_applied": False}

    @app.post("/generate/stream")
    async def generate_stream(request: GenerateRequest) -> StreamingResponse:
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "a generation is already running")
        try:
            run = await run_in_threadpool(engine.begin, request)
        except RequestError as exc:
            busy.release()
            raise HTTPException(400, str(exc)) from exc
        except BaseException:
            busy.release()
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
                busy.release()  # the run goes on to its record even if the client has hung up
                put(None)

        threading.Thread(target=work, name="darkroom-generate", daemon=True).start()

        async def stream():
            while (item := await queue.get()) is not None:
                yield sse(*item)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    # -- records and images ------------------------------------------------------------------------------
    @app.get("/records")
    def records(
        outcome: Outcome | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=MAX_PAGE),
    ) -> dict:
        page, total = engine.store.page(outcome, offset, limit)
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "records": [r.model_dump(mode="json") for r in page],
        }

    @app.get("/record/{record_id}")
    def record(record_id: str) -> GenerationRecord:
        return record_or_404(record_id)

    @app.get("/image/{record_id}")
    def image(record_id: str):
        found = record_or_404(record_id)
        path = engine.store.image_path(record_id)
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

        found = record_or_404(record_id)
        if found.removed_at is not None:
            return found
        if found.image is None or found.image.path is None:
            raise HTTPException(409, "this record has no stored image to remove")
        engine.store.remove_image(record_id)
        found.image.path = None
        found.removed_at = datetime.now(UTC)
        engine.store.save(found)
        return found

    @app.get("/archive")
    def archive() -> StreamingResponse:
        """Every record (JSON) and every stored image, as one zip. Deleted and omitted images were never
        written, so they are not in it."""
        store = engine.store

        def stream() -> Iterator[bytes]:
            sink = _Sink()
            with zipfile.ZipFile(sink, "w", zipfile.ZIP_DEFLATED) as zf:
                for path in sorted(store.root.glob("*.json")):
                    if not valid_id(path.stem):
                        continue
                    zf.write(path, f"records/{path.name}")
                    yield sink.take()
                    image_path = store.root / f"{path.stem}.png"
                    if image_path.is_file():
                        zf.write(image_path, f"images/{image_path.name}")
                        yield sink.take()
            yield sink.take()

        return StreamingResponse(
            stream(),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="darkroom-archive.zip"'},
        )

    # -- purge (devmode only) ----------------------------------------------------------------------------
    if cfg.devmode:

        @app.post("/purge")
        async def purge(request: Request) -> dict:
            """Delete every record and stored image, and clear darkroom's prompts from ComfyUI's history."""
            try:
                body = await request.json()
            except ValueError:
                body = None
            if body != {"confirm": "purge"}:
                raise HTTPException(400, 'the body must be {"confirm": "purge"}')
            if not busy.acquire(blocking=False):
                raise HTTPException(409, "a generation is running")
            try:
                removed = {"records": 0, "images": 0}
                for path in engine.store.root.iterdir():
                    if valid_id(path.stem) and path.suffix in (".json", ".png"):
                        path.unlink(missing_ok=True)
                        removed["records" if path.suffix == ".json" else "images"] += 1
                    elif path.name.endswith((".json.tmp", ".png.tmp")):
                        path.unlink(missing_ok=True)
                try:
                    history: int | None = await run_in_threadpool(engine.comfyui.purge_history)
                except GenerationError as exc:
                    logger.warning("purge.history_failed", error=str(exc))
                    history = None  # ComfyUI could not be reached: said so, not hidden
                return removed | {"comfyui_history": history}
            finally:
                busy.release()

    return app
