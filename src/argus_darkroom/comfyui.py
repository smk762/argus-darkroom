"""A minimal ComfyUI client: render one graph and get the image back over the websocket, in memory only.

The flow, in the order ComfyUI needs it:

1. open ``/ws?clientId=...`` **first**, so no message for the run is missed;
2. ``POST /prompt`` the graph;
3. read the socket: ``progress`` text messages, ``executing`` messages (which node is running, and ``node: null``
   when the run is done), and binary frames. A frame sent while the ``SaveImageWebsocket`` node is the running
   node is the image: an 8-byte header (event type, format), then the PNG. Latent previews are binary frames too
   and are ignored;
4. ``POST /history {"delete": [prompt_id]}``, always, so the prompt text does not stay behind in ComfyUI.

HTTP is stdlib; the websocket is ``websockets``' sync client. Proxy settings in the environment are ignored: the
image and the prompt go straight to the configured server.
"""

from __future__ import annotations

import json
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from http.client import HTTPException

import structlog
from websockets.exceptions import WebSocketException
from websockets.sync.client import ClientConnection, connect

logger = structlog.get_logger()

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
HEADER_BYTES = 8
# Every run uses a client id with this prefix. ComfyUI keeps it in the history entry, which is how a purge finds
# the prompts darkroom sent (and only those) if a delete was ever missed.
CLIENT_PREFIX = "argus-darkroom-"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
DEFAULT_TIMEOUT = 600.0  # one generation, from submit to image
HTTP_TIMEOUT = 30.0
RECV_POLL = 5.0  # how long one websocket read waits before the deadline is re-checked


class GenerationError(RuntimeError):
    """The image could not be generated. The message is recorded as the outcome reason."""


Progress = Callable[[int, int], None]  # (step, of)


def _client_id(entry: object) -> str:
    """The client id ComfyUI stored with a history entry (``prompt`` is ``[number, id, graph, extra_data, ...]``)."""
    prompt = entry.get("prompt") if isinstance(entry, dict) else None
    extra = prompt[3] if isinstance(prompt, list) and len(prompt) > 3 else None
    client = extra.get("client_id") if isinstance(extra, dict) else None
    return client if isinstance(client, str) else ""


@dataclass(frozen=True)
class RenderResult:
    image: bytes  # PNG, in memory only
    prompt_id: str


class ComfyUIClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        http_timeout: float = HTTP_TIMEOUT,
        ws_url: str | None = None,
    ) -> None:
        parts = urllib.parse.urlsplit(base_url.strip())
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("the ComfyUI URL must be an http(s) URL")
        self.base_url = base_url.strip().rstrip("/")
        # Normally derived from base_url. A deployment that serves the websocket elsewhere can say where.
        scheme = "wss" if parts.scheme == "https" else "ws"
        self.ws_url = (ws_url or f"{scheme}://{parts.netloc}{parts.path.rstrip('/')}/ws").rstrip("/")
        self.timeout = timeout
        self.http_timeout = http_timeout

    # -- HTTP -------------------------------------------------------------------------------------------

    def _request(self, method: str, path: str, body: object | None = None) -> object:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.base_url + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with _OPENER.open(req, timeout=self.http_timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", "replace")
            exc.close()
            raise GenerationError(f"ComfyUI returned HTTP {exc.code} for {method} {path}: {detail}".strip()) from exc
        except (urllib.error.URLError, HTTPException, TimeoutError, OSError) as exc:
            raise GenerationError(f"ComfyUI request failed: {exc}") from exc
        try:
            return json.loads(raw) if raw else None
        except ValueError as exc:
            raise GenerationError(f"ComfyUI returned a body that is not JSON for {method} {path}") from exc

    def engine_version(self) -> str:
        """The ComfyUI version from ``/system_stats``; ``unknown`` when it does not say."""
        try:
            stats = self._request("GET", "/system_stats")
        except GenerationError:
            return "unknown"
        system = stats.get("system") if isinstance(stats, dict) else None
        version = system.get("comfyui_version") if isinstance(system, dict) else None
        return version if isinstance(version, str) and version else "unknown"

    def output_classes(self, class_types: set[str]) -> dict[str, bool]:
        """For each class, whether ComfyUI says it is an output node. Raises if ComfyUI does not know a class."""
        out: dict[str, bool] = {}
        for class_type in sorted(class_types):
            info = self._request("GET", f"/object_info/{urllib.parse.quote(class_type, safe='')}")
            entry = info.get(class_type) if isinstance(info, dict) else None
            if not isinstance(entry, dict):
                raise GenerationError(f"ComfyUI does not know the node {class_type!r}")
            out[class_type] = bool(entry.get("output_node"))
        return out

    def sampling_choices(self) -> dict[str, list[str]]:
        """The samplers and schedulers ComfyUI's ``KSampler`` accepts, so a form offers only values that run.
        Empty lists when ComfyUI does not say."""
        try:
            info = self._request("GET", "/object_info/KSampler")
            required = info["KSampler"]["input"]["required"]
            return {
                "samplers": [str(name) for name in required["sampler_name"][0]],
                "schedulers": [str(name) for name in required["scheduler"][0]],
            }
        except (GenerationError, KeyError, IndexError, TypeError):
            return {"samplers": [], "schedulers": []}

    def delete_history(self, prompt_id: str) -> None:
        """Drop the prompt from ComfyUI's history. Best effort: it must not hide the run's own error."""
        try:
            self._request("POST", "/history", {"delete": [prompt_id]})
        except GenerationError as exc:
            logger.warning("comfyui.history_delete_failed", prompt_id=prompt_id, error=str(exc))

    def purge_history(self) -> int:
        """Delete every prompt in ComfyUI's history that darkroom sent, whatever run it belonged to.

        Other clients' prompts are left alone. Returns how many were deleted; raises :class:`GenerationError`
        when ComfyUI cannot be reached.
        """
        history = self._request("GET", "/history")
        ours = [
            prompt_id
            for prompt_id, entry in (history.items() if isinstance(history, dict) else [])
            if _client_id(entry).startswith(CLIENT_PREFIX)
        ]
        if ours:
            self._request("POST", "/history", {"delete": ours})
        return len(ours)

    def interrupt(self, prompt_id: str) -> None:
        """Stop *prompt_id* and drop it from the queue. Best effort."""
        for path, body in (("/queue", {"delete": [prompt_id]}), ("/interrupt", {})):
            try:
                self._request("POST", path, body)
            except GenerationError as exc:
                logger.warning("comfyui.interrupt_failed", prompt_id=prompt_id, error=str(exc))

    # -- render -----------------------------------------------------------------------------------------

    def render(self, graph: dict, save_node_id: str, on_progress: Progress | None = None) -> RenderResult:
        """Run *graph* and return the PNG sent by node *save_node_id*, held in memory only.

        The prompt is deleted from ComfyUI's history whether the run succeeds or not. Raises
        :class:`GenerationError`.
        """
        client_id = CLIENT_PREFIX + secrets.token_hex(8)
        deadline = time.monotonic() + self.timeout
        prompt_id: str | None = None
        try:
            with connect(
                f"{self.ws_url}?clientId={client_id}",
                max_size=None,  # a PNG frame is well over the 1 MiB default
                open_timeout=self.http_timeout,
                proxy=None,
            ) as ws:
                resp = self._request("POST", "/prompt", {"prompt": graph, "client_id": client_id})
                prompt_id = self._prompt_id(resp)
                image = self._read(ws, prompt_id, save_node_id, deadline, on_progress)
        except (WebSocketException, OSError) as exc:
            raise GenerationError(f"ComfyUI websocket failed: {exc}") from exc
        except GenerationError:
            if prompt_id is not None and time.monotonic() >= deadline:
                self.interrupt(prompt_id)
            raise
        finally:
            if prompt_id is not None:
                self.delete_history(prompt_id)
        return RenderResult(image=image, prompt_id=prompt_id)

    @staticmethod
    def _prompt_id(resp: object) -> str:
        if not isinstance(resp, dict):
            raise GenerationError("ComfyUI answered /prompt with something unexpected")
        if resp.get("node_errors"):
            raise GenerationError(f"ComfyUI rejected the workflow: {json.dumps(resp['node_errors'])[:1000]}")
        prompt_id = resp.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise GenerationError("ComfyUI returned no prompt_id")
        return prompt_id

    def _read(
        self, ws: ClientConnection, prompt_id: str, save_node_id: str, deadline: float, on_progress: Progress | None
    ) -> bytes:
        current: str | None = None
        images: list[bytes] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GenerationError(f"ComfyUI did not finish within {self.timeout:g}s")
            try:
                message = ws.recv(timeout=min(RECV_POLL, remaining))
            except TimeoutError:
                continue
            if isinstance(message, bytes):
                if current == save_node_id and len(message) > HEADER_BYTES:
                    images.append(message[HEADER_BYTES:])
                continue  # a latent preview, or something not from the save node
            try:
                event = json.loads(message)
            except ValueError:
                continue
            kind, data = event.get("type"), event.get("data") or {}
            if data.get("prompt_id") not in (None, prompt_id):
                continue  # another client's run
            if kind == "executing":
                current = data.get("node")
                if current is None:
                    return self._the_image(images)
            elif kind == "progress" and on_progress is not None:
                if isinstance(data.get("value"), int) and isinstance(data.get("max"), int):
                    on_progress(data["value"], data["max"])
            elif kind == "execution_error":
                detail = data.get("exception_message") or data.get("exception_type") or "unknown error"
                raise GenerationError(f"ComfyUI failed in node {data.get('node_type', '?')}: {str(detail).strip()}")
            elif kind == "execution_interrupted":
                raise GenerationError("ComfyUI run was interrupted")

    @staticmethod
    def _the_image(images: list[bytes]) -> bytes:
        if not images:
            raise GenerationError("ComfyUI finished without sending an image")
        if len(images) > 1:
            raise GenerationError(f"ComfyUI sent {len(images)} images; one image per run is expected")
        if not images[0].startswith(PNG_SIGNATURE):
            raise GenerationError("ComfyUI sent an image that is not a PNG")
        return images[0]
