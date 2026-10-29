"""Fake ComfyUI: the HTTP API on one port and the websocket on another, in the pattern of ``FakeSingGuard``.

It keeps a ``history`` like the real thing (the submitted graph stays in it until deleted), so tests can assert
that darkroom cleared it. The PNG it returns is a flat grey square, never real content.
"""

from __future__ import annotations

import io
import json
import struct
import threading
import urllib.parse
import uuid
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

# (fake, prompt_id, graph) -> messages to send: dicts become JSON text frames, bytes go as binary frames
Script = Callable[["FakeComfyUI", str, dict], list]

PREVIEW_HEADER = struct.pack(">II", 1, 1)  # event type PREVIEW_IMAGE, format JPEG
IMAGE_HEADER = struct.pack(">II", 1, 2)  # format PNG


def grey_png(size: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (128, 128, 128)).save(buf, "PNG")
    return buf.getvalue()


def event(kind: str, **data: object) -> dict:
    return {"type": kind, "data": data}


def default_script(fake: FakeComfyUI, prompt_id: str, graph: dict) -> list:
    save = next(i for i, n in graph.items() if n["class_type"] == "SaveImageWebsocket")
    return [
        event("execution_start", prompt_id=prompt_id),
        event("executing", node="3", prompt_id=prompt_id),
        PREVIEW_HEADER + b"latent preview, not the image",
        event("progress", value=1, max=2, node="3", prompt_id=prompt_id),
        event("progress", value=2, max=2, node="3", prompt_id=prompt_id),
        event("executing", node=save, prompt_id=prompt_id),
        IMAGE_HEADER + fake.png,
        event("executing", node=None, prompt_id=prompt_id),
    ]


class FakeComfyUI:
    def __init__(
        self,
        script: Script = default_script,
        output_nodes: frozenset[str] = frozenset({"SaveImageWebsocket"}),
        unknown_nodes: frozenset[str] = frozenset(),
    ) -> None:
        self.script = script
        self.output_nodes = output_nodes
        self.unknown_nodes = unknown_nodes
        self.png = grey_png()
        self.version = "0.3.test"
        self.reject_with: dict | None = None
        self.history: dict[str, dict] = {}
        self.graphs: list[dict] = []
        self.calls: list[tuple[str, str]] = []
        self.interrupted: list[str] = []
        self._conns: dict[str, object] = {}
        self._registered = threading.Condition()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, body: object) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                fake.calls.append(("GET", self.path))
                if self.path == "/system_stats":
                    self._send(200, {"system": {"comfyui_version": fake.version}})
                elif self.path.startswith("/object_info/"):
                    cls = urllib.parse.unquote(self.path.removeprefix("/object_info/"))
                    if cls in fake.unknown_nodes:
                        self._send(200, {})
                    else:
                        self._send(200, {cls: {"output_node": cls in fake.output_nodes}})
                else:
                    self._send(404, {})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                fake.calls.append(("POST", self.path))
                if self.path == "/prompt":
                    if fake.reject_with is not None:
                        self._send(400, fake.reject_with)
                        return
                    prompt_id = str(uuid.uuid4())
                    fake.graphs.append(body["prompt"])
                    fake.history[prompt_id] = {"prompt": body["prompt"]}
                    threading.Thread(
                        target=fake._run, args=(body["client_id"], prompt_id, body["prompt"]), daemon=True
                    ).start()
                    self._send(200, {"prompt_id": prompt_id, "number": 1, "node_errors": {}})
                elif self.path == "/history":
                    for prompt_id in body.get("delete", []):
                        fake.history.pop(prompt_id, None)
                    self._send(200, {})
                elif self.path in ("/queue", "/interrupt"):
                    fake.interrupted += body.get("delete", [])
                    self._send(200, {})
                else:
                    self._send(404, {})

            def log_message(self, *args: object) -> None:
                pass

        def ws_handler(conn) -> None:
            client_id = urllib.parse.parse_qs(urllib.parse.urlsplit(conn.request.path).query)["clientId"][0]
            with fake._registered:
                fake._conns[client_id] = conn
                fake._registered.notify_all()
            try:
                for _ in conn:
                    pass
            except ConnectionClosed:
                pass

        self._http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._ws = serve(ws_handler, "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self._http.server_address[1]}"
        self.ws_url = f"ws://127.0.0.1:{self._ws.socket.getsockname()[1]}/ws"
        self._threads = [
            threading.Thread(target=self._http.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True),
            threading.Thread(target=self._ws.serve_forever, daemon=True),
        ]

    def _run(self, client_id: str, prompt_id: str, graph: dict) -> None:
        with self._registered:
            self._registered.wait_for(lambda: client_id in self._conns, timeout=5)
            conn = self._conns.get(client_id)
        if conn is None:
            return
        try:
            for message in self.script(self, prompt_id, graph):
                conn.send(message if isinstance(message, bytes) else json.dumps(message))
        except ConnectionClosed:
            pass

    def __enter__(self) -> FakeComfyUI:
        for t in self._threads:
            t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._http.shutdown()
        self._http.server_close()
        self._ws.shutdown()
