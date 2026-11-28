"""Fake SingGuard server, in the pattern of cortex's ``_Fake``: a real HTTP server on a thread.

No real sexual, violent or illegal content goes in the tests. An illegal verdict comes from the fake's
canned answer for a harmless marker string, never from the content of a prompt or an image.
"""

from __future__ import annotations

import io
import json
import threading
import urllib.parse
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from argus_cortex.moderation import policies as policy_table
from PIL import Image

# {"input", "input_role", "image_base64"?, "policy"?, "thinking"} -> (status, answer)
Responder = Callable[[dict], tuple[int, dict]]


def safe_answer() -> dict:
    return {"safe": True, "verdict": "safe", "rule": None, "explanation": "ok", "categories": [], "model": "fake"}


def unsafe_answer(*rules: str, explanation: str = "because") -> dict:
    """An unsafe verdict naming *rules* (e.g. ``"A. Minors"``)."""
    return {
        "safe": False,
        "verdict": "unsafe",
        "rule": rules[0] if rules else None,
        "explanation": explanation,
        "categories": [{"code": r, "label": r, "why": explanation} for r in rules],
        "model": "fake",
    }


def policy_of(body: dict) -> str:
    """The darkroom policy name a request's policy text stands for."""
    text = body.get("policy")
    if text is None:
        return "default"
    for name in policy_table.POLICIES:
        if policy_table.policy_text(name) == text.strip():
            return name
    raise AssertionError("a request carried policy text that is not a named policy")


class FakeSingGuard:
    """Answers ``POST /api/guard/check`` from *responder*, and keeps every request body it saw."""

    def __init__(self, responder: Responder | None = None) -> None:
        self.responder: Responder = responder or (lambda body: (200, safe_answer()))
        self.requests: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                status, answer = fake.responder(body)
                data = json.dumps(answer).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    def __enter__(self) -> FakeSingGuard:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


class FakeNudeNet:
    """``POST /infer``: answers one image with *labels* (class -> score), as the NudeNet server does.

    ``GET /health`` names *model* (the shared 640m server's shape). ``model=None`` answers 404, as the upstream 320n
    image does; ``health_status`` sets another status.
    """

    def __init__(
        self, labels: dict[str, float] | None = None, status: int = 200, model: str | None = "640m.onnx"
    ) -> None:
        self.model = model
        self.health_status = 200
        self.health_calls = 0
        self.labels = labels or {}
        self.box = [0, 0, 10, 10]  # every detection's [x, y, w, h]
        self.status = status
        self.calls = 0
        self.bodies: list[bytes] = []  # every request body, as sent
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                fake.health_calls += 1
                status = 404 if self.path != "/health" or fake.model is None else fake.health_status
                body = {"ok": True, "model": fake.model, "resolution": 640} if status == 200 else {}
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.path != "/infer":  # as the real server: a client posting elsewhere gets no detections
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                fake.bodies.append(body)
                fake.calls += 1
                detections = [{"class": c, "score": s, "box": fake.box} for c, s in fake.labels.items()]
                data = json.dumps({"success": True, "prediction": [detections]}).encode()
                self.send_response(fake.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    def __enter__(self) -> FakeNudeNet:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


class FakeFaces:
    """Curator's ``GET /health`` and ``POST /faces`` for the age check."""

    def __init__(
        self,
        faces: list[dict] | None = None,
        *,
        people: list[dict] | None = None,
        people_check: str | None = None,
        people_error: str | None = None,
        omit_people: bool = False,
        health: dict | bytes | None = None,
        status: int = 200,
        health_status: int = 200,
    ) -> None:
        self.faces = faces or []
        self.people = people
        self.people_check = people_check
        self.people_error = people_error
        self.omit_people = omit_people  # send people_check without a people list (a malformed curator answer)
        self.health = health if health is not None else {"status": "ok", "faces": True, "mivolo": True}
        self.status = status
        self.health_status = health_status
        self.posts: list[bytes] = []
        self.paths: list[str] = []
        self.gets = 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                fake.gets += 1
                health = fake.health
                data = health if isinstance(health, bytes) else json.dumps(health).encode()
                self.send_response(fake.health_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self) -> None:
                fake.paths.append(self.path)
                fake.posts.append(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                payload: dict = {"source": "mivolo", "faces": fake.faces}
                if fake.people is not None or fake.people_check is not None:
                    if not fake.omit_people:
                        payload["people"] = fake.people or []
                    payload["people_check"] = fake.people_check
                    payload["people_error"] = fake.people_error
                data = json.dumps(payload).encode()
                self.send_response(fake.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    def __enter__(self) -> FakeFaces:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


class FakeOllama:
    """``POST /api/chat``: answers with *reply* (a string, or a callable of the request body), and keeps every body."""

    def __init__(
        self, reply: str | Callable[[dict], str] = "a cat, soft light, detailed fur", status: int = 200
    ) -> None:
        self.reply, self.status = reply, status
        self.requests: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                content = fake.reply(body) if callable(fake.reply) else fake.reply
                data = json.dumps({"message": {"role": "assistant", "content": content}}).encode()
                self.send_response(fake.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    def __enter__(self) -> FakeOllama:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


class FakeIris:
    """argus-iris's lease routes, enough for cortex's ``BrokerCoordinator``: it keeps every body it saw.

    ``queue_polls`` answers that many polls ``queued`` before granting; ``base_url`` is what a grant names; ``refuse``
    refuses every lease; ``no_leases`` answers 404 to everything, as an observe-only iris does.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url
        self.queue_polls = 0
        self.refuse = False
        self.no_leases = False
        self.tenants: list[dict] = []
        self.leases: list[dict] = []
        self.releases: list[tuple[str, dict]] = []
        self.unregistered: list[str] = []
        self.auth: list[str | None] = []
        self._polls: dict[str, int] = {}
        fake = self

        def status(lease_id: str, state: str) -> dict:
            body = {"lease_version": "1.0", "lease_id": lease_id, "state": state, "ttl_s": 60.0}
            if state == "granted":
                body |= {"node": "morpheus", "device": 0, "base_url": fake.base_url}
            elif state == "refused":
                body["reason"] = "it can never fit"
            return body

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, body: object) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _state(self, lease_id: str) -> str:
                if fake.refuse:
                    return "refused"
                if fake._polls.get(lease_id, 0) < fake.queue_polls:
                    return "queued"
                return "granted"

            def do_GET(self) -> None:
                lease_id = urllib.parse.urlsplit(self.path).path.removeprefix("/leases/")
                if fake.no_leases or lease_id not in fake._polls:
                    self._send(404, {"detail": "Not Found"})
                    return
                fake._polls[lease_id] += 1
                self._send(200, status(lease_id, self._state(lease_id)))

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                fake.auth.append(self.headers.get("Authorization"))
                if fake.no_leases:
                    self._send(404, {"detail": "Not Found"})
                elif self.path == "/tenants":
                    fake.tenants.append(body)
                    self._send(200, {})
                elif self.path == "/leases":
                    fake.leases.append(body)
                    lease_id = f"lease-{len(fake.leases)}"
                    fake._polls[lease_id] = 0
                    self._send(200, status(lease_id, self._state(lease_id)))
                elif self.path.endswith("/heartbeat"):
                    self._send(200, {"lease_version": "1.0", "state": "granted", "ttl_s": 60.0})
                elif self.path.endswith("/release"):
                    fake.releases.append((self.path.split("/")[2], body))
                    self._send(200, {})
                else:
                    self._send(404, {"detail": "Not Found"})

            def do_DELETE(self) -> None:
                fake.auth.append(self.headers.get("Authorization"))
                if fake.no_leases or not self.path.startswith("/tenants/"):
                    self._send(404, {"detail": "Not Found"})
                    return
                fake.unregistered.append(self.path.removeprefix("/tenants/"))
                self._send(200, {})

            def log_message(self, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    def __enter__(self) -> FakeIris:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


def skeleton_png(size: tuple[int, int], dots: int = 6) -> bytes:
    """A stand-in for puppet's drawing: black, with a few flat-colour squares. Nothing in it but colour."""
    image = Image.new("RGB", size, (0, 0, 0))
    for i in range(dots):
        colour = (255, 40 * i, 255 - 40 * i)
        x, y = (3 + 5 * i) % max(size[0] - 3, 1), (2 + 7 * i) % max(size[1] - 3, 1)
        for dx in range(3):
            for dy in range(3):
                image.putpixel((min(x + dx, size[0] - 1), min(y + dy, size[1] - 1)), colour)
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def person(x: float = 10.0, primary: bool = False) -> dict:
    """One person as puppet's ``PoseResult`` lists them: 133 (x, y, score) points. The numbers mean nothing."""
    return {
        "bbox": [x, 1.0, 20.0, 30.0],
        "det_score": 0.9,
        "primary": primary,
        "keypoints": [[x + i * 0.1, 2.0 + i * 0.2, 0.9] for i in range(133)],
        "part_scores": {"body": 0.9, "feet": 0.9, "face": 0.9, "left_hand": 0.9, "right_hand": 0.9},
    }


class FakePuppet:
    """argus-puppet's ``GET /health``, ``POST /pose`` and ``POST /render``, on a real socket.

    ``/pose`` answers a ``PoseResult`` for the posted image with *people*; ``/render`` answers *drawing* (bytes), or a
    stand-in skeleton of the canvas the request asks for. Every call is kept in order (``calls``), with the bodies
    (``posed``: the images; ``renders``: the JSON).
    """

    DETECTOR = {"name": "yolox_m", "sha256": "a" * 64, "license": "Apache-2.0", "bytes": 1}
    POSE = {"name": "dwpose-l-384x288", "sha256": "b" * 64, "license": "Apache-2.0", "bytes": 1}

    def __init__(self, people: list[dict] | None = None) -> None:
        self.people = [person(primary=True)] if people is None else people
        self.drawing: bytes | None = None
        self.pose_status = 200
        self.render_status = 200
        self.pose_answer: object | None = None  # sent instead of the PoseResult when set
        self.frame: tuple[int, int] | None = None  # the size /pose reports; default: the posted image's
        self.redirect_to: str | None = None
        self.calls: list[str] = []
        self.posed: list[bytes] = []
        self.renders: list[dict] = []
        self.headers: list[dict[str, str]] = []
        self.on_call: Callable[[str], None] | None = None
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, data: bytes, content_type: str = "application/json") -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                if fake.redirect_to and 300 <= status < 400:
                    self.send_header("Location", fake.redirect_to)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                fake.calls.append("GET " + self.path)
                self._send(200, b'{"status": "ok"}')

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                path = urllib.parse.urlsplit(self.path).path
                fake.calls.append("POST " + path)
                fake.headers.append({k.lower(): v for k, v in self.headers.items()})
                if fake.on_call:
                    fake.on_call(path)
                if fake.redirect_to:
                    self._send(307, b"")
                elif path == "/pose":
                    fake.posed.append(body)
                    if fake.pose_status != 200:
                        self._send(fake.pose_status, b'{"detail": "no"}')
                        return
                    width, height = fake.frame or Image.open(io.BytesIO(body)).size
                    answer = fake.pose_answer or {
                        "puppet_version": "1.3",
                        "image": "pose.png",
                        "image_sha256": "c" * 64,
                        "width": width,
                        "height": height,
                        "layout": "coco_wholebody_133",
                        "people": fake.people,
                        "primary": next((i for i, p in enumerate(fake.people) if p.get("primary")), None),
                        "estimator": {
                            "name": "topdown",
                            "layout": "coco_wholebody_133",
                            "detector": fake.DETECTOR,
                            "pose": fake.POSE,
                            "provider": "CPUExecutionProvider",
                            "package_version": "0.4.0",
                            "det_threshold": 0.5,
                            "nms_iou": 0.45,
                        },
                    }
                    self._send(200, json.dumps(answer).encode())
                elif path == "/render":
                    request = json.loads(body)
                    fake.renders.append(request)
                    if fake.render_status != 200:
                        self._send(fake.render_status, b'{"detail": "no"}')
                        return
                    options = request["options"]
                    drawing = fake.drawing or skeleton_png((options["width"], options["height"]))
                    self._send(200, drawing, "image/png")
                else:
                    self._send(404, b"{}")

            def log_message(self, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    def __enter__(self) -> FakePuppet:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
