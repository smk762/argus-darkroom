"""Fake SingGuard server, in the pattern of cortex's ``_Fake``: a real HTTP server on a thread.

No real sexual, violent or illegal content goes in the tests. An illegal verdict comes from the fake's
canned answer for a harmless marker string, never from the content of a prompt or an image.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from argus_cortex.moderation import policies as policy_table

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
        health: dict | bytes | None = None,
        status: int = 200,
        health_status: int = 200,
    ) -> None:
        self.faces = faces or []
        self.people = people
        self.people_check = people_check
        self.people_error = people_error
        self.health = health if health is not None else {"status": "ok", "faces": True, "mivolo": True}
        self.status = status
        self.health_status = health_status
        self.posts: list[bytes] = []
        self.paths: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
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
