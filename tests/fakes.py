"""Fake SingGuard server, in the pattern of cortex's ``_Fake``: a real HTTP server on a thread.

No real sexual, violent or illegal content goes in the tests. An illegal verdict comes from the fake's
canned answer for a harmless marker string, never from the content of a prompt or an image.
"""

from __future__ import annotations

import json
import threading
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
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> FakeSingGuard:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
