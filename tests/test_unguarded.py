"""No SingGuard URL: the server stays up, closed. Nothing is generated, checked or released, and no backend is reached.

The server used to exit, which under compose's ``restart: unless-stopped`` was a restart loop with the reason only in
the container's log (#79). Now ``create_app()`` returns ``unguarded_app``: ``/health`` says which variables to set, the
records and images that were already released can still be read, and every other route answers 503.

The backends here are real fake servers with their URLs in the environment, so a route that reached one would show
in its call count. Every URL and value is made up.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator
from dataclasses import replace

import pytest
import structlog
from argus_cortex.service import ServiceHealth
from fakes import FakeFaces
from rig import SAMPLING
from starlette.routing import Route
from starlette.testclient import TestClient

from argus_darkroom.config import GUARD_VARIABLES, ConfigError, GuardNotConfigured, Settings, Unguarded
from argus_darkroom.guard import Guard
from argus_darkroom.server import create_app
from argus_darkroom.server.app import unguarded_app

OWN, SUITE = GUARD_VARIABLES
SECRET = "hunter2-made-up"
MALFORMED = f"operator:{SECRET}@guard.invalid:9150/api"  # no scheme: not a URL darkroom will use
BODY = {
    "user_prompt": "a cat",
    "negative_prompt": "blurry",
    "seed": 3,
    "sampling": SAMPLING.model_dump(),
    "checkpoint": "base.safetensors",
    "loras": [{"name": "style.safetensors", "trigger_words": ["sks"]}],
    "expand": True,
}
# How the guard comes to be missing: (darkroom's own variable, the suite-wide one). None is unset.
STATES = {
    "unset": (None, None),
    "blank": ("", "  "),
    "off": ("off", None),
    "none": ("NONE", None),
    "off-beats-the-suite": ("off", "http://suite.invalid:9150"),
    "malformed": (MALFORMED, None),
    "malformed-suite-wide": (None, MALFORMED),
}
# The routes that still answer with no guard, and why. Everything else the app has, or ever gets, must answer 503:
# adding a route to the server does not add it here.
OPEN = {
    ("GET", "/health"): "says why the server is closed",
    ("GET", "/records"): "reads records that were written under a guard",
    ("GET", "/record/{record_id}"): "reads one such record",
    ("GET", "/image/{record_id}"): "serves an image that was judged and released under a guard",
    ("GET", "/depth/{record_id}"): "serves a depth map already made from a released image",
    ("GET", "/archive"): "a zip of those records, images and depth maps",
    ("DELETE", "/record/{record_id}"): "the user taking their own released image away; nothing is made or shown",
    ("GET", "/openapi.json"): "FastAPI's own docs",
    ("GET", "/docs"): "FastAPI's own docs",
    ("GET", "/docs/oauth2-redirect"): "FastAPI's own docs",
    ("GET", "/redoc"): "FastAPI's own docs",
}
# Named so the walk below cannot pass because the routes went missing.
MUST_REFUSE = {
    ("POST", "/generate/stream"),
    ("POST", "/preflight"),
    ("POST", "/depth/{record_id}"),
    ("GET", "/models"),
    ("GET", "/policies"),
    ("POST", "/purge"),
    ("POST", "/purge/{record_id}"),
}


class Closed:
    """The unguarded app over a store holding one released image, with every backend a live fake."""

    def __init__(self, rig, ollama, faces, monkeypatch, caplog) -> None:
        self.rig, self.ollama, self.faces = rig, ollama, faces
        self.monkeypatch, self.caplog = monkeypatch, caplog
        rig.settings = replace(rig.settings, devmode=True)  # so the reference app has the purge routes too
        engine = rig.engine()
        self.record_id = rig.generate(engine).id  # released under a guard, before the guard goes missing
        assert engine.store.image_path(self.record_id).is_file()
        self.guarded = create_app(engine)
        self.logs: list[dict] = []
        self.bodies: list[str] = []
        env = {
            "COMFYUI_URL": rig.comfyui.url,
            "NUDENET_URL": rig.nudenet.url,
            "OLLAMA_URL": ollama.url,
            "OLLAMA_MODEL": "tiny:1b",
            "AGE_URL": faces.url,
            "OUTPUT_ROOT": str(rig.out),
            "MODELS_ROOT": str(rig.settings.models_root),
            "WORKFLOW": str(rig.settings.workflow),
            "DEVMODE": "1",
        }
        for name, value in env.items():
            monkeypatch.setenv(f"ARGUS_DARKROOM_{name}", value)
        for name in ("POLICY", "ALLOW_NSFW", "READ_ONLY", "CORS_ORIGINS", "THINKING", "MINOR_EXPOSED"):
            monkeypatch.delenv(f"ARGUS_DARKROOM_{name}", raising=False)
        self.forget_calls()

    def forget_calls(self) -> None:
        self.rig.comfyui.calls.clear()
        self.rig.singguard.requests.clear()
        self.rig.nudenet.calls = self.rig.nudenet.health_calls = 0

    def start(self, state: str) -> TestClient:
        for name, value in zip(GUARD_VARIABLES, STATES[state], strict=True):
            if value is None:
                self.monkeypatch.delenv(name, raising=False)
            else:
                self.monkeypatch.setenv(name, value)
        with structlog.testing.capture_logs() as logs, self.caplog.at_level(logging.DEBUG):
            app = create_app()
        self.logs += logs
        return TestClient(app)

    def send(self, c: TestClient, method: str, path: str):
        url = path.replace("{record_id}", self.record_id)
        with self.caplog.at_level(logging.DEBUG):
            response = c.request(method, url, json=BODY if method in ("POST", "PUT", "PATCH") else None)
        self.bodies.append(response.text)
        return response

    def backend_calls(self) -> dict:
        return {
            "comfyui": list(self.rig.comfyui.calls),
            "singguard": list(self.rig.singguard.requests),
            "nudenet": self.rig.nudenet.calls + self.rig.nudenet.health_calls,
            "ollama": list(self.ollama.requests),
            "age": self.faces.gets + len(self.faces.posts),
        }

    def routes(self) -> set[tuple[str, str]]:
        """Every (method, path) the guarded server has, devmode routes included."""
        return {(m, r.path) for r in self.guarded.routes if isinstance(r, Route) for m in r.methods or ()}


NO_CALLS = {"comfyui": [], "singguard": [], "nudenet": 0, "ollama": [], "age": 0}


@pytest.fixture
def closed(rig, ollama, monkeypatch, caplog) -> Iterator[Closed]:
    with FakeFaces() as faces:
        yield Closed(rig, ollama, faces, monkeypatch, caplog)


def refusal(response) -> str:
    assert response.status_code == 503, (response.request.method, response.request.url.path, response.status_code)
    assert response.headers["content-type"].startswith("application/json")  # never an event stream that began
    detail = response.json()["detail"]
    assert "SingGuard is not configured" in detail and OWN in detail and SUITE in detail
    return detail


@pytest.mark.parametrize("state", STATES)
def test_without_a_guard_every_route_but_the_listed_reads_is_a_503_and_no_backend_is_reached(closed, state) -> None:
    c = closed.start(state)
    routes = closed.routes()
    assert routes >= MUST_REFUSE and not MUST_REFUSE & OPEN.keys()
    assert {route for route in OPEN if route not in routes} == set()  # a stale entry opens nothing, but says so
    walked = sorted(routes, key=lambda route: (route[0] == "DELETE", route))  # the removal last: the reads come first
    for method, path in walked:
        response = closed.send(c, method, path)
        if (method, path) in OPEN or (method == "HEAD" and ("GET", path) in OPEN):
            assert response.status_code != 503, (method, path)
        elif method == "HEAD":
            assert response.status_code == 503  # no body to read
        else:
            refusal(response)
    # Anything the server does not have is refused too, whatever the method: a new route starts closed.
    for method in ("GET", "POST", "PUT", "PATCH", "DELETE"):
        refusal(closed.send(c, method, "/a/route/added/later"))
    assert closed.backend_calls() == NO_CALLS


@pytest.mark.parametrize("state", STATES)
def test_without_a_guard_what_was_released_can_still_be_read_and_removed(closed, state) -> None:
    c = closed.start(state)
    rid = closed.record_id
    assert c.get("/records").json()["total"] == 1
    assert c.get(f"/record/{rid}").json()["outcome"] == "passed"
    assert c.get(f"/image/{rid}").content == closed.rig.comfyui.png
    assert c.get("/archive").status_code == 200
    assert c.delete(f"/record/{rid}").json()["removed_at"] is not None
    assert c.get(f"/image/{rid}").status_code == 404
    assert closed.backend_calls() == NO_CALLS


@pytest.mark.parametrize("state", STATES)
def test_without_a_guard_generation_is_refused_before_a_stream_opens(closed, state) -> None:
    c = closed.start(state)
    before = sorted(p.name for p in closed.rig.out.iterdir())
    with c.stream("POST", "/generate/stream", json=BODY) as response:
        response.read()
        refusal(response)
        assert "event:" not in response.text
    refusal(c.post("/preflight", json=BODY))
    refusal(c.post(f"/depth/{closed.record_id}"))
    refusal(c.post("/purge", json={"confirm": "purge"}))
    assert sorted(p.name for p in closed.rig.out.iterdir()) == before  # no record, no image, nothing purged
    assert closed.backend_calls() == NO_CALLS


@pytest.mark.parametrize("state", STATES)
def test_without_a_guard_health_is_a_degraded_service_health_that_names_the_variables(closed, state) -> None:
    from test_server import OLD_HEALTH_FIELDS

    c = closed.start(state)
    response = closed.send(c, "GET", "/health")
    body = response.json()
    assert response.status_code == 200  # as for an unreachable SingGuard: the body is the answer, and nothing restarts
    health = ServiceHealth.model_validate(body)
    assert (health.service, health.status) == ("argus-darkroom", "degraded")
    assert set(health.checks) == {"singguard"}
    check = health.checks["singguard"]
    assert (check.ok, check.required) == (False, True)
    assert check.detail.startswith("SingGuard is not configured") and OWN in check.detail and SUITE in check.detail
    assert set(body) >= OLD_HEALTH_FIELDS  # safelight, mint and iris read them
    assert body["guard_configured"] is False
    assert OWN in body["guard_unconfigured"] and SUITE in body["guard_unconfigured"]
    assert not any(body["backends"].values())
    assert (body["sfw_available"], body["expansion_enabled"], body["generating"]) == (False, False, False)
    # The cheap probe a container healthcheck uses runs no checks, so the reason is in darkroom's own fields too.
    shallow = closed.send(c, "GET", "/health?checks=0")
    assert shallow.status_code == 200 and shallow.json()["guard_configured"] is False
    assert closed.backend_calls() == NO_CALLS


def test_a_guarded_server_says_so_on_health(rig) -> None:
    assert TestClient(create_app(rig.engine())).get("/health").json()["guard_configured"] is True


@pytest.mark.parametrize("state", STATES)
def test_the_reason_is_logged_once_at_startup_by_variable_name(closed, state) -> None:
    closed.start(state)
    (entry,) = [log for log in closed.logs if log["event"] == "guard.not_configured"]
    assert entry["log_level"] == "error"
    assert OWN in entry["detail"] and SUITE in entry["detail"]


@pytest.mark.parametrize("state", ["malformed", "malformed-suite-wide"])
def test_a_malformed_guard_url_is_in_no_response_and_no_log(closed, state, capsys) -> None:
    c = closed.start(state)
    for method, path in sorted(closed.routes()):
        closed.send(c, method, path)
    closed.send(c, "GET", "/health?checks=0")
    named = SUITE if state == "malformed-suite-wide" else OWN
    assert named in closed.send(c, "POST", "/preflight").json()["detail"]  # which variable is wrong, by name
    output = capsys.readouterr()
    seen = [*closed.bodies, repr(closed.logs), closed.caplog.text, output.out, output.err]
    assert len(closed.bodies) > 10 and closed.logs
    for text in seen:
        assert SECRET not in text and MALFORMED not in text and "guard.invalid" not in text


# --- by construction: nothing that generates can exist without a guard ------------------------------------


def test_the_unguarded_configuration_holds_no_backend_url() -> None:
    assert not [f.name for f in dataclasses.fields(Unguarded) if "url" in f.name]


def test_the_unguarded_app_has_no_engine(closed) -> None:
    assert closed.start("unset").app.state.engine is None


@pytest.mark.parametrize("state", STATES)
def test_no_settings_and_so_no_engine_can_be_built_without_a_guard(closed, state) -> None:
    closed.start(state)
    with pytest.raises(GuardNotConfigured, match="unguarded") as refused:
        Settings.from_env()  # what the CLI's generate and preflight read
    assert SECRET not in str(refused.value)
    assert isinstance(refused.value, ConfigError)
    with pytest.raises(GuardNotConfigured):
        replace(closed.rig.settings, singguard_url=" ")
    with pytest.raises(ValueError, match="unguarded"):
        Guard("")


def test_another_configuration_error_still_stops_the_server_starting(closed) -> None:
    """Only a missing guard is served closed. A second mistake is reported at startup, not after the first is fixed."""
    closed.monkeypatch.setenv("ARGUS_DARKROOM_THINKING", "made-up-mode")
    for state in ("unset", "malformed"):
        with pytest.raises(ConfigError, match="THINKING") as refused:
            closed.start(state)
        assert not isinstance(refused.value, GuardNotConfigured)


def test_an_unguarded_app_built_directly_is_as_closed(tmp_path) -> None:
    cfg = Unguarded(
        problem="made up",
        output_root=tmp_path,
        minor_exposed="frame",
        policy="default",
        thinking="cascade",
        allow_nsfw=False,
        devmode=True,
        read_only=False,
        cors_origins=(),
    )
    c = TestClient(unguarded_app(cfg))
    assert refusal(c.post("/generate/stream", json=BODY)) == cfg.reason
    assert c.get("/records").json()["total"] == 0
