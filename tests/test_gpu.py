"""GPU leases (argus-iris, #53): what darkroom leases, at what priority, where the work goes, and what a refusal does."""

from __future__ import annotations

import contextlib
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from argus_cortex.gpu import GpuLease, GpuLeaseRefused, GpuLeaseTimeout, build_coordinator
from argus_cortex.gpu_wire import load_tenant_manifest
from fake_comfyui import FakeComfyUI, event
from fakes import FakeIris
from fastapi.testclient import TestClient
from rig import Rig
from test_server import BODY, generate, stream_events

from argus_darkroom import gpu
from argus_darkroom.comfyui import CLIENT_PREFIX, ComfyUIClient
from argus_darkroom.server.app import create_app

MANIFEST = Path(__file__).resolve().parent.parent / "gpu-tenant.yaml"


class Recording:
    """A coordinator that grants at once and keeps every lease's arguments.

    A grant names *base_url* (a URL, or a dict of URLs by target service).
    """

    def __init__(
        self, name: str = "none", base_url: str | dict[str, str] | None = None, raises: Exception | None = None
    ) -> None:
        self.name, self.base_url, self.raises = name, base_url, raises
        self.calls: list[dict] = []
        self.nested = False
        self._held = threading.local()

    @contextlib.contextmanager
    def lease(self, **kw):
        if getattr(self._held, "on", False):
            self.nested = True
        self.calls.append(kw)
        if self.raises is not None:
            raise self.raises
        self._held.on = True
        try:
            url = self.base_url
            if isinstance(url, dict):
                url = url.get(kw["target"].split(":", 1)[0])
            yield GpuLease(via="broker", base_url=url)
        finally:
            self._held.on = False

    def alease(self, **kw):  # pragma: no cover - darkroom leases from threads only
        raise NotImplementedError


def kinds(coordinator: Recording) -> list[tuple[str, str]]:
    return [(c["kind"], c["target"]) for c in coordinator.calls]


# --- the manifest and the figures ----------------------------------------------------------------------------


def test_the_manifest_validates_and_matches_the_lease_figures() -> None:
    manifest = load_tenant_manifest(MANIFEST)
    assert manifest.tenant == gpu.TENANT
    assert manifest.residents == ()
    for footprint, target in [
        (gpu.GENERATE, gpu.TARGET_COMFYUI),
        (gpu.DEPTH, gpu.TARGET_COMFYUI),
        (gpu.EXPAND, None),
    ]:
        kind = manifest.kind_for(footprint.kind)
        assert kind is not None, footprint.kind
        assert (kind.vram_mb, kind.model_mb) == (footprint.vram_mb, footprint.model_mb)
        assert kind.preemptible is False  # a render cannot stop part way
        if target:
            assert kind.target == target
    assert manifest.kind_for("expand").target.startswith("ollama:")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "interactive"),
        ("", "interactive"),
        ("batch", "batch"),
        (" Background ", "background"),
        ("interactive", "interactive"),
        ("realtime", "interactive"),  # darkroom's work is never realtime
    ],
)
def test_priority_from_the_caller(raw, expected) -> None:
    assert gpu.parse_priority(raw) == expected


def test_an_unknown_priority_is_refused() -> None:
    with pytest.raises(ValueError, match="X-Argus-Priority"):
        gpu.parse_priority("urgent")


# --- the engine ----------------------------------------------------------------------------------------------


def test_a_generation_leases_comfyui_at_the_callers_priority(rig) -> None:
    coordinator = Recording()
    engine = rig.engine(gpu_coordinator=coordinator)
    record = engine.generate(rig.request(), priority="batch")
    assert record.outcome == "passed"
    [call] = coordinator.calls
    assert call["kind"] == "generate" and call["target"] == "comfyui" and call["priority"] == "batch"
    assert (call["min_vram_mb"], call["model_mb"]) == (gpu.GENERATE.vram_mb, gpu.GENERATE.model_mb)


def test_a_person_is_interactive_by_default(rig) -> None:
    coordinator = Recording()
    rig.generate(rig.engine(gpu_coordinator=coordinator))
    assert coordinator.calls[0]["priority"] == "interactive"


def test_expansion_leases_its_ollama_model_first_and_leases_never_nest(rig, ollama) -> None:
    rig.settings = replace(rig.settings, ollama_url=ollama.url, ollama_model="tiny:1b")
    coordinator = Recording()
    record = rig.generate(rig.engine(gpu_coordinator=coordinator), expand=True)
    assert record.outcome == "passed"
    assert kinds(coordinator) == [("expand", "ollama:tiny:1b"), ("generate", "comfyui")]
    assert coordinator.calls[0]["min_vram_mb"] == gpu.EXPAND.vram_mb
    assert not coordinator.nested
    assert len(ollama.requests) == 1


def test_expansion_goes_to_the_ollama_the_lease_names(rig, ollama) -> None:
    rig.settings = replace(rig.settings, ollama_url="http://127.0.0.1:1", ollama_model="tiny:1b")
    record = rig.generate(rig.engine(gpu_coordinator=Recording(base_url={"ollama": ollama.url})), expand=True)
    assert record.outcome == "passed"
    assert len(ollama.requests) == 1


def test_a_refused_lease_is_a_failed_record_and_nothing_reaches_comfyui(rig) -> None:
    coordinator = Recording(raises=GpuLeaseRefused("GPU broker refused lease l1: it can never fit"))
    record = rig.generate(rig.engine(gpu_coordinator=coordinator))
    assert record.outcome == "failed"
    assert "refused" in record.outcome_reason and "never fit" in record.outcome_reason
    assert rig.comfyui.graphs == []
    assert list(rig.out.glob("*.png")) == []
    assert record.guard.preflight  # pre-flight ran before the lease was asked for
    assert len(coordinator.calls) == 1  # asked once: a refusal is a failed record, never retried like an OOM


def test_an_expansion_lease_that_times_out_is_a_failed_expansion(rig, ollama) -> None:
    rig.settings = replace(rig.settings, ollama_url=ollama.url, ollama_model="tiny:1b")
    coordinator = Recording(raises=GpuLeaseTimeout("not granted within 300s"))
    record = rig.generate(rig.engine(gpu_coordinator=coordinator), expand=True)
    assert record.outcome == "failed"
    assert record.outcome_reason.startswith("expansion failed: no GPU lease in time")
    assert ollama.requests == [] and rig.comfyui.graphs == []


def test_the_render_goes_to_the_comfyui_the_lease_names(rig) -> None:
    with FakeComfyUI() as other:
        coordinator = Recording(base_url=other.url + "/")
        engine = rig.engine(gpu_coordinator=coordinator)
        # The fakes serve the websocket on its own port; a real ComfyUI serves it at <base_url>/ws.
        engine._comfyui_at[other.url] = ComfyUIClient(other.url, ws_url=other.ws_url, timeout=5)
        record = rig.generate(engine)
        assert record.outcome == "passed"
        assert record.request.engine_version == other.version
        assert rig.comfyui.graphs == [] and len(other.graphs) == 1
        assert other.history == {}  # deleted from the ComfyUI that rendered it
        # The output-node check asked the ComfyUI that rendered, not the configured one.
        assert any(path.startswith("/object_info/") for _, path in other.calls)
        assert not any(path.startswith("/object_info/") for _, path in rig.comfyui.calls)
        assert [c.base_url for c in engine.comfyui_clients()] == [rig.comfyui.url, other.url]


def test_an_unusable_comfyui_url_from_the_broker_fails_the_run(rig) -> None:
    record = rig.generate(rig.engine(gpu_coordinator=Recording(base_url="ftp://elsewhere")))
    assert record.outcome == "failed" and "unusable ComfyUI URL" in record.outcome_reason
    assert rig.comfyui.graphs == []


def test_a_depth_map_leases_comfyui_as_depth(rig) -> None:
    coordinator = Recording()
    c = TestClient(create_app(rig.engine(gpu_coordinator=coordinator)))
    record = generate(c)
    r = c.post(f"/depth/{record['id']}", headers={"X-Argus-Priority": "batch"})
    assert r.status_code == 200
    assert coordinator.calls[-1]["kind"] == "depth" and coordinator.calls[-1]["priority"] == "batch"
    assert coordinator.calls[-1]["min_vram_mb"] == gpu.DEPTH.vram_mb


# --- the server ----------------------------------------------------------------------------------------------


def test_the_priority_header_reaches_the_lease(rig) -> None:
    coordinator = Recording()
    c = TestClient(create_app(rig.engine(gpu_coordinator=coordinator)))
    with c.stream("POST", "/generate/stream", json=BODY, headers={"X-Argus-Priority": "batch"}) as r:
        assert r.status_code == 200
        r.read()
    assert coordinator.calls[-1]["priority"] == "batch"


def test_an_unknown_priority_header_is_a_400_and_frees_the_gate(rig) -> None:
    coordinator = Recording()
    c = TestClient(create_app(rig.engine(gpu_coordinator=coordinator)))
    r = c.post("/generate/stream", json=BODY, headers={"X-Argus-Priority": "urgent"})
    assert r.status_code == 400 and "X-Argus-Priority" in r.json()["detail"]
    assert coordinator.calls == [] and not c.app.state.busy.locked()


@pytest.mark.parametrize("coordinator", [None, "lease"])
def test_without_the_broker_a_second_generation_is_still_a_409(tmp_path, singguard, nudenet, coordinator) -> None:
    """The default local coordinator, or a lease that does not queue: one generation at a time."""
    started, release = threading.Event(), threading.Event()

    def slow(fake, pid, graph):
        started.set()
        release.wait(10)
        return [
            event("executing", node="9", prompt_id=pid),
            b"\x00\x00\x00\x01\x00\x00\x00\x02" + fake.png,
            event("executing", node=None, prompt_id=pid),
        ]

    with FakeComfyUI(slow) as comfyui:
        rig = Rig(tmp_path, singguard, comfyui, nudenet)
        kw = {} if coordinator is None else {"gpu_coordinator": Recording(name=coordinator)}
        c = TestClient(create_app(rig.engine(**kw)))
        assert c.get("/health").json()["gpu"]["limit"] == 1
        results: list = []
        first = threading.Thread(target=lambda: results.append(stream_events(c)))
        first.start()
        try:
            assert started.wait(10)  # the first run holds the GPU, inside ComfyUI
            assert c.get("/health").json()["gpu"]["in_flight"] == 1
            r = c.post("/generate/stream", json=BODY)
            assert r.status_code == 409 and "already running" in r.json()["detail"]
        finally:
            release.set()
            first.join(10)
        assert results[0][-1][1]["record"]["outcome"] == "passed"
        assert c.get("/health").json()["gpu"]["in_flight"] == 0
        assert not c.app.state.busy.locked()


def test_with_the_broker_generations_queue_instead_of_a_409(tmp_path, singguard, nudenet) -> None:
    release = threading.Event()

    def slow(fake, pid, graph):
        release.wait(10)
        return [
            event("executing", node="9", prompt_id=pid),
            b"\x00\x00\x00\x01\x00\x00\x00\x02" + fake.png,
            event("executing", node=None, prompt_id=pid),
        ]

    with FakeComfyUI(slow) as comfyui:
        rig = Rig(tmp_path, singguard, comfyui, nudenet, devmode=True)
        c = TestClient(create_app(rig.engine(gpu_coordinator=Recording(name="broker"))))
        assert c.get("/health").json()["gpu"]["limit"] == gpu.MAX_IN_FLIGHT
        results: list = []
        threads = [threading.Thread(target=lambda: results.append(stream_events(c))) for _ in range(2)]
        for t in threads:
            t.start()
        for _ in range(200):
            if c.get("/health").json()["gpu"]["in_flight"] == 2:
                break
            threading.Event().wait(0.05)
        assert c.get("/health").json()["gpu"]["in_flight"] == 2
        # A purge still waits for nothing to be running.
        assert c.post("/purge", json={"confirm": "purge"}).status_code == 409
        release.set()
        for t in threads:
            t.join(10)
        assert [events[-1][1]["record"]["outcome"] for events in results] == ["passed", "passed"]
        assert not c.app.state.busy.locked()


def test_with_the_broker_the_gate_still_has_a_limit(rig) -> None:
    c = TestClient(create_app(rig.engine(gpu_coordinator=Recording(name="broker"))))
    gate = c.app.state.busy
    for _ in range(gate.limit):
        assert gate.enter()
    try:
        r = c.post("/generate/stream", json=BODY)
        assert r.status_code == 409 and "queued or running" in r.json()["detail"]
    finally:
        for _ in range(gate.limit):
            gate.leave()


def test_purge_clears_darkroom_prompts_on_every_comfyui_the_broker_used(tmp_path, singguard, nudenet) -> None:
    with FakeComfyUI() as comfyui, FakeComfyUI() as other:
        rig = Rig(tmp_path, singguard, comfyui, nudenet, devmode=True)
        engine = rig.engine(gpu_coordinator=Recording(base_url=other.url))
        engine._comfyui_at[other.url] = ComfyUIClient(other.url, ws_url=other.ws_url, timeout=5)
        c = TestClient(create_app(engine))
        assert generate(c)["outcome"] == "passed"
        other.history["left"] = {"prompt": [1, "left", {}, {"client_id": CLIENT_PREFIX + "x"}, []]}
        r = c.post("/purge", json={"confirm": "purge"})
        assert r.status_code == 200 and r.json()["comfyui_history"] == 1
        assert other.history == {}


# --- through cortex's broker client, against a fake iris -----------------------------------------------------


def broker(iris: FakeIris, fallback: str = "none"):
    return build_coordinator(
        "broker",
        broker_url=iris.url,
        broker_token="caller-token",
        manifest_path=str(MANIFEST),
        fallback=fallback,
        timeout_s=10,
    )


def test_through_the_broker_client(rig) -> None:
    with FakeIris() as iris:
        iris.queue_polls = 1  # queued once, then granted: darkroom waits instead of answering 409
        record = rig.engine(gpu_coordinator=broker(iris)).generate(rig.request(), priority="batch")
        assert record.outcome == "passed"
        [tenant] = iris.tenants
        assert tenant["tenant"] == "argus-darkroom"
        [lease] = iris.leases
        assert lease["tenant"] == "argus-darkroom"
        assert (lease["kind"], lease["target"], lease["priority"]) == ("generate", "comfyui", "batch")
        assert (lease["vram_mb"], lease["model_mb"]) == (gpu.GENERATE.vram_mb, gpu.GENERATE.model_mb)
        assert [body["outcome"] for _, body in iris.releases] == ["ok"]
        assert set(iris.auth) == {"Bearer caller-token"}


def test_an_iris_without_lease_routes_falls_back(rig) -> None:
    with FakeIris() as iris:
        iris.no_leases = True  # iris is observe-only today
        record = rig.engine(gpu_coordinator=broker(iris)).generate(rig.request())
        assert record.outcome == "passed"
        assert iris.releases == []


def test_a_refusal_from_iris_fails_the_run(rig) -> None:
    with FakeIris() as iris:
        iris.refuse = True
        record = rig.engine(gpu_coordinator=broker(iris)).generate(rig.request())
        assert record.outcome == "failed" and "refused" in record.outcome_reason
        assert rig.comfyui.graphs == []
