"""GPU leases from the suite's broker (argus-iris), through argus-cortex's coordinator.

Darkroom runs no model on a GPU itself: ComfyUI renders and Ollama expands. Each of those calls holds a lease, so the
broker can queue it by priority, place it on a node and make room first:

* one per render on ``comfyui`` (kind ``generate``), and one per depth map (kind ``depth``);
* one per prompt expansion on ``ollama:<model>`` (kind ``expand``).

A grant's ``base_url`` is where to send the work; ``None`` means darkroom's own configured URL. Leases are never
nested, so a local ``flock`` fallback cannot deadlock against itself.

The coordinator comes from the host-wide ``ARGUS_GPU_*`` environment (:func:`argus_cortex.gpu.coordinator_from_env`):
``ARGUS_GPU_BROKER_URL`` (and ``ARGUS_GPU_BROKER_TOKEN``) select the broker, whose tenant and kind defaults come from
``gpu-tenant.yaml`` (found in the working directory or ``/app``); unset is ``none``, no gating. The broker falls back
to ``ARGUS_GPU_BROKER_FALLBACK`` (``lease`` by default) when iris cannot be reached. The server registers the manifest
at startup (:func:`argus_cortex.gpu.aregistered` in its lifespan).

The figures below are the ones declared in ``gpu-tenant.yaml``; ``tests/test_gpu.py`` keeps the two in step. The broker
uses its learned p95 per kind when that is larger.

**A lease failure is not an OOM.** :class:`~argus_cortex.gpu.GpuLeaseRefused`,
:class:`~argus_cortex.gpu.GpuLeaseTimeout` and :class:`~argus_cortex.gpu_broker.BrokerError` become a ``failed``
record with the broker's reason. Nothing retries them.
"""

from __future__ import annotations

from dataclasses import dataclass

from argus_cortex.gpu import GpuCoordinator, GpuLeaseRefused, GpuLeaseTimeout, coordinator_from_env
from argus_cortex.gpu_broker import BrokerError
from argus_cortex.gpu_wire import PRIORITY_CLASSES, PriorityClass

TENANT = "argus-darkroom"
TARGET_COMFYUI = "comfyui"
PRIORITY_HEADER = "X-Argus-Priority"
DEFAULT_PRIORITY: PriorityClass = "interactive"
# Darkroom's work is never realtime: a caller asking for it gets interactive, the highest class darkroom leases at.
_CLAMP: dict[str, PriorityClass] = {"realtime": "interactive"}
# Generations allowed in flight at once when the broker queues them. Past this a new one is still a 409.
MAX_IN_FLIGHT = 8

LEASE_ERRORS: tuple[type[Exception], ...] = (GpuLeaseRefused, GpuLeaseTimeout, BrokerError)


@dataclass(frozen=True)
class Footprint:
    kind: str
    vram_mb: int  # the whole footprint: weights plus working set
    model_mb: int  # the weights the service keeps loaded after the lease


GENERATE = Footprint("generate", vram_mb=9000, model_mb=6500)  # an SDXL checkpoint in ComfyUI, and its latents
DEPTH = Footprint("depth", vram_mb=1500, model_mb=500)  # Depth Anything V2 Small
EXPAND = Footprint("expand", vram_mb=6000, model_mb=5200)  # an 8B model in Ollama, and its KV cache


def ollama_target(model: str) -> str:
    return f"ollama:{model.strip()}"


def parse_priority(raw: str | None) -> PriorityClass:
    """The priority class a caller asked for (``X-Argus-Priority``); :data:`DEFAULT_PRIORITY` when it asked for none.

    Raises :class:`ValueError` for a name that is not a class.
    """
    value = (raw or "").strip().lower()
    if not value:
        return DEFAULT_PRIORITY
    if value not in PRIORITY_CLASSES:
        raise ValueError(f"{PRIORITY_HEADER} must be one of {', '.join(PRIORITY_CLASSES)}, not {raw!r}")
    return _CLAMP.get(value, value)  # type: ignore[return-value]


def default_coordinator() -> GpuCoordinator:
    return coordinator_from_env()


def queues(coordinator: GpuCoordinator) -> bool:
    """Whether *coordinator* queues work itself, so the server may hold several generations waiting on it."""
    return coordinator.name == "broker"


def lease_failure(exc: Exception) -> str:
    """A record's reason for a lease that was not granted."""
    if isinstance(exc, GpuLeaseRefused):
        return f"the GPU broker refused the lease: {exc}"
    if isinstance(exc, GpuLeaseTimeout):
        return f"no GPU lease in time: {exc}"
    return f"the GPU broker could not be used: {exc}"
