from __future__ import annotations

import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fake_comfyui import FakeComfyUI  # noqa: E402
from fakes import FakeNudeNet, FakeOllama, FakeSingGuard  # noqa: E402
from rig import Rig  # noqa: E402


@pytest.fixture(autouse=True)
def no_gpu_broker(monkeypatch) -> None:
    """A host's ARGUS_GPU_* must not reach the tests: an engine built without a coordinator gates nothing."""
    for name in ("ARGUS_GPU_COORDINATOR", "ARGUS_GPU_BROKER_URL", "ARGUS_GPU_TENANT_MANIFEST", "ARGUS_GPU_TENANT"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_suite_settings(monkeypatch) -> None:
    """A developer's suite file and suite-wide peer URLs must not reach the tests: a test names every URL it uses."""
    monkeypatch.setenv("ARGUS_ENV_FILE", "off")
    for peer in ("SINGGUARD", "NUDENET", "COMFYUI", "OLLAMA", "CURATOR", "PUPPET"):
        monkeypatch.delenv(f"ARGUS_{peer}_URL", raising=False)


@pytest.fixture
def singguard() -> Iterator[FakeSingGuard]:
    with FakeSingGuard() as fake:
        yield fake


@pytest.fixture
def comfyui() -> Iterator[FakeComfyUI]:
    with FakeComfyUI() as fake:
        yield fake


@pytest.fixture
def nudenet() -> Iterator[FakeNudeNet]:
    with FakeNudeNet() as fake:
        yield fake


@pytest.fixture
def rig(tmp_path, singguard, comfyui, nudenet, monkeypatch) -> Rig:
    r = Rig(tmp_path, singguard, comfyui, nudenet)
    monkeypatch.setattr(tempfile, "tempdir", str(r.systmp))
    return r


@pytest.fixture
def ollama() -> Iterator[FakeOllama]:
    with FakeOllama() as fake:
        yield fake
