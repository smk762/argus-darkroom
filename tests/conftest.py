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
    monkeypatch.delenv("ARGUS_GPU_COORDINATOR", raising=False)


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
