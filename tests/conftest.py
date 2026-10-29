from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import FakeSingGuard  # noqa: E402


@pytest.fixture
def singguard() -> Iterator[FakeSingGuard]:
    with FakeSingGuard() as fake:
        yield fake
