from __future__ import annotations

import argus_darkroom


def test_version_is_exposed() -> None:
    assert isinstance(argus_darkroom.__version__, str)
    assert argus_darkroom.__version__
