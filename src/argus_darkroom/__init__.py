"""argus-darkroom — Guarded image generation for the Argus suite: prompts and outputs are checked before anything is released."""

from __future__ import annotations

try:
    # Written by hatch-vcs at build time (see pyproject [tool.hatch.build.hooks.vcs]).
    from argus_darkroom._version import __version__
except ImportError:  # running from a source checkout that hasn't been built
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("argus-darkroom")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
