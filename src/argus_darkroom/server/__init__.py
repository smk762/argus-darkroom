"""The darkroom HTTP server (the ``[server]`` extra)."""

from argus_darkroom.server.app import create_app

__all__ = ["create_app"]
