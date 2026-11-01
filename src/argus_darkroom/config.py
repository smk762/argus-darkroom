"""Server and engine configuration from ``ARGUS_DARKROOM_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from argus_cortex.moderation import THINKING_MODES
from argus_cortex.moderation import policies as policy_table
from argus_cortex.server import env_flag

PREFIX = "ARGUS_DARKROOM_"
# Safelight's dev origin: read from it by default, and the only origin trusted to write unless others are named.
LOCAL_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")


class ConfigError(ValueError):
    """The configuration cannot be run: the server refuses to start."""


@dataclass(frozen=True)
class Settings:
    singguard_url: str
    comfyui_url: str = "http://localhost:8188"
    ollama_url: str = ""  # empty turns prompt expansion off
    ollama_model: str = ""
    nudenet_url: str = ""  # empty: SFW generation is unavailable
    age_url: str = ""  # curator's POST /faces?body=true; empty falls back to the in-process [age] extra
    policy: str = "default"
    thinking: str = "cascade"
    output_root: Path = Path("darkroom-output")
    models_root: Path | None = None
    workflow: Path | None = None
    allow_nsfw: bool = False
    devmode: bool = False
    # Origins allowed to read and, if named, to write. ``*`` allows anonymous reads from anywhere, never writes.
    cors_origins: tuple[str, ...] = LOCAL_ORIGINS

    def __post_init__(self) -> None:
        if not self.singguard_url.strip():
            raise ConfigError(f"{PREFIX}SINGGUARD_URL is empty: darkroom never runs unguarded")
        if self.thinking not in THINKING_MODES:
            raise ConfigError(f"{PREFIX}THINKING must be one of {', '.join(THINKING_MODES)}")
        try:
            policy_table.policy_text(self.policy)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if policy_table.mode(self.policy) == "nsfw" and not self.allow_nsfw:
            raise ConfigError(f"{PREFIX}POLICY {self.policy!r} is NSFW: start with {PREFIX}ALLOW_NSFW=1 to enforce it")

    @property
    def expansion_enabled(self) -> bool:
        return bool(self.ollama_url.strip() and self.ollama_model.strip())

    @property
    def sfw_available(self) -> bool:
        """SFW output is gated on NudeNet, so without it no SFW request can be served."""
        return bool(self.nudenet_url.strip())

    @classmethod
    def from_env(cls) -> Settings:
        """Read ``os.environ``. Flags go through cortex's ``env_flag``, which warns about a mistyped value."""

        def get(name: str, default: str = "") -> str:
            return os.environ.get(PREFIX + name, default).strip()

        allow_nsfw = env_flag(PREFIX + "ALLOW_NSFW")
        defaults = cls.__dataclass_fields__
        return cls(
            singguard_url=get("SINGGUARD_URL"),
            comfyui_url=get("COMFYUI_URL", defaults["comfyui_url"].default),
            ollama_url=get("OLLAMA_URL"),
            ollama_model=get("OLLAMA_MODEL"),
            nudenet_url=get("NUDENET_URL"),
            age_url=get("AGE_URL"),
            policy=get("POLICY") or ("adult-granular" if allow_nsfw else "default"),
            thinking=get("THINKING", "cascade"),
            output_root=Path(get("OUTPUT_ROOT") or "darkroom-output"),
            models_root=Path(p) if (p := get("MODELS_ROOT")) else None,
            workflow=Path(p) if (p := get("WORKFLOW")) else None,
            allow_nsfw=allow_nsfw,
            devmode=env_flag(PREFIX + "DEVMODE"),
            cors_origins=tuple(o.strip() for o in get("CORS_ORIGINS").split(",") if o.strip()) or LOCAL_ORIGINS,
        )
