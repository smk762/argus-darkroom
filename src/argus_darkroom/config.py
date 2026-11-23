"""Server and engine configuration from ``ARGUS_DARKROOM_*`` environment variables.

The URL of another service (SingGuard, NudeNet, ComfyUI, Ollama, curator) is found with ``argus_cortex.peers``:
darkroom's own variable first, then the suite-wide ``ARGUS_<PEER>_URL``, each from the environment and then the suite
file. A URL that is set but malformed is a :class:`ConfigError`, never "not configured": a typo must not turn a
check off.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from argus_cortex.moderation import THINKING_MODES
from argus_cortex.moderation import policies as policy_table
from argus_cortex.peers import PeerRefused, resolve_peer
from argus_cortex.server import env_flag

from argus_darkroom.models import MinorExposedRule

PREFIX = "ARGUS_DARKROOM_"
# Safelight's origins, published (3000) and `make dev` (3097): read from them by default, and the only ones trusted to
# write unless others are named.
LOCAL_ORIGINS = tuple(f"http://{host}:{port}" for port in (3000, 3097) for host in ("localhost", "127.0.0.1"))


# How the minor-exposed rule places exposure (ARGUS_DARKROOM_MINOR_EXPOSED); the first is the default.
MINOR_EXPOSED_RULES: tuple[MinorExposedRule, ...] = ("frame", "overlap")


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
    # The suite-wide curator (ARGUS_CURATOR_URL), used for the age check only when age_url is empty and the in-process
    # extra cannot run: it can turn the check on, never replace an estimator that works.
    age_fallback_url: str = ""
    # frame: any age estimate under 18 plus exposure anywhere; overlap: exposure placed on a young person's box
    minor_exposed: MinorExposedRule = "frame"
    policy: str = "default"
    thinking: str = "cascade"
    output_root: Path = Path("darkroom-output")
    models_root: Path | None = None
    # The checkpoint a form starts on: an exact name, or a folder under checkpoints/ for the first model in it.
    default_checkpoint: str = "sdxl"
    # A template file used for every checkpoint, or a directory of them chosen per request (workflow.load_workflows).
    workflow: Path | None = None
    allow_nsfw: bool = False
    devmode: bool = False
    read_only: bool = False  # refuse every write; /health reports it
    # The demo server (demo/run.py, fake backends). Set only there, never from the environment, so a client can tell
    # the demo from a server that renders on a GPU: a test that generates checks /health before it sends anything.
    demo: bool = False
    # Origins allowed to read and, if named, to write. ``*`` allows anonymous reads from anywhere, never writes.
    cors_origins: tuple[str, ...] = LOCAL_ORIGINS

    def __post_init__(self) -> None:
        if not self.singguard_url.strip():
            raise ConfigError(f"{PREFIX}SINGGUARD_URL is empty: darkroom never runs unguarded")
        if self.minor_exposed not in MINOR_EXPOSED_RULES:
            raise ConfigError(
                f"{PREFIX}MINOR_EXPOSED is {self.minor_exposed!r}: it must be one of {', '.join(MINOR_EXPOSED_RULES)}"
            )
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
        """Read ``os.environ``. Flags go through cortex's ``env_flag``, which warns about a mistyped value.

        Peer URLs go through ``argus_cortex.peers``: darkroom's variable, then ``ARGUS_<PEER>_URL``. A blank one
        falls through, ``off`` stops the lookup (the peer is not configured), and a malformed one is a
        :class:`ConfigError` that names the variable and never the value.
        """

        def get(name: str) -> str:
            """The value, or "" when unset or blank (``VAR=${VAR:-}`` in compose): a blank never beats a default."""
            return os.environ.get(PREFIX + name, "").strip()

        def peer(name: str, var: str, default: str | None = None) -> tuple[str, bool]:
            """The peer's URL ("" when not configured), and whether darkroom's own variable named it."""
            own = PREFIX + var
            try:
                found = resolve_peer(name, own, default=default)
            except PeerRefused as exc:
                raise ConfigError(str(exc)) from None  # names the variable only: a URL can carry a credential
            return (found.url, found.source.endswith(":" + own)) if found else ("", False)

        allow_nsfw = env_flag(PREFIX + "ALLOW_NSFW")
        defaults = cls.__dataclass_fields__
        # No default here: `off` and "nothing set" both leave it empty, which __post_init__ refuses.
        singguard_url, _ = peer("singguard", "SINGGUARD_URL")
        comfyui_url, _ = peer("comfyui", "COMFYUI_URL", defaults["comfyui_url"].default)
        if not comfyui_url:
            raise ConfigError(f"{PREFIX}COMFYUI_URL is off: darkroom renders through ComfyUI")
        curator_url, own_age_url = peer("curator", "AGE_URL")
        return cls(
            singguard_url=singguard_url,
            comfyui_url=comfyui_url,
            ollama_url=peer("ollama", "OLLAMA_URL")[0],
            ollama_model=get("OLLAMA_MODEL"),
            nudenet_url=peer("nudenet", "NUDENET_URL")[0],
            age_url=curator_url if own_age_url else "",
            age_fallback_url="" if own_age_url else curator_url,
            minor_exposed=get("MINOR_EXPOSED") or defaults["minor_exposed"].default,
            policy=get("POLICY") or ("adult-granular" if allow_nsfw else "default"),
            thinking=get("THINKING") or defaults["thinking"].default,
            output_root=Path(get("OUTPUT_ROOT") or "darkroom-output"),
            models_root=Path(p) if (p := get("MODELS_ROOT")) else None,
            default_checkpoint=get("DEFAULT_CHECKPOINT") or defaults["default_checkpoint"].default,
            workflow=Path(p) if (p := get("WORKFLOW")) else None,
            allow_nsfw=allow_nsfw,
            devmode=env_flag(PREFIX + "DEVMODE"),
            read_only=env_flag(PREFIX + "READ_ONLY"),
            cors_origins=tuple(o.strip() for o in get("CORS_ORIGINS").split(",") if o.strip()) or LOCAL_ORIGINS,
        )
