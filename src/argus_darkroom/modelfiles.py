"""Find and hash the models a request names, under the ComfyUI models root (``ARGUS_DARKROOM_MODELS_ROOT``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from argus_darkroom.hashing import sha256_cached
from argus_darkroom.models import LoRARef, ModelRef
from argus_darkroom.request import RequestError

Kind = Literal["checkpoint", "lora", "vae"]
# ComfyUI's own layout under its ``models`` directory.
SUBDIRS: dict[Kind, str] = {"checkpoint": "checkpoints", "lora": "loras", "vae": "vae"}
SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".sft")


def pick_default(names: list[str], preferred: str) -> str | None:
    """The name *preferred* selects: itself, or the first in the folder it names. None when nothing matches."""
    if preferred in names:
        return preferred
    folder = preferred.strip("/") + "/"
    return next((name for name in names if preferred and name.startswith(folder)), None)


class ModelLibrary:
    def __init__(self, root: str | Path | None) -> None:
        self.root = Path(root) if root else None

    def _dir(self, kind: Kind) -> Path:
        if self.root is None:
            raise RequestError("ARGUS_DARKROOM_MODELS_ROOT is not set, so models cannot be found and hashed")
        return self.root / SUBDIRS[kind]

    def path(self, kind: Kind, name: str) -> Path:
        """The file for *name* (as ComfyUI loads it, a path relative to the kind's folder).

        Raises :class:`RequestError` when it is not there, or tries to leave the folder.
        """
        folder = self._dir(kind).resolve()
        path = (folder / name).resolve()
        if not name or folder not in path.parents or not path.is_file():
            raise RequestError(f"{kind} {name!r} was not found under the models root")
        return path

    def ref(self, kind: Kind, name: str) -> ModelRef:
        return ModelRef(name=name, sha256=sha256_cached(self.path(kind, name)))

    def lora_ref(self, name: str, weight: float, trigger_words: list[str] | None = None) -> LoRARef:
        return LoRARef(
            name=name,
            sha256=sha256_cached(self.path("lora", name)),
            weight=weight,
            trigger_words=list(trigger_words or []),
        )

    def names(self, kind: Kind) -> list[str]:
        """The model names found for *kind*, sorted. Empty when the folder is missing.

        A diffusers folder (one holding ``model_index.json``) is skipped: its files are the parts of a model, and
        none of them loads alone.
        """
        folder = self._dir(kind)
        if not folder.is_dir():
            return []
        files = [p for p in folder.rglob("*") if p.is_file()]
        diffusers = {p.parent for p in files if p.name == "model_index.json"}
        return sorted(
            p.relative_to(folder).as_posix() for p in files if p.suffix in SUFFIXES and diffusers.isdisjoint(p.parents)
        )
