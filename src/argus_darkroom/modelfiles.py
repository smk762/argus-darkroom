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

    def lora_ref(self, name: str, weight: float) -> LoRARef:
        return LoRARef(name=name, sha256=sha256_cached(self.path("lora", name)), weight=weight)

    def names(self, kind: Kind) -> list[str]:
        """The model names found for *kind*, sorted. Empty when the folder is missing."""
        folder = self._dir(kind)
        if not folder.is_dir():
            return []
        return sorted(
            p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p.suffix in SUFFIXES
        )
