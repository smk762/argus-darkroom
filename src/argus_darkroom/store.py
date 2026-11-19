"""The record store: one flat ``<id>.json`` per record, images beside them as ``<id>.png`` and an image's depth map,
when one was asked for, as ``<id>.depth.png``.

The same pattern as curator's ``ScanStore``. Ids are generated here and checked on the way in, never
joined into a path as given, so a crafted id cannot leave the root.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Literal

from argus_darkroom.models import GenerationRecord, Outcome

_ID = re.compile(r"^[0-9a-f]{32}$")
DEPTH_SUFFIX = ".depth.png"


def new_id() -> str:
    return uuid.uuid4().hex


def valid_id(record_id: str) -> bool:
    return bool(_ID.fullmatch(record_id))


def _fsync_dir(directory: Path) -> None:
    """Make a rename in *directory* durable. Best effort: not every platform can open a directory."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_atomic(path: Path, data: bytes) -> str:
    """Write *data* to *path* so a reader, or a crash, sees the old file or the whole new one, never part of one.

    The temp file is this write's own (``<id>.<random>.<suffix>.tmp``, so concurrent writers to one path cannot share
    it, and ``/purge`` and :meth:`RecordStore.delete` still find what a crash left). Its bytes reach the disk before
    the rename, and the rename before this returns.
    """
    record_id, _, rest = path.name.partition(".")
    tmp = path.with_name(f"{record_id}.{uuid.uuid4().hex}.{rest}.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)
    return path.name


class RecordStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def record_path(self, record_id: str) -> Path:
        if not valid_id(record_id):
            raise ValueError("not a record id")
        return self.root / f"{record_id}.json"

    def image_path(self, record_id: str) -> Path:
        if not valid_id(record_id):
            raise ValueError("not a record id")
        return self.root / f"{record_id}.png"

    def depth_path(self, record_id: str) -> Path:
        if not valid_id(record_id):
            raise ValueError("not a record id")
        return self.root / f"{record_id}{DEPTH_SUFFIX}"

    def save(self, record: GenerationRecord) -> Path:
        path = self.record_path(record.id)
        # Replace atomically so a reader never sees a half-written record.
        _write_atomic(path, record.model_dump_json().encode("utf-8"))
        return path

    def write_image(self, record_id: str, data: bytes) -> str:
        """Write a released image, atomically, and return its file name under the root.

        Only the release step calls this, after the verdict. A deleted or omitted image never gets here.
        """
        return _write_atomic(self.image_path(record_id), data)

    def write_depth(self, record_id: str, data: bytes) -> str:
        """Write a depth map of a stored image, atomically, and return its file name under the root."""
        return _write_atomic(self.depth_path(record_id), data)

    def remove_image(self, record_id: str) -> None:
        """Remove a stored image and the depth map derived from it (the depth map first: never one without its
        image)."""
        self.depth_path(record_id).unlink(missing_ok=True)
        self.image_path(record_id).unlink(missing_ok=True)

    def exists(self, record_id: str) -> bool:
        """Whether *record_id* has a record file, readable or not."""
        return valid_id(record_id) and self.record_path(record_id).is_file()

    def delete(self, record_id: str) -> int:
        """Remove a record, any image it kept (and its depth map) and what a crash left. The number of images removed,
        0 or 1.

        The image goes first: if a removal fails, the record stays to find it by, rather than an image with no record.
        """
        record, image = self.record_path(record_id), self.image_path(record_id)
        kept = image.is_file()
        self.remove_image(record_id)
        for tmp in self.root.glob(f"{record_id}.*.tmp"):  # an interrupted write's temp file, of any of its files
            tmp.unlink(missing_ok=True)
        record.unlink(missing_ok=True)
        return int(kept)

    def load(self, record_id: str) -> GenerationRecord | None:
        if not valid_id(record_id):
            return None
        try:
            return GenerationRecord.model_validate_json(self.record_path(record_id).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def page(
        self,
        outcome: Outcome | None = None,
        mode: Literal["sfw", "nsfw"] | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[GenerationRecord], int]:
        """One page of records, newest first, and the total that matched. Skips unreadable files."""
        records: list[GenerationRecord] = []
        for path in self.root.glob("*.json"):
            if not valid_id(path.stem):
                continue
            try:
                record = GenerationRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if (outcome is None or record.outcome == outcome) and (mode is None or record.request.mode == mode):
                records.append(record)
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[offset : offset + limit], len(records)
