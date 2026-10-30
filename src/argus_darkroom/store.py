"""The record store: one flat ``<id>.json`` per record, images beside them as ``<id>.png``.

The same pattern as curator's ``ScanStore``. Ids are generated here and checked on the way in, never
joined into a path as given, so a crafted id cannot leave the root.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Literal

from argus_darkroom.models import GenerationRecord, Outcome

_ID = re.compile(r"^[0-9a-f]{32}$")


def new_id() -> str:
    return uuid.uuid4().hex


def valid_id(record_id: str) -> bool:
    return bool(_ID.match(record_id))


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

    def save(self, record: GenerationRecord) -> Path:
        path = self.record_path(record.id)
        # Replace atomically so a reader never sees a half-written record.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(record.model_dump_json(), encoding="utf-8")
        tmp.replace(path)
        return path

    def write_image(self, record_id: str, data: bytes) -> str:
        """Write a released image, atomically, and return its file name under the root.

        Only the release step calls this, after the verdict. A deleted or omitted image never gets here.
        """
        path = self.image_path(record_id)
        tmp = path.with_suffix(".png.tmp")
        try:
            tmp.write_bytes(data)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        return path.name

    def remove_image(self, record_id: str) -> None:
        self.image_path(record_id).unlink(missing_ok=True)

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
