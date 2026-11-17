from __future__ import annotations

import pytest
from argus_cortex.wire import VersionError

from argus_darkroom.models import (
    Cell,
    GenerationRecord,
    GuardRecord,
    RequestRecord,
    render_wire_schema,
)
from argus_darkroom.store import RecordStore, new_id, valid_id


def record(outcome="refused", **kw) -> GenerationRecord:
    return GenerationRecord(
        id=kw.pop("id", new_id()),
        darkroom_version="0",
        outcome=outcome,
        request=RequestRecord(mode="sfw", user_prompt="a", final_prompt="a"),
        guard=GuardRecord(enforcing_policy="default", thinking="cascade", preflight={"default": Cell(state="safe")}),
        **kw,
    )


def test_roundtrip_and_paging(tmp_path) -> None:
    store = RecordStore(tmp_path)
    saved = [record("refused"), record("deleted"), record("refused")]
    for r in saved:
        store.save(r)
    page, total = store.page()
    assert total == 3
    assert {r.id for r in page} == {r.id for r in saved}
    assert [r.created_at for r in page] == sorted((r.created_at for r in page), reverse=True)
    only, total = store.page(outcome="deleted")
    assert total == 1 and only[0].outcome == "deleted"
    assert store.page(offset=2, limit=5)[0] == page[2:]


def test_a_crafted_id_cannot_leave_the_root(tmp_path) -> None:
    store = RecordStore(tmp_path / "root")
    (tmp_path / "secret.json").write_text("{}")
    assert store.load("../secret") is None
    with pytest.raises(ValueError):
        store.record_path("../secret")
    with pytest.raises(ValueError):
        store.image_path("a/b")


def test_an_id_with_a_trailing_newline_is_refused(tmp_path) -> None:
    # `$` also matches before a final newline; the id check matches the whole string.
    record_id = new_id()
    assert valid_id(record_id)
    assert not valid_id(record_id + "\n")
    store = RecordStore(tmp_path / "root")
    with pytest.raises(ValueError):
        store.record_path(record_id + "\n")


def test_unreadable_files_are_skipped(tmp_path) -> None:
    store = RecordStore(tmp_path)
    store.save(record())
    (tmp_path / f"{new_id()}.json").write_text("not json")
    assert store.page()[1] == 1


def test_an_incompatible_major_is_refused() -> None:
    data = record().model_dump(mode="json") | {"record_version": "2.0"}
    with pytest.raises(VersionError):
        GenerationRecord.model_validate(data)


def test_no_pixels_in_the_schema_of_a_deleted_record() -> None:
    """A deleted record holds hashes only: no field in the contract can carry the image again."""
    schema = render_wire_schema()
    for banned in ("thumbnail", "preview", "base64", "data_url"):
        assert banned not in schema
    assert '"sha256"' in schema and '"phash"' in schema


def test_a_record_from_before_the_overlap_rule_still_loads() -> None:
    """``people``, ``minor_exposed_rule`` and NudeNet ``detections`` are new: an older record has none of them."""
    data = record("deleted").model_dump(mode="json")
    data["guard"]["age"] = {
        "source": "mivolo",
        "faces": [{"age_estimate_years": 16.0, "bbox": [0, 0, 10, 10]}],
        "minor_exposed": True,
    }
    data["guard"]["nudenet"] = {
        "exposed_score": 0.9,
        "labels": {"FEMALE_BREAST_EXPOSED": 0.9},
        "threshold": 0.2,
        "blur": True,
    }
    loaded = GenerationRecord.model_validate(data)
    assert loaded.guard.nudenet is not None and loaded.guard.nudenet.detections == []
    assert loaded.guard.age is not None
    assert loaded.guard.age.people == [] and loaded.guard.age.minor_exposed_rule == "frame"
    assert loaded.guard.age.minor_exposed


def test_a_write_uses_its_own_temp_file(tmp_path) -> None:
    # Another writer's temp file for the same image, mid-write: this write must neither publish nor remove it.
    store = RecordStore(tmp_path)
    record_id = new_id()
    theirs = store.image_path(record_id).with_suffix(".png.tmp")
    theirs.write_bytes(b"half of another write")
    store.write_image(record_id, b"mine")
    assert store.image_path(record_id).read_bytes() == b"mine"
    assert theirs.read_bytes() == b"half of another write"
    store.delete(record_id)  # and what a crash left is still found by the record's id
    assert list(tmp_path.iterdir()) == []


def test_concurrent_writes_to_one_image_each_publish_whole(tmp_path) -> None:
    import threading

    store = RecordStore(tmp_path)
    record_id = new_id()
    payloads = [bytes([i]) * 200_000 for i in range(8)]
    errors: list[BaseException] = []

    def write(data: bytes) -> None:
        try:
            store.write_image(record_id, data)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(p,)) for p in payloads]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert errors == []
    assert store.image_path(record_id).read_bytes() in payloads
    assert [p.name for p in tmp_path.iterdir()] == [f"{record_id}.png"]


def test_a_write_reaches_the_disk_before_its_rename(tmp_path, monkeypatch) -> None:
    import os

    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: events.append("fsync") or real_fsync(fd))
    monkeypatch.setattr(os, "replace", lambda a, b: events.append("replace") or real_replace(a, b))
    store = RecordStore(tmp_path)
    store.write_image(new_id(), b"png")
    # The file's bytes, then the rename, then the directory entry.
    assert events == ["fsync", "replace", "fsync"]
