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
from argus_darkroom.store import RecordStore, new_id


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
