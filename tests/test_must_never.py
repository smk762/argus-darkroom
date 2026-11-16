"""Checks for must-nevers that had none (#58). Each is a row in CLAUDE.md's "Must never happen" table."""

from __future__ import annotations

import base64
import binascii
import json
import re
import subprocess
import tempfile
from pathlib import Path

from rig import EXAMPLES, Rig
from test_img2img import FakeAge, init

from argus_darkroom.models import RECORD_VERSION, GenerationRecord

ROOT = Path(__file__).resolve().parent.parent
FROZEN_FIELDS = Path(__file__).resolve().parent / "data" / f"record-{RECORD_VERSION.split('.')[0]}.0-fields.txt"


# --- the age is never sent to SingGuard --------------------------------------------------------------------------

AGE = 37.25  # an estimate no frame label, prompt or score in the run could produce by chance
# What darkroom sends SingGuard (cortex's SingGuardClient): the text, its role, the image, the policy text, the mode.
GUARD_KEYS = {"input", "input_role", "image_base64", "policy", "thinking"}


def test_the_age_estimate_never_reaches_singguard(tmp_path, singguard, comfyui, nudenet, monkeypatch) -> None:
    """Ages are read for the input and the output, and none of it is in any SingGuard request: no extra field, and
    nothing in the text or the policy."""
    rig = Rig(tmp_path, singguard, comfyui, nudenet, workflow=EXAMPLES / "sdxl+lora1+img2img.json")
    monkeypatch.setattr(tempfile, "tempdir", str(rig.systmp))
    record = rig.generate(rig.engine(age=FakeAge(AGE)), init_image=init(), advisory_policies=["adult-granular"])

    assert record.outcome == "passed"
    assert [f.age_estimate_years for f in record.guard.input.age.faces] == [AGE]  # the input's age was read
    assert [f.age_estimate_years for f in record.guard.age.faces] == [AGE]  # and the output's
    assert len(singguard.requests) >= 8  # pre-flight, then input and output, isolated and with the prompt
    for body in singguard.requests:
        assert set(body) <= GUARD_KEYS, set(body) - GUARD_KEYS
        text = json.dumps({k: v for k, v in body.items() if k != "image_base64"}).lower()
        for leak in (str(AGE), "37", "age_estimate", "mivolo"):
            assert leak not in text, (leak, body.get("input_role"))


# --- a record field is never renamed or removed within a major ---------------------------------------------------


def field_paths(schema: dict) -> set[str]:
    """Every field of *schema* as a dotted path (``guard.nudenet.blur``; ``[]`` for list items, ``{}`` for map
    values), following ``$ref``s."""
    defs = schema.get("$defs", {})
    out: set[str] = set()

    def walk(node: dict, prefix: str, seen: frozenset[str]) -> None:
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            if name not in seen:
                walk(defs[name], prefix, seen | {name})
        for key in ("anyOf", "oneOf", "allOf"):
            for sub in node.get(key, []):
                walk(sub, prefix, seen)
        if "items" in node:
            walk(node["items"], prefix + "[]", seen)
        if isinstance(node.get("additionalProperties"), dict):
            walk(node["additionalProperties"], prefix + "{}", seen)
        for name, sub in node.get("properties", {}).items():
            path = f"{prefix}.{name}" if prefix else name
            out.add(path)
            walk(sub, path, seen)

    walk(schema, "", frozenset())
    return out


def test_record_fields_only_grow_within_a_major() -> None:
    """The fields of ``record_version`` 1.x as first frozen, all still there. Adding a field needs nothing here;
    renaming or removing one fails, and is a new major (``RECORD_VERSION``) with its own frozen list."""
    assert FROZEN_FIELDS.is_file(), f"freeze the fields of record_version {RECORD_VERSION} in {FROZEN_FIELDS.name}"
    frozen = set(FROZEN_FIELDS.read_text(encoding="utf-8").split())
    missing = frozen - field_paths(GenerationRecord.model_json_schema())
    assert not missing, f"renamed or removed within record_version {RECORD_VERSION}: {sorted(missing)}"


# --- no imagery enters the repo ----------------------------------------------------------------------------------

MEDIA = re.compile(
    r"\.(png|jpe?g|webp|gif|bmp|tiff?|avif|heic|ico|mp4|webm|mov|safetensors|ckpt|pt|pth|onnx|npy|npz)$", re.I
)
BASE64_RUN = re.compile(rb"[A-Za-z0-9+/]{120,}={0,2}")
IMAGE_MAGIC = (b"\x89PNG", b"\xff\xd8\xff", b"RIFF", b"GIF8", b"BM", b"II*\x00", b"MM\x00*")


def tracked() -> list[str]:
    return subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()


def test_no_image_video_or_model_file_is_tracked() -> None:
    """Tests draw their images in code (PIL), and the illegal path is driven by a fake verdict, so no image file has
    a reason to be in the repo, and one that is cannot be reviewed in a diff."""
    assert [p for p in tracked() if MEDIA.search(p)] == []


def test_no_image_is_hidden_in_a_base64_literal() -> None:
    """Nor is an image pasted into source as base64."""
    found = []
    for path in tracked():
        if not path.endswith((".py", ".json", ".md", ".toml", ".yaml", ".yml", ".txt", ".html", ".js", ".ts")):
            continue
        for run in BASE64_RUN.findall((ROOT / path).read_bytes()):
            try:
                head = base64.b64decode(run[:64] + b"=" * (-len(run[:64]) % 4), validate=True)
            except (binascii.Error, ValueError):
                continue
            if head.startswith(IMAGE_MAGIC):
                found.append(path)
    assert found == []


# --- policy texts live in cortex only ----------------------------------------------------------------------------


def test_no_policy_text_is_copied_out_of_cortex() -> None:
    """Darkroom names policies; their texts, the illegal table and ``mode()`` are cortex's. No line of any policy text
    appears in darkroom's source, so a copy cannot drift from what SingGuard is sent."""
    from argus_cortex.moderation import policies as policy_table

    source = "".join(p.read_text(encoding="utf-8") for p in (ROOT / "src").rglob("*.py"))
    copied = [
        (name, line)
        for name in policy_table.POLICIES
        for line in (s.strip() for s in (policy_table.policy_text(name) or "").splitlines())
        if len(line) > 40 and line in source
    ]
    assert copied == []
