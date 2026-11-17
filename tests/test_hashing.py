"""The model-file hash cache: a hit only for the same bytes."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from argus_darkroom import hashing
from argus_darkroom.hashing import sha256_cached


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rewrite_keeping_mtime(path: Path, data: bytes) -> None:
    """New bytes of the same length, with the old mtime put back, as ``touch -d`` would."""
    before = path.stat()
    time.sleep(0.02)  # past the kernel's coarse timestamp tick, so ctime moves
    path.write_bytes(data)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_same_length_bytes_with_the_mtime_put_back_are_hashed_again(tmp_path) -> None:
    model = tmp_path / "m.safetensors"
    model.write_bytes(b"A" * 64)
    assert sha256_cached(model) == sha(b"A" * 64)
    rewrite_keeping_mtime(model, b"B" * 64)
    assert sha256_cached(model) == sha(b"B" * 64)


def test_a_file_that_changes_while_it_is_hashed_is_not_cached(tmp_path, monkeypatch) -> None:
    model = tmp_path / "m.safetensors"
    model.write_bytes(b"A" * 64)
    real = hashing.sha256_file
    calls = []

    def racing(path: Path) -> str:
        calls.append(path)
        if len(calls) == 1:  # the file is replaced between the stat and the read
            rewrite_keeping_mtime(path, b"B" * 64)
        return real(path)

    monkeypatch.setattr(hashing, "sha256_file", racing)
    sha256_cached(model)
    # Its digest is of bytes the first stat never saw, so it was not stored: the next call reads the file again.
    assert sha256_cached(model) == sha(b"B" * 64)
    assert len(calls) == 2
    # And the old bytes back, mtime and all, are not served the new bytes' digest.
    rewrite_keeping_mtime(model, b"A" * 64)
    assert sha256_cached(model) == sha(b"A" * 64)


def test_an_unchanged_file_is_hashed_once(tmp_path, monkeypatch) -> None:
    model = tmp_path / "m.safetensors"
    model.write_bytes(b"A" * 64)
    real = hashing.sha256_file
    calls = []
    monkeypatch.setattr(hashing, "sha256_file", lambda p: calls.append(p) or real(p))
    assert sha256_cached(model) == sha256_cached(model) == sha(b"A" * 64)
    assert len(calls) == 1
