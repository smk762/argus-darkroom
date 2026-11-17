"""SHA256 of model files: the identity a record pins.

Hashing a multi-GB checkpoint on every request is too slow, so results are cached for the life of the process,
keyed by path and what ``stat`` says of the file: device, inode, size, mtime and ctime. Writing the file or
setting its mtime back (``os.utime``) both move ctime, which user space cannot set, so new bytes are hashed again.
A file that changed while it was read is not cached. One entry is kept per path.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

_CHUNK = 1 << 20
_Key = tuple[int, int, int, int, int]
_cache: dict[str, tuple[_Key, str]] = {}
_lock = threading.Lock()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _key(stat: os.stat_result) -> _Key:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def sha256_cached(path: Path) -> str:
    resolved = path.resolve()
    key = _key(resolved.stat())
    with _lock:
        entry = _cache.get(str(resolved))
    if entry is not None and entry[0] == key:
        return entry[1]
    digest = sha256_file(resolved)
    if _key(resolved.stat()) == key:  # unchanged while it was read, so the digest is this key's
        with _lock:
            _cache[str(resolved)] = (key, digest)
    return digest
