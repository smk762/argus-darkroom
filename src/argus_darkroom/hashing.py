"""SHA256 of model files: the identity a record pins.

Hashing a multi-GB checkpoint on every request is too slow, so results are cached by path, size and mtime for
the life of the process; an edited file is hashed again.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

_CHUNK = 1 << 20
_cache: dict[tuple[str, int, int], str] = {}
_lock = threading.Lock()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_cached(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    with _lock:
        cached = _cache.get(key)
    if cached is None:
        cached = sha256_file(resolved)
        with _lock:
            _cache[key] = cached
    return cached
