"""What is kept of an image whether or not its pixels are: sha256, perceptual hash and size."""

from __future__ import annotations

import hashlib
import io

import imagehash
from PIL import Image

from argus_darkroom.models import ImageRecord


class UnreadableImage(ValueError):
    """The bytes are not an image Pillow can decode."""


def describe(data: bytes) -> ImageRecord:
    """The identity of *data*: lets a repeat of the same bad output be recognised without keeping it."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            width, height = img.size
            phash = str(imagehash.phash(img, hash_size=8))
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise UnreadableImage(f"the image cannot be decoded: {exc}") from exc
    return ImageRecord(sha256=hashlib.sha256(data).hexdigest(), phash=phash, width=width, height=height)
