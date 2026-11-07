"""What is kept of an image whether or not its pixels are: sha256, perceptual hash and size."""

from __future__ import annotations

import hashlib
import io

import imagehash
from PIL import Image

from argus_darkroom.models import ImageRecord


class UnreadableImage(ValueError):
    """The bytes are not an image Pillow can decode."""


class BlankImage(UnreadableImage):
    """Every pixel is the same colour: a broken render, usually NaNs cast to black. There is nothing to judge."""


def describe(data: bytes) -> ImageRecord:
    """The identity of *data*: lets a repeat of the same bad output be recognised without keeping it.

    Raises :class:`UnreadableImage`, or :class:`BlankImage` for a frame of one colour, which the guards would
    otherwise pass. Only an exactly uniform frame counts, so a dark scene is never failed for being dark.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            width, height = img.size
            blank = all(low == high for low, high in img.convert("RGB").getextrema())
            phash = str(imagehash.phash(img, hash_size=8))
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise UnreadableImage(f"the image cannot be decoded: {exc}") from exc
    if blank:
        raise BlankImage("the render came back blank: every pixel is the same colour")
    return ImageRecord(sha256=hashlib.sha256(data).hexdigest(), phash=phash, width=width, height=height)
