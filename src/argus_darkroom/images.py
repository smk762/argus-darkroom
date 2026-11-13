"""What is kept of an image whether or not its pixels are: sha256, perceptual hash and size.

Also the image-to-image input: decoded from the request's base64 once, into the one PNG that is both judged and sent
to ComfyUI, so what was checked is exactly what is rendered from.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
from dataclasses import dataclass
from functools import cached_property

import imagehash
from PIL import Image, ImageOps

from argus_darkroom.models import ImageRecord

MAX_INPUT_BYTES = 16 * 1024 * 1024  # decoded
MAX_INPUT_EDGE = 4096  # pixels, either side
INPUT_FORMATS = ("PNG", "JPEG", "WEBP")
# The longest base64 that can decode to MAX_INPUT_BYTES: anything longer is refused before it is decoded.
_MAX_INPUT_BASE64 = 4 * -(-MAX_INPUT_BYTES // 3)


class UnreadableImage(ValueError):
    """The bytes are not an image Pillow can decode."""


class BlankImage(UnreadableImage):
    """Every pixel is the same colour: a broken render, usually NaNs cast to black. There is nothing to judge."""


def describe(data: bytes, *, allow_blank: bool = False) -> ImageRecord:
    """The identity of *data*: lets a repeat of the same bad output be recognised without keeping it.

    Raises :class:`UnreadableImage`, or :class:`BlankImage` for a frame of one colour, which the guards would
    otherwise pass. Only an exactly uniform frame counts, so a dark scene is never failed for being dark. An
    image-to-image input may be one colour (*allow_blank*): it is judged like any other.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            width, height = img.size
            blank = all(low == high for low, high in img.convert("RGB").getextrema())
            phash = str(imagehash.phash(img, hash_size=8))
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise UnreadableImage(f"the image cannot be decoded: {exc}") from exc
    if blank and not allow_blank:
        raise BlankImage("the render came back blank: every pixel is the same colour")
    return ImageRecord(sha256=hashlib.sha256(data).hexdigest(), phash=phash, width=width, height=height)


@dataclass(frozen=True)
class InputImage:
    """An image-to-image input, held in memory only.

    *png* is the input as it is judged and rendered from: decoded, turned upright by its EXIF orientation, made RGB
    (alpha dropped, first frame only) and re-encoded as a PNG with no metadata. Scanning the bytes ComfyUI will
    decode, rather than the upload, means no second frame, colour profile or hidden channel can differ between the
    two.
    """

    png: bytes
    record: ImageRecord
    source_sha256: str

    def __repr__(self) -> str:  # never the pixels, in a log or a traceback
        return f"InputImage(sha256={self.record.sha256}, {self.record.width}x{self.record.height})"

    @cached_property  # encoded once: it fills the graph both when the request is prepared and when it is rendered
    def base64(self) -> str:
        return base64.b64encode(self.png).decode("ascii")


def load_input(data_base64: str) -> InputImage:
    """Decode an image-to-image input from base64. Raises :class:`UnreadableImage` with the reason it is refused:
    not base64, over :data:`MAX_INPUT_BYTES` decoded, not a PNG, JPEG or WebP, an edge over :data:`MAX_INPUT_EDGE`,
    or pixels that do not decode."""
    if len(data_base64) > _MAX_INPUT_BASE64:
        raise UnreadableImage(f"the init image is over {MAX_INPUT_BYTES // (1024 * 1024)} MiB")
    try:
        data = base64.b64decode(data_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise UnreadableImage("the init image is not valid base64") from exc
    if not data:
        raise UnreadableImage("the init image is empty")
    if len(data) > MAX_INPUT_BYTES:
        raise UnreadableImage(f"the init image is over {MAX_INPUT_BYTES // (1024 * 1024)} MiB")
    try:
        with Image.open(io.BytesIO(data), formats=INPUT_FORMATS) as img:
            # The size is known from the header: refuse before the pixels are decoded.
            if max(img.size) > MAX_INPUT_EDGE:
                raise UnreadableImage(
                    f"the init image is {img.width}x{img.height}; an edge may be {MAX_INPUT_EDGE} at most"
                )
            img.load()
            upright = ImageOps.exif_transpose(img)
            rgb = upright.convert("RGB")
    except UnreadableImage:
        raise
    except Exception as exc:  # a malformed header or EXIF block can raise anything (struct.error, KeyError, ...)
        raise UnreadableImage(f"the init image cannot be decoded as PNG, JPEG or WebP: {exc}") from exc
    buf = io.BytesIO()
    rgb.save(buf, "PNG")
    png = buf.getvalue()
    return InputImage(png=png, record=describe(png, allow_blank=True), source_sha256=hashlib.sha256(data).hexdigest())
