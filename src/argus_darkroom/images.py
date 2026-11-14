"""What is kept of an image whether or not its pixels are: sha256, perceptual hash and size.

Also the image-to-image input: decoded from the request's base64 once, into the one PNG that is both judged and sent
to ComfyUI, so what was checked is exactly what is rendered from. And the inpainting mask, decoded the same way into a
black-and-white PNG; it selects a region and carries no imagery, so it is never scanned.
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


def _decode_upright(data_base64: str, what: str) -> tuple[Image.Image, str]:
    """Decode *data_base64* (the request's *what*, e.g. "the init image") into an upright image and the sha256 of the
    bytes sent. Raises :class:`UnreadableImage` with the reason it is refused: not base64, over
    :data:`MAX_INPUT_BYTES` decoded, not a PNG, JPEG or WebP, an edge over :data:`MAX_INPUT_EDGE`, or pixels that do
    not decode."""
    if len(data_base64) > _MAX_INPUT_BASE64:
        raise UnreadableImage(f"{what} is over {MAX_INPUT_BYTES // (1024 * 1024)} MiB")
    try:
        data = base64.b64decode(data_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise UnreadableImage(f"{what} is not valid base64") from exc
    if not data:
        raise UnreadableImage(f"{what} is empty")
    if len(data) > MAX_INPUT_BYTES:
        raise UnreadableImage(f"{what} is over {MAX_INPUT_BYTES // (1024 * 1024)} MiB")
    try:
        with Image.open(io.BytesIO(data), formats=INPUT_FORMATS) as img:
            # The size is known from the header: refuse before the pixels are decoded.
            if max(img.size) > MAX_INPUT_EDGE:
                raise UnreadableImage(f"{what} is {img.width}x{img.height}; an edge may be {MAX_INPUT_EDGE} at most")
            img.load()
            upright = ImageOps.exif_transpose(img)
            upright.load()
            if upright.mode.startswith("I"):
                # A 16-bit greyscale PNG: Pillow's conversions to L or RGB clip instead of scaling, which would make
                # every grey above 255 of 65535 white. Scale it down to 8 bits first.
                upright = upright.convert("I").point(lambda v: v * (1 / 256)).convert("L")
    except UnreadableImage:
        raise
    except Exception as exc:  # a malformed header or EXIF block can raise anything (struct.error, KeyError, ...)
        raise UnreadableImage(f"{what} cannot be decoded as PNG, JPEG or WebP: {exc}") from exc
    return upright, hashlib.sha256(data).hexdigest()


def load_input(data_base64: str) -> InputImage:
    """Decode an image-to-image input from base64. Raises :class:`UnreadableImage` with the reason it is refused:
    not base64, over :data:`MAX_INPUT_BYTES` decoded, not a PNG, JPEG or WebP, an edge over :data:`MAX_INPUT_EDGE`,
    or pixels that do not decode."""
    upright, source_sha256 = _decode_upright(data_base64, "the init image")
    try:
        rgb = upright.convert("RGB")
    except Exception as exc:
        raise UnreadableImage(f"the init image cannot be decoded as PNG, JPEG or WebP: {exc}") from exc
    buf = io.BytesIO()
    rgb.save(buf, "PNG")
    png = buf.getvalue()
    return InputImage(png=png, record=describe(png, allow_blank=True), source_sha256=source_sha256)


@dataclass(frozen=True)
class InputMask:
    """An inpainting mask, held in memory only: white (set) repaints, black keeps.

    It selects a region of the input image; it carries no imagery of its own, so it is not scanned. *png* is the mask
    as ComfyUI gets it: decoded, turned upright by its EXIF orientation like the image, made greyscale, cut at
    :data:`MASK_THRESHOLD` into black and white, and re-encoded as a 1-bit PNG with no metadata.
    """

    png: bytes
    sha256: str
    source_sha256: str
    width: int
    height: int
    # The share of pixels that are repainted, 0..1, to 4 decimals.
    masked_fraction: float

    def __repr__(self) -> str:  # never the pixels, in a log or a traceback
        return f"InputMask(sha256={self.sha256}, {self.width}x{self.height}, masked={self.masked_fraction})"

    @cached_property  # encoded once, like InputImage.base64
    def base64(self) -> str:
        return base64.b64encode(self.png).decode("ascii")


MASK_THRESHOLD = 128  # a grey at or above this repaints


def load_mask(data_base64: str, size: tuple[int, int]) -> InputMask:
    """Decode an inpainting mask from base64, for an input image of *size* (as judged: upright).

    Raises :class:`UnreadableImage` with the reason it is refused: everything :func:`load_input` refuses, a size
    (once upright) that is not exactly *size*, or nothing masked.
    """
    upright, source_sha256 = _decode_upright(data_base64, "the mask")
    if upright.size != size:
        raise UnreadableImage(
            f"the mask is {upright.width}x{upright.height} but the init image is {size[0]}x{size[1]}: they must match"
        )
    try:
        grey = upright.convert("L")
    except Exception as exc:
        raise UnreadableImage(f"the mask cannot be decoded as PNG, JPEG or WebP: {exc}") from exc
    cut = grey.point(lambda v: 255 if v >= MASK_THRESHOLD else 0)
    masked = cut.histogram()[255]
    if masked == 0:
        raise UnreadableImage("the mask masks nothing: paint the region to repaint white")
    binary = cut.convert("1", dither=Image.Dither.NONE)
    buf = io.BytesIO()
    binary.save(buf, "PNG")
    png = buf.getvalue()
    return InputMask(
        png=png,
        sha256=hashlib.sha256(png).hexdigest(),
        source_sha256=source_sha256,
        width=binary.width,
        height=binary.height,
        # Never 0 once something is masked: a few pixels of a large mask would otherwise round to "masks nothing".
        masked_fraction=max(round(masked / (binary.width * binary.height), 4), 0.0001),
    )
