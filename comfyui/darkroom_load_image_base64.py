"""ComfyUI custom nodes for argus-darkroom's image-to-image and inpainting: an image, or a mask, from a base64 string,
never from disk.

ComfyUI's stock ``LoadImage`` reads from its ``input/`` folder, so a caller's image would have to be written there
first and would stay behind. This node takes the image inside the prompt graph instead: darkroom fills its ``image``
input with the input image it has already checked, as base64, and deletes the prompt from ComfyUI's history after the
run. It reads nothing from disk and writes nothing, and it is not an output node.

``DarkroomLoadMaskBase64`` does the same for an inpainting mask: base64 of a black-and-white PNG in, a ``MASK`` out
(1.0 where the image is repainted, 0.0 where it is kept).

Install: copy this file into ComfyUI's ``custom_nodes/`` and restart ComfyUI. Check with
``curl -s localhost:8188/object_info/DarkroomLoadImageBase64`` (and ``.../DarkroomLoadMaskBase64``).
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def decode(image: str) -> np.ndarray:
    """*image* (base64 of a PNG, JPEG or WebP) as float32 RGB in 0..1, shape (H, W, 3). Only the first frame."""
    data = base64.b64decode(image, validate=True)
    with Image.open(io.BytesIO(data), formats=("PNG", "JPEG", "WEBP")) as img:
        rgb = img.convert("RGB")
    return np.asarray(rgb, dtype=np.float32) / 255.0


def decode_mask(mask: str) -> np.ndarray:
    """*mask* (base64 of a PNG, JPEG or WebP) as float32 greyscale in 0..1, shape (H, W). Only the first frame."""
    data = base64.b64decode(mask, validate=True)
    with Image.open(io.BytesIO(data), formats=("PNG", "JPEG", "WEBP")) as img:
        grey = img.convert("L")
    return np.asarray(grey, dtype=np.float32) / 255.0


class DarkroomLoadImageBase64:
    CATEGORY = "argus-darkroom"
    DESCRIPTION = "An image from a base64 string in the graph. Reads and writes no file."
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802 (ComfyUI's name)
        return {"required": {"image": ("STRING", {"multiline": False})}}

    def load(self, image: str) -> tuple:
        import torch  # ComfyUI always has it; importing here keeps decode() testable without it

        return (torch.from_numpy(decode(image))[None, ...],)  # IMAGE is [batch, H, W, C]


class DarkroomLoadMaskBase64:
    CATEGORY = "argus-darkroom"
    DESCRIPTION = "A mask from a base64 string in the graph (white repaints). Reads and writes no file."
    RETURN_TYPES = ("MASK",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802 (ComfyUI's name)
        return {"required": {"mask": ("STRING", {"multiline": False})}}

    def load(self, mask: str) -> tuple:
        import torch  # ComfyUI always has it; importing here keeps decode_mask() testable without it

        return (torch.from_numpy(decode_mask(mask))[None, ...],)  # MASK is [batch, H, W]


NODE_CLASS_MAPPINGS = {
    "DarkroomLoadImageBase64": DarkroomLoadImageBase64,
    "DarkroomLoadMaskBase64": DarkroomLoadMaskBase64,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DarkroomLoadImageBase64": "Load Image (base64, darkroom)",
    "DarkroomLoadMaskBase64": "Load Mask (base64, darkroom)",
}
