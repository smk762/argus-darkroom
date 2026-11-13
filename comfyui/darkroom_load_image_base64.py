"""A ComfyUI custom node for argus-darkroom's image-to-image: an image from a base64 string, never from disk.

ComfyUI's stock ``LoadImage`` reads from its ``input/`` folder, so a caller's image would have to be written there
first and would stay behind. This node takes the image inside the prompt graph instead: darkroom fills its ``image``
input with the input image it has already checked, as base64, and deletes the prompt from ComfyUI's history after the
run. It reads nothing from disk and writes nothing, and it is not an output node.

Install: copy this file into ComfyUI's ``custom_nodes/`` and restart ComfyUI. Check with
``curl -s localhost:8188/object_info/DarkroomLoadImageBase64``.
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


NODE_CLASS_MAPPINGS = {"DarkroomLoadImageBase64": DarkroomLoadImageBase64}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomLoadImageBase64": "Load Image (base64, darkroom)"}
