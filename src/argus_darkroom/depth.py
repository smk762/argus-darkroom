"""Depth maps: a derived artefact of a record's stored image, computed by ComfyUI and kept beside the image.

``POST /depth/{id}`` sends the stored image through a graph built here, never one from ``ARGUS_DARKROOM_WORKFLOW``:
the image goes in as base64 to ``DarkroomLoadImageBase64`` (the same way image-to-image passes ``init_image``), through
``DarkroomDepthAnything`` (``comfyui/darkroom_depth.py``, Depth Anything V2 Small, Apache-2.0), and back over the
websocket from ``SaveImageWebsocket``. It is not in the workflow set, so generation never selects it and ``/models``
does not list it.

**No guard runs.** A depth map is derived from an image darkroom already judged and released (a ``passed`` or
``review`` record that still has its file), with no new prompt and no new imagery: the prompt pre-flight and the image
scans have nothing new to judge. The guard for generations is unchanged.

What ComfyUI sends back is cut down here to the stored form: 8-bit greyscale at the record image's size, near bright
(255) and far dark (0), stretched min to max, re-encoded as a PNG with no metadata.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
from dataclasses import dataclass

from PIL import Image

from argus_darkroom.images import UnreadableImage
from argus_darkroom.workflow import LOADER_NODE, SAVE_NODE, Workflow, template_sha256

MODEL = "depth-anything-v2-small"
MODEL_LICENSE = "Apache-2.0"
# The folder under ComfyUI's ``models/depth/`` the node loads from (``huggingface-cli download ... --local-dir``).
MODEL_DIR = "Depth-Anything-V2-Small-hf"
DEPTH_NODE = "DarkroomDepthAnything"
WORKFLOW_NAME = "depth-anything-v2-small"
_IMAGE = "$image"

_TEMPLATE: dict = {
    "1": {"class_type": LOADER_NODE, "inputs": {"image": _IMAGE}},
    "2": {"class_type": DEPTH_NODE, "inputs": {"image": ["1", 0], "model": MODEL_DIR}},
    "9": {"class_type": SAVE_NODE, "inputs": {"images": ["2", 0]}},
}
# A Workflow only so the engine's output-node check (``Engine._verify_outputs``) can run on it. Never in a WorkflowSet.
WORKFLOW = Workflow(template=_TEMPLATE, sha256=template_sha256(_TEMPLATE), save_node_id="9", name=WORKFLOW_NAME)


def graph(image: bytes) -> dict:
    """The depth graph for the PNG *image*."""
    out = copy.deepcopy(_TEMPLATE)
    out["1"]["inputs"]["image"] = base64.b64encode(image).decode("ascii")
    return out


@dataclass(frozen=True)
class DepthMap:
    png: bytes
    width: int
    height: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.png).hexdigest()


def to_depth_png(data: bytes, width: int, height: int) -> DepthMap:
    """ComfyUI's depth image as the stored PNG: greyscale, *width* x *height*, min to max, no metadata.

    Raises :class:`UnreadableImage` when *data* does not decode.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            grey = img.convert("L")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise UnreadableImage(f"the depth map cannot be decoded: {exc}") from exc
    if grey.size != (width, height):
        grey = grey.resize((width, height), Image.Resampling.BICUBIC)
    low, high = grey.getextrema()
    if high > low:
        grey = grey.point(lambda v: round((v - low) * 255 / (high - low)))
    else:
        grey = Image.new("L", (width, height), 0)  # no depth to tell apart: all far
    buf = io.BytesIO()
    grey.save(buf, "PNG")  # Pillow writes no text chunks unless asked
    return DepthMap(png=buf.getvalue(), width=width, height=height)
