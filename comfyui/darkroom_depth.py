"""ComfyUI custom node for argus-darkroom's depth maps: Depth Anything V2 Small over an ``IMAGE``, out as an ``IMAGE``.

Darkroom sends a record's stored image (already judged and released) through ``DarkroomLoadImageBase64``, this node and
``SaveImageWebsocket``. The output is relative inverse depth (disparity) stretched min to max per image, at the input's
size, grey in all three channels: near is bright (1.0), far is dark (0.0). Darkroom cuts it to 8-bit greyscale.

Only the Small model loads: it is Apache-2.0. Depth Anything V2 Base and Large are CC-BY-NC-4.0, so any other name, or
a folder whose config is not the Small model's, is refused rather than loaded. The weights are read from a local folder
under ComfyUI's models directory, never downloaded at run time::

    huggingface-cli download depth-anything/Depth-Anything-V2-Small-hf \\
        --local-dir /path/to/ComfyUI/models/depth/Depth-Anything-V2-Small-hf

Install: copy this file into ComfyUI's ``custom_nodes/`` (beside ``darkroom_load_image_base64.py``) and restart ComfyUI.
Check with ``curl -s localhost:8188/object_info/DarkroomDepthAnything``. It reads only the model folder, writes nothing,
and is not an output node.
"""

from __future__ import annotations

import json
import os
import threading

import numpy as np

# The one model this node loads, by its folder name under ``models/depth/``.
SMALL = "Depth-Anything-V2-Small-hf"
ALLOWED = (SMALL,)
# Depth Anything V2 Small's backbone (DINOv2 ViT-S/14). Base is 768 and Large 1024: a folder renamed to pass as Small is
# caught by its config.
SMALL_HIDDEN_SIZE = 384

_lock = threading.Lock()
_cache: dict[tuple[str, str], tuple[object, object]] = {}


def depth_root() -> str:
    """ComfyUI's ``models/depth`` folder."""
    import folder_paths  # ComfyUI's; imported here so the checks below are testable without it

    return os.path.join(folder_paths.models_dir, "depth")


def check_model(root: str, name: str) -> str:
    """The folder of the model *name* under *root*, once it is known to be Depth Anything V2 Small. Raises
    ``ValueError`` for any other name or a folder that does not hold the Small model."""
    if name not in ALLOWED:
        raise ValueError(
            f"only {SMALL} (Apache-2.0) may be loaded; {name!r} is refused (Depth Anything V2 Base and Large are "
            "CC-BY-NC-4.0)"
        )
    path = os.path.join(root, name)
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"{path} does not hold the model; download it once with `huggingface-cli download "
            f"depth-anything/{SMALL} --local-dir {path}`"
        ) from exc
    config = config if isinstance(config, dict) else {}
    backbone = config.get("backbone_config")
    hidden = backbone.get("hidden_size") if isinstance(backbone, dict) else None
    if config.get("model_type") != "depth_anything" or hidden != SMALL_HIDDEN_SIZE:
        raise ValueError(f"{path} is not Depth Anything V2 Small (backbone hidden_size {hidden!r}); refused")
    return path


def normalise(depth: np.ndarray) -> np.ndarray:
    """*depth* (H, W) stretched to 0..1, min to max, as float32. A flat map is all 0."""
    depth = depth.astype(np.float32)
    low, high = float(depth.min()), float(depth.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(depth)
    return (depth - low) / (high - low)


def _load(path: str, device: object) -> tuple[object, object]:
    from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation

    key = (path, str(device))
    with _lock:
        if key not in _cache:
            processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
            model = DepthAnythingForDepthEstimation.from_pretrained(path, local_files_only=True)
            _cache.clear()  # one model, on one device
            _cache[key] = (processor, model.to(device).eval())
        return _cache[key]


class DarkroomDepthAnything:
    CATEGORY = "argus-darkroom"
    DESCRIPTION = "Depth Anything V2 Small (Apache-2.0): near bright, far dark, min to max per image. Writes no file."
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate"

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802 (ComfyUI's name)
        return {"required": {"image": ("IMAGE",), "model": (list(ALLOWED),)}}

    def estimate(self, image, model: str) -> tuple:
        import torch  # ComfyUI always has it; importing here keeps the checks testable without it
        from comfy import model_management
        from PIL import Image

        path = check_model(depth_root(), model)
        device = model_management.get_torch_device()
        processor, net = _load(path, device)
        out = []
        for frame in image:  # IMAGE is [batch, H, W, C], 0..1
            height, width = int(frame.shape[0]), int(frame.shape[1])
            pixels = (frame[..., :3].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            inputs = processor(images=Image.fromarray(pixels), return_tensors="pt").to(device)
            with torch.inference_mode():
                predicted = net(**inputs).predicted_depth  # [1, h, w], relative inverse depth: near is large
            resized = torch.nn.functional.interpolate(
                predicted[:, None].float(), size=(height, width), mode="bicubic", align_corners=False
            )[0, 0]
            grey = torch.from_numpy(normalise(resized.cpu().numpy()))
            out.append(grey[..., None].expand(height, width, 3))
        return (torch.stack(out).contiguous(),)


NODE_CLASS_MAPPINGS = {"DarkroomDepthAnything": DarkroomDepthAnything}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomDepthAnything": "Depth Anything V2 Small (darkroom)"}
