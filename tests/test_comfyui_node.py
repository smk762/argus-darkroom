"""The vendored ComfyUI node, ``comfyui/darkroom_load_image_base64.py``. It is not part of the package: ComfyUI loads it
from its ``custom_nodes/``. Loaded here by path."""

from __future__ import annotations

import base64
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

NODE = Path(__file__).resolve().parent.parent / "comfyui" / "darkroom_load_image_base64.py"


def load_node():
    spec = importlib.util.spec_from_file_location("darkroom_load_image_base64", NODE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def png(mode: str = "RGB", size=(3, 2)) -> str:
    image = Image.new(mode, size, (255, 0, 0, 0)[: len(mode)] if mode != "L" else 128)
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_the_node_is_registered_as_a_loader_not_an_output() -> None:
    node = load_node()
    cls = node.NODE_CLASS_MAPPINGS["DarkroomLoadImageBase64"]
    assert cls.RETURN_TYPES == ("IMAGE",) and not getattr(cls, "OUTPUT_NODE", False)
    assert cls.INPUT_TYPES() == {"required": {"image": ("STRING", {"multiline": False})}}


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_decode_gives_rgb_floats_in_0_to_1(mode) -> None:
    out = load_node().decode(png(mode))
    assert out.shape == (2, 3, 3) and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_decode_refuses_what_is_not_an_image() -> None:
    node = load_node()
    with pytest.raises(ValueError):
        node.decode("not base64!")
    with pytest.raises(OSError):
        node.decode(base64.b64encode(b"text").decode())


def test_the_node_never_touches_a_file() -> None:
    source = NODE.read_text()
    assert "open(" not in source.replace("Image.open(", "") and "folder_paths" not in source


def test_load_returns_a_batch_of_one() -> None:
    torch = pytest.importorskip("torch")
    (image,) = load_node().DarkroomLoadImageBase64().load(png())
    assert isinstance(image, torch.Tensor) and tuple(image.shape) == (1, 2, 3, 3)


# --- the mask loader ------------------------------------------------------------------------------------------


def mask_png(mode: str = "1", size=(3, 2)) -> str:
    image = Image.new(mode, size, 0)
    image.putpixel((0, 0), 255 if mode in ("1", "L") else (255,) * len(mode))
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_the_mask_node_is_registered_as_a_loader_not_an_output() -> None:
    node = load_node()
    cls = node.NODE_CLASS_MAPPINGS["DarkroomLoadMaskBase64"]
    assert cls.RETURN_TYPES == ("MASK",) and not getattr(cls, "OUTPUT_NODE", False)
    assert cls.INPUT_TYPES() == {"required": {"mask": ("STRING", {"multiline": False})}}
    assert set(node.NODE_DISPLAY_NAME_MAPPINGS) == set(node.NODE_CLASS_MAPPINGS)


@pytest.mark.parametrize("mode", ["1", "L", "RGB", "RGBA"])
def test_decode_mask_gives_one_channel_of_floats_white_is_one(mode) -> None:
    out = load_node().decode_mask(mask_png(mode))
    assert out.shape == (2, 3) and out.dtype == np.float32
    assert out[0, 0] == 1.0 and out.sum() == 1.0  # one white pixel, the rest kept


def test_decode_mask_refuses_what_is_not_an_image() -> None:
    node = load_node()
    with pytest.raises(ValueError):
        node.decode_mask("not base64!")
    with pytest.raises(OSError):
        node.decode_mask(base64.b64encode(b"text").decode())


def test_the_canonical_mask_decodes_to_ones_where_it_repaints() -> None:
    from argus_darkroom.images import load_mask

    image = Image.new("L", (8, 4), 0)
    image.paste(200, (0, 0, 4, 4))
    buf = io.BytesIO()
    image.save(buf, "PNG")
    mask = load_mask(base64.b64encode(buf.getvalue()).decode(), (8, 4))
    out = load_node().decode_mask(mask.base64)
    assert out[:, :4].min() == 1.0 and out[:, 4:].max() == 0.0


def test_load_mask_returns_a_batch_of_one() -> None:
    torch = pytest.importorskip("torch")
    (mask,) = load_node().DarkroomLoadMaskBase64().load(mask_png())
    assert isinstance(mask, torch.Tensor) and tuple(mask.shape) == (1, 2, 3) and mask.dtype == torch.float32
