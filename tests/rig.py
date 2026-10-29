"""Shared test rig: fake SingGuard, ComfyUI and NudeNet wired to an engine, in a tmp dir."""

from __future__ import annotations

import base64
from pathlib import Path

from fake_comfyui import FakeComfyUI
from fakes import FakeNudeNet, FakeSingGuard, policy_of, safe_answer, unsafe_answer

from argus_darkroom.comfyui import ComfyUIClient
from argus_darkroom.config import Settings
from argus_darkroom.engine import Engine
from argus_darkroom.guard import Guard
from argus_darkroom.models import GenerationRecord, SamplingParams
from argus_darkroom.request import GenerateRequest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "workflows"
SAMPLING = SamplingParams(sampler="euler", scheduler="normal", steps=4, cfg=5.0, width=64, height=64)
EXPOSED = {"FEMALE_BREAST_EXPOSED": 0.9}
ILLEGAL_A = unsafe_answer("A. Minors")


class Rig:
    """Fake SingGuard, ComfyUI and NudeNet wired to an engine, in a tmp dir."""

    def __init__(
        self, tmp_path: Path, singguard: FakeSingGuard, comfyui: FakeComfyUI, nudenet: FakeNudeNet, **kw
    ) -> None:
        self.root = tmp_path
        self.out = tmp_path / "out"
        self.systmp = tmp_path / "systmp"
        self.systmp.mkdir()
        models = tmp_path / "models"
        for kind, name in [("checkpoints", "base.safetensors"), ("loras", "style.safetensors")]:
            (models / kind).mkdir(parents=True, exist_ok=True)
            (models / kind / name).write_bytes(name.encode())
        self.singguard, self.comfyui, self.nudenet = singguard, comfyui, nudenet
        settings = Settings(
            singguard_url=singguard.url,
            comfyui_url=comfyui.url,
            nudenet_url=kw.pop("nudenet_url", nudenet.url),
            output_root=self.out,
            models_root=models,
            workflow=EXAMPLES / "sdxl_lora.json",
            **kw,
        )
        self.settings = settings
        self.events: list[tuple[str, dict]] = []

    def engine(self, **kw) -> Engine:
        return Engine(
            self.settings,
            guard=Guard(self.settings.singguard_url),
            comfyui=ComfyUIClient(self.comfyui.url, ws_url=self.comfyui.ws_url, timeout=5),
            **kw,
        )

    def request(self, **kw) -> GenerateRequest:
        base = {
            "user_prompt": "a cat",
            "negative_prompt": "blurry",
            "seed": 3,
            "sampling": SAMPLING,
            "checkpoint": "base.safetensors",
            "loras": [{"name": "style.safetensors", "trigger_words": ["sks"]}],
        }
        return GenerateRequest(**(base | kw))

    def generate(self, engine: Engine, **kw) -> GenerationRecord:
        return engine.generate(self.request(**kw), lambda kind, data: self.events.append((kind, data)))

    def files(self) -> list[str]:
        """Every file under the output root and the (redirected) system temp dir."""
        return sorted(str(p.relative_to(self.root)) for base in (self.out, self.systmp) for p in base.rglob("*"))

    def holds_the_image(self) -> bool:
        """Whether the PNG, or any base64 of it, is in any file under the tmp dir (models excluded)."""
        png = self.comfyui.png
        for path in self.root.rglob("*"):
            if path.is_file() and "models" not in path.relative_to(self.root).parts:
                data = path.read_bytes()
                if png in data or base64.b64encode(png) in data:
                    return True
        return False


def nsfw(rig: Rig) -> None:
    rig.settings = Settings(**{**rig.settings.__dict__, "allow_nsfw": True, "policy": "adult-granular"})


def is_image_scan(body: dict) -> bool:
    return "image_base64" in body


def answers(preflight=None, isolated=None, with_prompt=None):
    """A SingGuard responder: an answer (or per-policy dict of answers) per context; default safe."""

    def pick(spec, body):
        if spec is None:
            return safe_answer()
        return spec.get(policy_of(body), safe_answer()) if isinstance(spec, dict) and "safe" not in spec else spec

    def respond(body: dict):
        if not is_image_scan(body):
            return 200, pick(preflight, body)
        return 200, pick(isolated if body["input"] == "" else with_prompt, body)

    return respond
