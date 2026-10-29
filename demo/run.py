"""Run the real darkroom server against fake backends, so the whole thing can be tried without a GPU.

    uv run --no-sync python demo/run.py            # darkroom on http://127.0.0.1:8106

Everything darkroom does is real here: pre-flight, the ComfyUI websocket protocol, both scans per policy, the
outcome table, the record store and the HTTP API. Only the four backends are fakes, from the same fakes the tests
use. Their verdicts are chosen by a harmless marker in the prompt, never by content, so nothing flagged,
sexual or violent exists anywhere in this demo. The images are flat colour with a label.

The age check is fake too: it reports a made-up estimate, so safelight can show how an estimate is worded.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import shutil
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))  # the fakes the tests use

import uvicorn  # noqa: E402
from fake_comfyui import IMAGE_HEADER, FakeComfyUI, event  # noqa: E402
from fakes import FakeNudeNet, FakeOllama, FakeSingGuard, policy_of, safe_answer, unsafe_answer  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from argus_darkroom.age import AgeUnavailable  # noqa: E402
from argus_darkroom.comfyui import ComfyUIClient  # noqa: E402
from argus_darkroom.config import Settings  # noqa: E402
from argus_darkroom.engine import Engine  # noqa: E402
from argus_darkroom.models import FaceAge  # noqa: E402
from argus_darkroom.server import create_app  # noqa: E402

MARKERS = {
    "flag": "the with-prompt scan flags rule B, the isolated scan does not: review, the prompt changed the verdict",
    "flag-both": "both scans flag rule B: review, the scans agree",
    "illegal": "the with-prompt scan hits rule A (use an adult policy; `default` has no illegal rules): deleted",
    "unchecked": "the with-prompt scan fails: deleted, because an unchecked enforcing cell fails closed",
    "exposed": "NudeNet scores 0.41: omitted under an SFW policy, blurred under an NSFW one",
    "nonude": "NudeNet does not answer: omitted under SFW, blurred under NSFW",
    "minor": "an age estimate of 17 with NudeNet 0.45: deleted, minor-exposed",
    "noage": "the age check has nothing for this image: the record says `age: null`",
    "refuse": "pre-flight flags rule B: refused under SFW, a warning under NSFW",
    "refuse-illegal": "pre-flight hits rule A: refused (use an adult policy)",
    "preflight-unchecked": "the pre-flight check fails: refused, unchecked",
    "fail": "ComfyUI fails mid-way: failed",
}
EXPOSED = "FEMALE_BREAST_EXPOSED"  # a NudeNet class name; the score is made up
ILLEGAL_RULE = {"adult-granular": "A. Minors"}
MARKER = re.compile(r"\[([a-z-]+)\]")
DEFAULT_ORIGINS = tuple(f"http://{host}:{port}" for port in (3000, 3100) for host in ("localhost", "127.0.0.1"))
WHY = "Demo verdict: chosen by a marker in the prompt, not by any content."


def markers_in(text: str) -> set[str]:
    return {m for m in MARKER.findall(text) if m in MARKERS}


class Demo:
    """What the fake backends share: the markers of the generation in flight (one runs at a time)."""

    def __init__(self, latency: float, step_delay: float) -> None:
        self.latency, self.step_delay = latency, step_delay
        self.markers: set[str] = set()
        self.nudenet: FakeNudeNet | None = None

    # -- SingGuard ----------------------------------------------------------------------------------

    def singguard(self, body: dict) -> tuple[int, dict]:
        time.sleep(self.latency)  # so the shield grid visibly fills in
        policy = policy_of(body)
        illegal = unsafe_answer(ILLEGAL_RULE.get(policy, "A. Illegal or Extreme Sexual Content"), explanation=WHY)
        flagged = unsafe_answer("B. Real-World Crimes & Public Safety", explanation=WHY)
        if "image_base64" not in body:  # pre-flight: the text is the request
            found = markers_in(body["input"])
            if "preflight-unchecked" in found:
                return 500, {"error": "demo: SingGuard did not answer"}
            if "refuse-illegal" in found:
                return 200, illegal
            return 200, flagged if "refuse" in found else safe_answer()
        found, with_prompt = self.markers, body["input"] != ""
        if with_prompt and "unchecked" in found:
            return 500, {"error": "demo: SingGuard did not answer"}
        if with_prompt and "illegal" in found:
            return 200, illegal
        if "flag-both" in found or (with_prompt and "flag" in found):
            return 200, flagged
        return 200, safe_answer()

    # -- ComfyUI ------------------------------------------------------------------------------------

    def comfyui(self, fake: FakeComfyUI, prompt_id: str, graph: dict):
        """Messages for one render, one at a time with a pause, so the progress bar moves."""
        sampler = next(n for n in graph.values() if n["class_type"] == "KSampler")
        latent = next(n for n in graph.values() if n["class_type"] == "EmptyLatentImage")["inputs"]
        positive = graph[sampler["inputs"]["positive"][0]]["inputs"]["text"]
        self.markers = markers_in(str(positive))
        if self.nudenet is not None:
            score = 0.45 if "minor" in self.markers else 0.41 if "exposed" in self.markers else None
            self.nudenet.labels = {EXPOSED: score} if score else {}
            self.nudenet.status = 500 if "nonude" in self.markers else 200
        save = next(i for i, n in graph.items() if n["class_type"] == "SaveImageWebsocket")
        steps = int(sampler["inputs"]["steps"])

        yield event("execution_start", prompt_id=prompt_id)
        yield event("executing", node="3", prompt_id=prompt_id)
        for step in range(1, steps + 1):
            time.sleep(self.step_delay)
            yield event("progress", value=step, max=steps, node="3", prompt_id=prompt_id)
            if "fail" in self.markers and step >= max(1, steps // 2):
                yield event(
                    "execution_error",
                    prompt_id=prompt_id,
                    node="3",
                    node_type="KSampler",
                    exception_message="CUDA out of memory (a demo error, chosen by [fail])",
                )
                return
        yield event("executing", node=save, prompt_id=prompt_id)
        yield IMAGE_HEADER + flat_png(int(latent["width"]), int(latent["height"]), int(sampler["inputs"]["seed"]))
        yield event("executing", node=None, prompt_id=prompt_id)


def flat_png(width: int, height: int, seed: int) -> bytes:
    """A flat colour field with a label and nothing else, the colour taken from the seed."""
    digest = hashlib.sha256(str(seed).encode()).digest()
    colour = tuple(70 + b % 120 for b in digest[:3])
    image = Image.new("RGB", (width, height), colour)
    ImageDraw.Draw(image).text((12, 12), f"darkroom demo image\nseed {seed}\n{width}x{height}", fill=(245, 245, 245))
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


class DemoAge:
    """A made-up MiVOLO estimate, so the UI can show how one is worded. Not a face detector."""

    def __init__(self, demo: Demo) -> None:
        self.demo = demo

    def estimate(self, data: bytes) -> list[FaceAge]:
        found = self.demo.markers
        if "noage" in found:
            raise AgeUnavailable("demo: no age check for this image")
        years = 17.0 if "minor" in found else 34.0
        return [FaceAge(age_estimate_years=years, bbox=(24, 24, 64, 64))]


def reply(body: dict) -> str:
    """What the fake Ollama says: the user's text, expanded. Any marker survives, as it would for a real model."""
    return f"{body['messages'][-1]['content']}, soft window light, 35mm photo, shallow depth of field"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8106)
    parser.add_argument(
        "--cors-origin", action="append", help="a UI origin allowed to write (repeatable; default: safelight's ports)"
    )
    parser.add_argument("--data", type=Path, default=ROOT / "demo-data", help="records, images and fake models")
    parser.add_argument("--fresh", action="store_true", help="start from an empty history")
    parser.add_argument(
        "--workflow", default="sdxl.json", help="a file under examples/workflows (sdxl_lora.json adds a LoRA)"
    )
    parser.add_argument("--latency", type=float, default=0.4, help="seconds each fake SingGuard answer takes")
    parser.add_argument("--step-delay", type=float, default=0.1, help="seconds each fake ComfyUI step takes")
    args = parser.parse_args()

    if args.fresh:
        shutil.rmtree(args.data, ignore_errors=True)
    models = args.data / "models"
    for kind, name in [("checkpoints", "demo-checkpoint.safetensors"), ("loras", "demo-style.safetensors")]:
        (models / kind).mkdir(parents=True, exist_ok=True)
        (models / kind / name).write_bytes(name.encode())  # stand-ins: only their hashes are used

    demo = Demo(args.latency, args.step_delay)
    with (
        FakeSingGuard(demo.singguard) as singguard,
        FakeNudeNet() as nudenet,
        FakeOllama(reply) as ollama,
        FakeComfyUI(script=demo.comfyui) as comfyui,
    ):
        demo.nudenet = nudenet
        settings = Settings(
            singguard_url=singguard.url,
            comfyui_url=comfyui.url,
            nudenet_url=nudenet.url,
            ollama_url=ollama.url,
            ollama_model="demo-model",
            policy="default",
            allow_nsfw=True,
            devmode=True,
            cors_origins=tuple(args.cors_origin or DEFAULT_ORIGINS),
            output_root=args.data / "records",
            models_root=models,
            workflow=ROOT / "examples" / "workflows" / args.workflow,
        )
        engine = Engine(
            settings, comfyui=ComfyUIClient(comfyui.url, ws_url=comfyui.ws_url, timeout=60), age=DemoAge(demo)
        )
        print(banner(args))
        uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="warning")


def banner(args: argparse.Namespace) -> str:
    rows = "\n".join(f"  [{m}]".ljust(26) + textwrap.shorten(text, 90) for m, text in MARKERS.items())
    return (
        f"\nargus-darkroom demo on http://{args.host}:{args.port}  (devmode and NSFW policies on; history in {args.data})\n"
        "Real darkroom, fake backends: SingGuard, NudeNet, ComfyUI, Ollama and the age check are stand-ins.\n"
        "Put a marker in a prompt to choose what the fake guards say:\n\n"
        f"{rows}\n"
    )


if __name__ == "__main__":
    main()
