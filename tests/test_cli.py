from __future__ import annotations

import json
from pathlib import Path

import pytest
from fakes import unsafe_answer
from rig import answers
from typer.testing import CliRunner

from argus_darkroom import cli
from argus_darkroom.cli import app

runner = CliRunner()
SAMPLING = [
    "--sampler",
    "euler",
    "--scheduler",
    "normal",
    "--steps",
    "4",
    "--cfg",
    "5",
    "--width",
    "64",
    "--height",
    "64",
]
ROOT = Path(__file__).resolve().parent.parent


def use(monkeypatch, rig, **kw) -> None:
    engine = rig.engine(**kw)
    monkeypatch.setattr(cli, "_engine", lambda: engine)


def gen(*extra: str):
    return runner.invoke(
        app, ["generate", "a cat", "--checkpoint", "base.safetensors", "--seed", "3", *SAMPLING, *extra]
    )


def test_the_committed_schema_is_current() -> None:
    """The check CI runs: `argus-darkroom schema --check` against the committed file."""
    result = runner.invoke(app, ["schema", "--check", "--output", str(ROOT / "schema" / "darkroom-wire.schema.json")])
    assert result.exit_code == 0, result.output


def test_a_stale_schema_fails_the_check(tmp_path) -> None:
    stale = tmp_path / "s.json"
    stale.write_text("{}")
    result = runner.invoke(app, ["schema", "--check", "--output", str(stale)])
    assert result.exit_code == 1 and "stale" in result.output
    assert runner.invoke(app, ["schema", "--output", str(stale)]).exit_code == 0
    assert runner.invoke(app, ["schema", "--check", "--output", str(stale)]).exit_code == 0


def test_version() -> None:
    assert runner.invoke(app, ["version"]).stdout.strip()


def test_preflight_prints_the_verdict(monkeypatch, rig) -> None:
    use(monkeypatch, rig)
    result = runner.invoke(app, ["preflight", "a cat", "-n", "blurry", "--trigger-word", "sks"])
    assert result.exit_code == 0
    body = json.loads(result.stdout)
    assert body["verdict"] == "proceed" and body["final_prompt"] == "a cat, sks"


def test_preflight_exits_2_when_refused(monkeypatch, rig) -> None:
    rig.singguard.responder = answers(preflight=unsafe_answer("B. Real-World Crimes & Public Safety"))
    use(monkeypatch, rig)
    result = runner.invoke(app, ["preflight", "a cat"])
    assert result.exit_code == 2 and json.loads(result.stdout)["reason"] == "default:B"


def test_preflight_rejects_an_nsfw_policy_without_the_flag(monkeypatch, rig) -> None:
    use(monkeypatch, rig)
    result = runner.invoke(app, ["preflight", "a cat", "--policy", "adult-granular"])
    assert result.exit_code == 1 and "allow-nsfw" in result.output


def test_generate_releases_an_image_and_says_where(monkeypatch, rig) -> None:
    use(monkeypatch, rig)
    result = gen("--lora", "style.safetensors:0.6", "--trigger", "sks")
    assert result.exit_code == 0, result.output
    assert "outcome  passed" in result.stdout
    (png,) = rig.out.glob("*.png")
    assert str(png) in result.stdout and png.read_bytes() == rig.comfyui.png
    assert "step 2/2" in result.stderr and "pre-flight: proceed" in result.stderr


def test_generate_json_is_the_whole_record(monkeypatch, rig) -> None:
    use(monkeypatch, rig)
    body = json.loads(gen("--lora", "style.safetensors", "--json").stdout)
    assert body["outcome"] == "passed" and body["request"]["seed"] == 3
    assert body["request"]["loras"][0]["weight"] == 1.0


def test_generate_exits_2_and_writes_nothing_when_not_released(monkeypatch, rig) -> None:
    rig.nudenet.labels = {"FEMALE_BREAST_EXPOSED": 0.9}
    use(monkeypatch, rig)
    result = gen("--lora", "style.safetensors")
    assert result.exit_code == 2 and "omitted (sfw-exposed)" in result.stdout
    assert list(rig.out.glob("*.png")) == []


def test_generate_reports_a_request_error(monkeypatch, rig) -> None:
    use(monkeypatch, rig)
    result = runner.invoke(
        app, ["generate", "a cat", "--checkpoint", "gone.safetensors", *SAMPLING, "--lora", "style.safetensors"]
    )
    assert result.exit_code == 1 and "not found" in result.output
    assert rig.comfyui.calls == []


def test_the_sampling_options_have_no_invented_defaults() -> None:
    result = runner.invoke(app, ["generate", "a cat", "--checkpoint", "x"])
    assert result.exit_code == 2 and "--sampler" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--lora", "a:heavy"], "weight"),
        (["--lora", "a", "--trigger", "b=word"], "not one of the --lora"),
        (["--trigger", "word"], "NAME=words"),
    ],
)
def test_lora_and_trigger_options_are_checked(args, message) -> None:
    result = gen(*args)
    assert result.exit_code == 2 and message in result.output


def test_serve_refuses_to_start_without_singguard(monkeypatch) -> None:
    monkeypatch.delenv("ARGUS_DARKROOM_SINGGUARD_URL", raising=False)
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 1 and "unguarded" in result.output


def test_serve_maps_its_flags_to_the_environment_and_binds_to_localhost(monkeypatch, tmp_path, singguard) -> None:
    import uvicorn

    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda application, **kw: started.update(app=application, **kw))
    monkeypatch.setenv("ARGUS_DARKROOM_SINGGUARD_URL", singguard.url)
    for name in ("POLICY", "ALLOW_NSFW", "OUTPUT_ROOT", "CORS_ORIGINS"):
        monkeypatch.delenv(f"ARGUS_DARKROOM_{name}", raising=False)
    result = runner.invoke(
        app, ["serve", "--allow-nsfw", "--output-root", str(tmp_path / "o"), "--cors-origin", "https://s.example"]
    )
    assert result.exit_code == 0, result.output
    assert (started["host"], started["port"]) == ("127.0.0.1", 8106)
    settings = started["app"].state.engine.settings
    assert settings.allow_nsfw and settings.policy == "adult-granular"
    assert settings.output_root == tmp_path / "o" and settings.cors_origins == ("https://s.example",)


def test_expand_flag_expands_the_prompt(monkeypatch, rig, ollama) -> None:
    from dataclasses import replace

    rig.settings = replace(rig.settings, ollama_url=ollama.url, ollama_model="tiny:1b")
    use(monkeypatch, rig)
    result = gen("--expand", "--lora", "style.safetensors")
    assert result.exit_code == 0, result.output
    assert len(ollama.requests) == 1
    assert "a cat, soft light, detailed fur" in rig.comfyui.graphs[0]["6"]["inputs"]["text"]


def test_expand_without_ollama_is_a_usage_error(monkeypatch, rig) -> None:
    use(monkeypatch, rig)
    result = gen("--expand", "--lora", "style.safetensors")
    assert result.exit_code != 0 and "not configured" in result.output
