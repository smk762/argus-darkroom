"""The ``argus-darkroom`` command line: ``generate``, ``preflight``, ``serve`` and ``schema``.

Configuration is the same ``ARGUS_DARKROOM_*`` environment the server reads. The CLI runs the same engine, so a
generation from here is guarded exactly as one from the server is, and leaves the same record.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from typer import Argument, Option

from argus_darkroom import __version__
from argus_darkroom.config import PREFIX, ConfigError, Settings
from argus_darkroom.request import GenerateRequest, LoRASpec, RequestError

app = typer.Typer(help="Guarded image generation: prompts and outputs are checked before anything is released.")

DEFAULT_SCHEMA_PATH = Path("schema/darkroom-wire.schema.json")
# What a run exits with, so a script can tell a released image from one that was not.
EXIT_NOT_RELEASED = 2  # refused, deleted, omitted or failed
EXIT_USAGE = 1


def _engine():
    from argus_darkroom.engine import Engine

    try:
        return Engine(Settings.from_env())
    except ConfigError as exc:
        typer.echo(f"configuration error: {exc}", err=True)
        raise typer.Exit(EXIT_USAGE) from exc


def _parse_loras(specs: list[str], triggers: list[str]) -> list[LoRASpec]:
    """``NAME`` or ``NAME:WEIGHT`` per ``--lora``. Trigger words belong to the first LoRA unless there are
    several, in which case each ``--trigger`` is ``NAME=word`` (or ``NAME=word,word``)."""
    loras: list[LoRASpec] = []
    for spec in specs:
        name, _, weight = spec.rpartition(":") if ":" in spec else (spec, "", "")
        try:
            loras.append(LoRASpec(name=name, weight=float(weight) if weight else 1.0))
        except ValueError as exc:
            raise typer.BadParameter(f"--lora {spec!r}: the weight after ':' must be a number") from exc
    for trigger in triggers:
        name, sep, words = trigger.partition("=")
        if not sep:
            if len(loras) != 1:
                raise typer.BadParameter("--trigger needs NAME=words when there is not exactly one --lora")
            name, words = loras[0].name, trigger
        target = next((lora for lora in loras if lora.name == name), None)
        if target is None:
            raise typer.BadParameter(f"--trigger names {name!r}, which is not one of the --lora options")
        target.trigger_words += [w.strip() for w in words.split(",") if w.strip()]
    return loras


def _selection(policy: str | None, advisory: list[str], thinking: str | None) -> dict:
    return {"enforcing_policy": policy, "advisory_policies": advisory, "thinking": thinking}


@app.command()
def preflight(
    prompt: str = Argument(..., help="The prompt to check"),
    negative: str = Option("", "--negative", "-n", help="Negative prompt: always part of the check"),
    policy: str | None = Option(None, "--policy", help="Enforcing policy (default: the server default)"),
    advisory: list[str] = Option([], "--advisory", help="Extra policy to run for comparison (repeatable)"),
    thinking: str | None = Option(None, "--thinking", help="fast, fast-slow, slow or cascade"),
    trigger: list[str] = Option([], "--trigger-word", help="LoRA trigger word added to the final prompt"),
) -> None:
    """Run the text-only pre-flight check alone. Exits 2 when the prompt would be refused."""
    request = GenerateRequest(
        user_prompt=prompt,
        negative_prompt=negative,
        loras=[LoRASpec(name="-", trigger_words=trigger)] if trigger else [],
        **_selection(policy, advisory, thinking),
    )
    engine = _engine()
    try:
        result = engine.preflight(request)
    except RequestError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    typer.echo(json.dumps(engine.preflight_view(request, result), indent=2))
    if result.refused:
        raise typer.Exit(EXIT_NOT_RELEASED)


@app.command()
def generate(
    prompt: str = Argument(..., help="The prompt"),
    checkpoint: str = Option(..., "--checkpoint", help="Checkpoint name under <models root>/checkpoints"),
    sampler: str = Option(..., "--sampler"),
    scheduler: str = Option(..., "--scheduler"),
    steps: int = Option(..., "--steps"),
    cfg: float = Option(..., "--cfg"),
    width: int = Option(..., "--width"),
    height: int = Option(..., "--height"),
    clip_skip: int = Option(1, "--clip-skip"),
    expand: bool = Option(False, "--expand", help="Expand the prompt with Ollama first (the expansion is checked too)"),
    negative: str = Option("", "--negative", "-n", help="Negative prompt: always part of the checks"),
    seed: int | None = Option(None, "--seed", help="Fixed seed (default: random, recorded)"),
    vae: str | None = Option(None, "--vae"),
    lora: list[str] = Option([], "--lora", help="NAME or NAME:WEIGHT (repeatable)"),
    trigger: list[str] = Option([], "--trigger", help="Trigger words: WORDS, or NAME=WORDS with several LoRAs"),
    policy: str | None = Option(None, "--policy", help="Enforcing policy (default: the server default)"),
    advisory: list[str] = Option([], "--advisory", help="Extra policy to run for comparison (repeatable)"),
    thinking: str | None = Option(None, "--thinking", help="fast, fast-slow, slow or cascade"),
    nsfw_threshold: float = Option(0.2, "--nsfw-threshold", help="NudeNet score at which an NSFW image is blurred"),
    as_json: bool = Option(False, "--json", help="Print the whole record instead of a summary"),
) -> None:
    """Generate one image through every guard. Exits 0 for a released image (passed or review), 2 otherwise.

    The image is written under the output root only when it is released; the record is kept either way.
    """
    request = GenerateRequest(
        user_prompt=prompt,
        negative_prompt=negative,
        expand=expand,
        seed=seed,
        sampling={
            "sampler": sampler,
            "scheduler": scheduler,
            "steps": steps,
            "cfg": cfg,
            "clip_skip": clip_skip,
            "width": width,
            "height": height,
        },
        checkpoint=checkpoint,
        vae=vae,
        loras=_parse_loras(lora, trigger),
        nsfw_threshold=nsfw_threshold,
        **_selection(policy, advisory, thinking),
    )

    engine = _engine()

    def progress(kind: str, data: dict) -> None:
        if kind == "progress":
            typer.echo(f"step {data['step']}/{data['total']}", err=True)
        elif kind == "preflight":
            typer.echo(f"pre-flight: {data['outcome']}", err=True)
        elif kind == "scan":
            typer.echo(f"scan {data['policy']} {data['context']}: {data['cell']['state']}", err=True)

    try:
        record = engine.generate(request, progress)
    except RequestError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_USAGE) from exc

    if as_json:
        typer.echo(record.model_dump_json(indent=2))
    else:
        released = record.image is not None and record.image.path is not None
        typer.echo(f"record   {record.id}")
        typer.echo(f"outcome  {record.outcome}" + (f" ({record.outcome_reason})" if record.outcome_reason else ""))
        if released:
            typer.echo(f"image    {engine.store.root / record.image.path}")
            if record.guard.nudenet is not None and record.guard.nudenet.blur:
                typer.echo("blur     the image is marked to be blurred until revealed")
    if record.outcome not in ("passed", "review"):
        raise typer.Exit(EXIT_NOT_RELEASED)


@app.command()
def serve(
    port: int = Option(8106, "--port", "-p", help="Port to listen on"),
    host: str = Option("127.0.0.1", "--host", help="Host to bind to (localhost by default: not for public hosting)"),
    output_root: Path | None = Option(None, "--output-root", help=f"Where records and images go ({PREFIX}OUTPUT_ROOT)"),
    allow_nsfw: bool = Option(False, "--allow-nsfw", help=f"Let an NSFW policy enforce ({PREFIX}ALLOW_NSFW=1)"),
    cors_origin: list[str] = Option(
        [], "--cors-origin", help=f"Origin allowed to read and write (or {PREFIX}CORS_ORIGINS)"
    ),
) -> None:
    """Start the darkroom server. Refuses to start without a SingGuard URL: darkroom never runs unguarded."""
    try:
        import uvicorn

        from argus_darkroom.server import create_app
    except ImportError as exc:
        typer.echo("The server needs: pip install argus-darkroom[server]", err=True)
        raise typer.Exit(EXIT_USAGE) from exc

    if allow_nsfw:
        os.environ[PREFIX + "ALLOW_NSFW"] = "1"
    if output_root is not None:
        os.environ[PREFIX + "OUTPUT_ROOT"] = str(output_root)
    if cors_origin:
        os.environ[PREFIX + "CORS_ORIGINS"] = ",".join(cors_origin)
    try:
        application = create_app()
    except ConfigError as exc:
        typer.echo(f"configuration error: {exc}", err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    uvicorn.run(application, host=host, port=port)


@app.command()
def schema(
    output: Path = Option(DEFAULT_SCHEMA_PATH, "--output", "-o", help="Where to write the JSON Schema"),
    check: bool = Option(False, "--check", help="Exit non-zero if the committed schema is stale (for CI)"),
) -> None:
    """Emit the wire-contract JSON Schema consumers codegen against."""
    from argus_darkroom.models import render_wire_schema

    rendered = render_wire_schema()
    if check:
        existing = output.read_text(encoding="utf-8") if output.exists() else ""
        if existing != rendered:
            typer.echo(f"{output} is stale: run `argus-darkroom schema` and commit the result.", err=True)
            raise typer.Exit(1)
        typer.echo(f"{output} is up to date.")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    typer.echo(f"Wrote wire schema -> {output}")


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
