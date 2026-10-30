# argus-darkroom

Guarded image generation for the Argus suite. A prompt is checked before anything is generated. Each output is checked
twice, once on its own and once alongside the prompt that made it, and only then released or destroyed. A record is kept
for every attempt.

ComfyUI generates, Ollama optionally expands the prompt, SingGuard and NudeNet judge. Darkroom orchestrates them, decides
and keeps the record. Its demo UI is `argus-safelight`. Part of the
[Argus suite](https://github.com/smk762?tab=repositories&q=argus). The design is in [`spec.md`](spec.md).

```
safelight :3000 ──▶ argus-darkroom :8106 ──▶ ComfyUI :8188
                     record store            Ollama :11434 (optional)
                     output root             SingGuard :9150
                                             NudeNet
```

## How a generation runs

1. **Assemble** the final prompt: your text (or its Ollama expansion), then the LoRA trigger words.
2. **Pre-flight**: SingGuard reads the final prompt and the negative prompt, framed together, under every selected
   policy. An illegal rule refuses. A failed check refuses. Nothing has been rendered yet.
3. **Render** through ComfyUI. The image comes back over the websocket and is held in memory only.
4. **Scan**: per policy, SingGuard judges the image alone and with the prompt. NudeNet scores exposure. If the age check
   is installed, MiVOLO estimates an age per face.
5. **Decide and release or destroy.** `passed` and `review` images are written to the output root. `deleted` and
   `omitted` images are never written.

| Outcome | Means |
|---|---|
| `refused` | pre-flight stopped it; nothing was rendered |
| `passed` | every cell safe |
| `review` | some cell flagged; the image is kept and served behind its shield |
| `deleted` | an illegal rule in any policy, `minor-exposed`, or an enforcing check that could not run |
| `omitted` | SFW only: NudeNet saw exposure, or did not answer |
| `failed` | expansion or generation failed; no image |

Nothing is written before the verdict: no temp file, no `SaveImage` node. A workflow containing a node that saves,
previews, exports or writes is refused when it is loaded.

**The age estimate is a tool's guess**, not a fact. It is stored as `age_estimate_years` from `mivolo`, and an
under-18 estimate with exposure deletes the image. The age check is an extra signal on top of SingGuard's minors rule.

## Run it

You need the services darkroom reaches by URL:

- **ComfyUI**, with the `SaveImageWebsocket` node (below).
- **SingGuard** (agent-composer's `/api/guard/check`). **Required**: darkroom never runs unguarded, and an empty
  `ARGUS_DARKROOM_SINGGUARD_URL` makes it refuse to start.
- **NudeNet**. Without it, SFW generation is unavailable (HTTP 400) because every SFW output would be omitted.
- **Ollama**, optional, for `--expand`.

### Compose

```bash
cp .env.example .env     # set SINGGUARD_URL and MODELS_ROOT at least
docker compose up --build
```

This runs darkroom (`:8106`) and safelight (`:3000`), both published on `127.0.0.1` only (`DARKROOM_PUBLISH` /
`SAFELIGHT_PUBLISH` in `.env`). Safelight is built from
`SAFELIGHT_CONTEXT` (default `../argus-safelight`); its darkroom URL is baked in at build time from
`DARKROOM_PUBLIC_URL`. Records and images live in the `darkroom-data` volume. `MODELS_ROOT` is mounted read-only.

`DEVMODE=1` in `.env` becomes `ARGUS_DARKROOM_DEVMODE=1`. It adds `POST /purge` (and safelight's purge button), which
deletes every record and image. Leave it off anywhere you care about the output.

### Without Docker

```bash
uv pip install "argus-darkroom[server,cli]"
export ARGUS_DARKROOM_SINGGUARD_URL=http://localhost:9150
export ARGUS_DARKROOM_MODELS_ROOT=/path/to/ComfyUI/models
export ARGUS_DARKROOM_WORKFLOW=examples/workflows/sdxl.json
argus-darkroom serve            # 127.0.0.1:8106
```

The server binds to localhost by default. It is not meant for public hosting.

### ComfyUI: the `SaveImageWebsocket` node

Darkroom takes the image from the websocket instead of from disk, so ComfyUI needs a `SaveImageWebsocket` node. It
ships with ComfyUI as an example node, `websocket_image_save.py` under `custom_nodes/`. Confirm that ComfyUI knows it:

```bash
curl -s localhost:8188/object_info/SaveImageWebsocket | head -c 200
```

An empty `{}` means the node is not installed. The node does not write to disk, which is why darkroom can use it.

### The workflow template

A ComfyUI API-format workflow with `$placeholders`: `$base_checkpoint`, `$vae`, `$positive`, `$negative`, `$seed`,
`$steps`, `$cfg`, `$sampler`, `$scheduler`, `$width`, `$height`, `$clip_skip`, and `$lora` / `$lora_weight`,
`$lora_2` / `$lora_2_weight` and so on. It must contain exactly one `SaveImageWebsocket` node and no other node that
saves. See [`examples/workflows/`](examples/workflows).

## Configuration

All settings are `ARGUS_DARKROOM_*` environment variables.

| Variable | Default | |
|---|---|---|
| `SINGGUARD_URL` | | **Required.** |
| `COMFYUI_URL` | `http://localhost:8188` | |
| `NUDENET_URL` | | Empty: SFW generation unavailable. |
| `OLLAMA_URL`, `OLLAMA_MODEL` | | Both set: `expand` is available. |
| `POLICY` | `default`, or `adult-granular` with `ALLOW_NSFW` | The enforcing policy when a request names none. |
| `THINKING` | `cascade` | `fast`, `fast-slow`, `slow`, `cascade`. |
| `ALLOW_NSFW` | off | Lets an NSFW policy enforce. |
| `OUTPUT_ROOT` | `darkroom-output` | `<id>.json` records and `<id>.png` images. |
| `MODELS_ROOT` | | ComfyUI's models folder: checkpoints and LoRAs are listed and hashed from it. |
| `WORKFLOW` | | The template path. |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | May read and write. `*` allows reads only. |
| `DEVMODE` | off | Adds `POST /purge`. |

`ALLOW_NSFW` and `DEVMODE` accept `1`/`true`/`yes`/`on`. A mistyped value is warned about, not silently ignored.

Policies are chosen by name (`default`, `adult`, `adult-granular`, `adult-cautious`); a request cannot carry policy
text. `default` and `sfw-*` are SFW, the rest NSFW. SFW refuses on any pre-flight flag and omits exposed output. NSFW
warns on non-illegal flags and blurs exposed output, and illegal content is refused and deleted in both.

## CLI

```bash
argus-darkroom preflight "a cat on a sofa" -n "blurry"
argus-darkroom generate "a cat on a sofa" --checkpoint base.safetensors \
    --sampler euler --scheduler normal --steps 20 --cfg 5 --width 1024 --height 1024 \
    --lora style.safetensors:0.8 --trigger sks --expand
argus-darkroom serve
argus-darkroom schema --check
```

`generate` exits `0` for a released image (`passed` or `review`), `2` for anything else, and `1` for a usage or
configuration error. Sampling options are required: darkroom does not invent them.

`--expand` has Ollama rewrite the prompt. The expansion replaces your text (trigger words are appended) and is checked
like anything else, so it cannot route around the gate. If Ollama fails, the attempt is a `failed` record, never a
silent fall back to the unexpanded prompt.

## HTTP API

| | |
|---|---|
| `GET /health` | version, whether each backend answers (booleans), `nsfw_allowed`, `sfw_available`, `devmode`, age-check status |
| `GET /policies` | each policy's name, modes, `enforceable` flag and rule names (`A`, `B`, ...) |
| `GET /models` | checkpoints and LoRAs as `{name}` objects, plus the samplers and schedulers ComfyUI offers |
| `POST /preflight` | the text check alone. It does not expand: an expansion differs per run, so `expansion_applied` is `false` |
| `POST /generate/stream` | SSE: `preflight`, `queued`, `progress {step,total}`, `scan {policy,context,cell}`, `complete {record}`, `error {detail}`. One at a time; a second gets 409 |
| `GET /records`, `GET /record/{id}` | `/records` takes `page`, `page_size`, `outcome` and `mode` |
| `GET /image/{id}` | 404 for a deleted, omitted or refused record |
| `DELETE /record/{id}` | removes the user's own image; the record stays |
| `GET /archive` | streamed zip of every record and stored image |
| `POST /purge` | `{"confirm": "purge"}`; only with `DEVMODE` |

Writes are refused from origins that are not in `CORS_ORIGINS`.

Every record under an NSFW policy carries a `guard.nudenet` block (`exposed_score`, `labels`, `threshold`, `blur`);
`blur` is the only place the blur decision lives. `exposed_score` is `null` when NudeNet could not be reached, and the
image is then blurred. Each LoRA in `params.loras` keeps the `trigger_words` the request supplied.

## Demo (no GPU)

`demo/run.py` runs the real server and engine against the same fake SingGuard, NudeNet, ComfyUI and Ollama servers the
tests use, so a reviewer can click through every outcome. A marker in the prompt chooses what the fakes say:

```bash
make demo          # http://127.0.0.1:8106, history in ./demo-data
make smoke         # in another shell: 16 prompts, each checked against the outcome it should reach
```

| Marker | Fakes say | Outcome |
|---|---|---|
| none | everything safe | `passed` |
| `[flag]` / `[flag-both]` | the image (or image and prompt) trips rule B | `review` |
| `[unchecked]` | the scan cannot be read | `deleted` (`unchecked`) |
| `[exposed]` | NudeNet reports exposure | SFW: `omitted`; NSFW: `passed`, blurred |
| `[nonude]` | NudeNet is down | SFW: `omitted`; NSFW: `passed`, blurred |
| `[noage]` | MiVOLO has no estimate | `passed` |
| `[minor]` | exposure and an age under 18 | `deleted` (`minor-exposed`) |
| `[illegal]` | the illegal rule fires | `deleted` |
| `[refuse]`, `[refuse-illegal]` | pre-flight flags | `refused` |
| `[preflight-unchecked]` | pre-flight cannot be read | `refused` (`unchecked`) |
| `[fail]` | ComfyUI errors in `KSampler` | `failed` |

`python demo/run.py --help` lists `--host`, `--port`, `--cors-origin`, `--workflow` (use `sdxl_lora.json` to try LoRAs;
the default `sdxl.json` has no LoRA slot, so a request with one is a 400), `--latency`, `--step-delay` and `--fresh`.
The demo turns `DEVMODE` and `ALLOW_NSFW` on and uses fake models. Bind it to a non-loopback host only on a network you
trust. No real content is involved; the verdicts are canned.

[argus-safelight](https://github.com/smk762/argus-safelight) is the UI for it: point `NEXT_PUBLIC_DARKROOM_URL` at the
demo.

## Age check (optional)

```bash
uv pip install "argus-darkroom[age]"
```

This runs curator's face pipeline in process and needs the MiVOLO ONNX file (`argus-curator mivolo-convert`). Without
it records carry `age: null` and `/health` says why. The Docker image leaves it out.

## Wire schema

Records are versioned (`record_version`) and the schema is committed at
[`schema/darkroom-wire.schema.json`](schema/darkroom-wire.schema.json) for consumers to generate code from.
`argus-darkroom schema` rewrites it; `schema --check` fails if it is stale. Change a model, regenerate, commit both.

## Develop

```bash
make install   # venv + editable install with the "dev" extras
make test
make lint
argus-darkroom schema --check
```

Tests run against fake SingGuard, NudeNet, ComfyUI and Ollama servers. No real sexual, violent or illegal content goes in
the repo or its tests: the illegal-to-deleted path is driven by a fake verdict.
[`scripts/preflight_eval.py`](scripts/preflight_eval.py) scores the pre-flight framing against a real SingGuard server on
the labelled prompts in `eval/`; run it by hand, not in CI.

## CI / Release

- **CI** runs via the shared [`argus-ci`](https://github.com/smk762/argus-ci) reusable workflow, plus a `schema --check`
  job.
- **Release** publishes to PyPI (OIDC trusted publishing) and GHCR on `v*` tags.
- Versioning is derived from git tags via `hatch-vcs`: tag `vX.Y.Z` to cut a release.

This repo was scaffolded from [`argus-pkg-template`](https://github.com/smk762/argus-pkg-template).
Run `copier update` to pull template changes (CI, release, tooling).
