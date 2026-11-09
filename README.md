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
| `failed` | expansion or generation failed, or the render came back blank; no image |

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
export ARGUS_DARKROOM_WORKFLOW=examples/workflows   # a directory: one template per family and LoRA count
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

### Workflow templates

A template is a ComfyUI API-format workflow with `$placeholders`: `$base_checkpoint`, `$vae`, `$positive`, `$negative`,
`$seed`, `$steps`, `$cfg`, `$guidance`, `$sampler`, `$scheduler`, `$width`, `$height`, `$clip_skip`, and `$lora` /
`$lora_weight`, `$lora_2` / `$lora_2_weight` and so on. It must contain exactly one `SaveImageWebsocket` node and no
other node that saves, and a `$positive` slot.

`WORKFLOW` names a file or a directory:

- **A directory** holds one template per checkpoint family and LoRA count, named `<family>.json` (no LoRA slots) or
  `<family>+lora<N>.json` (exactly N). A checkpoint's family is its top-level folder under `checkpoints/`, so
  `sdxl/juggernaut.safetensors` with two LoRAs renders with `sdxl+lora2.json`. When nothing fits, including a
  checkpoint outside a family folder, the request is a 400. Darkroom never falls back to another family's template or
  LoRA count. A misnamed or broken file stops the server from starting.
- **A file** is used for every checkpoint, as before. Every run must then fill its LoRA slots exactly.

A template never silently drops a value the request sets. A request is a 400 if it sets something the chosen template
has no slot for: a negative prompt (Flux takes none), `guidance`, or a `clip_skip` above 2. `/models` lists each
template's `slots`, and each checkpoint's `family` and the LoRA counts it can render with (`lora_counts`, empty when
none). Records name the template in `request.workflow`, beside `workflow_sha256`.

[`examples/workflows/`](examples/workflows) ships these. They are the image's default directory:

| Family | Files | Sampling |
|---|---|---|
| `sdxl` | `sdxl`, `sdxl+lora1`, `sdxl+lora2` | the usual: a negative prompt, `cfg` around 5, `clip_skip` |
| `flux` | `flux`, `flux+lora1`, `flux+lora2` | `guidance` (around 3.5 for dev), `cfg` 1, `simple` or `beta`, no negative prompt. Schnell: about 4 steps |
| `flux-klein` | `flux-klein` | FLUX.2 Klein 9B (distilled): `cfg` 1, about 4 steps, no negative prompt |
| `sd35` | `sd35` | SD3.5 Large: a negative prompt, `cfg` around 4, `euler` with `sgm_uniform` |

The Flux files under `checkpoints/` hold only the diffusion model, so `flux*.json` and `flux-klein.json` load the text
encoders and VAE by name: `t5xxl_fp8_e4m3fn` and `clip_l` with `ae` for Flux, and `qwen_3_8b_fp8mixed` with
`flux2-vae` for Klein. Edit those names to match your ComfyUI. The record pins them through `workflow_sha256`, not
by their own hash. The non-SDXL families need more VRAM than SDXL and may run out of memory on smaller GPUs.

`$clip_skip` is read as A1111 reads it for SDXL: `clip_skip` 1 and 2 both send `stop_at_clip_layer: -2`, the layer SDXL
conditions on, and `n` above 2 sends `-n`. A request may ask for 1 to 12.

## Configuration

All settings are `ARGUS_DARKROOM_*` environment variables.

| Variable | Default | |
|---|---|---|
| `SINGGUARD_URL` | | **Required.** |
| `COMFYUI_URL` | `http://localhost:8188` | |
| `NUDENET_URL` | | Empty: SFW generation unavailable. |
| `AGE_URL` | | Curator's `POST /faces?body=true`. Empty: the in-process `[age]` extra, or no age check. |
| `OLLAMA_URL`, `OLLAMA_MODEL` | | Both set: `expand` is available. |
| `POLICY` | `default`, or `adult-granular` with `ALLOW_NSFW` | The enforcing policy when a request names none. |
| `THINKING` | `cascade` | `fast`, `fast-slow`, `slow`, `cascade`. |
| `ALLOW_NSFW` | off | Lets an NSFW policy enforce. |
| `OUTPUT_ROOT` | `darkroom-output` | `<id>.json` records and `<id>.png` images. |
| `MODELS_ROOT` | | ComfyUI's models folder: checkpoints and LoRAs are listed and hashed from it. |
| `DEFAULT_CHECKPOINT` | `sdxl` | The checkpoint a form starts on: an exact name, or a folder under `checkpoints/` for the first model in it. When nothing matches, `/models` reports null. |
| `WORKFLOW` | | A template file, used for every checkpoint, or a directory of them chosen per request (see [Workflow templates](#workflow-templates)). The image defaults to `/app/examples/workflows`. |
| `CORS_ORIGINS` | `localhost` and `127.0.0.1` on ports 3000 and 3097 | May read and write. `*` allows reads only. Compose defaults to port 3000 only. |
| `DEVMODE` | off | Adds `POST /purge` and `POST /purge/{id}`. |
| `READ_ONLY` | off | Answers 403 to every write from every origin: `POST /preflight`, `POST /generate/stream`, `DELETE /record/{id}` and both purges. Records and images stay readable. |

`ALLOW_NSFW`, `DEVMODE` and `READ_ONLY` accept `1`/`true`/`yes`/`on`. A mistyped value is warned about, not silently ignored.

Policies are chosen by name (`default`, `adult`, `adult-granular`, `adult-cautious`, `adult-cinematic`); a request cannot carry policy
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
| `GET /health` | version, whether each backend answers (booleans), `nsfw_allowed`, `sfw_available`, `devmode`, `read_only`, age-check status, `workflows` (names) |
| `GET /policies` | each policy's name, modes, `enforceable` flag and rule names (`A`, `B`, ...) |
| `GET /models` | checkpoints as `{name, family, lora_counts}`, LoRAs as `{name}`, `workflows` as `{name, family, loras, slots}`, `default_checkpoint`, plus the samplers and schedulers ComfyUI offers |
| `POST /preflight` | the text check alone. It does not expand: an expansion differs per run, so `expansion_applied` is `false` |
| `POST /generate/stream` | SSE: `preflight`, `queued`, `progress {step,total}`, `scan {policy,context,cell}`, `complete {record}`, `error {detail}`. One at a time; a second gets 409 |
| `GET /records`, `GET /record/{id}` | `/records` takes `page`, `page_size`, `outcome` and `mode` |
| `GET /image/{id}` | 404 for a deleted, omitted or refused record |
| `DELETE /record/{id}` | removes the user's own image; the record stays |
| `GET /archive` | streamed zip of every record and stored image |
| `POST /purge` | `{"confirm": "purge"}`; only with `DEVMODE` |
| `POST /purge/{id}` | the same body: deletes one record and its image; only with `DEVMODE` |

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

`python demo/run.py --help` lists `--host`, `--port`, `--cors-origin`, `--workflow` (by default the whole examples
directory, with fake `sdxl/` and `flux/` checkpoints and two LoRAs; name one file to use it for every checkpoint),
`--latency`, `--step-delay` and `--fresh`.
The demo turns `DEVMODE` and `ALLOW_NSFW` on and uses fake models. Bind it to a non-loopback host only on a network you
trust. No real content is involved; the verdicts are canned.

[argus-safelight](https://github.com/smk762/argus-safelight) is the UI for it: point `NEXT_PUBLIC_DARKROOM_URL` at the
demo.

## Age check (optional)

The age check is an extra signal on top of SingGuard's minors rule, never a replacement for it. SingGuard judges every
image for itself, with or without an estimate: rule A of each adult policy covers anyone who appears to be a minor. So
when the age check is missing, that rule still applies.

The estimate is a tool's reading, and it is never sent to SingGuard. It has one effect of its own: an estimate under 18
on an image where NudeNet scores an exposed class at 0.2 or more deletes the image (`minor-exposed`). Without exposure
it is only recorded and shown.

Set `AGE_URL` to a curator 0.10+ that serves `POST /faces?body=true` (InsightFace + MiVOLO, a body crop per face, and
unmatched people). The Docker image is meant to be used this way. A failed `people_check` is treated as age unavailable,
not as "no one found", and so is an unmatched person with no estimate. An unavailable age check never deletes on its
own: the record says `age: null` and the image is judged by SingGuard and NudeNet.

```bash
uv pip install "argus-darkroom[age]"   # in-process fallback when AGE_URL is empty
```

The extra runs curator's face pipeline in this process and needs the MiVOLO ONNX file (`argus-curator mivolo-convert`).
Without a URL or the extra, records carry `age: null` and `/health` says why.

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
- **Release** publishes to PyPI (OIDC trusted publishing) and GHCR on `v*` tags, then creates the GitHub Release.
- Versioning is derived from git tags via `hatch-vcs`: tag `vX.Y.Z` to cut a release.
- Release notes are the version's section of `CHANGELOG.md`. Add a `## [X.Y.Z]` section before tagging: without one,
  the build fails and nothing is published.

This repo was scaffolded from [`argus-pkg-template`](https://github.com/smk762/argus-pkg-template).
Run `copier update` to pull template changes (CI, release, tooling).
