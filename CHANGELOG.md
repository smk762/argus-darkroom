# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `scripts/nudenet_compare.py` (#54), run by hand: scores darkroom's stored images (`--records`) and/or a hand-sorted
  `blur/`/`clear/` folder (`--labelled`) on an old and a new NudeNet server, and sweeps the new one's cutoff against
  the old 0.2 (`EXPOSED_FLOOR`, `DEFAULT_NSFW_THRESHOLD`). Reports names and scores, never image bytes.

- **The NudeNet detector is recorded with its score** (#54). `guard.nudenet` gains `detector` (`640m`, `320n`, or
  `null`) and `detector_model` (the weights file the server named), from cortex's `NudeNetClient.info()`, asked after
  each successful score, for outputs and input images alike. 640m and 320n score the same image differently, so
  records from before and after the suite's move to one shared 640m server (argus-cortex#79) can be told apart.
  Metadata only: a server that does not say, or whose `/health` fails, scores and decides exactly as before.
  `record_version` stays 1.0 and the wire schema is regenerated. `/health` gains `nudenet_detector`.

- **GPU leases from argus-iris** (#53). Each render and depth map holds a lease on `comfyui` and each prompt expansion
  one on `ollama:<model>`, through argus-cortex's coordinator (`ARGUS_GPU_COORDINATOR`, `ARGUS_GPU_BROKER_URL`,
  `ARGUS_GPU_BROKER_TOKEN`, `ARGUS_GPU_TENANT_MANIFEST`, `ARGUS_GPU_BROKER_FALLBACK`; unset gates nothing, as before).
  The work goes to the URL a grant names, else darkroom's own. The priority class comes from the caller's
  `X-Argus-Priority` (`interactive` by default; mint's sample runs send `batch`), or `generate --priority`.
- With the broker, up to 8 generations and depth runs are in flight at once, each waiting for its lease, so a second
  generation queues behind iris instead of getting a 409. Without it, one at a time as before. A purge still waits
  for none to run.
- `gpu-tenant.yaml`, validated by cortex's `TenantManifest`: kinds `generate` and `depth` on `comfyui`, `expand` on
  Ollama. The image ships it at `/app/gpu-tenant.yaml`, compose's default manifest path.
- `/health` gains `gpu` (`coordinator`, `in_flight`, `limit`).
- A refused lease, one not granted in time, or a broker error is a `failed` record that says so. Nothing retries it as
  an OOM.

- **Depth maps** (for argus-mask's plate backdrops). `POST /depth/{id}` makes a depth map of a record's stored image
  with Depth Anything V2 Small (Apache-2.0) and returns the record's new `depth`; `GET /depth/{id}` serves it as an
  8-bit greyscale PNG at the image's size, near bright and far dark, min to max, no metadata. Idempotent; shares the
  generation lock (409 while one runs); 404 for no such record, 409 when the record has no stored image, 502 when
  ComfyUI fails. A write, so cross-site and read-only refuse it. No pre-flight or scan: the image was already judged
  and released, and nothing new goes in.
- The record gains an optional `depth` (`model`, `model_license`, `sha256`, `width`, `height`, `created_at`, `workflow`,
  `engine_version`); `record_version` stays 1.0 and the wire schema is regenerated. The PNG is kept as
  `<id>.depth.png` beside the image, is in `/archive` as `depth/<id>.png`, and goes when the image is removed
  (`DELETE /record/{id}`, `depth` back to `null`) or purged.
- `comfyui/darkroom_depth.py` (`DarkroomDepthAnything`), a vendored ComfyUI node: copy it into `custom_nodes/` and
  download the model once into `models/depth/Depth-Anything-V2-Small-hf`. It loads only the Small model, from that
  local folder, and refuses Base and Large (CC-BY-NC-4.0). The depth graph is built into the package, never a
  generation template.

- `ARGUS_DARKROOM_MINOR_EXPOSED` picks how the `minor-exposed` rule places exposure: `frame` (the default, unchanged:
  any age estimate under 18 plus a NudeNet exposed score of 0.2 or more) or `overlap` (argus-curator's original rule:
  an exposed NudeNet detection deletes when its box overlaps a person, else face, box estimated under 18, or no box at
  all, and fails closed when a young face has no body or the exposure has no box). Any other value refuses to start.
  `/health` reports it as `minor_exposed`.
- The record's `guard.age` (and `guard.input.age`) gains `people`, every person box with its estimate and the face it
  was matched to, and `minor_exposed_rule`. Both are optional: older records load as `[]` and `"frame"`.
  `guard.nudenet` (and `guard.input.nudenet`) gains `detections`: every NudeNet detection's label, score and box
  (never pixels), so an `overlap` decision can be recomputed from the record. Older records load as `[]`.
  `record_version` stays 1.0; the wire schema is regenerated.
- The age estimators keep curator's matched person boxes (`HttpAgeEstimator.read`, `CuratorAgeEstimator.read`).
  `estimate` still returns the faces alone.

### Changed

- argus-cortex `>=0.7,<0.8` for `NudeNetClient.info()`, and argus-curator `>=0.17,<0.18`, the first curator on
  cortex 0.7 (#54).
- argus-cortex `>=0.6,<0.7`, with the `[remote]` (broker client) and `[yaml]` (manifest) extras.
- argus-curator `>=0.16,<0.17` for the `[age]` extra and the tests: 0.16 is the first curator on cortex 0.6
  (argus-curator#157). Curator 0.15 and later carry more face measurements; darkroom reads only the age estimates and
  boxes, so nothing else changes.
- Record ids and workflow placeholder and file names are checked with `fullmatch` (argus-cortex's `fullmatch`
  codemod, ai-code-auditor#56), so they match the whole string: `$` also matches before a final newline, so an id
  followed by `\n` used to pass `valid_id`.

### Fixed

- The spec said darkroom applied curator's `minor_exposed` rule; it applied the whole-frame rule. It now describes
  both rules and the default.

## [0.2.0] - 2026-11-14

### Added

- **Image-to-image.** A request may carry `init_image` (`data_base64`: PNG, JPEG or WebP, at most 16 MiB and 4096 px
  per edge; `denoise`: 0.05 to 1.0, default 0.5). The input gets the same checks as an output (SingGuard isolated and
  with the prompt under every policy, NudeNet, the age estimate) after the text pre-flight and before anything reaches
  ComfyUI, and goes on only if its outcome is `passed`. Anything else refuses the attempt (`refused`,
  `outcome_reason` `input image: <reason>`, `critical` as for an output). The SSE stream carries one `input_scan`
  event per input cell.
- The input is never written to disk: it travels to ComfyUI as base64 inside the graph, read by a vendored custom node,
  `comfyui/darkroom_load_image_base64.py` (`DarkroomLoadImageBase64`), and the graph is deleted from ComfyUI's history
  after the run. Darkroom keeps only its hashes: `request.init_image` and `guard.input` in the record (new optional
  fields; `record_version` stays 1.0).
- Image-to-image templates, named `<family>[+lora<N>]+img2img.json`, with `$init_image` and `$denoise`. Examples
  `sdxl+img2img` and `sdxl+lora1+img2img`. `/models` gives each checkpoint's `img2img_lora_counts` and each
  workflow's `img2img` flag.

- **Inpainting.** `init_image` may carry `mask_base64` (PNG, JPEG or WebP, the image's limits, exactly the input's
  size once upright; white repaints, black keeps). Only the masked region is repainted; the example templates composite
  the result over the scaled input, so every pixel outside the (8 px grown) mask is the input's. The input is judged
  exactly as for image-to-image. The mask is a region selector with no imagery, so it is not scanned; it is cut at 128
  into a 1-bit PNG, sent inside the graph like the input, and never written. A mask that does not decode, is the wrong
  size or masks nothing is a 400. Records gain `request.init_image.mask` (`sha256`, `source_sha256`,
  `masked_fraction`; additive, `record_version` stays 1.0).
- `DarkroomLoadMaskBase64`, in the same vendored file as `DarkroomLoadImageBase64`: copy the file into ComfyUI's
  `custom_nodes/` again and restart ComfyUI.
- Inpainting templates, named `<family>[+lora<N>]+inpaint.json`, with `$init_image`, `$init_mask` and `$denoise`;
  examples `sdxl+inpaint` and `sdxl+lora1+inpaint`. A mask with no inpainting template is a 400. `/models` gives each
  checkpoint's `inpaint_lora_counts` and each workflow's `mode` (`txt2img`, `img2img`, `inpaint`).

### Changed

- A workflow's `img2img` flag in `/models` is true for plain image-to-image only, not for inpainting.
  `WorkflowSet.select` and `lora_counts` take a `mode` instead of an `img2img` flag.
- A template with a node that loads an image from ComfyUI's disk (`LoadImage` and the like) is refused at load: an
  image reaches a run only as a checked input.
- **argus-cortex `>=0.4.6,<0.6`** (was `<0.5`). The Docker image and a plain `pip install argus-darkroom[server,cli]` get
  cortex 0.5. The `[age]` and `[dev]` extras stay on 0.4.x while argus-curator caps cortex below 0.5. darkroom uses
  nothing 0.5 changed, and the tests and `make smoke` pass on both 0.4.7 and 0.5.0.
- **Docs:** the README lists every CLI option and the real exit codes (a malformed option also exits 2), and gives the
  full `/policies` and `/models` shapes (#45).

### Fixed

- The ComfyUI websocket no longer sends keepalive pings. ComfyUI answers them on its event loop, which can stall for
  minutes while a large checkpoint loads under memory pressure, so a 20 s ping timeout failed SD3.5 runs that ComfyUI
  went on to finish (`ComfyUI websocket failed: sent 1011 (internal error) keepalive ping timeout`). The render
  deadline (600 s) still bounds a hung ComfyUI.

## [0.1.3] - 2026-11-09

### Added

- **A workflow per checkpoint family and LoRA count (#29).** `ARGUS_DARKROOM_WORKFLOW` may now name a directory of
  `<family>.json` and `<family>+lora<N>.json` templates. Each request gets the one for its checkpoint's top-level
  folder and number of LoRAs. When nothing fits, the request is a 400, never a substitute. A single file still serves
  every checkpoint. The Docker image now defaults to the examples directory, so LoRAs are optional and the Flux, Flux
  Klein and SD3.5 checkpoints can render.
- Example templates `sdxl+lora2`, `flux`, `flux+lora1`, `flux+lora2`, `flux-klein` and `sd35`.
- `sampling.guidance` fills a template's new `$guidance` slot (Flux's `FluxGuidance`). `request.workflow` in records
  names the template used.
- `/models` gives each checkpoint's `family` and `lora_counts`, and lists the `workflows` with their slots.
  `/health` lists the workflow names.

### Changed

- `$negative` is optional in a template. A request with a negative prompt, `guidance` or a `clip_skip` above 2 is a
  400 when the chosen template has no slot for it. Pre-flight still checks the negative prompt.
- `examples/workflows/sdxl_lora.json` is now `sdxl+lora1.json`. Its content and hash are unchanged.
- Under the Docker image's new default, a checkpoint renders only from a folder a template is named for (`sdxl/`,
  `flux/`, `flux-klein/`, `sd35/`). One at the top of `checkpoints/` or in another folder (`pony/`, say) is now a 400,
  where `sdxl.json` used to serve it. Set `ARGUS_DARKROOM_WORKFLOW` to a single file to keep one template for all.

## [0.1.2] - 2026-11-08

### Added

- **GitHub Releases.** Tagging a version now creates its GitHub Release once PyPI has it, with notes from this file.

### Changed

- **Safelight's dev origin is trusted by default.** `CORS_ORIGINS` now defaults to ports 3000 and 3097, the port
  safelight's `make dev` serves on. Compose still defaults to 3000 only.

### Fixed

- **SDXL renders use SDXL's own text-encoder layer.** `clip_skip` 1 sent `stop_at_clip_layer: -1`, the last CLIP
  layer, which SDXL never uses. Some checkpoints ship that layer as all NaN, so every render came back blank (#19).
  `clip_skip` is now read as A1111 reads it for SDXL: 1 and 2 both send -2, ComfyUI's default. Every SDXL render with
  `clip_skip` 1 changes, so rerunning an older record with its seed gives a different image.
- A request's `clip_skip` must be from 1 to 12. Records outside that range still load.
- `DELETE /record/{id}` can no longer bring back a record that a devmode purge deleted while the removal was running.

## [0.1.1] - 2026-11-07

### Added

- **`POST /purge/{id}`.** A devmode route that deletes one record and its image (#18).
- **`DEFAULT_CHECKPOINT`.** `GET /models` now names `default_checkpoint`, the checkpoint a form starts on, `sdxl` by
  default. It is null on an install with no `sdxl` folder (#21).

### Fixed

- A render that comes back as one exact colour is now recorded as `failed`, not `passed`, and is neither scanned nor
  kept. A dark scene is not failed for being dark (#19).
- `GET /models` no longer lists the parts of a diffusers folder as checkpoints (#20).
- `POST /purge/{id}` can remove a record that no longer validates, and clears files left by a crash.
- If removing an image fails, the record stays, so the image is not orphaned.
- `GET /models` no longer walks the models folder on the event loop.
- Compose publishes safelight's real port, 3097 rather than 3000 (#17). The demo trusts that origin too.

## [0.1.0] - 2026-11-06

First release of argus-darkroom: guarded image generation for the Argus suite. A prompt is checked before anything is
generated, each output is checked before it is released or destroyed, and a record is kept for every attempt.

### Added

- **Pre-flight.** SingGuard checks the final prompt, including any expansion and the negative prompt, before ComfyUI
  renders.
- **Output scans.** Each image is checked twice by SingGuard, once alone and once with its prompt, and scored by
  NudeNet. Nothing is written until the verdict, and a deleted or omitted image is never written.
- **Records.** Every attempt has a versioned JSON record, read through `/records`, `/record/{id}` and `/archive`.
- **Policies by name.** `default`, `adult`, `adult-granular`, `adult-cautious` and `adult-cinematic`, from argus-cortex.
  A request can't carry policy text.
- **Optional age check.** MiVOLO estimates from curator, over `AGE_URL` or in process with the `[age]` extra. It adds to
  SingGuard's minors rule and never replaces it. If the check is unavailable, the record shows `age: null` and the scan
  carries on.
- **Read-only mode.** `ARGUS_DARKROOM_READ_ONLY` answers 403 to every write. `/health` reports it as `read_only`.
- **HTTP API, CLI and Docker image.** The server binds to localhost by default and refuses to start without
  `SINGGUARD_URL`.
