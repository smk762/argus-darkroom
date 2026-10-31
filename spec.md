# argus-darkroom — build doc

Guarded image generation for the Argus suite. A prompt is checked before anything is generated. Each output is checked twice, once on its own and once alongside the prompt that made it, and only then released or destroyed. Generation is done by ComfyUI and prompt expansion by Ollama, both external; darkroom orchestrates them, judges the results and keeps the record. Its demo UI is [argus-safelight](../argus-safelight/spec.md). This doc is the build plan; the repo does not exist yet.

```
argus-quarry -> argus-curator -> argus-lens -> argus-forge -> trainer -> argus-proof
                                                                  │
                                                     validated LoRA│
                                                                  ▼
                                                           argus-darkroom
```

Darkroom sits off the main line, like argus-atlas: it uses what the line produces (a checkpoint and a LoRA) and feeds nothing back into it. It is not proof. Proof generates from prompts it trusts (the curator manifest) and returns scores. Darkroom generates from prompts it does not trust and enforces verdicts by destroying output.

## What already exists

Do not rebuild these.

- **ComfyUI** (`:8188`) generates. Darkroom sends it a workflow and collects the image over the websocket.
- **Ollama** (`:11434`), optional, expands a short prompt. It is never a guard.
- **SingGuard server**: agent-composer's `/api/guard/check` (`:9150`, Ollama GGUF `SingGuard-4b`). It accepts `input`, an optional `image_base64`, `input_role`, `policy` text and `thinking` (`fast`, `fast-slow`, `slow`, `cascade`), and answers `{safe, verdict, rule, rule_text, explanation, categories, model, thinking, elapsed_ms}`. A text-only check (no image) is how pre-flight works. Darkroom depends on that PR (agent-composer #4) being merged. A separate argus moderation-service package may replace it later; the client must not care which.
- **NudeNet server**, over HTTP. Never imported (AGPL).
- **argus-cortex** (`>=0.4.2,<0.5`):
  - `moderation.SingGuardClient` / `NudeNetClient` / `exposed_score`.
  - New in 0.4.2: `SingGuardClient.check_text()`, and `moderation.policies`: the named policy texts (moved from curator's `guard.py`) and the per-policy table of illegal rules.
  - `server.WriteGuard` / `cross_site_refuse` / `env_flag`.
  - `wire.make_versioned_base` and `render_schema` for the record's wire schema.
- **curator's face and age pipeline** (`argus_curator.faces.detect_faces`, `argus_curator.mivolo`). It is optional, behind darkroom's `[age]` extra, which pulls `argus-curator[faces]` and runs in process. `detect_faces` takes `(rel, abs, bytes)` items, so the image is judged from memory and nothing is written to disk. MiVOLO needs its ONNX file (`mivolo-convert`). When the extra or the weights are missing, the age check is skipped and `/health` says so.
- **proof's generation models.** `ModelRef`, `LoRARef` and `SamplingParams` already describe a reproducible run. Darkroom copies their field names exactly and does not import proof. They move into cortex with the ComfyUI transport later, and identical fields make that move a rename.

## Flow

One generation runs through these steps, and the user sees each one.

1. **Assemble.** The user's prompt, then Ollama's expansion (if on), then LoRA trigger words, give the **final prompt**: the exact string ComfyUI will receive. The negative prompt rides along. Every later check judges the final prompt, never only the user's text, so the expansion step cannot route around the gate.
2. **Pre-flight.** A text-only SingGuard check of the final prompt and the negative prompt, framed together (see [Prompt framing](#prompt-framing)), under every selected policy.
   - Any illegal rule under any policy: **refused**. Nothing is generated. The user is told which rule fired and why.
   - Unsafe under the enforcing policy on non-illegal rules only: **refused** in SFW mode, with the reason. In NSFW mode the verdict is shown as a warning and generation proceeds, and the output scan decides (see [Mode](#mode)).
   - Advisory policies that flag are shown, not acted on.
   - A failed check is a refusal, recorded as `unchecked`.
3. **Generate.** ComfyUI renders with `SaveImageWebsocket`. The bytes come back over the websocket and are held in memory only. Darkroom then deletes the prompt from ComfyUI's history.
4. **Scan.** For each selected policy, SingGuard judges the image twice: **isolated** (no text) and **with prompt** (the framed final prompt as the user turn). NudeNet scores the exposed classes once. If the age check is available, MiVOLO estimates an age for each detected face. Requests use the configured thinking mode, `cascade` by default. The age estimate is never sent to SingGuard as context: the isolated scan has to stay text-free, and the with-prompt scan measures the prompt alone.
5. **Decide**, from the scans (see [Verdicts](#verdicts)): **passed**, **review** or **deleted**.
6. **Release or destroy.** Passed and review images are written to the output root and served. A deleted image is never written: its buffer is dropped. Its record is kept either way.

The comparison between the isolated scan and the with-prompt scan is the point of step 4: it shows how much the prompt's intent changes the verdict. The isolated scan is also the defence against a prompt written to talk the guard round ("an artistic study of consenting adults"). Text can sway the with-prompt verdict but not the isolated one.

## Policies

- Requests name policies; they never send policy text. The names come from `argus_cortex.moderation.policies`.
- **Enforcing policy:** one per request. Its verdict decides pre-flight, and its name sets the [mode](#mode). The server default (`ARGUS_DARKROOM_POLICY`) is `adult-granular` when NSFW is allowed, and `default` when it is not.
- **Advisory policies:** zero or more extra policies run for comparison. They never block or release an image, except that an illegal rule under any policy still deletes.
- During development, all five names are available: `default`, `adult`, `adult-granular`, `adult-cautious` and `adult-cinematic`. At release only `default` and one adult policy remain, whichever adult variant the ongoing evaluation picks. The default moves to that one.
- "Configurable" means picking the enforcing and advisory policies and the thinking mode. SingGuard returns a verdict, not a score, so it has no threshold to tune. The NudeNet threshold is a real number and is configurable (see [Verdicts](#verdicts)).

### Mode

The mode comes from the enforcing policy's name; it is not a separate field. SingGuard's `default` rules treat sexual content as a risk, so `default` is SFW. Any policy named `sfw-*` is SFW too: that prefix is reserved for finer-grained SFW policies added later. Every other policy is NSFW. Cortex owns this rule (`policies.mode(name)`), next to the illegal table, so a new policy cannot end up with a mode that doesn't match its name.

| | SFW | NSFW |
|---|---|---|
| Pre-flight: illegal rule, any policy | refused | refused |
| Pre-flight: non-illegal flag, enforcing policy | refused, with the reason | warning, generation proceeds |
| Output scans | as in [Verdicts](#verdicts) | as in [Verdicts](#verdicts) |
| NudeNet exposed ≥ 0.2, or NudeNet down | omitted | served, blurred |

NSFW policies can enforce only when the server is started with `--allow-nsfw` (`ARGUS_DARKROOM_ALLOW_NSFW=1`), the same pattern as curator's `--allow-move`. Without the flag:

- a request that names an NSFW policy as enforcing gets HTTP 400;
- `/health` reports `nsfw_allowed: false`, and `/policies` marks the NSFW policies as advisory-only.

NSFW policies can still run as advisory under an SFW enforcing policy, for comparison. Mode changes two things: what pre-flight blocks, and whether NudeNet exposure omits the image or blurs it. The illegal table and the delete rules are the same in both modes.

### Illegal rules

Each policy has its own table, because rule letters mean different things under each one:

| Policy | Illegal (delete) | Everything else |
|---|---|---|
| `adult-granular` | A Minors, H Non-Consensual, I Sexual Violence & Extreme | review |
| `adult`, `adult-cautious`, `adult-cinematic` | A Illegal or Extreme Sexual Content | review |
| `default` | none | review |

Under SingGuard's own `default` rules, A is "Sexual Content Risk", which also covers consensual adult nudity. Treating it as illegal would delete every adult image whenever `default` is chosen for comparison, so `default` can only send an image to review. B (Real-World Crimes) and G (Animal Abuse) go to review, not deletion, under every policy. The table lives in cortex next to the policy texts, so a policy and its illegal rules cannot drift apart.

## Verdicts

Each scan cell (one policy × isolated or with prompt) has one of four states:

| Cell | Means |
|---|---|
| `safe` | SingGuard passed it |
| `flagged` | unsafe, on non-illegal rules only |
| `illegal` | at least one illegal rule fired |
| `unchecked` | the check failed: timeout, HTTP error, no `<answer>` |

The outcome comes from all the cells:

- **deleted** if any cell is `illegal`, under any policy, enforcing or advisory, in either context.
- **deleted** if any *enforcing* cell is `unchecked`. This fails closed, and costs little: the record holds every parameter, so the same seed regenerates the same image once the guard is back. An `unchecked` advisory cell is shown as grey and changes nothing.
- **review** if any cell is `flagged`. The image is served with its shields. This covers the scans disagreeing and the scans all agreeing on a non-illegal rule.
- **passed** if every cell is `safe`.

**NudeNet under an SFW enforcing policy.** An SFW request must not return nudity, even behind a shield, so NudeNet acts as a gate:

- The image's exposed score is under 0.2 (NudeNet's reporting floor, on the five default exposed classes): the outcome is decided by the cells above. A non-safe cell still means review behind its shield.
- The score is 0.2 or more: **omitted**, with `outcome_reason: "sfw-exposed"`. The user's `nsfw_threshold` has no effect on this.
- NudeNet does not respond, or `NUDENET_URL` is empty: **omitted**, with `outcome_reason: "sfw-unchecked"`.

An omitted image is dropped the same way a deleted one is: never written, its buffer discarded. That is to save space and keep SFW output clean, not because the content is criminal, so its record is marked `critical: false`. Records for illegal or `minor-exposed` deletions are `critical: true`. When record retention is decided, the flag lets non-critical records be pruned first. Deletion rules still take priority: an image that is both illegal and exposed is **deleted**, not omitted.

Without a `NUDENET_URL`, every SFW output would be omitted. So the server refuses SFW generation (HTTP 400, naming the missing NudeNet), and `/health` reports `sfw_available: false`.

**NudeNet under an NSFW enforcing policy.** NudeNet never omits or deletes on its own here, because it cannot see age or consent. Its exposed score against the request's `nsfw_threshold` (default `0.2`) sets `blur: true` on the served image, so safelight blurs it until the user clicks to reveal. If NudeNet does not respond, `blur` is `true`.

**Age estimate (when available).** Darkroom applies curator's `minor_exposed` rule: any face with a MiVOLO age under 18, on an image where NudeNet scores an exposed class at 0.2 or more, makes the outcome **deleted**, with `outcome_reason: "minor-exposed"`. The rule ignores the user's NudeNet threshold. An age under 18 without exposure does not delete; it is recorded and shown. When the age check is unavailable, the record says `age: null` and the minors rule rests on SingGuard alone.

The age is a tool's estimate from a face, and darkroom says so everywhere it appears. The record names its source (`mivolo`) and stores the estimate, never a stated age. The API field is `age_estimate_years`, not `age`. Safelight labels it as "MiVOLO estimate". A wrong estimate can delete an adult's image or, worse, pass a minor's. The second case is why the age check is an extra signal on top of SingGuard's minors rule, never a replacement for it.

## Prompt framing

SingGuard reads a bare description as a request; curator's `guard_context.py` exists to tell it otherwise. In darkroom the text really is a request, and the framing says so.

```
Image generation request.
Prompt: <final prompt>
Negative prompt (what the image must not show): <negative prompt>
```

The negative prompt is always part of pre-flight. The with-prompt scan sends the same text as the user turn, followed by the image. Labelling the negative prompt matters: a negative of `child, minor, underage` is a common safety practice and must not read as asking for those things. The other direction matters too: negating clothing in the negative prompt is a known way to get nudity. The pre-flight evaluation covers both directions on labelled prompts. If this framing fails either, the framing changes; the negative prompt stays in the check.

## Record

A record is kept for every attempt, including refused and deleted ones. It is the only thing left of a deleted image, so it must be enough to regenerate it exactly.

```
GenerationRecord (record_version "1.0", wire schema committed)
  id, created_at, darkroom_version
  outcome: refused | passed | review | deleted | omitted | failed
  outcome_reason: rule codes, "minor-exposed", "unchecked", "sfw-exposed", "sfw-unchecked", or the generation error
  critical: bool                # true for illegal and minor-exposed deletions; false for everything else
  request
    mode: sfw | nsfw            # derived from guard.enforcing_policy, stored for filtering
    user_prompt, expanded_prompt (null if Ollama is off), final_prompt, negative_prompt
    expansion: {model, system_prompt_sha256} | null
    seed
    sampling: SamplingParams    # sampler, scheduler, steps, cfg, clip_skip, width, height
    checkpoint: ModelRef        # name + sha256
    vae: ModelRef | null
    loras: [LoRARef]            # name, sha256, weight
    workflow_sha256, engine "comfyui", engine_version
  guard
    enforcing_policy, advisory_policies, thinking
    preflight: {policy: GuardResult}
    scans: {policy: {isolated: Cell, with_prompt: Cell}}   # Cell = state + GuardResult | error
    nudenet: {exposed_score, labels, threshold, blur} | null
    age: {source "mivolo", faces: [{age_estimate_years, bbox}], minor_exposed} | null   # tool estimate
  image
    sha256, phash, width, height      # always, including when deleted
    path                              # only when passed or review
```

A deleted record has no pixels, thumbnail or preview of any kind; those would be the image again. Its sha256 and pHash let a repeat of the same bad output be recognised. Model sha256s are read from the ComfyUI models root (`ARGUS_DARKROOM_MODELS_ROOT`) and cached by path and mtime, since hashing a checkpoint on every request is too slow.

## What it does not do

- **Write anything before the verdict.** No temporary file, no cortex blob or lineage write, no `SaveImage` or `PreviewImage` node. A workflow that contains a node that writes to disk is refused when it is loaded, not when it runs.
- **Image-to-image, inpainting or uploaded reference images** in v1. Each of these is a second input that would need its own pre-flight.
- **Training, captioning or scoring against a dataset.** Those are forge, lens and proof.
- **Accept policy text in a request.** Policies are chosen by name only.
- **Public hosting.** Not planned in the short or medium term. The server binds to localhost by default.

## Shape

```
safelight :3000 ──▶ argus-darkroom :8106 ──▶ ComfyUI :8188
                     record store            Ollama :11434 (optional)
                     output root             SingGuard :9150 (agent-composer)
                                             NudeNet
```

The package is scaffolded from `argus-pkg-template` with the CLI, server and Dockerfile options on. It has three layers: an engine (pydantic plus cortex, stdlib HTTP for ComfyUI), an optional Typer CLI (`generate`, `preflight`, `serve`, `schema`) and an optional FastAPI server.

**Server** (`argus-darkroom serve --port 8106 --output-root …`):

- `GET /health`: version, which backends answer, the policy list, `nsfw_allowed`.
- `GET /policies`: names, the enforcing default and each policy's illegal rules. Safelight shows the wording.
- `GET /models`: checkpoints and LoRAs found under the models root.
- `POST /preflight`: the text check alone, so the form can show a verdict before Generate.
- `POST /generate/stream`: SSE events `preflight`, `queued`, `progress` (ComfyUI step), `scan` (one per finished cell), `complete` (the record). One generation at a time; a second gets HTTP 409.
- `GET /records`: paged, filterable by outcome. `GET /record/{id}`.
- `GET /image/{id}`: ids only, no path input. A deleted, omitted or refused record answers 404 with the record id.
- `DELETE /record/{id}`: the user removing their own output. This deletes the file; the record stays, marked as removed by the user.
- `GET /archive`: a streamed zip of every record (JSON) and every stored image, which means passed and review images only, because deleted and omitted ones were never written. It is read-only and always available.
- `POST /purge`: deletes every record and stored image, and clears darkroom's prompts from ComfyUI's history. A clean slate. It exists only when `ARGUS_DARKROOM_DEVMODE=1`, and answers 404 otherwise, so hiding the button is not the only thing preventing a purge. The body must be `{"confirm": "purge"}`. A purge during a generation gets HTTP 409. `/health` reports `devmode`.
- `WriteGuard` and `cross_site_refuse` from cortex. CORS allows reads by default; writes are allowed only from the configured origins (`ARGUS_DARKROOM_CORS_ORIGINS`).

**Config** (`ARGUS_DARKROOM_*`): `COMFYUI_URL`, `OLLAMA_URL` (empty turns expansion off), `OLLAMA_MODEL`, `SINGGUARD_URL`, `NUDENET_URL`, `POLICY`, `THINKING` (default `cascade`), `OUTPUT_ROOT`, `MODELS_ROOT`, `WORKFLOW` (the template path), `ALLOW_NSFW`, `DEVMODE`. Compose maps the `.env` value `DEVMODE=1` to `ARGUS_DARKROOM_DEVMODE`, and both flags are read with cortex's `env_flag`, which warns about a mistyped value instead of silently ignoring it. An empty `SINGGUARD_URL` makes the server refuse to start. Darkroom never runs unguarded.

**Store:** one flat `<id>.json` per record, the same pattern as curator's `ScanStore`. Images go under `<output_root>/<id>.png`.

## Testing

No real sexual, violent or illegal content goes in the repo or the tests.

- Fake SingGuard, NudeNet and ComfyUI servers, in the pattern of cortex's `_Fake`. The illegal-to-deleted path is driven by a fake verdict, not by an image.
- SFW NudeNet tests: an exposed score of 0.19 is served, 0.2 is omitted, a NudeNet timeout is omitted, an image that is both exposed and illegal is deleted (with `critical: true`), and SFW generation without `NUDENET_URL` gets HTTP 400.
- Purge tests: `POST /purge` answers 404 without `DEVMODE`, 409 during a generation, and 400 without the confirm body. After a purge, records, images and darkroom's ComfyUI history are empty. `GET /archive` taken before the purge holds every record.
- A test that a deleted or omitted outcome leaves no bytes anywhere: the output root, the system temp directory and the fake ComfyUI history. The test lists all three before and after.
- A test that a workflow containing a disk-writing node is refused at load.
- Mode tests:
  - `default` and `sfw-*` names are SFW, and everything else is NSFW.
  - A non-illegal pre-flight flag refuses under an SFW enforcing policy and proceeds under an NSFW one; an illegal flag refuses under both.
  - Without `--allow-nsfw`, an NSFW enforcing policy gets HTTP 400, and the same policy as advisory is accepted.
- Unit tests over the full outcome table (every combination of cell states), and per policy for the illegal table.
- The age check through a fake `detect_faces`: an under-18 estimate with exposure deletes, one without exposure does not, and with the extra missing the record gets `age: null` and the scan carries on.
- A pre-flight evaluation script, in the spirit of curator's `scripts/guard_eval.py`: labelled text prompts, including negative-prompt cases in both directions, run against the real server. Run by hand, not in CI.

## Open questions

None are blocking the build.

Settled: the enforcing policy sets the mode (`default` and `sfw-*` are SFW; the rest are NSFW), and non-illegal pre-flight flags block in SFW mode only ([Mode](#mode)). The negative prompt is always part of pre-flight ([Prompt framing](#prompt-framing)). The MiVOLO age estimate is used when available, and presented as a tool's estimate ([Verdicts](#verdicts)).

Deferred until before a public launch:

- **Which adult policy ships.** This follows the evaluation of the three variants. The default moves with it.
- **Record retention.** Keep records forever, or prune by age? Refused and deleted records are the audit trail. Until this is decided, records are kept.

## Build order

1. **Cortex 0.4.2**, a separate PR in argus-core: `check_text()` and `moderation.policies` (the texts, the illegal table and `mode()`). Curator then imports its policies from cortex (a PR into curator `dev`).
2. **Engine without generation.** Assemble the prompt, frame it, run pre-flight, compute the outcome table and write records, all tested against fakes. Run the pre-flight evaluation against :9150.
3. **ComfyUI.** Load the workflow and refuse disk-writing nodes, render through `SaveImageWebsocket`, clear history, hash the models. Use darkroom's own minimal client for now.
4. **Post-generation scans and release or destroy**, with the no-bytes test. Then the `[age]` extra and the `minor-exposed` rule.
5. **Server and SSE**, then the CLI.
6. **Ollama expansion.**
7. **Schema, README, CLAUDE.md, compose.** Compose runs darkroom and safelight; ComfyUI, Ollama, SingGuard and NudeNet are reached by URL.
8. **Later, in their own PRs:** lift the ComfyUI transport and the generation models into cortex, and move atlas and proof onto them.

Repo conventions from day one: `main` for releases, `dev` for work, PRs into `dev` (`.cursor/rules/git-workflow.mdc`). CI runs on main only, so lint, tests and `schema --check` run locally before every PR into `dev`. Versions come from git tags (`hatch-vcs`).

## Done when

- A prompt flagged under the enforcing policy is refused with its rule and explanation, and ComfyUI never sees a request.
- A generated image whose scan comes back illegal (from a fake verdict) leaves no bytes on disk or in ComfyUI's history, and its record holds every parameter needed to regenerate it.
- A record's seed and parameters, replayed, give the same image sha256 on the same ComfyUI and models.
- A review image is served with every cell's verdict, and a disagreement between the isolated and with-prompt scans shows up in the record.
- With SingGuard unreachable, nothing is served.
- No SFW request returns an image with a NudeNet exposed score of 0.2 or more, or one NudeNet never scored.
