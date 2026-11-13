# argus-darkroom

Guarded image generation. Read `spec.md` for the design and `README.md` for running it. This file is the rules that must
not be broken and where things are.

## Rules that must hold

- **Nothing is written before the verdict.** No temp file, no cortex blob, no `SaveImage`/`PreviewImage` node. A
  deleted or omitted image is never written, and its buffer is dropped. `tests/test_generate.py` lists the output root,
  the system temp directory and the fake ComfyUI history before and after to prove it.
- **Never run unguarded.** An empty `SINGGUARD_URL` refuses to start (`Settings.__post_init__`). A failed check is an
  `unchecked` cell, never a pass. An unchecked enforcing cell deletes; an unchecked advisory cell is only shown.
- **Every check judges the final prompt**, expansion included. The negative prompt is always part of pre-flight. Do not
  add a path that checks only the user's text.
- **Policies are chosen by name only.** Never accept policy text in a request. Policy texts, the illegal table and
  `mode()` live in `argus_cortex.moderation.policies`; do not copy them here.
- **Fail closed.** If you are unsure whether an error should refuse, delete or pass, it refuses or deletes.
- **Never import NudeNet** (AGPL). Reach it over HTTP through cortex's `NudeNetClient`.
- **The server binds to localhost by default.** Writes need `WriteGuard` and `cross_site_refuse`. `POST /purge` exists
  only with `ARGUS_DARKROOM_DEVMODE` and answers 404 otherwise.
- **The age is a tool's estimate.** Name it `age_estimate_years`, source `mivolo`. Never call it an age, and never send
  it to SingGuard as context.
- **No real sexual, violent or illegal content** in the repo or tests. Drive the illegal path with a fake verdict.
- **An input image is judged like an output, and goes on only on `passed`.** Image-to-image runs the same scans and
  `decide` on the input after the text pre-flight and before anything reaches ComfyUI; any other outcome (review,
  omitted, deleted, unchecked) refuses. The input is never written to disk: it travels as base64 inside the graph to
  `DarkroomLoadImageBase64` (`comfyui/`), the history delete removes it, and the record keeps only its hashes. No
  template may load an image any other way.

## Argus suite
<!-- argus-suite: keep this block; argus-ci checks for it -->
This repo is part of the Argus suite. The suite map below lists every sibling and what it owns; check it before
building something a sibling may already do. **On every release, update this repo's row and capabilities in
argus-cortex's SUITE.md** (a PR to argus-cortex in the same release).

@../argus-core/SUITE.md

Not checked out? Read https://github.com/smk762/argus-cortex/blob/main/SUITE.md

## Layout

| | |
|---|---|
| `engine.py` | `Engine`: `begin` validates, `_run` expands, pre-flights, scans an input image (`check_input`), renders, scans, decides, releases or destroys |
| `outcome.py` | the outcome table (`decide`). Precedence: deleted, omitted (SFW), review, passed. Unit-test any change by cell combination |
| `preflight.py`, `scan.py`, `guard.py` | the text check, the image scans, and the SingGuard wrapper that turns failures into `unchecked` |
| `prompt.py` | `assemble_prompt`, and `frame` (folds line breaks so text cannot pose as the frame's labels) |
| `expansion.py` | `OllamaExpander`. Ollama is never a guard |
| `comfyui.py`, `workflow.py`, `modelfiles.py`, `hashing.py` | render over the websocket, the `$placeholder` templates (one file, or a directory chosen by checkpoint family, LoRA count and `+img2img`), model sha256s |
| `images.py` | an image's hashes (`describe`), and `load_input`: the image-to-image input decoded once into the PNG that is both scanned and sent |
| `comfyui/` | not in the package: `DarkroomLoadImageBase64`, the ComfyUI custom node image-to-image needs (base64 in, `IMAGE` out, no disk) |
| `age.py` | curator over `AGE_URL` (`POST /faces?body=true`), or in-process `detect_faces` behind the `[age]` extra |
| `store.py` | `<id>.json` and `<id>.png` in the output root; atomic image write |
| `models.py` | the versioned wire models. `schema/darkroom-wire.schema.json` is generated from them |
| `demo/run.py`, `demo/smoke.py` | the real server and engine over the test fakes; a marker in the prompt picks the verdicts. Demo only: it turns `DEVMODE` on and uses fake models |
| `server/app.py`, `cli.py`, `config.py`, `request.py` | HTTP, Typer, `ARGUS_DARKROOM_*`, the request model (`extra=forbid`) |

## Working

```bash
make install && make test && make lint
uv run --no-sync argus-darkroom schema --check   # after any change to models.py: run `schema`, commit the file
```

- `main` is releases, `dev` is work. Work in a worktree from `dev`, and open PRs into `dev`. CI runs on `main` only, so
  run lint, tests and `schema --check` locally before every PR.
- Use `docker compose`, not `docker-compose`.
- The record is a wire contract (`record_version`). Adding a field is fine; renaming or removing one is a new version.
  The blur decision lives only in `guard.nudenet.blur`, which is recorded for every NSFW-policy image (`exposed_score`
  is `null` when NudeNet was unreachable). Each LoRA records the `trigger_words` the request supplied.
- argus-safelight codes against the HTTP shapes in the README's API table (`/health`, `/policies`, `/models`, `/records`
  paging, SSE event bodies). Change one and `make demo` plus `make smoke` show whether the UI still works; update
  safelight's `types.ts` in step.
- Tests use fake servers on real sockets (`tests/fakes.py`, `tests/fake_comfyui.py`, `tests/rig.py`). Prefer extending
  them to mocking.
- `scripts/preflight_eval.py` is run by hand against a real SingGuard that supports text-only checks (agent-composer
  PR #4). It is not a test.
