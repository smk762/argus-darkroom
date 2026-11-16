# argus-darkroom

Guarded image generation. Read `spec.md` for the design and `README.md` for running it. This file is what must never
happen and what checks it.

## Must never happen

Each row names what enforces it. An **UNENFORCED** row is a known gap: when one bites, add a check, not more prose.
`suite-rules` is argus-ci's lint catalogue (the CI job of that name, with an empty `.suite-rules-baseline.json`).

| Never | Enforced by |
|---|---|
| An image written before the verdict (temp file, cortex blob, `SaveImage`/`PreviewImage` node), or a deleted or omitted image written at all | `test_generate::test_an_illegal_cell_deletes_and_leaves_no_bytes` (output root, system temp and ComfyUI history, before and after); `test_workflow::test_a_disk_writing_node_is_refused_at_load`; at run time, ComfyUI's own output-node list (`Engine._verify_outputs`) |
| Running without SingGuard | `test_server::test_an_empty_singguard_url_stops_the_server_starting` (`Settings.__post_init__`) |
| A failed check counted as a pass. An unchecked enforcing cell deletes; an unchecked advisory cell is only shown | `test_preflight::test_a_failed_check_is_a_refusal_recorded_as_unchecked`; `test_outcome::test_unchecked_enforcing_deletes_but_is_not_critical`, `::test_unchecked_advisory_is_grey_and_changes_nothing` |
| A check that judges the user's text instead of the final prompt (expansion included), or a pre-flight without the negative prompt | `test_preflight::test_the_expansion_is_what_is_judged`, `::test_frame_always_carries_a_negative_line`; `test_expansion::test_expansion_cannot_route_around_the_gate` |
| Policy text accepted in a request (policies are chosen by name) | `test_server::test_a_request_cannot_carry_policy_text` |
| A policy text copied out of `argus_cortex.moderation.policies` (with the illegal table and `mode()`, it lives there only) | `test_must_never::test_no_policy_text_is_copied_out_of_cortex` |
| An outcome less strict than the table: unsure means refuse or delete | `test_outcome::test_every_combination_of_cell_states`. Precedence is deleted, omitted (SFW), review, passed; test any change by cell combination |
| NudeNet (AGPL) imported in-process: reach it over HTTP through cortex's `NudeNetClient` | `suite-rules` → `agpl-import` |
| The server bound beyond localhost by default, a write without `WriteGuard` and `cross_site_refuse`, or `POST /purge` without `ARGUS_DARKROOM_DEVMODE` (it answers 404) | `test_cli::test_serve_maps_its_flags_to_the_environment_and_binds_to_localhost`; `test_server::test_a_cross_site_write_is_refused_and_a_cross_site_read_is_not_granted`, `::test_purge_does_not_exist_without_devmode` |
| The age estimate called an age: it is `age_estimate_years`, source `mivolo`, a tool's estimate | `AgeRecord.source` is `Literal["mivolo"]`; a renamed field fails `test_must_never::test_record_fields_only_grow_within_a_major` |
| The age estimate sent to SingGuard as context | `test_must_never::test_the_age_estimate_never_reaches_singguard` |
| Real sexual, violent or illegal imagery in the repo or tests: tests draw their images in code and drive the illegal path with a fake verdict | `test_must_never::test_no_image_video_or_model_file_is_tracked`, `::test_no_image_is_hidden_in_a_base64_literal` |
| Real sexual, violent or illegal text in the repo or tests | **UNENFORCED** (smk762/argus-ci#20: a suite-wide rule) |
| An input image used on any outcome but `passed`: review, omitted, deleted and unchecked all refuse, after pre-flight and before ComfyUI | `test_img2img::test_an_illegal_input_is_refused_and_critical`, `::test_an_input_flagged_under_sfw_is_refused_not_sent_to_review`, `::test_an_input_singguard_cannot_check_is_refused`, `::test_engine_render_will_not_send_an_input_that_was_not_scanned` |
| An input image (or mask) written to disk, left in ComfyUI's history, or loaded by a template any way but base64 in the graph to `DarkroomLoadImageBase64` | `test_img2img::test_a_passed_input_reaches_comfyui_in_the_graph_and_the_output_is_judged_as_usual` (no bytes left behind, history empty); `test_inpaint::test_an_inpaint_run_sends_the_input_and_the_mask_and_scans_only_the_input` (history empty); `test_comfyui_node::test_the_node_never_touches_a_file`; `test_workflow::test_a_node_that_loads_an_unchecked_image_is_refused_at_load` |
| The input or the mask in a repr, an error or a record (the record keeps hashes, and `masked_fraction`) | `test_img2img::test_the_input_never_shows_in_a_repr`, `::test_a_comfyui_error_that_echoes_the_input_does_not_put_it_in_the_record`; `test_inpaint::test_the_mask_never_shows_in_a_repr` |
| A mask scanned, sent to a guard or written. It selects a region and has no imagery: it is checked as a file only (decodes, the upright input's exact size, something masked, else 400) | `test_inpaint::test_an_inpaint_run_sends_the_input_and_the_mask_and_scans_only_the_input`, `::test_a_refused_input_with_a_mask_reaches_nothing_and_leaves_no_bytes`, `::test_a_mask_that_cannot_be_used_is_a_request_error_and_touches_nothing` |
| The mask loader outside an `+inpaint` template, or a mask falling back to an image-to-image template | `test_workflow::test_a_mask_loader_outside_an_inpaint_template_is_refused`; `test_inpaint::test_a_mask_with_no_inpaint_template_is_a_request_error` |
| A record field renamed or removed within a `record_version` major (adding one is fine) | `test_must_never::test_record_fields_only_grow_within_a_major` (against `tests/data/record-1.0-fields.txt`); `test_store_and_schema::test_an_incompatible_major_is_refused` |
| The depth workflow offered for generation | `test_depth::test_the_depth_workflow_is_never_offered_for_generation` |
| A GPU lease nested inside another | `test_gpu::test_expansion_leases_its_ollama_model_first_and_leases_never_nest` |
| A refused or timed-out lease retried like an OOM | `test_gpu::test_a_refused_lease_is_a_failed_record_and_nothing_reaches_comfyui` |
| The NudeNet detector used to decide (it is recorded, never read) | `test_nudenet_detector::test_the_upstream_320n_image_has_no_health_route_and_records_no_detector`, `::test_a_failing_health_route_costs_only_the_detector` |

## Argus suite
<!-- argus-suite: keep this block; argus-ci checks for it -->
This repo is part of the Argus suite. The suite map below lists every sibling and what it owns; check it before
building something a sibling may already do. **On every release, update this repo's row and capabilities in
argus-cortex's SUITE.md** (a PR to argus-cortex in the same release).

@../argus-core/SUITE.md

Not checked out? Read https://github.com/smk762/argus-cortex/blob/main/SUITE.md

## Not shown by the code

- `comfyui/` is not in the package: the ComfyUI custom nodes image-to-image, inpainting and depth maps need
  (`DarkroomLoadImageBase64`, `DarkroomLoadMaskBase64`, `DarkroomDepthAnything`), copied into ComfyUI by hand.
- `depth.py` runs no guard: the image was already released, and nothing new goes in.
- `gpu.py`'s footprints are kept in step with `gpu-tenant.yaml` (`test_gpu::test_the_manifest_validates_and_matches_the_lease_figures`).
- `demo/` is demo only: it turns `DEVMODE` on and uses fake models.

## Working

```bash
make install && make test && make lint
uv run --no-sync argus-darkroom schema --check   # after any change to models.py: run `schema`, commit the file
```

- Work in a worktree from `origin/dev` and open PRs into `dev` (see Branches below). Run lint, tests and
  `schema --check` locally before every PR; CI runs them again on the PR.
- Use `docker compose`, not `docker-compose`.
- The blur decision lives only in `guard.nudenet.blur`, recorded for every NSFW-policy image (`exposed_score` is `null`
  when NudeNet was unreachable).
- argus-safelight codes against the HTTP shapes in the README's API table (`/health`, `/policies`, `/models`, `/records`
  paging, SSE event bodies). Change one and `make demo` plus `make smoke` show whether the UI still works; update
  safelight's `types.ts` in step.
- Tests use fake servers on real sockets (`tests/fakes.py`, `tests/fake_comfyui.py`, `tests/rig.py`). Prefer extending
  them to mocking.
- `scripts/preflight_eval.py` is run by hand against a real SingGuard that supports text-only checks (agent-composer
  PR #4). It is not a test.
- `scripts/nudenet_compare.py` is run by hand: it scores darkroom's stored images (or a hand-sorted `blur/`/`clear/`
  folder outside the repo) on two NudeNet servers and sweeps the new one's cutoff against the old 0.2. It reports
  names and scores only. Stored images are all released ones, so that sample has no deleted or omitted images. It
  is not a test, and its sample never enters the repo.

## Branches

`dev` is the default branch and the base for every PR. Cut feature branches from `origin/dev` and bring them up to date
by merging `origin/dev`, never `main`. Never push work to `main` or open a PR against it, except the release PR from
`dev`: `main` must never be ahead of `dev`, with one exception. A release fast-forwards `main` to a `dev` commit that
passed CI (`git push origin origin/dev:main`), then tags it. If `main` is protected, the release is a reviewed PR from
`dev` into `main`, and merging it leaves a merge commit on `main` alone. That merge commit is the one exception to the
rule: it leaves `main` one commit ahead of `dev` with an identical tree. Leave it there and don't merge `main` back into
`dev`. Anything else on `main` that `dev` lacks is a mistake (`git diff origin/dev origin/main` must be empty). Once
`main` carries such a commit it can't be fast-forwarded, so later releases are PRs too.
