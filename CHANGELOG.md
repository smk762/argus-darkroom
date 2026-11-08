# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
