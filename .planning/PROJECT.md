# Tiresias — ASLM Simulation & PEP 723 Examples

## What This Is

Tiresias is a GPU-first Python package for blind PSF estimation and CuPy Richardson-Lucy deconvolution from 3-D TIFF microscopy volumes. This milestone extends its PSF seed generation from static light-sheet illumination to also model Axially Swept Light-Sheet Microscopy (ASLM): a rolling-shutter acquisition mode where only a narrow slit around the swept beam waist is illuminated and read out at any instant. It also brings standalone, PEP 723-compliant example scripts to the project so users can run demos with `uv run` and no separate install step.

## Core Value

Users can generate an ASLM-mode PSF seed (narrow rolling-shutter slit, perfectly synchronized to the beam sweep) through the same `generate_psf_seed()` API and CLI they already use for static light-sheet PSFs — without breaking existing "single" or "light_sheet" behavior.

## Requirements

### Validated

- ✓ Theoretical PSF generation via `psfmodels.make_psf` (`generate_theoretical_psf`) — existing
- ✓ Calibrated PSF loading and fitting to a target support shape (`load_psf_seed`) — existing
- ✓ Static light-sheet PSF seed generation: detection PSF × rotated illumination PSF (`generate_psf_seed`, `psf_mode="light_sheet"`) — existing
- ✓ Single-detection PSF seed generation (`psf_mode="single"`) — existing
- ✓ CuPy tile-based blind PSF estimation and Richardson-Lucy deconvolution with VRAM-aware chunking — existing
- ✓ CLI entrypoints `tiresias-estimate-psf` and `tiresias-deconvolve` — existing

### Active

- [ ] User can generate an ASLM-mode PSF seed via `generate_psf_seed(psf_mode="aslm", ...)`, combining the detection PSF with an illumination PSF masked to a narrow rolling-shutter slit
- [ ] User can control the ASLM slit width via a `slit_width` parameter that limits the effective illumination/detection extent along the scan axis (narrower FOV than static light-sheet)
- [ ] ASLM simulation assumes perfect shutter synchronization (slit tracks the beam waist exactly) — no timing jitter or desync modeling in this milestone
- [ ] Invalid ASLM parameters (e.g. non-positive or out-of-range `slit_width`) raise clear errors, consistent with existing `generate_psf_seed` validation
- [ ] `tiresias-estimate-psf` and `tiresias-deconvolve` CLIs expose flags to select ASLM mode and set slit width
- [ ] Existing "single" and "light_sheet" CLI/API behavior is unchanged (regression-safe)
- [ ] Repository includes standalone, PEP 723-compliant example script(s) (inline `# /// script` metadata block) demonstrating static light-sheet vs. ASLM PSF generation, runnable via `uv run` without installing the package first
- [ ] Documentation (`docs/usage.md`) covers the new ASLM mode and the PEP 723 example script(s)

### Out of Scope

- Modeling imperfect shutter synchronization, timing jitter, or desync artifacts — deliberately simplified to perfect sync for this milestone
- Full raw image-formation simulation (sample volume + beam sweep + rolling-shutter readout → synthetic camera frames) — this milestone only extends PSF *seed* generation, not a full acquisition simulator
- Converting the installable package itself (pyproject.toml/src layout) to PEP 723 — PEP 723 applies to the new standalone example scripts only; the package keeps its normal packaging
- MATLAB runtime compatibility and non-3D image support — already excluded per existing package scope

## Context

- Tiresias follows a layered architecture: CLI (`cli.py`) → tiling/chunking (`tiling.py`) → PSF seed initialization (`seeds.py`) → FFT/RL algorithm layer (`blind_rl.py`), with dual SciPy (reference) and CuPy (GPU production) backends.
- The existing `generate_psf_seed()` in `src/tiresias/seeds.py` already supports `psf_mode="single"` and `psf_mode="light_sheet"` (static beam: detection PSF multiplied by a rotated illumination PSF via `rotate_illumination_psf`). ASLM will be added as a third `psf_mode="aslm"` value alongside these, reusing the existing detection/illumination PSF generation and normalization helpers where possible.
- `generate_psf_seed` is currently exported from the package's public API (`src/tiresias/__init__.py`) but not yet wired into the CLI (`cli.py` calls `generate_theoretical_psf` directly) — CLI wiring for `psf_mode` (including the new `aslm` mode) is in scope for this milestone.
- No simulation or ASLM-related code currently exists in the codebase (confirmed via search) — this is new capability, not a bug fix or refactor.
- No `examples/` directory currently exists; it will be created for the PEP 723 example script(s).
- ASLM (Axially Swept Light-Sheet Microscopy) systems use a camera rolling shutter synchronized to a swept light-sheet beam waist so that, at any instant, only a thin in-focus slit is exposed — giving more uniform axial resolution than a static light-sheet across the field of view.

## Constraints

- **Compatibility**: New `psf_mode="aslm"` must not change behavior for existing `psf_mode="single"` and `psf_mode="light_sheet"` callers — additive only.
- **Simplification**: Perfect shutter/beam synchronization is assumed; no timing-jitter modeling in this milestone (explicit user decision).
- **Packaging**: PEP 723 compliance applies to new standalone example scripts only, not the installable package's own build/dependency metadata.
- **GPU dependency**: Production PSF/deconvolution paths require a CUDA-capable GPU (`cupy-cuda11x`); the SciPy backend remains the CPU-friendly reference path, per existing project constraints.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| ASLM is a new `psf_mode="aslm"` value, not a parameter extension of `light_sheet` | Keeps static light-sheet code path untouched and easy to reason about; explicit mode is clearer than an implicit slit_width toggle | — Pending |
| ASLM narrower FOV is controlled by a `slit_width` parameter | Directly models the physical rolling-shutter slit rather than shrinking the whole simulated frame | — Pending |
| Assume perfect shutter synchronization | User explicitly deferred timing-jitter/desync modeling to keep this milestone tractable | — Pending |
| ASLM wired into CLI (not API-only) | User wants command-line access, not just library-level usage | — Pending |
| PEP 723 applies to new example scripts only | Package already uses a proper pyproject.toml/src-layout installable structure; PEP 723 targets single-file scripts, not installable packages | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-09-08 after initialization*
