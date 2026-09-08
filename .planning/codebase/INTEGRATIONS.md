# External Integrations

**Analysis Date:** 2026-09-08

## APIs & External Services

**Third-Party APIs:**
- Not detected - Tiresias is a self-contained scientific package with no external API integrations

**Optical PSF Modeling:**
- `psfmodels` library - Provides theoretical PSF seed generation for blind estimation
  - SDK/Client: Python package import `psfmodels as pm` in `seeds.py:9`
  - Usage: `pm.make_psf()` called with optical parameters (NA, wavelength, refractive index, etc.)
  - Supports models: vectorial (default), scalar, Gaussian
  - No API key or authentication required

## Data Storage

**Databases:**
- Not used - Tiresias is not a database-backed application

**File Storage:**
- **Local filesystem only** - All I/O is local file operations
  - Input: 3-D TIFF volume files (memory-mapped access via `tifffile.memmap()` in `tiling.py`)
  - Output: TIFF files written via `tifffile.imwrite()` in `cli.py` and `tiling.py`
  - Intermediate: PSF cache stored locally (controlled by `--cache-dir` CLI flag, default `.psf_cache/`)
  - Logs: Console output only (no file logging detected)

**Caching:**
- **Local PSF merge cache** - Deterministic file-based cache for repeated PSF estimation runs
  - Location: `.psf_cache/` directory or custom location via `--cache-dir` CLI flag
  - Implementation: SHA256 hash-based cache keys in `tiling.py` (hashlib usage)
  - Control: `--no-psf-cache` flag disables caching

## Authentication & Identity

**Auth Provider:**
- Not applicable - No authentication system; package is used locally or in HPC environments

**Identity Management:**
- Not applicable - Single-user command-line and API usage pattern

## Monitoring & Observability

**Error Tracking:**
- Not detected - No error tracking service integration

**Logs:**
- **Console only** - All output to stdout/stderr
- No structured logging framework detected
- CLI provides progress/status output during long-running operations

**Metrics:**
- Not detected - No metrics collection or telemetry

## CI/CD & Deployment

**Hosting:**
- **GitHub Pages** - Documentation deployed to GitHub Pages
  - Trigger: Push to `main` branch
  - Workflow: `.github/workflows/docs.yml` builds Sphinx docs and deploys to GitHub Pages
  - Permissions: `pages:write`, `id-token:write` for deployment

**CI Pipeline:**
- **GitHub Actions** (`.github/workflows/ci.yml`)
  - Trigger: Push to `main` or pull request to `main`
  - Matrix: Python 3.10, 3.11, 3.12
  - Runs: Ubuntu latest
  - Steps: Setup Python (with pip cache), install dependencies, run pytest
  - No external status checks or notifications detected

**Repository Badges:**
- CI and Docs workflow status badges in README.md link to GitHub Actions workflows

## Environment Configuration

**Required env vars:**
- `CUPY_CACHE_DIR` (optional, auto-configured in `blind_rl.py`)
  - If set: Uses specified directory for CuPy JIT compilation artifacts
  - If not set: Falls back to `SLURM_TMPDIR` (HPC), `TMPDIR` (system), or `/tmp`
  - Path construction: `{cache_root}/cupy-kernel-cache-{uid}`
  - Purpose: Isolate kernel cache from read-only container home mounts (Singularity, etc.)

**Secrets location:**
- Not applicable - No API keys, tokens, or secrets in configuration
- `.env` files: Not present in repository
- Secrets: No secrets detected

## Webhooks & Callbacks

**Incoming:**
- Not applicable - Standalone package; no webhook endpoints

**Outgoing:**
- Not applicable - No outbound webhooks or callbacks

## SLURM HPC Integration

**HPC Environment Variables:**
- `SLURM_TMPDIR` - Checked in `blind_rl.py` to configure CuPy cache in HPC environments
  - Purpose: Use compute-node temporary storage instead of home mount for kernel cache

**Subprocess/Process Management:**
- Multiprocessing via `multiprocessing.cpu_count()` (standard library)
- ProcessPoolExecutor for parallel CPU operations in `blind_rl.py`
- No job submission or HPC-specific task scheduling detected

## No External Integrations Summary

Tiresias is intentionally designed as a **self-contained scientific package** with:
- No cloud dependencies
- No external API calls
- No authentication requirements
- No database
- No message queues or event systems
- No webhooks
- Minimal third-party integrations (only numerical libraries: NumPy, SciPy, CuPy, psfmodels)

All computation occurs locally with optional GPU acceleration via CUDA/CuPy.

---

*Integration audit: 2026-09-08*
