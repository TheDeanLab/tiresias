# Tiresias Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `../tiresias` as an Apache-2.0 open-source Python package for CuPy/SciPy blind PSF estimation and optional cuCIM Richardson-Lucy restoration.

**Architecture:** Extract the reusable numerical code into a conventional `src/tiresias` package with small modules for blind RL, PSF seed generation, tiled estimation, and CLI entry points. Keep Astrocyte, Nextflow, MATLAB, and OME-Zarr workflow glue out of the new public package.

**Tech Stack:** Python 3.10+, NumPy, SciPy, tifffile, psfmodels, optional CuPy/cupyx, optional cuCIM, pytest.

## Global Constraints

- Do not modify files under `deconvolution-gpu`.
- License the new repository under Apache License 2.0.
- Preserve UTSW copyright attribution.
- Keep GPU dependencies optional so CPU-only tests run without CUDA.
- Include cuCIM support as an optional adapter.
- Exclude MATLAB compatibility and Nextflow/Astrocyte workflow wiring from the public package.

---

### Task 1: Repository Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `LICENSE`
- Create: `NOTICE`
- Create: `.gitignore`
- Create: `src/tiresias/__init__.py`
- Create: `tests/test_public_api.py`

**Interfaces:**
- Produces: package import `tiresias`
- Produces: public names `estimate_blind_psf_scipy`, `estimate_blind_psf_cupy`, `estimate_psf_array_cupy`, `deconvolve_with_cucim`, `generate_theoretical_psf`, `generate_psf_seed`, `estimate_psf_from_chunks`

- [ ] **Step 1: Write the failing public API test**

```python
def test_public_api_exports_core_functions():
    import tiresias

    for name in [
        "estimate_blind_psf_scipy",
        "estimate_blind_psf_cupy",
        "estimate_psf_array_cupy",
        "deconvolve_with_cucim",
        "generate_theoretical_psf",
        "generate_psf_seed",
        "estimate_psf_from_chunks",
    ]:
        assert hasattr(tiresias, name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_public_api.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tiresias'`.

- [ ] **Step 3: Add minimal package scaffold and metadata**

Create the files listed above with package metadata, Apache-2.0 license text, and stub exports.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_public_api.py -q`
Expected: PASS.

### Task 2: Blind RL Engine

**Files:**
- Create: `src/tiresias/blind_rl.py`
- Create: `tests/test_blind_rl.py`
- Modify: `src/tiresias/__init__.py`

**Interfaces:**
- Consumes: package scaffold from Task 1.
- Produces: `estimate_blind_psf`, `estimate_blind_psf_scipy`, `estimate_blind_psf_cupy`, `estimate_psf_array_cupy`, `deconvolve_with_cucim`, `clear_cupy_memory`, `trim_cupy_memory_pool`.

- [ ] **Step 1: Write failing tests for adjoints, PSF constraints, damping, lazy latent updates, and cuCIM cleanup**

Adapt the current CPU and fake-GPU tests from `deconvolution-gpu/tests/test_blind_rl.py` to import from `tiresias.blind_rl`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_blind_rl.py -q`
Expected: FAIL because functions are not implemented.

- [ ] **Step 3: Implement blind RL and cuCIM adapter**

Move the reusable CuPy/SciPy implementation from `workflow/scripts/blind_rl.py`, preserving memory cleanup and optional import errors.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_blind_rl.py -q`
Expected: PASS.

### Task 3: PSF Seeds and Tiled Estimation

**Files:**
- Create: `src/tiresias/seeds.py`
- Create: `src/tiresias/tiling.py`
- Create: `tests/test_seeds.py`
- Create: `tests/test_tiling.py`
- Modify: `src/tiresias/__init__.py`

**Interfaces:**
- Consumes: `tiresias.blind_rl.estimate_psf_array_cupy`, `clear_cupy_memory`, `trim_cupy_memory_pool`.
- Produces: `generate_theoretical_psf`, `generate_psf_seed`, `resolve_dxy`, `estimate_psf_from_chunks`, `open_tiff_memmap`, `resolve_cupy_blind_chunk_xy`.

- [ ] **Step 1: Write failing tests for seed validation, light-sheet rotation, tile selection, cache-key separation, CuPy direct array path, and OOM chunk retry helpers**

Use synthetic arrays and mocks so tests remain CPU-only.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_seeds.py tests/test_tiling.py -q`
Expected: FAIL because modules are not implemented.

- [ ] **Step 3: Implement seed generation and tiled estimation**

Move the reusable non-MATLAB code from `psf_estimation.py` and `psf_modes.py`; remove MATLAB backend branches and OME-Zarr imports.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_seeds.py tests/test_tiling.py -q`
Expected: PASS.

### Task 4: CLI and Documentation

**Files:**
- Create: `src/tiresias/cli.py`
- Create: `tests/test_cli.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

**Interfaces:**
- Consumes: `generate_theoretical_psf`, `estimate_psf_from_chunks`, `deconvolve_with_cucim`.
- Produces: console scripts `tiresias-estimate-psf` and `tiresias-deconvolve`.

- [ ] **Step 1: Write failing CLI parser tests**

Test argument parsing and that CLI functions call the library API with expected values using mocks.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -q`
Expected: FAIL because CLI module and entry points are missing.

- [ ] **Step 3: Implement CLI entry points and README usage**

Add TIFF-based PSF estimation and cuCIM deconvolution commands.

- [ ] **Step 4: Run full verification**

Run: `python -m pytest -q`
Expected: PASS.
