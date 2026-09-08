# Technology Stack

**Analysis Date:** 2026-09-08

## Languages

**Primary:**
- Python 3.10+ - GPU-accelerated scientific package for blind PSF estimation and deconvolution

**Secondary:**
- Shell/Bash - Build and CI/CD scripts in GitHub Actions workflows

## Runtime

**Environment:**
- Python 3.10, 3.11, 3.12 (as declared in `pyproject.toml`)
- NVIDIA CUDA 11.x (for GPU acceleration; production path requires CUDA-capable GPU)
- Optional CPU-only path using SciPy (reference/validation only, no GPU required)

**Package Manager:**
- pip (primary)
- uv (recommended in README.md for installation)
- Lockfile: Not detected (uses pyproject.toml version pinning)

## Frameworks

**Core:**
- NumPy 1.24+ - Array operations and computation foundation
- SciPy 1.10+ - Scientific computing (FFT, signal processing with `scipy.fft`, `scipy.signal`, `scipy.ndimage`)
- CuPy-CUDA11x 13-14 - GPU acceleration for NumPy-compatible arrays and FFT operations

**Testing:**
- pytest 8+ - Test framework and runner (configured in `pyproject.toml` with testpaths in `tests/`)

**Build/Dev:**
- setuptools 68+ - Python package building and distribution
- wheel - Binary distribution format

**Documentation:**
- Sphinx 7+ - Documentation generation
- MyST-Parser 2+ - Markdown support for Sphinx
- Furo 2024+ - Sphinx HTML theme

## Key Dependencies

**Critical:**
- `cupy-cuda11x` (13-14) - GPU computation backend; production path requires this; CPU fallback uses SciPy
- `tifffile` (2024.0+) - 3-D TIFF volume I/O; used in `cli.py` for `imread`/`imwrite` and `tiling.py` for memory-mapped TIFF access via `TiffFile` and `memmap`
- `psfmodels` (0.3+) - Theoretical PSF seed generation via `pm.make_psf()` in `seeds.py`; supports vectorial, scalar, and Gaussian PSF models with optical parameter handling

**Infrastructure:**
- NumPy 1.24+ - Array manipulation, broadcasting, FFT utility helpers in `blind_rl.py`
- SciPy 1.10+ - FFT operations (`scipy.fft.rfftn`, `scipy.fft.irfftn`, `scipy.fft.next_fast_len`), signal processing (`scipy.signal.fftconvolve`), and spatial operations (`scipy.ndimage.rotate`)

## Configuration

**Environment:**
- Python version specified via `.python-version` (if using pyenv/uv) or virtual environment
- CuPy CUDA cache location controlled via `CUPY_CACHE_DIR` environment variable (set automatically in `blind_rl.py` if not present; checks `SLURM_TMPDIR`, `TMPDIR`, or defaults to `/tmp`)
- GPU device selection via CUDA environment variables (standard NVIDIA driver config)
- VRAM management: `DEFAULT_CUPY_VRAM_FRACTION = 0.72` and `DEFAULT_CUPY_FFT_BYTES_PER_VOXEL = 208` in `tiling.py`

**Build:**
- `pyproject.toml` - Single source of truth for project metadata, dependencies, scripts, and pytest config
- Build backend: `setuptools.build_meta`
- Package discovery: `setuptools.packages.find` from `src/` directory
- CLI entry points defined in `[project.scripts]`:
  - `tiresias-estimate-psf` → `tiresias.cli:estimate_psf_main`
  - `tiresias-deconvolve` → `tiresias.cli:deconvolve_main`

## Platform Requirements

**Development:**
- Python 3.10+ with pip or uv
- Optional: CUDA 11.x and NVIDIA GPU (required for CuPy path; SciPy-only validation works on CPU)
- Optional: C compiler for any compiled extensions
- Git for version control

**Production:**
- Python 3.10+
- CUDA 11.x-compatible NVIDIA GPU (required for `cupy-cuda11x`)
- Compatible NVIDIA driver for installed CUDA version
- Sufficient VRAM (default 72% available memory allocated; configurable via `--vram-gb` CLI flag)

## Notes

- The package targets CUDA 11.x specifically; different CUDA versions require different CuPy variants (`cupy-cuda12x` for CUDA 12.x, etc.)
- CPU-only SciPy backend available for reference and validation workloads via direct API usage; CLI defaults to CuPy for production
- TIFF I/O is memory-mapped for large volumes (`tiff_memmap` in `tiling.py`)
- Build uses `--no-deps` in CI to install from source without transitive dependencies, then adds only specified requirements

---

*Stack analysis: 2026-09-08*
