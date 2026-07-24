<h1 align="center">Tiresias</h1>
<h2 align="center">GPU-first blind PSF estimation for 3-D microscopy.</h2>

<p align="center">
  <a href="https://github.com/TheDeanLab/tiresias/actions/workflows/ci.yml?query=branch%3Amain"><img src="https://github.com/TheDeanLab/tiresias/actions/workflows/ci.yml/badge.svg?branch=main" alt="Tests"></a>
  <a href="https://github.com/TheDeanLab/tiresias/actions/workflows/docs.yml?query=branch%3Amain"><img src="https://github.com/TheDeanLab/tiresias/actions/workflows/docs.yml/badge.svg?branch=main" alt="Docs"></a>
</p>

Tiresias is an open source Python package for blind point-spread-function (PSF)
estimation and cuCIM Richardson-Lucy deconvolution from 3-D TIFF image
volumes.

## Package Scope

Tiresias is a reusable Python package for blind PSF estimation and deconvolution.
It provides compact, GPU-accelerated kernels and command-line entrypoints for
direct use in microscopy image-processing pipelines.

The package intentionally excludes:

- MATLAB runtime compatibility.
- Non-3D image support.

Production users should use:

- `tiresias-estimate-psf` for GPU blind PSF estimation.
- `tiresias-deconvolve` for cuCIM-based Richardson-Lucy restoration.

## Current Functionality

- Single-detection PSF seed generation via `psfmodels.make_psf`.
- Blind PSF estimation by alternating latent-image and kernel updates.
- CuPy backend for tile-based blind estimation with automatic VRAM-aware chunk
  sizing.
- Optional deterministic merge caching for repeated PSF estimation runs.
- SNR-weighted selection of representative XY tiles.
- Optional prefetch controls for overlapping CPU preparation.
- cuCIM-backed Richardson-Lucy restoration (`cucim.skimage.restoration.richardson_lucy`).
- A SciPy backend implementation retained as a numerical reference and for
  validation.
- CLI entrypoints in `project.scripts`:
  - `tiresias-estimate-psf`
  - `tiresias-deconvolve`
- Python API for direct integration (`generate_theoretical_psf`, `estimate_psf_from_chunks`,
  `deconvolve_with_cucim`, etc.).
- Built-in CuPy cache cleanup helpers to stabilize long-running GPU jobs.

## Requirements

- Python `>=3.10`.
- A CUDA-capable GPU for the production path.
- The SciPy path is available without GPU but intended for reference and small
  validation workloads.
- The package metadata targets CUDA 11.x through `cupy-cuda11x`.
- Core dependencies include:
  - `numpy`
  - `scipy`
  - `tifffile`
  - `psfmodels`
  - `cupy-cuda11x`
  - `cucim`

The production path can fail quickly when:

- CuPy/CUDA and the installed NVIDIA driver are incompatible.
- cuCIM is missing in the active Python environment.
- The input image is not a 3-D volume.

## Installation

### Requirements install (recommended)

Install `uv`:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

Create a Python 3.10 virtual environment and install Tiresias:

```bash
uv python install 3.10
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

```powershell
# Windows (PowerShell)
uv python install 3.10
uv venv --python 3.10
.venv\Scripts\Activate.ps1
uv pip install -e .
```

### Alternative pip install

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

### Optional installation variants

```bash
python -m pip install -e ".[dev]"
```

## Input and Data Expectations

- Input volumes must be TIFF stacks.
- Volumes are interpreted as `Z, Y, X` in all estimation and restoration steps.
- Intensity values should be finite and mostly non-negative after cleanup.
- Pixel units are in micrometers.
- The PSF seed is adapted automatically if it is larger than the selected blind
  estimation volume.

## Command Line Interface

The package exposes two command-line entrypoints:

```text
tiresias-estimate-psf   --image-path IMAGE --output-path PSF --wavelength ... --ni ... --ns ... --dz ...
tiresias-deconvolve     --image-path IMAGE --psf-path PSF --output-path RESTORED [--n-iters N] [--device-id D]
```

See `docs/usage.md` for the full parameter list. The key flags and defaults are
below.

## CLI: PSF estimation

```bash
tiresias-estimate-psf \
  --image-path volume.tif \
  --output-path estimated_psf.tif \
  --wavelength 0.561 \
  --ni 1.33 \
  --ns 1.33 \
  --dz 0.300 \
  --detection-na 1.0 \
  --dxy 0.108
```

Core options:

- `--n-iters`: number of blind-RL iterations (default `10`).
- `--chunk-xy`: requested XY tile core size (default `256`).
- `--blind-max-tiles`: max representative tiles (`16`, `0` uses all tiles).
- `--blind-z-slices`: number of z planes to use (`128`, `0` uses full Z).
- `--pad-xy`: halo around XY tile reads (default `32`).
- `--pad-z`: Z padding inside each tile (default `20`).
- `--prefetch-chunks`: queued worker concurrency prefetch.
- `--vram-gb`: override detected free VRAM.
- `--cache-dir`: cache root for merged PSF results.
- `--no-psf-cache`: disable cache reads/writes.
- `--peak-normalization`: `none`, `gamma`, or `unit`.
- `--peak-gamma-max`: gamma cap for peak normalization.
- `--latent-update-period`: alternating update period (default `2`).
- `--snr-weight-cap`: tile weight cap for merge.

Optics and seed options:

- `--wavelength` (required), `--dz` (required), `--ni` (required), `--ns` (required).
- `--na`, `--detection-na`, `--illumination-na`.
- `--camera-pixel-size` and `--magnification` can replace `--dxy`.
- `--psf-size-z`, `--psf-size-xy`, `--psf-model`, `--background`.
- `--psf-model` choices are `vectorial`, `scalar`, `gaussian`.
- Optional objective/medium overrides: `--ni0`, `--tg`, `--tg0`, `--ng`, `--ng0`, `--ti0`.
- `--oversample-factor` tuning.

## CLI: restoration

```bash
tiresias-deconvolve \
  --image-path volume.tif \
  --psf-path estimated_psf.tif \
  --output-path restored.tif \
  --n-iters 20 \
  --device-id 0
```

- `--n-iters`: RL restoration iterations (default `20`).
- `--device-id`: CUDA device index for cuCIM (default `0`).

## Python API

```python
from pathlib import Path
from tifffile import imread, imwrite

from tiresias import (
    generate_psf_seed,
    estimate_psf_from_chunks,
    deconvolve_with_cucim,
)

seed = generate_psf_seed(
    psf_mode="single",
    na=1.0,
    detection_na=1.0,
    illumination_na=None,
    wavelength=0.561,
    ni=1.33,
    ns=1.33,
    ni0=None,
    tg=None,
    tg0=None,
    ng=None,
    ng0=None,
    ti0=None,
    oversample_factor=3,
    psf_model="vectorial",
    dxy=0.108,
    dz=0.300,
    psf_size_z=61,
    psf_size_xy=128,
    background=0.0,
    light_sheet_angle=90.0,
)

psf = estimate_psf_from_chunks(
    image_path=Path("volume.tif"),
    psf_seed=seed,
    n_iters=10,
    chunk_xy=256,
    blind_max_tiles=16,
    blind_z_slices=128,
    pad_xy=32,
    pad_z=20,
)
imwrite("estimated_psf.tif", psf)

image = imread("volume.tif")
restored = deconvolve_with_cucim(image, psf, n_iters=20, device_id=0)
imwrite("restored.tif", restored)
```

```python
from tiresias import (
    estimate_blind_psf_scipy,
    clear_cupy_memory,
    trim_cupy_memory_pool,
    resolve_dxy,
)

dxy = resolve_dxy(dxy=None, camera_pixel_size=6.5, magnification=60)
clear_cupy_memory(device_id=0)
trimmed = trim_cupy_memory_pool(8 * 1024**3, device_id=0)
```

## Performance and Runtime Notes

- The blind estimator is aligned to a single GPU process worker to avoid
  multi-process CUDA contention.
- VRAM is used to estimate a safe maximum chunk size before estimation starts.
- `--chunk-xy` is a request and may be reduced automatically to fit available
  memory.
- On OOM during the first tiles, Tiresias retries with smaller chunk size before
  failing.
- The cache key includes image metadata, seed content, and major algorithmic
  settings.
- Tile weighting uses SNR statistics and weight capping to avoid outlier chunk
  domination.
- Restored TIFF outputs are written as `float32`.

## Output Layout

Tiresias writes TIFF outputs by default:

- Blind PSF outputs are written to the `--output-path` argument as a normalized
  `float32` TIFF.
- Restored volumes are written to the `--output-path` argument as a `float32`
  TIFF.
- Cached merged PSFs are stored under `.psf_cache` beside the input volume unless
  `--cache-dir` is provided.

## Troubleshooting

`cudaErrorInsufficientDriver`

This usually means a host CUDA driver/runtime mismatch. Confirm with
`nvidia-smi`, then run the GPU verification snippet from `docs/usage.md`.

`CuPy is missing from the worker environment`

Install Tiresias in an environment that includes `cupy-cuda11x` for the same
Python and CUDA runtime.

`Restoration requires both cupy and cucim`

Install cuCIM into the exact runtime that executes `tiresias-deconvolve`.

`Observed image has no positive finite signal`

Tiresias requires valid positive signal in the selected blind-estimation window.
Verify preprocessing and window selection.

`All chunks failed during PSF estimation`

Reduce `--chunk-xy`, reduce `--blind-z-slices`, or verify GPU availability and
input signal quality.

## Development

Run the test suite:

```bash
python -m pytest
```

## License

Tiresias is licensed under the Apache 2.0 License.

See `LICENSE` for details.
