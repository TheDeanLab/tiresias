# Tiresias Usage Guide

Tiresias estimates a blind point-spread function (PSF) from a 3-D TIFF volume
with CuPy, then can use that PSF for cuCIM Richardson-Lucy restoration.

The production path requires a CUDA-capable GPU. The SciPy implementation is
included as a numerical reference and for tests, not as the intended production
backend.

## Install

The package metadata targets CUDA 11.x through `cupy-cuda11x`.

```bash
cd tiresias
python -m pip install .
```

Verify the GPU stack:

```bash
python - <<'PY'
import cupy
import cucim

print("cupy", cupy.__version__)
print("cucim", cucim.__version__)
print("gpu_count", cupy.cuda.runtime.getDeviceCount())
PY
```

If `getDeviceCount()` raises a CUDA driver/runtime error, fix the host CUDA
driver or install a package build that matches the driver before running
Tiresias.

## Input Expectations

- Input image volumes are TIFF stacks.
- Arrays are interpreted as `(z, y, x)`.
- Pixel sizes are in micrometers.
- Intensities should be finite and non-negative after background handling.
- The PSF seed must fit inside the selected blind-estimation volume. If the
  seed has more Z planes than the blind window, Tiresias center-crops it in Z
  and renormalizes it.

Tiresias currently excludes MATLAB fallback and OME-Zarr workflow I/O.

## Estimate a PSF

Minimal CLI example:

```bash
tiresias-estimate-psf \
  --image-path volume.tif \
  --output-path estimated_psf.tif \
  --dxy 0.108 \
  --dz 0.300 \
  --wavelength 0.561 \
  --detection-na 1.0 \
  --ni 1.33 \
  --ns 1.33
```

Useful tuning flags:

```bash
tiresias-estimate-psf \
  --image-path volume.tif \
  --output-path estimated_psf.tif \
  --dxy 0.108 \
  --dz 0.300 \
  --wavelength 0.561 \
  --detection-na 1.0 \
  --ni 1.33 \
  --ns 1.33 \
  --n-iters 20 \
  --chunk-xy 256 \
  --blind-max-tiles 16 \
  --blind-z-slices 128 \
  --pad-xy 32 \
  --pad-z 20 \
  --latent-update-period 2 \
  --vram-gb 24
```

Common PSF-estimation options:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--n-iters` | `10` | Blind Richardson-Lucy iterations per tile. |
| `--chunk-xy` | `256` | Requested XY core tile size. Tiresias may reduce it to fit VRAM. |
| `--blind-max-tiles` | `16` | Maximum representative high-SNR tiles. Use `0` for the full grid. |
| `--blind-z-slices` | `128` | Maximum Z planes used for blind estimation. Use `0` for full Z. |
| `--pad-xy` | `32` | XY halo around each tile. |
| `--pad-z` | `20` | Symmetric Z padding inside each CuPy blind-RL tile. |
| `--latent-update-period` | `2` | Update the latent image every N iterations. Use `1` for full alternating updates. |
| `--snr-weight-cap` | `100` | Caps any one tile's contribution to the merge. |
| `--vram-gb` | auto | Override detected free VRAM for tile sizing. |
| `--cache-dir` | `.psf_cache` next to input | Cache directory for merged PSFs. |
| `--no-psf-cache` | off | Force recomputation. |

The output PSF is a float32 TIFF normalized to sum to one.

## Deconvolve a TIFF Volume

```bash
tiresias-deconvolve \
  --image-path volume.tif \
  --psf-path estimated_psf.tif \
  --output-path restored.tif \
  --n-iters 20 \
  --device-id 0
```

The deconvolution command loads the image and PSF from TIFF, runs
`cucim.skimage.restoration.richardson_lucy` with `clip=False`, and writes a
float32 TIFF.

## Python API

Estimate a PSF:

```python
from pathlib import Path

from tifffile import imwrite

from tiresias import estimate_psf_from_chunks, generate_theoretical_psf

seed = generate_theoretical_psf(
    detection_na=1.0,
    wavelength=0.561,
    ni=1.33,
    ns=1.33,
    dxy=0.108,
    dz=0.300,
    psf_size_z=61,
    psf_size_xy=128,
)

psf = estimate_psf_from_chunks(
    image_path=Path("volume.tif"),
    psf_seed=seed,
    n_iters=20,
    chunk_xy=256,
    blind_max_tiles=16,
)

imwrite("estimated_psf.tif", psf)
```

Run cuCIM deconvolution:

```python
from tifffile import imread, imwrite

from tiresias import deconvolve_with_cucim

image = imread("volume.tif")
psf = imread("estimated_psf.tif")
restored = deconvolve_with_cucim(image, psf, n_iters=20, device_id=0)
imwrite("restored.tif", restored)
```

Use the SciPy reference implementation for small validation cases:

```python
from tiresias import estimate_blind_psf_scipy

estimated_psf = estimate_blind_psf_scipy(observed, initial_psf, n_iters=4)
```

## PSF Seed Modes

The CLI currently generates a single-detection theoretical PSF seed. The Python
API also exposes `generate_psf_seed()` for light-sheet seed construction:

```python
from tiresias import generate_psf_seed

seed = generate_psf_seed(
    psf_mode="light_sheet",
    na=1.0,
    detection_na=1.0,
    illumination_na=0.2,
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
```

`psf_mode="single"` returns the detection seed. `psf_mode="light_sheet"`
multiplies the detection seed by a rotated illumination seed and normalizes the
result.

## Performance Notes

- Tiresias clamps blind PSF estimation to one CuPy tile worker. This avoids
  multiple Python processes competing for one CUDA context and cuFFT workspace.
- `--chunk-xy` is a request, not a guarantee. Tiresias estimates cuFFT memory
  use and lowers the tile size when needed.
- If CuPy raises an out-of-memory error, Tiresias discards the partial pass and
  retries every tile with the next smaller aligned tile size.
- `--blind-max-tiles 16` is usually much faster than processing the whole grid.
  Use `--blind-max-tiles 0` for comparison or reproducibility studies where
  full-grid coverage is required.
- The cache key includes image metadata, seed content, tile sizing, selected Z
  window, iteration count, and tile-selection settings.

## Troubleshooting

`cudaErrorInsufficientDriver`

The installed CUDA runtime is newer than the host driver supports, or the
driver is not visible inside the job/container. Use `nvidia-smi` and the CuPy
verification snippet above to confirm driver/runtime compatibility.

`CuPy is missing from the worker environment`

Install Tiresias in an environment that includes the required CuPy wheel. The
default package metadata installs `cupy-cuda11x`.

`Restoration requires both cupy and cucim`

Install cuCIM in the same Python environment used to run `tiresias-deconvolve`.

`Observed image has no positive finite signal`

The selected tile contains no usable positive signal after NaN/Inf cleanup.
Check the input volume, channel selection, and background handling.

`All chunks failed during PSF estimation`

Reduce `--chunk-xy`, reduce `--blind-z-slices`, check GPU availability, and
confirm that the input TIFF is a 3-D volume with non-zero signal.
