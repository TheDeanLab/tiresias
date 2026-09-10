# Tiresias Usage Guide

Tiresias estimates a blind point-spread function (PSF) from a 3-D TIFF volume
with CuPy, then can use that PSF for CuPy Richardson-Lucy restoration.

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

print("cupy", cupy.__version__)
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

To continue estimation from a calibrated or previously estimated PSF, use a
TIFF seed. Optical-model arguments are not required in this mode:

```bash
tiresias-estimate-psf \
  --image-path volume.tif \
  --output-path estimated_psf.tif \
  --psf-seed-path calibrated_psf.tif \
  --psf-size-z 61 \
  --psf-size-xy 128
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
  --cupy-fft-engine scout \
  --pad-xy 32 \
  --pad-z 20 \
  --latent-update-period 2 \
  --vram-gb 24
```

Run in `aslm` mode with a physical-units slit width:

```bash
tiresias-estimate-psf \
  --image-path volume.tif \
  --output-path estimated_psf.tif \
  --dxy 0.108 \
  --dz 0.300 \
  --wavelength 0.561 \
  --detection-na 1.0 \
  --illumination-na 0.2 \
  --ni 1.33 \
  --ns 1.33 \
  --psf-mode aslm \
  --slit-width 2.0 \
  --light-sheet-angle 90.0
```

Common PSF-estimation options:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--psf-seed-path` | none | Calibrated TIFF seed. Tiresias center-crops or pads it to the configured PSF support and normalizes its energy. |
| `--n-iters` | `10` | Blind Richardson-Lucy iterations per tile. |
| `--chunk-xy` | `256` | Requested XY core tile size. Tiresias may reduce it to fit VRAM. |
| `--blind-max-tiles` | `16` | Maximum representative high-SNR tiles. Use `0` for the full grid. |
| `--blind-z-slices` | `128` | Maximum Z planes used for blind estimation. Use `0` for full Z. |
| `--cupy-fft-engine` | `scout` | `scout` runs a short pass over selected tiles, keeps the most self-consistent PSFs, then finishes with CuPy. Use `cupyx` to run every selected tile for the full iteration count. |
| `--adaptive-scout-iters` | `2` | Number of short scout iterations used by `--cupy-fft-engine scout`. |
| `--adaptive-keep-tiles` | `4` | Number of scout-approved tiles to finish. |
| `--pad-xy` | `32` | XY halo around each tile. |
| `--pad-z` | `20` | Symmetric Z padding inside each CuPy blind-RL tile. |
| `--latent-update-period` | `2` | Update the latent image every N iterations. Use `1` for full alternating updates. |
| `--snr-weight-cap` | `100` | Caps any one tile's contribution to the merge. |
| `--vram-gb` | auto | Override detected free VRAM for tile sizing. |
| `--cache-dir` | `.psf_cache` next to input | Cache directory for merged PSFs. |
| `--no-psf-cache` | off | Force recomputation. |
| `--psf-mode` | `single` | Theoretical seed mode: `single`, `light_sheet`, or `aslm`. The default preserves existing behavior exactly. |
| `--slit-width` | none | ASLM slit gate FWHM in physical units (same units as `--dxy`/`--dz`). Applies only to `aslm` mode; exactly one of `--slit-width` or `--slit-width-px` must be supplied. |
| `--slit-width-px` | none | ASLM slit gate FWHM as a pixel count, converted to physical units via `--dxy`. Applies only to `aslm` mode; exactly one of `--slit-width` or `--slit-width-px` must be supplied. |
| `--slit-axis` | auto | Override the auto-detected ASLM gate axis (`0` for Z, `2` for X). Applies only to `aslm` mode. |
| `--light-sheet-angle` | `90.0` | Illumination rotation angle in degrees, used by both `light_sheet` and `aslm` modes. |

These five flags are shared by both `tiresias-estimate-psf` and `tiresias-deconvolve` via the same optical-argument parser.

The output PSF is a float32 TIFF normalized to sum to one.
Without `--psf-seed-path`, theoretical seed generation requires
`--wavelength`, `--dz`, `--ni`, `--ns`, and either `--dxy` or the
camera-pixel-size/magnification pair.

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
accelerated CuPy FFT Richardson-Lucy restoration, and writes a uint16 TIFF.

`--psf-path` is optional. Supplying it loads that TIFF as the PSF. Omitting it
makes `tiresias-deconvolve` build a theoretical seed through the same
`generate_psf_seed()` path and the same optical and slit flags as
`tiresias-estimate-psf` (`--psf-mode`, `--slit-width`, `--slit-width-px`,
`--slit-axis`, `--light-sheet-angle`, and the other optical arguments). When
both `--psf-path` and optical flags are given, the loaded PSF wins.

Generate an ASLM seed on the fly, with no PSF path:

```bash
tiresias-deconvolve \
  --image-path volume.tif \
  --output-path restored.tif \
  --dxy 0.108 \
  --dz 0.300 \
  --wavelength 0.561 \
  --detection-na 1.0 \
  --illumination-na 0.2 \
  --ni 1.33 \
  --ns 1.33 \
  --psf-mode aslm \
  --slit-width 2.0 \
  --n-iters 20 \
  --device-id 0
```

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

Run CuPy deconvolution:

```python
from tifffile import imread, imwrite

from tiresias import deconvolve_with_cupy

image = imread("volume.tif")
psf = imread("estimated_psf.tif")
restored = deconvolve_with_cupy(image, psf, n_iters=20, device_id=0)
imwrite("restored.tif", restored)
```

Use the SciPy reference implementation for small validation cases:

```python
from tiresias import estimate_blind_psf_scipy

estimated_psf = estimate_blind_psf_scipy(observed, initial_psf, n_iters=4)
```

## PSF Seed Modes

`generate_psf_seed()` builds the theoretical seed for all three modes —
`single`, `light_sheet`, and `aslm`. Both `tiresias-estimate-psf` and
`tiresias-deconvolve` route their theoretical seed generation through this same
function, selected by `--psf-mode`.

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

```python
from tiresias import generate_psf_seed

seed = generate_psf_seed(
    psf_mode="aslm",
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
    slit_width=2.0,
)
```

`psf_mode="single"` returns the detection seed. `psf_mode="light_sheet"`
multiplies the detection seed by a rotated illumination seed and normalizes the
result. `psf_mode="aslm"` builds the same detection-times-rotated-illumination
product as `light_sheet`, but first multiplies the illumination PSF, in its
pre-rotation frame, by a narrow Gaussian slit gate — modeling the rolling
shutter that follows the swept beam waist, assumed perfectly synchronized to
it.

The slit gate multiplies the illumination PSF in its pre-rotation frame,
before `rotate_illumination_psf` runs. Applying the gate before rotation is
what keeps the result correct at oblique `light_sheet_angle` values, not only
at 90 degrees.

The gate narrows exactly one axis of the `(z, y, x)` volume. By default that
axis is derived from `light_sheet_angle` by rounding the angle to the nearest
cardinal quadrant: even quadrants gate axis 0 (Z), odd quadrants gate axis 2
(X). The default `light_sheet_angle=90.0` gates axis 2 (X); `0.0` or `180.0`
gate axis 0 (Z). Exact tie angles resolve through Python's round-half-to-even
rule, so 45, 135, 225, and 315 degrees all resolve to axis 0. This rounding is
unconditional — an oblique angle still snaps to a cardinal gating axis, which
is exactly why the `slit_axis` override exists.

Pass `slit_axis` to bypass auto-detection entirely. It accepts only `0` (Z) or
`2` (X); `1` (Y) is rejected with a `ValueError`, because `rotate_illumination_psf`
only rotates the Z/X plane, so gating Y would not correspond to any
rolling-shutter direction. Use the override for oblique `light_sheet_angle`
values where the snapped axis is not the one you want, or to match a
lab-specific axis convention.

The gate itself is a Gaussian taper, not a hard binary mask — pixels outside
the slit are attenuated smoothly rather than zeroed. `slit_width` is the
taper's full width at half maximum (FWHM), and the taper is centered on the
geometric midpoint of the gated axis, matching the centered beam waist that
`psfmodels` produces. The pixel spacing used to convert the physical FWHM into
pixels is `dz` when the gate axis is 0 (Z) and `dxy` when it is 2 (X).

`slit_width` is a physical width in the same units as `dxy` and `dz`
(micrometers, per Input Expectations). `slit_width_px` is the equivalent
pixel-count form. Exactly one of the two must be supplied for `psf_mode="aslm"`
— supplying both, or neither, raises a `ValueError` — and both must be
positive.

`slit_width_px` is always converted to physical units through `dxy`, even when
the resolved gate axis is 0 (Z, whose samples are spaced by `dz`) — this is a
deliberate design decision, not an oversight. If you are gating Z, the
pixel-count form does not mean "this many Z planes"; prefer the physical
`slit_width` form when gating Z unless you specifically want the
`dxy`-scaled behavior.

The two forms produce identical seeds when the pixel count is scaled by the
same spacing used for conversion (`dxy`). This example builds the same `aslm`
seed two ways — once with a physical `slit_width`, once with an equivalent
`slit_width_px` — and confirms they match:

```python
import numpy as np

from tiresias import generate_psf_seed

common = dict(
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

seed_px = generate_psf_seed(psf_mode="aslm", slit_width_px=20, **common)
seed_physical = generate_psf_seed(psf_mode="aslm", slit_width=20 * 0.108, **common)

print(np.array_equal(seed_px, seed_physical))
# True
```

At the wide end of the `slit_width` range, `aslm` reduces exactly to
`light_sheet`. Once `slit_width` reaches or exceeds the full extent of the
gated axis, the Gaussian taper described above is skipped entirely rather than
merely widened, so `aslm` output is bit-identical to the equivalent
`light_sheet` output, not just numerically close. Compute that full extent for
your own parameters from the gated axis's sample count times its pixel
spacing: `psf_size_xy * dxy` when the gate axis is 2 (X), or `psf_size_z * dz`
when it is 0 (Z). This is a useful sanity check that your ASLM parameters are
wired correctly, and it means `aslm` degrades gracefully into the mode you
already know at wide slit widths rather than failing or producing something
you cannot reason about.

```python
import numpy as np

from tiresias import generate_psf_seed

common = dict(
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

# light_sheet_angle=90.0 gates axis 2 (X), whose full extent is
# psf_size_xy * dxy.
full_extent = 128 * 0.108

seed_light_sheet = generate_psf_seed(psf_mode="light_sheet", **common)
seed_aslm_full_width = generate_psf_seed(
    psf_mode="aslm", slit_width=full_extent, **common
)

print(np.array_equal(seed_light_sheet, seed_aslm_full_width))
# True
```

The ASLM slit gate is a fixed spatial taper, not a time-resolved acquisition
simulation. The rolling shutter is assumed to be perfectly synchronized with
the swept beam waist, so the illuminated slit always sits exactly at the
waist. There is no basis in this codebase for a credible timing-error
distribution, so encoding one would mean inventing numbers rather than
modeling physics. Timing jitter, shutter/beam desynchronization, and
sweep-velocity error are therefore not modeled and are explicitly out of
scope for this milestone.

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
- `--cupy-fft-engine scout` is the default speed-oriented mode. It is most useful
  when high-SNR tile selection may include contaminated, saturated, edge-heavy,
  or otherwise unrepresentative regions; the scout pass filters tiles by PSF
  shape agreement before spending the remaining iterations.
- `--cupy-fft-engine cupyx` is the direct full-iteration path. Use it when you
  want every selected tile to contribute, or when a dataset may contain real
  spatial PSF variation that the scout filter could treat as disagreement.
- Lowering `--blind-max-tiles` below `16` can increase speed significantly
  because fewer CuPy blind-RL tiles are run. Treat it as an explicit throughput
  knob: validate against `16` tiles, MATLAB, or a known-good reference before
  using lower values as a production default.
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

`Restoration requires cupy`

Install CuPy in the same Python environment used to run `tiresias-deconvolve`.

`Observed image has no positive finite signal`

The selected tile contains no usable positive signal after NaN/Inf cleanup.
Check the input volume, channel selection, and background handling.

`All chunks failed during PSF estimation`

Reduce `--chunk-xy`, reduce `--blind-z-slices`, check GPU availability, and
confirm that the input TIFF is a 3-D volume with non-zero signal.
