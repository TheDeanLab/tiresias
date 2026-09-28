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
  --illumination-polar-deg 90.0 \
  --illumination-azimuthal-deg 0.0
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
| `--ni0` | none (psfmodels uses `1.515`) | Immersion refractive index the objective was designed for. Leaving this unset lets psfmodels apply its own design value (`1.515`, oil), which adds spherical aberration and a focal shift for water-immersion setups. Pass the same value as `--ni` (for example `1.33`) for an aberration-free seed. |
| `--slit-width` | none | ASLM slit window FWHM (same physical units as `--dz`), the Gaussian window the illumination is convolved with along the beam propagation direction. Applies only to `aslm` mode; exactly one of `--slit-width` or `--slit-width-px` must be supplied. |
| `--slit-width-px` | none | ASLM slit window FWHM as a sample count along the propagation axis, converted to physical units via `--dz`. Applies only to `aslm` mode; exactly one of `--slit-width` or `--slit-width-px` must be supplied. |
| `--illumination-polar-deg` | `90.0` | Polar angle, in degrees, of the illumination propagation direction, measured from the illumination's own pre-rotation +Z propagation axis. Used by both `light_sheet` and `aslm` modes. The default `90.0` (paired with the azimuthal default below) is the broadside default. |
| `--illumination-azimuthal-deg` | `0.0` | Azimuthal angle, in degrees, of the illumination propagation direction, measured in the X-Y plane from +X. Used by both `light_sheet` and `aslm` modes. The default `0.0` (paired with the polar default above) is the broadside default. |

These seven flags are shared by both `tiresias-estimate-psf` and `tiresias-deconvolve` via the same optical-argument parser.

The output PSF is a float32 TIFF normalized to sum to one.
Without `--psf-seed-path`, theoretical seed generation requires
`--detection-na` (or `--na`), `--wavelength`, `--dz`, `--ni`, `--ns`, and
either `--dxy` or the camera-pixel-size/magnification pair.

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
`--illumination-polar-deg`, `--illumination-azimuthal-deg`, and
the other optical arguments). When both `--psf-path` and optical flags are
given, the loaded PSF wins.

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

Every example below passes `ni0` equal to `ni`. `ni0` should equal `ni` unless
you are deliberately modeling an index mismatch between the objective's design
immersion and the sample's actual immersion — see the immersion-index entry
under Known limitations below.

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
    ni0=1.33,
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
    polar_deg=90.0,
    azimuthal_deg=0.0,
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
    ni0=1.33,
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
    polar_deg=90.0,
    azimuthal_deg=0.0,
    slit_width=2.0,
)
```

`psf_mode="single"` returns the detection seed. `psf_mode="light_sheet"`
multiplies the detection seed by a rotated illumination seed and normalizes the
result. `psf_mode="aslm"` models an axially swept light-sheet acquisition: a
rolling shutter reads out only a narrow slit around the swept beam waist at
any instant, assumed perfectly synchronized to the sweep. The effective
illumination is therefore the pre-rotation illumination PSF convolved with a
Gaussian slit window along the beam propagation direction — a sweep-integrated
model, not a static per-axis multiply — before `rotate_illumination` runs.
This applies for every illumination direction, including oblique ones, never
only at the broadside default. The illumination itself is the same 3-D pencil
beam that `light_sheet` uses; see the pencil-beam entry under Known
limitations below.

The illumination propagation direction is set by two angles in degrees:
`polar_deg`, measured from the illumination's own pre-rotation +Z optical
axis, and `azimuthal_deg`, measured in the X-Y plane from +X — the standard
physics spherical convention. The defaults `polar_deg=90.0, azimuthal_deg=0.0`
reproduce the pre-v1.1 broadside geometry (a beam propagating along X).

The canonical range for `polar_deg` is `[0, 180]` and for `azimuthal_deg` is
`[0, 360)`, but these ranges are documentation guidance, not enforced: any
finite float is accepted, and a value outside the canonical range describes an
equivalent, redundant geometry rather than being rejected. Non-finite values
(`NaN`, infinity) ARE rejected with a `ValueError`, before any PSF is
generated. At the poles (`polar_deg` of `0` or `180`) the beam points along
the Z axis and `azimuthal_deg` has no effect — this is an expected geometric
degeneracy, not a bug.

Rotation is performed in physical space, not voxel-index space: the
pre-rotation illumination PSF is resampled onto an isotropic grid at the `dxy`
spacing, rotated there, and resampled back to the native `(dz, dxy)` grid, so
a requested angle is an angle in micrometers, not in voxel indices. This
matters for this package specifically because `dz` is typically several times
`dxy`; a naive index-space rotation would tilt the beam by a visibly different
angle than requested. This physical-space resample applies to oblique
directions.

The four axis-aligned directions matching the pre-v1.1 cardinal cases (the old
single-angle rotation parameter's `0`, `90`, `180`, and `270` degree values —
including the default broadside direction) instead use an axis relabel that
reproduces pre-v1.1 output exactly. That relabel is geometrically exact only
when `dz == dxy`: with `dz != dxy`, the relabelled beam's propagation and
cross-section axes are mis-scaled by `dz`/`dxy`, because the relabel carries no
resampling step to correct for the anisotropic voxel size. Fixing this with a
proper physical-space resample at the cardinal directions is deferred (see
Known limitations below); every non-cardinal direction — including
orientations with a Y component, which the old single-angle parameter could
not express at all — already uses the physical-space pipeline described
above and is unaffected.

The slit window always integrates along the true beam propagation direction —
pre-rotation axis 0 — regardless of the requested `(polar_deg, azimuthal_deg)`
orientation or where that direction ends up after rotation. The window
is a Gaussian taper, not a hard binary mask — pixels outside the slit are
attenuated smoothly rather than zeroed. `slit_width` is the taper's full width
at half maximum (FWHM), and the taper is centered on the geometric midpoint of
the propagation axis, matching the centered beam waist that `psfmodels`
produces.

`slit_width` is a physical width in the same units as `dxy` and `dz`
(micrometers, per Input Expectations). `slit_width_px` is the equivalent
pixel-count form. Exactly one of the two must be supplied for `psf_mode="aslm"`
— supplying both, or neither, raises a `ValueError` — and both must be
positive.

`slit_width_px` is converted to physical units via `--dz`/`dz`, the sample
spacing of the pre-rotation propagation axis — never `dxy`, regardless of
which axis the beam ends up on after rotation.

The two forms produce identical seeds when the pixel count is scaled by the
same spacing used for conversion (`dz`). This example builds the same `aslm`
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
    ni0=1.33,
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
    polar_deg=90.0,
    azimuthal_deg=0.0,
)

seed_px = generate_psf_seed(psf_mode="aslm", slit_width_px=20, **common)
seed_physical = generate_psf_seed(psf_mode="aslm", slit_width=20 * 0.300, **common)

print(np.array_equal(seed_px, seed_physical))
# True
```

### Narrow and wide slit limits

Below one `dz` sample, `slit_width` applies no convolution: the seed is
identical to `light_sheet` — the waist-limited limit. As `slit_width`
grows, the seed converges instead to the sweep-average over the simulated
propagation window; it is not identical to `light_sheet` at any finite width,
and the value it converges toward depends on how long that window is.
Between the two limits, axial FWHM is monotonic non-decreasing in
`slit_width` — see `examples/slit_width_sweep.py`, which measures and
prints both limits and the trend between them for its own parameters, rather
than a number fixed here.

`slit_width` is the FWHM of a Gaussian window. Dean et al. 2015 used a
hard rectangular slit of width `W = 2*xR`; a Gaussian window with the same
second moment as that rectangle has FWHM ≈ `0.68*W`. Keep this mapping in mind
when relating `slit_width` to a rectangular-slit description from the
literature.

### emitter_offset

`emitter_offset` is a Python-API-only keyword on `generate_psf_seed()` — it is
not a CLI flag. It is the emitter's offset from the illumination waist along
the beam propagation direction, in the same units as `dxy`/`dz`. Its sign
follows `psfmodels.tot_psf`'s `x_offset` convention: a positive value moves
the waist to a smaller pre-rotation axis-0 index. `light_sheet` evaluates the
beam at this off-waist position, so its axial resolution degrades away from
the waist. `aslm` is invariant to `emitter_offset`: under the
perfectly synchronized sweep, the slit tracks the emitter across the whole
simulated window, so the effective illumination relative to the emitter never
depends on where the emitter sits. `single` has no illumination beam and
requires `emitter_offset=0.0`.

At the waist, `aslm` matches `light_sheet`; the difference between the two
modes shows up only off the waist, where `light_sheet` loses optical
sectioning and `aslm` does not — see `examples/light_sheet_vs_aslm.py`.

The ASLM slit window is a sweep-integrated convolution along the beam's
propagation direction, not a time-resolved acquisition simulation. The
rolling shutter is assumed to be perfectly synchronized with the swept beam
waist, so the illuminated slit always sits exactly at the waist. There is no
basis in this codebase for a credible timing-error distribution, so encoding
one would mean inventing numbers rather than modeling physics. Timing
jitter, shutter/beam desynchronization, and sweep-velocity error are
therefore not modeled and are explicitly out of scope for this milestone.

### Known limitations

- **Pencil-beam illumination.** `light_sheet` and `aslm` use a 3-D pencil
  beam, not a sheet integrated over the in-plane lateral axis the way DSLM or
  cylindrical-lens optics are. Lateral widths are therefore optimistic, and
  off-waist `light_sheet` axial profiles show Fresnel structure from the
  pencil beam's on-axis intensity zeros. The fix is deferred.
- **Cardinal-direction sampling.** The axis-relabel rotation used at the four
  legacy cardinal directions is exact only when `dz == dxy` (see the rotation
  discussion above) — the shipped example scripts use cubic voxels for this
  reason.
- **Immersion index.** Leaving `ni0` unset (`None`) means psfmodels applies
  its own design immersion index (`1.515`, oil). Pass `ni0` equal to `ni`
  unless you are deliberately modeling an index mismatch.

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

`Exactly one of slit_width or slit_width_px must be provided for psf_mode='aslm'`

`aslm` mode needs exactly one width form. Supplying both `slit_width` and
`slit_width_px`, or neither, raises this error. See the physical-unit vs.
pixel-count discussion in PSF Seed Modes above to choose the form that fits
the beam propagation axis.

`ASLM slit-integrated illumination has no positive finite energy`

The slit-integrated illumination came out non-finite or all-zero after the
Gaussian window was applied (or skipped, at a sub-`dz` `slit_width`). This
guard exists because normalizing such a result would otherwise pass silently
into blind estimation; the error fires after the detection and illumination
PSFs have already been generated — `generate_psf_seed` calls
`generate_theoretical_psf` twice before this guard runs — but before the seed
is returned, so with `psf_model="vectorial"` two potentially slow optical
computations will already have run. The ASLM model assumes the rolling
shutter is perfectly synchronized to the swept beam waist, so there is no
timing to adjust — check the illumination parameters (`illumination_na`,
`wavelength`, `ni`) and `slit_width`/`slit_width_px` instead.
