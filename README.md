# Tiresias

Tiresias is a GPU-first Python package for blind point-spread-function (PSF)
estimation and cuCIM Richardson-Lucy restoration.

It extracts the reusable CuPy, SciPy, and cuCIM components from an internal
Astrocyte/Nextflow workflow into a standalone open-source library. Workflow
orchestration, MATLAB compatibility, and Astrocyte-specific file handling are
not part of this package.

## Features

- SciPy reference implementation of fixed-support blind Richardson-Lucy PSF
  estimation.
- CuPy implementation using `cupyx.scipy.signal.fftconvolve`.
- cuCIM adapter for Richardson-Lucy image restoration.
- Theoretical PSF seed generation with `psfmodels`.
- Single-detection and light-sheet PSF seed modes.
- TIFF-based chunked PSF estimation with SNR-weighted merge and cache support.

## Requirements

Tiresias requires a CUDA-capable GPU for production PSF estimation and
deconvolution. The SciPy implementation is retained as a numerical reference
and test helper.

The default package metadata targets CUDA 11.x, matching the source workflow's
CUDA 11.8 environment.

```bash
python -m pip install .
```

## CLI

Estimate a PSF from a TIFF volume:

```bash
tiresias-estimate-psf \
  --image-path volume.tif \
  --output-path estimated_psf.tif \
  --dxy 0.108 \
  --dz 0.3 \
  --wavelength 0.561 \
  --detection-na 1.0 \
  --ni 1.33 \
  --ns 1.33
```

Run cuCIM Richardson-Lucy restoration on a TIFF volume:

```bash
tiresias-deconvolve \
  --image-path volume.tif \
  --psf-path estimated_psf.tif \
  --output-path restored.tif \
  --n-iters 20
```

## License

Tiresias is licensed under the Apache License, Version 2.0.
