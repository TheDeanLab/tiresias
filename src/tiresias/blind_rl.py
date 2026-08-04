"""Shared SciPy/CuPy blind Richardson-Lucy PSF estimation."""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.fft import next_fast_len

Array = Any
Convolve = Callable[..., Array]


def _configure_cupy_cache() -> None:
    """Keep CuPy JIT artifacts off read-only container home mounts."""
    if os.environ.get("CUPY_CACHE_DIR"):
        return
    cache_root = (
        os.environ.get("SLURM_TMPDIR")
        or os.environ.get("TMPDIR")
        or "/tmp"
    )
    os.environ["CUPY_CACHE_DIR"] = str(
        Path(cache_root) / f"cupy-kernel-cache-{os.getuid()}"
    )


_configure_cupy_cache()


def _shape(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"Invalid shape: {result}")
    return result


def _fft_shape(*shapes: Sequence[int]) -> tuple[int, ...]:
    return tuple(
        next_fast_len(
            sum(int(shape[axis]) for shape in shapes) - len(shapes) + 1
        )
        for axis in range(len(shapes[0]))
    )


def _crop_slices(starts: Sequence[int], shape: Sequence[int]) -> tuple[slice, ...]:
    return tuple(
        slice(int(start), int(start) + int(size))
        for start, size in zip(starts, shape)
    )


class FftConvolutionEngine:
    """Fixed-shape real FFT convolutions for blind-RL tile updates."""

    def __init__(
        self,
        xp: Any,
        image_shape: Sequence[int],
        kernel_shape: Sequence[int],
    ):
        self.xp = xp
        self.image_shape = _shape(image_shape)
        self.kernel_shape = _shape(kernel_shape)
        if len(self.image_shape) != len(self.kernel_shape):
            raise ValueError("Image and kernel dimensionality must match")

        if self.xp is np:
            from scipy import fft

            self._fft_module = fft
        else:
            self._fft_module = self.xp.fft
        self.full_shape = tuple(
            int(image) + int(kernel) - 1
            for image, kernel in zip(self.image_shape, self.kernel_shape)
        )
        self.forward_fft_shape = tuple(next_fast_len(size) for size in self.full_shape)
        self.forward_same_slices = _crop_slices(
            ((size - 1) // 2 for size in self.kernel_shape),
            self.image_shape,
        )
        self.embed_slices = self.forward_same_slices
        self.image_adjoint_fft_shape = _fft_shape(self.full_shape, self.kernel_shape)
        self.image_adjoint_slices = _crop_slices(
            (size - 1 for size in self.kernel_shape),
            self.image_shape,
        )
        self.psf_adjoint_fft_shape = _fft_shape(self.full_shape, self.image_shape)
        self.psf_adjoint_slices = _crop_slices(
            (size - 1 for size in self.image_shape),
            self.kernel_shape,
        )
        self._embed_buffer: Array | None = None

    def _check_shape(self, values: Array, expected: Sequence[int], name: str) -> None:
        if tuple(int(size) for size in values.shape) != tuple(expected):
            raise ValueError(
                f"{name} shape {tuple(values.shape)} does not match expected "
                f"{tuple(expected)}"
            )

    def _convolve_crop(
        self,
        first: Array,
        second: Array,
        fft_shape: Sequence[int],
        crop: tuple[slice, ...],
    ) -> Array:
        fft = self._fft_module
        first_fft = fft.rfftn(first, s=tuple(fft_shape))
        second_fft = fft.rfftn(second, s=tuple(fft_shape))
        result = fft.irfftn(first_fft * second_fft, s=tuple(fft_shape))
        return result[crop].astype(first.dtype, copy=False)

    def _embed_same_adjoint(self, values: Array) -> Array:
        self._check_shape(values, self.image_shape, "Values")
        if (
            self._embed_buffer is None
            or tuple(int(size) for size in self._embed_buffer.shape) != self.full_shape
            or self._embed_buffer.dtype != values.dtype
        ):
            self._embed_buffer = self.xp.zeros(self.full_shape, dtype=values.dtype)
        else:
            self._embed_buffer.fill(0)
        self._embed_buffer[self.embed_slices] = values
        return self._embed_buffer

    def convolve_same(self, image: Array, kernel: Array) -> Array:
        self._check_shape(image, self.image_shape, "Image")
        self._check_shape(kernel, self.kernel_shape, "Kernel")
        return self._convolve_crop(
            image,
            kernel,
            self.forward_fft_shape,
            self.forward_same_slices,
        )

    def image_adjoint(self, values: Array, kernel: Array) -> Array:
        self._check_shape(kernel, self.kernel_shape, "Kernel")
        embedded = self._embed_same_adjoint(values)
        return self._convolve_crop(
            embedded,
            self.xp.flip(kernel, axis=tuple(range(kernel.ndim))),
            self.image_adjoint_fft_shape,
            self.image_adjoint_slices,
        )

    def psf_adjoint(self, values: Array, image: Array) -> Array:
        self._check_shape(image, self.image_shape, "Image")
        embedded = self._embed_same_adjoint(values)
        return self._convolve_crop(
            embedded,
            self.xp.flip(image, axis=tuple(range(image.ndim))),
            self.psf_adjoint_fft_shape,
            self.psf_adjoint_slices,
        )


class BatchedFftConvolutionEngine:
    """Fixed-shape real FFT convolutions with a leading tile batch dimension."""

    def __init__(
        self,
        xp: Any,
        image_shape: Sequence[int],
        kernel_shape: Sequence[int],
        *,
        batch_size: int,
    ):
        self.xp = xp
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.image_shape = _shape(image_shape)
        self.kernel_shape = _shape(kernel_shape)
        if len(self.image_shape) != len(self.kernel_shape):
            raise ValueError("Image and kernel dimensionality must match")

        if self.xp is np:
            from scipy import fft

            self._fft_module = fft
        else:
            self._fft_module = self.xp.fft
        self.spatial_axes = tuple(range(1, len(self.image_shape) + 1))
        self.full_shape = tuple(
            int(image) + int(kernel) - 1
            for image, kernel in zip(self.image_shape, self.kernel_shape)
        )
        self.forward_fft_shape = tuple(next_fast_len(size) for size in self.full_shape)
        self.forward_same_slices = (slice(None),) + _crop_slices(
            ((size - 1) // 2 for size in self.kernel_shape),
            self.image_shape,
        )
        self.embed_slices = self.forward_same_slices
        self.image_adjoint_fft_shape = _fft_shape(self.full_shape, self.kernel_shape)
        self.image_adjoint_slices = (slice(None),) + _crop_slices(
            (size - 1 for size in self.kernel_shape),
            self.image_shape,
        )
        self.psf_adjoint_fft_shape = _fft_shape(self.full_shape, self.image_shape)
        self.psf_adjoint_slices = (slice(None),) + _crop_slices(
            (size - 1 for size in self.image_shape),
            self.kernel_shape,
        )
        self._embed_buffer: Array | None = None

    def _check_shape(self, values: Array, expected: Sequence[int], name: str) -> None:
        expected_shape = (self.batch_size,) + tuple(expected)
        if tuple(int(size) for size in values.shape) != expected_shape:
            raise ValueError(
                f"{name} shape {tuple(values.shape)} does not match expected "
                f"{expected_shape}"
            )

    def _convolve_crop(
        self,
        first: Array,
        second: Array,
        fft_shape: Sequence[int],
        crop: tuple[slice, ...],
    ) -> Array:
        fft = self._fft_module
        first_fft = fft.rfftn(
            first,
            s=tuple(fft_shape),
            axes=self.spatial_axes,
        )
        second_fft = fft.rfftn(
            second,
            s=tuple(fft_shape),
            axes=self.spatial_axes,
        )
        result = fft.irfftn(
            first_fft * second_fft,
            s=tuple(fft_shape),
            axes=self.spatial_axes,
        )
        return result[crop].astype(first.dtype, copy=False)

    def _embed_same_adjoint(self, values: Array) -> Array:
        self._check_shape(values, self.image_shape, "Values")
        expected_shape = (self.batch_size,) + self.full_shape
        if (
            self._embed_buffer is None
            or tuple(int(size) for size in self._embed_buffer.shape) != expected_shape
            or self._embed_buffer.dtype != values.dtype
        ):
            self._embed_buffer = self.xp.zeros(expected_shape, dtype=values.dtype)
        else:
            self._embed_buffer.fill(0)
        self._embed_buffer[self.embed_slices] = values
        return self._embed_buffer

    def convolve_same(self, image: Array, kernel: Array) -> Array:
        self._check_shape(image, self.image_shape, "Image")
        self._check_shape(kernel, self.kernel_shape, "Kernel")
        return self._convolve_crop(
            image,
            kernel,
            self.forward_fft_shape,
            self.forward_same_slices,
        )

    def image_adjoint(self, values: Array, kernel: Array) -> Array:
        self._check_shape(kernel, self.kernel_shape, "Kernel")
        embedded = self._embed_same_adjoint(values)
        return self._convolve_crop(
            embedded,
            self.xp.flip(kernel, axis=self.spatial_axes),
            self.image_adjoint_fft_shape,
            self.image_adjoint_slices,
        )

    def psf_adjoint(self, values: Array, image: Array) -> Array:
        self._check_shape(image, self.image_shape, "Image")
        embedded = self._embed_same_adjoint(values)
        return self._convolve_crop(
            embedded,
            self.xp.flip(image, axis=self.spatial_axes),
            self.psf_adjoint_fft_shape,
            self.psf_adjoint_slices,
        )


def _normalise_psf(psf: Array, xp: Any, epsilon: float) -> Array:
    psf = xp.maximum(
        xp.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    total = psf.sum(dtype=xp.float64)
    if float(total) <= epsilon:
        raise ValueError("PSF has no positive finite energy")
    return (psf / total).astype(xp.float32, copy=False)


def _normalise_psf_batch(psf: Array, xp: Any, epsilon: float) -> Array:
    psf = xp.maximum(
        xp.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    axes = tuple(range(1, psf.ndim))
    totals = psf.sum(axis=axes, dtype=xp.float64, keepdims=True)
    if float(totals.min()) <= epsilon:
        raise ValueError("At least one PSF has no positive finite energy")
    return (psf / totals).astype(xp.float32, copy=False)


def _embed_same_adjoint(values: Array, kernel_shape: Sequence[int], xp: Any) -> Array:
    """Adjoint of scipy.signal's centered ``mode='same'`` crop."""
    kernel_shape = _shape(kernel_shape)
    full_shape = tuple(
        int(n) + k - 1 for n, k in zip(values.shape, kernel_shape)
    )
    embedded = xp.zeros(full_shape, dtype=values.dtype)
    starts = tuple((k - 1) // 2 for k in kernel_shape)
    slices = tuple(
        slice(start, start + int(n)) for start, n in zip(starts, values.shape)
    )
    embedded[slices] = values
    return embedded


def convolve_same(image: Array, psf: Array, fftconvolve: Convolve) -> Array:
    if image.ndim != psf.ndim:
        raise ValueError("Image and PSF dimensionality must match")
    return fftconvolve(image, psf, mode="same")


def image_adjoint(
    values: Array, psf: Array, xp: Any, fftconvolve: Convolve
) -> Array:
    embedded = _embed_same_adjoint(values, psf.shape, xp)
    return fftconvolve(
        embedded, xp.flip(psf, axis=tuple(range(psf.ndim))), mode="valid"
    )


def psf_adjoint(
    values: Array,
    image: Array,
    psf_shape: Sequence[int],
    xp: Any,
    fftconvolve: Convolve,
) -> Array:
    psf_shape = _shape(psf_shape)
    embedded = _embed_same_adjoint(values, psf_shape, xp)
    result = fftconvolve(
        embedded, xp.flip(image, axis=tuple(range(image.ndim))), mode="valid"
    )
    if tuple(result.shape) != psf_shape:
        raise RuntimeError(
            f"PSF adjoint returned {tuple(result.shape)}, expected {psf_shape}"
        )
    return result


def _error_ratio(
    observed: Array,
    model: Array,
    xp: Any,
    epsilon: float,
    dampar: float,
) -> Array:
    ratio = observed / xp.maximum(model, epsilon)
    if dampar > 0.0:
        ratio = xp.where(xp.abs(observed - model) <= dampar, 1.0, ratio)
    return xp.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)


def estimate_blind_psf(
    observed: Array,
    initial_psf: Array,
    n_iters: int,
    *,
    xp: Any,
    fftconvolve: Convolve,
    dampar: float = 0.0,
    return_history: bool = False,
    latent_update_period: int = 1,
    fft_engine: str | FftConvolutionEngine | None = None,
) -> Array | tuple[Array, list[float]]:
    """Estimate a fixed-support PSF with alternating Richardson-Lucy updates."""
    if n_iters < 1:
        raise ValueError("n_iters must be at least 1")
    if dampar < 0.0:
        raise ValueError("dampar cannot be negative")
    latent_update_period = int(latent_update_period)
    if latent_update_period < 1:
        raise ValueError("latent_update_period must be at least 1")

    observed = xp.asarray(observed, dtype=xp.float32)
    initial_psf = xp.asarray(initial_psf, dtype=xp.float32)
    if observed.ndim != initial_psf.ndim:
        raise ValueError("Observed image and PSF dimensionality must match")
    if any(int(p) > int(i) for p, i in zip(initial_psf.shape, observed.shape)):
        raise ValueError(
            f"PSF shape {tuple(initial_psf.shape)} exceeds image shape "
            f"{tuple(observed.shape)}"
        )

    observed = xp.maximum(
        xp.nan_to_num(observed, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    observed_peak = float(observed.max())
    if observed_peak <= 0.0:
        raise ValueError("Observed image has no positive finite signal")
    epsilon = max(float(np.finfo(np.float32).eps), observed_peak * 1.0e-7)

    psf = _normalise_psf(initial_psf, xp, epsilon)
    if fft_engine == "auto":
        fft_engine = FftConvolutionEngine(xp, observed.shape, psf.shape)
    elif isinstance(fft_engine, str):
        raise ValueError(f"Unknown FFT engine: {fft_engine}")
    latent = xp.maximum(observed.copy(), epsilon)
    ones = xp.ones_like(observed)
    image_sensitivity = None
    psf_sensitivity = None
    latent_changed = True
    history: list[float] = []

    for iteration in range(int(n_iters)):
        update_latent = (iteration % latent_update_period) == 0
        if update_latent:
            if image_sensitivity is None:
                image_sensitivity = (
                    fft_engine.image_adjoint(ones, psf)
                    if fft_engine is not None
                    else image_adjoint(ones, psf, xp, fftconvolve)
                )
            model = (
                fft_engine.convolve_same(latent, psf)
                if fft_engine is not None
                else convolve_same(latent, psf, fftconvolve)
            )
            ratio = _error_ratio(observed, model, xp, epsilon, dampar)
            image_correction = (
                fft_engine.image_adjoint(ratio, psf)
                if fft_engine is not None
                else image_adjoint(ratio, psf, xp, fftconvolve)
            )
            latent *= image_correction / xp.maximum(
                image_sensitivity, epsilon
            )
            del image_correction, model, ratio, image_sensitivity
            image_sensitivity = None
            psf_sensitivity = None
            latent_changed = True
            latent = xp.maximum(
                xp.nan_to_num(latent, nan=0.0, posinf=0.0, neginf=0.0), epsilon
            )

        model = (
            fft_engine.convolve_same(latent, psf)
            if fft_engine is not None
            else convolve_same(latent, psf, fftconvolve)
        )
        ratio = _error_ratio(observed, model, xp, epsilon, dampar)
        correction = (
            fft_engine.psf_adjoint(ratio, latent)
            if fft_engine is not None
            else psf_adjoint(ratio, latent, psf.shape, xp, fftconvolve)
        )
        del model, ratio
        if psf_sensitivity is None or latent_changed:
            psf_sensitivity = (
                fft_engine.psf_adjoint(ones, latent)
                if fft_engine is not None
                else psf_adjoint(ones, latent, psf.shape, xp, fftconvolve)
            )
            latent_changed = False
        psf *= correction / xp.maximum(psf_sensitivity, epsilon)
        psf = _normalise_psf(psf, xp, epsilon)
        del correction

        if return_history:
            model = xp.maximum(
                (
                    fft_engine.convolve_same(latent, psf)
                    if fft_engine is not None
                    else convolve_same(latent, psf, fftconvolve)
                ),
                epsilon,
            )
            likelihood = xp.sum(
                observed * xp.log(model) - model, dtype=xp.float64
            )
            history.append(float(likelihood))

    return (psf, history) if return_history else psf


def estimate_blind_psf_batch(
    observed: Array,
    initial_psf: Array,
    n_iters: int,
    *,
    xp: Any,
    dampar: float = 0.0,
    latent_update_period: int = 1,
) -> Array:
    """Estimate independent PSFs for a leading batch of observed tiles."""
    if n_iters < 1:
        raise ValueError("n_iters must be at least 1")
    if dampar < 0.0:
        raise ValueError("dampar cannot be negative")
    latent_update_period = int(latent_update_period)
    if latent_update_period < 1:
        raise ValueError("latent_update_period must be at least 1")

    observed = xp.asarray(observed, dtype=xp.float32)
    initial_psf = xp.asarray(initial_psf, dtype=xp.float32)
    if observed.ndim < 2:
        raise ValueError("Observed batch must include a leading batch dimension")
    batch_size = int(observed.shape[0])
    image_shape = tuple(int(size) for size in observed.shape[1:])
    if initial_psf.ndim == observed.ndim - 1:
        psf_shape = tuple(int(size) for size in initial_psf.shape)
        initial_psf = xp.broadcast_to(
            initial_psf,
            (batch_size,) + psf_shape,
        ).copy()
    elif initial_psf.ndim == observed.ndim:
        if int(initial_psf.shape[0]) != batch_size:
            raise ValueError(
                "Initial PSF batch size must match observed batch size"
            )
        psf_shape = tuple(int(size) for size in initial_psf.shape[1:])
    else:
        raise ValueError("Initial PSF dimensionality does not match observed batch")
    if len(image_shape) != len(psf_shape):
        raise ValueError("Observed image and PSF dimensionality must match")
    if any(int(p) > int(i) for p, i in zip(psf_shape, image_shape)):
        raise ValueError(
            f"PSF shape {psf_shape} exceeds image shape {image_shape}"
        )

    observed = xp.maximum(
        xp.nan_to_num(observed, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    observed_peak = float(observed.max())
    if observed_peak <= 0.0:
        raise ValueError("Observed image batch has no positive finite signal")
    epsilon = max(float(np.finfo(np.float32).eps), observed_peak * 1.0e-7)

    psf = _normalise_psf_batch(initial_psf, xp, epsilon)
    engine = BatchedFftConvolutionEngine(
        xp,
        image_shape,
        psf_shape,
        batch_size=batch_size,
    )
    latent = xp.maximum(observed.copy(), epsilon)
    ones = xp.ones_like(observed)
    psf_sensitivity = None
    latent_changed = True

    for iteration in range(int(n_iters)):
        update_latent = (iteration % latent_update_period) == 0
        if update_latent:
            image_sensitivity = engine.image_adjoint(ones, psf)
            model = engine.convolve_same(latent, psf)
            ratio = _error_ratio(observed, model, xp, epsilon, dampar)
            image_correction = engine.image_adjoint(ratio, psf)
            latent *= image_correction / xp.maximum(image_sensitivity, epsilon)
            del image_correction, model, ratio, image_sensitivity
            psf_sensitivity = None
            latent_changed = True
            latent = xp.maximum(
                xp.nan_to_num(latent, nan=0.0, posinf=0.0, neginf=0.0), epsilon
            )

        model = engine.convolve_same(latent, psf)
        ratio = _error_ratio(observed, model, xp, epsilon, dampar)
        correction = engine.psf_adjoint(ratio, latent)
        del model, ratio
        if psf_sensitivity is None or latent_changed:
            psf_sensitivity = engine.psf_adjoint(ones, latent)
            latent_changed = False
        psf *= correction / xp.maximum(psf_sensitivity, epsilon)
        psf = _normalise_psf_batch(psf, xp, epsilon)
        del correction

    return psf


def estimate_shared_blind_psf_batch(
    observed: Array,
    initial_psf: Array,
    n_iters: int,
    *,
    xp: Any,
    dampar: float = 0.0,
    latent_update_period: int = 1,
    fft_batch_size: int | None = None,
) -> Array:
    """Estimate one shared PSF from a leading batch of observed tiles."""
    if n_iters < 1:
        raise ValueError("n_iters must be at least 1")
    if dampar < 0.0:
        raise ValueError("dampar cannot be negative")
    latent_update_period = int(latent_update_period)
    if latent_update_period < 1:
        raise ValueError("latent_update_period must be at least 1")

    observed = xp.asarray(observed, dtype=xp.float32)
    initial_psf = xp.asarray(initial_psf, dtype=xp.float32)
    if observed.ndim < 2:
        raise ValueError("Observed batch must include a leading batch dimension")
    batch_size = int(observed.shape[0])
    image_shape = tuple(int(size) for size in observed.shape[1:])
    psf_shape = tuple(int(size) for size in initial_psf.shape)
    if len(image_shape) != len(psf_shape):
        raise ValueError("Observed image and PSF dimensionality must match")
    if any(int(p) > int(i) for p, i in zip(psf_shape, image_shape)):
        raise ValueError(
            f"PSF shape {psf_shape} exceeds image shape {image_shape}"
        )

    observed = xp.maximum(
        xp.nan_to_num(observed, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    observed_peak = float(observed.max())
    if observed_peak <= 0.0:
        raise ValueError("Observed image batch has no positive finite signal")
    epsilon = max(float(np.finfo(np.float32).eps), observed_peak * 1.0e-7)

    if fft_batch_size is None:
        fft_batch_size = batch_size
    fft_batch_size = min(batch_size, max(1, int(fft_batch_size)))
    batch_slices = [
        slice(start, min(start + fft_batch_size, batch_size))
        for start in range(0, batch_size, fft_batch_size)
    ]
    engines = {
        int(batch_slice.stop - batch_slice.start): BatchedFftConvolutionEngine(
            xp,
            image_shape,
            psf_shape,
            batch_size=int(batch_slice.stop - batch_slice.start),
        )
        for batch_slice in batch_slices
    }

    psf = _normalise_psf(initial_psf, xp, epsilon)
    latent = xp.maximum(observed.copy(), epsilon)
    ones = xp.ones_like(observed)

    for iteration in range(int(n_iters)):
        update_latent = (iteration % latent_update_period) == 0
        if update_latent:
            for batch_slice in batch_slices:
                microbatch_size = int(batch_slice.stop - batch_slice.start)
                engine = engines[microbatch_size]
                psf_batch = xp.broadcast_to(psf, (microbatch_size,) + psf_shape)
                ones_batch = ones[batch_slice]
                latent_batch = latent[batch_slice]
                observed_batch = observed[batch_slice]
                image_sensitivity = engine.image_adjoint(ones_batch, psf_batch)
                model = engine.convolve_same(latent_batch, psf_batch)
                ratio = _error_ratio(observed_batch, model, xp, epsilon, dampar)
                image_correction = engine.image_adjoint(ratio, psf_batch)
                latent[batch_slice] = latent_batch * (
                    image_correction / xp.maximum(image_sensitivity, epsilon)
                )
                del (
                    image_correction,
                    image_sensitivity,
                    model,
                    observed_batch,
                    ones_batch,
                    psf_batch,
                    ratio,
                )
            latent = xp.maximum(
                xp.nan_to_num(latent, nan=0.0, posinf=0.0, neginf=0.0), epsilon
            )

        update_sum = xp.zeros(psf_shape, dtype=xp.float32)
        for batch_slice in batch_slices:
            microbatch_size = int(batch_slice.stop - batch_slice.start)
            engine = engines[microbatch_size]
            psf_batch = xp.broadcast_to(psf, (microbatch_size,) + psf_shape)
            ones_batch = ones[batch_slice]
            latent_batch = latent[batch_slice]
            observed_batch = observed[batch_slice]
            model = engine.convolve_same(latent_batch, psf_batch)
            ratio = _error_ratio(observed_batch, model, xp, epsilon, dampar)
            correction = engine.psf_adjoint(ratio, latent_batch)
            psf_sensitivity = engine.psf_adjoint(ones_batch, latent_batch)
            tile_updates = correction / xp.maximum(psf_sensitivity, epsilon)
            update_sum += tile_updates.sum(axis=0)
            del (
                correction,
                model,
                observed_batch,
                ones_batch,
                psf_batch,
                psf_sensitivity,
                ratio,
                tile_updates,
            )
        shared_update = update_sum / float(batch_size)
        psf *= shared_update
        psf = _normalise_psf(psf, xp, epsilon)
        del shared_update, update_sum

    return psf


def _free_current_cupy_cached_blocks(cp: Any, *, clear_plan_cache: bool = False) -> None:
    cp.cuda.Stream.null.synchronize()
    if clear_plan_cache:
        cp.fft.config.get_plan_cache().clear()
    cp.get_default_memory_pool().free_all_blocks()


def _estimate_shared_blind_psf_streamed_cupy(
    cp: Any,
    observed_host: np.ndarray,
    initial_psf: Array,
    n_iters: int,
    *,
    dampar: float = 0.0,
    latent_update_period: int = 1,
    fft_batch_size: int = 1,
) -> Array:
    """Estimate one shared PSF while keeping only one tile microbatch on the GPU."""
    if n_iters < 1:
        raise ValueError("n_iters must be at least 1")
    if dampar < 0.0:
        raise ValueError("dampar cannot be negative")
    latent_update_period = int(latent_update_period)
    if latent_update_period < 1:
        raise ValueError("latent_update_period must be at least 1")

    observed_host = np.asarray(observed_host, dtype=np.float32)
    if observed_host.ndim < 2:
        raise ValueError("Observed batch must include a leading batch dimension")
    batch_size = int(observed_host.shape[0])
    if batch_size < 1:
        raise ValueError("Observed batch must include at least one tile")
    image_shape = tuple(int(size) for size in observed_host.shape[1:])
    psf_shape = tuple(int(size) for size in initial_psf.shape)
    if len(image_shape) != len(psf_shape):
        raise ValueError("Observed image and PSF dimensionality must match")
    if any(int(p) > int(i) for p, i in zip(psf_shape, image_shape)):
        raise ValueError(
            f"PSF shape {psf_shape} exceeds image shape {image_shape}"
        )

    observed_host = np.maximum(
        np.nan_to_num(observed_host, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    ).astype(np.float32, copy=False)
    observed_peak = float(observed_host.max())
    if observed_peak <= 0.0:
        raise ValueError("Observed image batch has no positive finite signal")
    epsilon = max(float(np.finfo(np.float32).eps), observed_peak * 1.0e-7)

    fft_batch_size = min(batch_size, max(1, int(fft_batch_size)))
    batch_slices = [
        slice(start, min(start + fft_batch_size, batch_size))
        for start in range(0, batch_size, fft_batch_size)
    ]

    psf = _normalise_psf(initial_psf, cp, epsilon)
    latent_host = np.maximum(observed_host.copy(), epsilon).astype(
        np.float32, copy=False
    )

    for iteration in range(int(n_iters)):
        update_latent = (iteration % latent_update_period) == 0
        if update_latent:
            for batch_slice in batch_slices:
                microbatch_size = int(batch_slice.stop - batch_slice.start)
                engine = BatchedFftConvolutionEngine(
                    cp,
                    image_shape,
                    psf_shape,
                    batch_size=microbatch_size,
                )
                observed_gpu = cp.asarray(observed_host[batch_slice], dtype=cp.float32)
                latent_gpu = cp.asarray(latent_host[batch_slice], dtype=cp.float32)
                ones_gpu = cp.ones_like(observed_gpu)
                psf_batch = cp.broadcast_to(psf, (microbatch_size,) + psf_shape)
                image_sensitivity = engine.image_adjoint(ones_gpu, psf_batch)
                model = engine.convolve_same(latent_gpu, psf_batch)
                ratio = _error_ratio(observed_gpu, model, cp, epsilon, dampar)
                image_correction = engine.image_adjoint(ratio, psf_batch)
                latent_gpu *= image_correction / cp.maximum(image_sensitivity, epsilon)
                latent_gpu = cp.maximum(
                    cp.nan_to_num(latent_gpu, nan=0.0, posinf=0.0, neginf=0.0),
                    epsilon,
                )
                latent_host[batch_slice] = cp.asnumpy(latent_gpu).astype(
                    np.float32, copy=False
                )
                del (
                    engine,
                    image_correction,
                    image_sensitivity,
                    latent_gpu,
                    model,
                    observed_gpu,
                    ones_gpu,
                    psf_batch,
                    ratio,
                )
                _free_current_cupy_cached_blocks(cp)

        update_sum = cp.zeros(psf_shape, dtype=cp.float32)
        for batch_slice in batch_slices:
            microbatch_size = int(batch_slice.stop - batch_slice.start)
            engine = BatchedFftConvolutionEngine(
                cp,
                image_shape,
                psf_shape,
                batch_size=microbatch_size,
            )
            observed_gpu = cp.asarray(observed_host[batch_slice], dtype=cp.float32)
            latent_gpu = cp.asarray(latent_host[batch_slice], dtype=cp.float32)
            ones_gpu = cp.ones_like(observed_gpu)
            psf_batch = cp.broadcast_to(psf, (microbatch_size,) + psf_shape)
            model = engine.convolve_same(latent_gpu, psf_batch)
            ratio = _error_ratio(observed_gpu, model, cp, epsilon, dampar)
            correction = engine.psf_adjoint(ratio, latent_gpu)
            psf_sensitivity = engine.psf_adjoint(ones_gpu, latent_gpu)
            tile_updates = correction / cp.maximum(psf_sensitivity, epsilon)
            update_sum += tile_updates.sum(axis=0)
            del (
                correction,
                engine,
                latent_gpu,
                model,
                observed_gpu,
                ones_gpu,
                psf_batch,
                psf_sensitivity,
                ratio,
                tile_updates,
            )
            _free_current_cupy_cached_blocks(cp)
        shared_update = update_sum / float(batch_size)
        psf *= shared_update
        psf = _normalise_psf(psf, cp, epsilon)
        del shared_update, update_sum

    return psf


def estimate_blind_psf_scipy(
    observed: np.ndarray,
    initial_psf: np.ndarray,
    n_iters: int,
    *,
    dampar: float = 0.0,
    return_history: bool = False,
    latent_update_period: int = 1,
) -> np.ndarray | tuple[np.ndarray, list[float]]:
    from scipy.signal import fftconvolve

    return estimate_blind_psf(
        observed,
        initial_psf,
        n_iters,
        xp=np,
        fftconvolve=fftconvolve,
        dampar=dampar,
        return_history=return_history,
        latent_update_period=latent_update_period,
    )


def estimate_blind_psf_cupy(
    observed: Array,
    initial_psf: Array,
    n_iters: int,
    *,
    dampar: float = 0.0,
    latent_update_period: int = 1,
    fft_engine: str = "auto",
) -> Array:
    try:
        import cupy as cp
        from cupyx.scipy.signal import fftconvolve
    except ImportError as exc:
        raise RuntimeError(
            "blind_backend='cupy' requires CuPy with cupyx.scipy"
        ) from exc
    return estimate_blind_psf(
        observed,
        initial_psf,
        n_iters,
        xp=cp,
        fftconvolve=fftconvolve,
        dampar=dampar,
        latent_update_period=latent_update_period,
        fft_engine=None if str(fft_engine) == "cupyx" else str(fft_engine),
    )


def deconvolve_with_cucim(
    observed: Array, psf: Array, n_iters: int, *, device_id: int = 0
) -> np.ndarray:
    """Restore an image with cuCIM using the estimated PSF."""
    try:
        import cupy as cp
        from cucim.skimage.restoration import richardson_lucy
    except ImportError as exc:
        raise RuntimeError("Restoration requires both cupy and cucim") from exc

    image_gpu = None
    psf_gpu = None
    restored = None
    pending_error: BaseException | None = None
    with cp.cuda.Device(int(device_id)):
        try:
            image_gpu = cp.asarray(observed, dtype=cp.float32)
            psf_gpu = cp.asarray(psf, dtype=cp.float32)
            epsilon = max(
                float(np.finfo(np.float32).eps),
                float(image_gpu.max()) * 1.0e-7,
            )
            psf_gpu = _normalise_psf(psf_gpu, cp, epsilon)
            restored = richardson_lucy(
                image_gpu,
                psf_gpu,
                num_iter=int(n_iters),
                clip=False,
                filter_epsilon=epsilon,
            )
            cp.cuda.Stream.null.synchronize()
            restored_host = cp.asnumpy(restored).astype(np.float32, copy=False)
            return restored_host
        except BaseException as exc:
            pending_error = exc
            raise
        finally:
            restored = None
            psf_gpu = None
            image_gpu = None
            try:
                cp.cuda.Stream.null.synchronize()
                cp.fft.config.get_plan_cache().clear()
                cp.get_default_memory_pool().free_all_blocks()
            except BaseException:
                if pending_error is None:
                    raise


def _prepare_observed(
    observed: np.ndarray, mode: str, gamma_max: float
) -> np.ndarray:
    observed = np.asarray(observed, dtype=np.float32)
    if mode == "none":
        return observed
    peak = float(np.nanmax(observed))
    if not np.isfinite(peak) or peak <= 0.0:
        return observed
    unit = np.clip(observed / peak, 0.0, 1.0)
    if mode == "unit":
        return unit
    if mode == "gamma":
        if gamma_max <= 0.0:
            raise ValueError("blind_peak_gamma_max must be positive")
        return np.power(unit, 1.0 / float(gamma_max)).astype(np.float32)
    raise ValueError(f"Unknown blind peak normalization mode: {mode}")


def clear_cupy_memory(
    *,
    device_id: int | None = None,
    clear_plan_cache: bool = True,
    free_memory_pool: bool = True,
) -> None:
    """Release CuPy FFT plans and pooled allocations for one GPU."""
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("CuPy is missing from the worker environment") from exc

    if device_id is None:
        device_id = int(os.environ.get("DECON_CUDA_DEVICE", "0"))
    with cp.cuda.Device(int(device_id)):
        cp.cuda.Stream.null.synchronize()
        if clear_plan_cache:
            cp.fft.config.get_plan_cache().clear()
        if free_memory_pool:
            cp.get_default_memory_pool().free_all_blocks()


def trim_cupy_memory_pool(
    max_total_bytes: int | None,
    *,
    device_id: int | None = None,
) -> bool:
    """Free cached CuPy blocks only when the retained pool exceeds a run budget."""
    if max_total_bytes is None or int(max_total_bytes) <= 0:
        return False
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("CuPy is missing from the worker environment") from exc

    if device_id is None:
        device_id = int(os.environ.get("DECON_CUDA_DEVICE", "0"))
    with cp.cuda.Device(int(device_id)):
        pool = cp.get_default_memory_pool()
        total_bytes = getattr(pool, "total_bytes", None)
        if total_bytes is None:
            return False
        if int(total_bytes()) <= int(max_total_bytes):
            return False
        cp.cuda.Stream.null.synchronize()
        pool.free_all_blocks()
        return True


def estimate_psf_array_cupy(
    observed: np.ndarray,
    initial_psf: np.ndarray,
    n_iters: int,
    pad_z: int,
    *,
    peak_normalization: str = "none",
    peak_gamma_max: float = 2.5,
    dampar: float = 0.0,
    device_id: int | None = None,
    clear_plan_cache: bool = True,
    free_memory_pool: bool = True,
    latent_update_period: int = 1,
    fft_engine: str = "auto",
) -> np.ndarray:
    """Estimate one PSF directly from host arrays on a single CuPy GPU."""
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("CuPy is missing from the worker environment") from exc

    if device_id is None:
        device_id = int(os.environ.get("DECON_CUDA_DEVICE", "0"))

    image_gpu = None
    seed_gpu = None
    psf_gpu = None
    pending_error: BaseException | None = None
    with cp.cuda.Device(int(device_id)):
        try:
            observed = _prepare_observed(
                observed, str(peak_normalization), float(peak_gamma_max)
            )
            image_gpu = cp.asarray(observed, dtype=cp.float32)
            if int(pad_z) > 0:
                image_gpu = cp.pad(
                    image_gpu,
                    ((int(pad_z), int(pad_z)), (0, 0), (0, 0)),
                    mode="symmetric",
                )
            seed_gpu = cp.asarray(initial_psf, dtype=cp.float32)
            psf_gpu = estimate_blind_psf_cupy(
                image_gpu,
                seed_gpu,
                int(n_iters),
                dampar=float(dampar),
                latent_update_period=int(latent_update_period),
                fft_engine=str(fft_engine),
            )
            cp.cuda.Stream.null.synchronize()
            return cp.asnumpy(psf_gpu).astype(np.float32, copy=False)
        except BaseException as exc:
            pending_error = exc
            raise
        finally:
            psf_gpu = None
            seed_gpu = None
            image_gpu = None
            try:
                if pending_error is not None or clear_plan_cache or free_memory_pool:
                    clear_cupy_memory(
                        device_id=device_id,
                        clear_plan_cache=clear_plan_cache or pending_error is not None,
                        free_memory_pool=free_memory_pool or pending_error is not None,
                    )
            except BaseException:
                if pending_error is None:
                    raise


def estimate_psf_array_batch_cupy(
    observed_batch: np.ndarray,
    initial_psf: np.ndarray,
    n_iters: int,
    pad_z: int,
    *,
    peak_normalization: str = "none",
    peak_gamma_max: float = 2.5,
    dampar: float = 0.0,
    device_id: int | None = None,
    clear_plan_cache: bool = True,
    free_memory_pool: bool = True,
    latent_update_period: int = 1,
) -> np.ndarray:
    """Estimate independent PSFs for a leading batch of host arrays on one GPU."""
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("CuPy is missing from the worker environment") from exc

    if device_id is None:
        device_id = int(os.environ.get("DECON_CUDA_DEVICE", "0"))

    image_gpu = None
    seed_gpu = None
    psf_gpu = None
    pending_error: BaseException | None = None
    with cp.cuda.Device(int(device_id)):
        try:
            observed_batch = np.stack(
                [
                    _prepare_observed(chunk, str(peak_normalization), float(peak_gamma_max))
                    for chunk in np.asarray(observed_batch)
                ],
                axis=0,
            )
            image_gpu = cp.asarray(observed_batch, dtype=cp.float32)
            if int(pad_z) > 0:
                image_gpu = cp.pad(
                    image_gpu,
                    ((0, 0), (int(pad_z), int(pad_z)), (0, 0), (0, 0)),
                    mode="symmetric",
                )
            seed_gpu = cp.asarray(initial_psf, dtype=cp.float32)
            psf_gpu = estimate_blind_psf_batch(
                image_gpu,
                seed_gpu,
                int(n_iters),
                xp=cp,
                dampar=float(dampar),
                latent_update_period=int(latent_update_period),
            )
            cp.cuda.Stream.null.synchronize()
            return cp.asnumpy(psf_gpu).astype(np.float32, copy=False)
        except BaseException as exc:
            pending_error = exc
            raise
        finally:
            psf_gpu = None
            seed_gpu = None
            image_gpu = None
            try:
                if pending_error is not None or clear_plan_cache or free_memory_pool:
                    clear_cupy_memory(
                        device_id=device_id,
                        clear_plan_cache=clear_plan_cache or pending_error is not None,
                        free_memory_pool=free_memory_pool or pending_error is not None,
                    )
            except BaseException:
                if pending_error is None:
                    raise


def estimate_shared_psf_array_cupy(
    observed_batch: np.ndarray,
    initial_psf: np.ndarray,
    n_iters: int,
    pad_z: int,
    *,
    peak_normalization: str = "none",
    peak_gamma_max: float = 2.5,
    dampar: float = 0.0,
    device_id: int | None = None,
    clear_plan_cache: bool = True,
    free_memory_pool: bool = True,
    latent_update_period: int = 1,
    fft_batch_size: int | None = None,
) -> np.ndarray:
    """Estimate one shared PSF from a leading batch of host arrays on one GPU."""
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("CuPy is missing from the worker environment") from exc

    if device_id is None:
        device_id = int(os.environ.get("DECON_CUDA_DEVICE", "0"))
    if fft_batch_size is None:
        fft_batch_size = int(os.environ.get("DECON_SHARED_FFT_BATCH_SIZE", "1"))

    image_gpu = None
    seed_gpu = None
    psf_gpu = None
    pending_error: BaseException | None = None
    with cp.cuda.Device(int(device_id)):
        try:
            observed_batch = np.stack(
                [
                    _prepare_observed(chunk, str(peak_normalization), float(peak_gamma_max))
                    for chunk in np.asarray(observed_batch)
                ],
                axis=0,
            )
            if int(pad_z) > 0:
                observed_batch = np.pad(
                    observed_batch,
                    ((0, 0), (int(pad_z), int(pad_z)), (0, 0), (0, 0)),
                    mode="symmetric",
                ).astype(np.float32, copy=False)
            seed_gpu = cp.asarray(initial_psf, dtype=cp.float32)
            if int(fft_batch_size) < int(observed_batch.shape[0]):
                psf_gpu = _estimate_shared_blind_psf_streamed_cupy(
                    cp,
                    observed_batch,
                    seed_gpu,
                    int(n_iters),
                    dampar=float(dampar),
                    latent_update_period=int(latent_update_period),
                    fft_batch_size=int(fft_batch_size),
                )
            else:
                image_gpu = cp.asarray(observed_batch, dtype=cp.float32)
                psf_gpu = estimate_shared_blind_psf_batch(
                    image_gpu,
                    seed_gpu,
                    int(n_iters),
                    xp=cp,
                    dampar=float(dampar),
                    latent_update_period=int(latent_update_period),
                    fft_batch_size=int(fft_batch_size),
                )
            cp.cuda.Stream.null.synchronize()
            return cp.asnumpy(psf_gpu).astype(np.float32, copy=False)
        except BaseException as exc:
            pending_error = exc
            raise
        finally:
            psf_gpu = None
            seed_gpu = None
            image_gpu = None
            try:
                if pending_error is not None or clear_plan_cache or free_memory_pool:
                    clear_cupy_memory(
                        device_id=device_id,
                        clear_plan_cache=clear_plan_cache or pending_error is not None,
                        free_memory_pool=free_memory_pool or pending_error is not None,
                    )
            except BaseException:
                if pending_error is None:
                    raise


def _estimate_psf_file_worker(
    chunk_path: str,
    seed_path: str,
    output_path: str,
    n_iters: int,
    pad_z: int,
    peak_mode: str,
    gamma_max: float,
    dampar: float,
    device_id: int,
    latent_update_period: int,
) -> dict[str, Any]:
    from tifffile import imread, imwrite

    psf = estimate_psf_array_cupy(
        imread(chunk_path),
        imread(seed_path),
        int(n_iters),
        int(pad_z),
        peak_normalization=peak_mode,
        peak_gamma_max=float(gamma_max),
        dampar=float(dampar),
        device_id=int(device_id),
        latent_update_period=int(latent_update_period),
    )
    imwrite(output_path, psf, photometric="minisblack")
    return {
        "shape": tuple(int(value) for value in psf.shape),
        "sum": float(psf.sum(dtype=np.float64)),
        "device_id": int(device_id),
    }


def estimate_psf_file_in_process(
    chunk_path: str | Path,
    seed_path: str | Path,
    output_path: str | Path,
    n_iters: int,
    pad_z: int,
    *,
    peak_normalization: str = "none",
    peak_gamma_max: float = 2.5,
    dampar: float = 0.0,
    device_id: int | None = None,
    latent_update_period: int = 1,
) -> dict[str, Any]:
    """Estimate one chunk's PSF in an isolated spawned process."""
    if device_id is None:
        device_id = int(os.environ.get("DECON_CUDA_DEVICE", "0"))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        future = executor.submit(
            _estimate_psf_file_worker,
            str(chunk_path),
            str(seed_path),
            str(output_path),
            int(n_iters),
            int(pad_z),
            str(peak_normalization),
            float(peak_gamma_max),
            float(dampar),
            int(device_id),
            int(latent_update_period),
        )
        return future.result()
