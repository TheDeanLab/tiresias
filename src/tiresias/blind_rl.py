"""Shared SciPy/CuPy blind Richardson-Lucy PSF estimation."""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

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


def _normalise_psf(psf: Array, xp: Any, epsilon: float) -> Array:
    psf = xp.maximum(
        xp.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    total = psf.sum(dtype=xp.float64)
    if float(total) <= epsilon:
        raise ValueError("PSF has no positive finite energy")
    return (psf / total).astype(xp.float32, copy=False)


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
                image_sensitivity = image_adjoint(ones, psf, xp, fftconvolve)
            model = convolve_same(latent, psf, fftconvolve)
            ratio = _error_ratio(observed, model, xp, epsilon, dampar)
            image_correction = image_adjoint(ratio, psf, xp, fftconvolve)
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

        model = convolve_same(latent, psf, fftconvolve)
        ratio = _error_ratio(observed, model, xp, epsilon, dampar)
        correction = psf_adjoint(ratio, latent, psf.shape, xp, fftconvolve)
        del model, ratio
        if psf_sensitivity is None or latent_changed:
            psf_sensitivity = psf_adjoint(ones, latent, psf.shape, xp, fftconvolve)
            latent_changed = False
        psf *= correction / xp.maximum(psf_sensitivity, epsilon)
        psf = _normalise_psf(psf, xp, epsilon)
        del correction

        if return_history:
            model = xp.maximum(
                convolve_same(latent, psf, fftconvolve), epsilon
            )
            likelihood = xp.sum(
                observed * xp.log(model) - model, dtype=xp.float64
            )
            history.append(float(likelihood))

    return (psf, history) if return_history else psf


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
