"""Theoretical PSF seed generation."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import psfmodels as pm
from scipy.ndimage import rotate
from tifffile import imread

RIGHT_ANGLE_TOLERANCE = 1e-6


def normalise_psf(psf: np.ndarray) -> np.ndarray:
    psf = np.nan_to_num(psf.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    psf = np.clip(psf, 0, None)
    total = float(psf.sum())
    if total > 0:
        psf = psf / total
    return psf.astype(np.float32, copy=False)


def resolve_dxy(
    dxy: float | None,
    camera_pixel_size: float | None = None,
    magnification: float | None = None,
) -> float:
    if dxy is not None and dxy > 0:
        return dxy
    if camera_pixel_size and magnification and camera_pixel_size > 0 and magnification > 0:
        return camera_pixel_size / magnification
    raise ValueError("dxy must be > 0, or camera_pixel_size and magnification must be provided")


def generate_theoretical_psf(
    na: float | None = None,
    detection_na: float | None = None,
    illumination_na: float | None = None,
    wavelength: float | None = None,
    ni: float | None = None,
    ns: float | None = None,
    ni0: float | None = None,
    tg: float | None = None,
    tg0: float | None = None,
    ng: float | None = None,
    ng0: float | None = None,
    ti0: float | None = None,
    oversample_factor: int = 3,
    psf_model: str = "vectorial",
    dxy: float | None = None,
    dz: float | None = None,
    psf_size_z: int = 61,
    psf_size_xy: int = 128,
    background: float = 0.0,
) -> np.ndarray:
    """Generate a normalized 3-D PSF seed with ``psfmodels.make_psf``."""
    del illumination_na
    detection_na = detection_na if detection_na is not None else na
    required_values = {
        "detection_na": detection_na,
        "wavelength": wavelength,
        "ni": ni,
        "ns": ns,
        "dxy": dxy,
        "dz": dz,
    }
    missing = [name for name, value in required_values.items() if value is None or value <= 0]
    if missing:
        raise ValueError(
            "Missing required optical/acquisition parameter(s): " + ", ".join(missing)
        )

    requested_kwargs = {
        "z": psf_size_z,
        "nx": psf_size_xy,
        "dz": dz,
        "dxy": dxy,
        "NA": detection_na,
        "wvl": wavelength,
        "ni": ni,
        "oversample_factor": oversample_factor,
        "model": psf_model,
    }
    optional_kwargs = {
        "ns": ns,
        "ni0": ni0,
        "tg": tg,
        "tg0": tg0,
        "ng": ng,
        "ng0": ng0,
        "ti0": ti0,
    }
    requested_kwargs.update(
        {name: value for name, value in optional_kwargs.items() if value is not None}
    )
    signature = inspect.signature(pm.make_psf)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_kwargs:
        missing_params = [
            name for name in requested_kwargs if name not in signature.parameters
        ]
        if missing_params:
            raise RuntimeError(
                "psfmodels.make_psf API mismatch; missing expected parameter(s): "
                + ", ".join(missing_params)
            )

    psf = pm.make_psf(**requested_kwargs).astype(np.float32)
    return normalise_psf(np.maximum(psf - background, 0))


def _center_crop_or_pad(volume: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    output = np.zeros(shape, dtype=volume.dtype)
    source_slices = []
    dest_slices = []
    for current, target in zip(volume.shape, shape):
        if current >= target:
            source_start = (current - target) // 2
            dest_start = 0
            length = target
        else:
            source_start = 0
            dest_start = (target - current) // 2
            length = current
        source_slices.append(slice(source_start, source_start + length))
        dest_slices.append(slice(dest_start, dest_start + length))
    output[tuple(dest_slices)] = volume[tuple(source_slices)]
    return output


def load_psf_seed(path: str | Path, shape: tuple[int, int, int]) -> np.ndarray:
    """Load a calibrated TIFF PSF and fit it to the configured support."""
    source = np.asarray(imread(path), dtype=np.float32)
    if source.ndim == 2:
        source = source[np.newaxis, :, :]
    if source.ndim != 3:
        raise ValueError(
            f"External PSF seed must be 3-D, got shape {source.shape} from {path}"
        )
    target_shape = tuple(int(axis) for axis in shape)
    if len(target_shape) != 3 or any(axis <= 0 for axis in target_shape):
        raise ValueError(f"External PSF target shape must be positive 3-D: {shape}")
    fitted = _center_crop_or_pad(source, target_shape)
    if not np.any(np.isfinite(fitted) & (fitted > 0)):
        raise ValueError(f"External PSF seed has no positive finite energy: {path}")
    return normalise_psf(fitted)


def rotate_illumination_psf(illumination: np.ndarray, angle: float) -> np.ndarray:
    """Rotate illumination coordinates in the Z/X plane."""
    right_angle_units = angle / 90.0
    if abs(right_angle_units - round(right_angle_units)) <= RIGHT_ANGLE_TOLERANCE:
        rotated = np.rot90(illumination, k=int(round(right_angle_units)), axes=(0, 2))
        return _center_crop_or_pad(rotated, illumination.shape)
    return rotate(
        illumination,
        angle=angle,
        axes=(0, 2),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def generate_psf_seed(
    *,
    psf_mode: str,
    na: float,
    detection_na: float | None,
    illumination_na: float | None,
    wavelength: float,
    ni: float,
    ns: float | None,
    ni0: float | None,
    tg: float | None,
    tg0: float | None,
    ng: float | None,
    ng0: float | None,
    ti0: float | None,
    oversample_factor: int,
    psf_model: str,
    dxy: float,
    dz: float,
    psf_size_z: int,
    psf_size_xy: int,
    background: float,
    light_sheet_angle: float = 90.0,
) -> np.ndarray:
    """Create a single-detection or light-sheet blind-estimation seed PSF."""
    detection = generate_theoretical_psf(
        na=na,
        detection_na=detection_na,
        illumination_na=illumination_na,
        wavelength=wavelength,
        ni=ni,
        ns=ns,
        ni0=ni0,
        tg=tg,
        tg0=tg0,
        ng=ng,
        ng0=ng0,
        ti0=ti0,
        oversample_factor=oversample_factor,
        psf_model=psf_model,
        dxy=dxy,
        dz=dz,
        psf_size_z=psf_size_z,
        psf_size_xy=psf_size_xy,
        background=background,
    )

    if psf_mode == "single":
        return normalise_psf(detection)
    if psf_mode != "light_sheet":
        raise ValueError(f"Unsupported psf_mode={psf_mode!r}")

    illumination = generate_theoretical_psf(
        na=na,
        detection_na=illumination_na if illumination_na is not None else detection_na,
        illumination_na=illumination_na,
        wavelength=wavelength,
        ni=ni,
        ns=ns,
        ni0=ni0,
        tg=tg,
        tg0=tg0,
        ng=ng,
        ng0=ng0,
        ti0=ti0,
        oversample_factor=oversample_factor,
        psf_model=psf_model,
        dxy=dxy,
        dz=dz,
        psf_size_z=psf_size_z,
        psf_size_xy=psf_size_xy,
        background=background,
    )
    rotated = rotate_illumination_psf(illumination, light_sheet_angle)
    return normalise_psf(detection * rotated)
