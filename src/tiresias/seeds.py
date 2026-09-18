"""Theoretical PSF seed generation."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import psfmodels as pm
from scipy.ndimage import affine_transform, zoom
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


def _rotation_matrix(polar_deg: float, azimuthal_deg: float) -> np.ndarray:
    """Build the (Z, Y, X)-ordered rotation matrix ``Rz(azimuthal) @ Ry(polar)``."""
    # D-01: polar_deg is measured from the pre-rotation +Z propagation axis;
    # azimuthal_deg is measured from +X in the X-Y plane -- the standard
    # physics spherical convention, embedded into this project's (Z, Y, X)
    # array index order.
    # D-02: both angles are in degrees, not radians, matching how the old
    # single-angle rotation parameter was always specified/documented.
    if not math.isfinite(polar_deg):
        raise ValueError(f"polar_deg must be finite, got {polar_deg!r}")
    if not math.isfinite(azimuthal_deg):
        raise ValueError(f"azimuthal_deg must be finite, got {azimuthal_deg!r}")
    theta = np.radians(polar_deg)
    phi = np.radians(azimuthal_deg)
    ry = np.array(
        [
            [np.cos(theta), 0.0, -np.sin(theta)],
            [0.0, 1.0, 0.0],
            [np.sin(theta), 0.0, np.cos(theta)],
        ]
    )
    rz = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(phi), np.sin(phi)],
            [0.0, -np.sin(phi), np.cos(phi)],
        ]
    )
    return rz @ ry


def _spherical_direction(polar_deg: float, azimuthal_deg: float) -> np.ndarray:
    """Return the (Z, Y, X) propagation unit vector for (polar_deg, azimuthal_deg)."""
    # D-03: (90.0, 0.0) is the fixed reference point mapping to the old
    # broadside default (formerly a single 90.0-degree angle parameter),
    # yielding (Z=0, Y=0, X=1). This is the single source of truth for the
    # direction vector -- callers never re-derive the trig themselves.
    return _rotation_matrix(polar_deg, azimuthal_deg) @ np.array([1.0, 0.0, 0.0])


def _match_legacy_cardinal(direction: np.ndarray) -> int | None:
    """Return the numpy.rot90 ``k`` for a legacy cardinal direction, else None."""
    # Matches on the direction vector, not the (polar_deg, azimuthal_deg) pair,
    # so pole degeneracy at polar_deg 0/180 needs no special-case branch.
    # RIGHT_ANGLE_TOLERANCE is now the direction-space analogue of the old
    # angle-space tolerance the pre-v1.1 fast path used.
    legacy_directions = (
        (np.array([1.0, 0.0, 0.0]), 0),
        (np.array([0.0, 0.0, 1.0]), 1),
        (np.array([-1.0, 0.0, 0.0]), 2),
        (np.array([0.0, 0.0, -1.0]), 3),
    )
    for legacy_direction, k in legacy_directions:
        if np.max(np.abs(direction - legacy_direction)) <= RIGHT_ANGLE_TOLERANCE:
            return k
    return None


def _legacy_rot90_rotation(illumination: np.ndarray, k: int) -> np.ndarray:
    """Delegate to the exact pre-v1.1 rotation code, unchanged (ROT-04)."""
    rotated = np.rot90(illumination, k=k, axes=(0, 2))
    return _center_crop_or_pad(rotated, illumination.shape)


def _rotate_isotropic(
    illumination: np.ndarray, rotation: np.ndarray, dxy: float, dz: float
) -> np.ndarray:
    """Rotate a physically anisotropic volume via an isotropic resample round trip."""
    # D-04: resample onto an isotropic grid (index-space rotation is only
    # physically valid once voxels are cubic), rotate there, then resample
    # back to the native (dz, dxy) grid. The single-combined-affine
    # alternative (rotation + anisotropic scaling in one transform) was
    # explicitly rejected as the primary approach.
    z_zoom = dz / dxy
    iso = zoom(
        illumination, zoom=(z_zoom, 1.0, 1.0), order=1, mode="constant", cval=0.0, prefilter=False
    )

    # rotation is orthogonal, so its transpose is the exact inverse;
    # affine_transform's matrix is the output->input ("pull") map, not the
    # forward rotation -- passing rotation itself would silently mirror the
    # result.
    matrix = rotation.T
    center = (np.asarray(iso.shape, dtype=np.float64) - 1.0) / 2.0
    offset = center - matrix @ center
    rotated_iso = affine_transform(
        iso,
        matrix,
        offset=offset,
        output_shape=iso.shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    back = zoom(
        rotated_iso,
        zoom=(dxy / dz, 1.0, 1.0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    # The two independent zoom-shape roundings are not guaranteed inverses of
    # each other, so always finish with the shared crop/pad helper rather than
    # assuming shape equality.
    return _center_crop_or_pad(back, illumination.shape)


def rotate_illumination(
    illumination: np.ndarray,
    *,
    polar_deg: float,
    azimuthal_deg: float,
    dxy: float,
    dz: float,
) -> np.ndarray:
    """Rotate the pre-rotation illumination PSF to the requested 3D direction.

    Delegates to the exact, unmodified legacy ``numpy.rot90`` fast path at the
    four legacy cardinal directions (bit-identical output, ROT-04); every
    other direction -- including new azimuthal orientations the old 1-DOF API
    could never reach -- routes through the isotropic resample/rotate/resample
    pipeline (ROT-02).
    """
    rotation = _rotation_matrix(polar_deg, azimuthal_deg)
    direction = rotation @ np.array([1.0, 0.0, 0.0])
    k = _match_legacy_cardinal(direction)
    if k is not None:
        return _legacy_rot90_rotation(illumination, k)
    return _rotate_isotropic(illumination, rotation, dxy, dz)


def _resolve_slit_axis(direction: np.ndarray, slit_axis: int | None = None) -> int:
    """Resolve which pre-rotation illumination axis the slit gate narrows."""
    # D-05: an explicit override bypasses auto-detection entirely, now
    # accepting all three axes. Otherwise the resolver snaps the 3D
    # propagation direction to whichever single coordinate axis it is closest
    # to and gates along that one pre-rotation axis.
    if slit_axis is not None:
        if slit_axis not in (0, 1, 2):
            raise ValueError(f"slit_axis must be 0, 1, or 2, got {slit_axis!r}")
        return slit_axis
    # Correction to 07-RESEARCH.md's plain-argmax recommendation (Assumption
    # A1): a raw numpy.argmax(numpy.abs(direction)) is NOT a reliable
    # tie-break at the legacy quadrant-tie angles (45/135/225/315 degrees).
    # Ry(135deg)'s cos/sin components are independently rounded floats that
    # differ from each other by ~1 ULP (e.g. abs(cos)=0.7071067811865475 vs
    # abs(sin)=0.7071067811865476), so a bare argmax picks whichever
    # component happened to round up -- axis 0 at some tie angles, axis 2 at
    # others -- instead of the legacy round-half-to-even quadrant heuristic's
    # consistent axis-0 preference at all four tie angles (pinned by the
    # 07-01 fixture). Snap any axis within RIGHT_ANGLE_TOLERANCE of the true
    # maximum magnitude into a tie, then let argmax's first-occurrence
    # behavior break the tie in favor of the lowest axis index (Z).
    magnitudes = np.abs(direction)
    is_near_max = magnitudes >= magnitudes.max() - RIGHT_ANGLE_TOLERANCE
    return int(np.argmax(is_near_max))


def _resolve_slit_fwhm(
    slit_width: float | None, slit_width_px: int | None, dxy: float
) -> float:
    """Resolve the ASLM slit gate's FWHM in physical units from whichever form was supplied."""
    provided = [value for value in (slit_width, slit_width_px) if value is not None]
    if len(provided) != 1:
        raise ValueError(
            "Exactly one of slit_width or slit_width_px must be provided for psf_mode='aslm'"
        )
    if slit_width is not None:
        if slit_width <= 0:
            raise ValueError(f"slit_width must be > 0, got {slit_width!r}")
        return slit_width
    if slit_width_px <= 0:
        raise ValueError(f"slit_width_px must be > 0, got {slit_width_px!r}")
    # D-09: this conversion always uses dxy, even when the resolved gate axis is
    # 0 (Z, spaced by dz) — deliberate per locked decision D-09, not an
    # oversight. test_aslm_slit_width_px_converts_via_dxy_even_on_the_z_axis
    # (plan 01-04) pins this so it goes red if someone "fixes" it later.
    return slit_width_px * dxy


def _gaussian_slit_window(size: int, fwhm: float, pixel_size: float) -> np.ndarray | None:
    """Build a geometric-midpoint-centered Gaussian taper, or None to skip the gate."""
    full_extent = size * pixel_size
    if fwhm >= full_extent:
        return None  # D-06: skip the gate entirely for exact light_sheet reduction
    sigma_px = (fwhm / pixel_size) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    center = (size - 1) / 2.0  # D-05: geometric midpoint, matches psfmodels' centered beam waist
    idx = np.arange(size, dtype=np.float64)
    window = np.exp(-0.5 * ((idx - center) / sigma_px) ** 2)
    return window.astype(np.float32)


def _apply_aslm_slit_gate(
    illumination: np.ndarray, axis: int, fwhm: float, dxy: float, dz: float
) -> np.ndarray:
    """Multiply the pre-rotation illumination by a Gaussian slit gate along one axis."""
    pixel_size = dz if axis == 0 else dxy
    size = illumination.shape[axis]
    window = _gaussian_slit_window(size, fwhm, pixel_size)
    if window is None:
        gated = illumination
    else:
        shape = [1, 1, 1]
        shape[axis] = size
        gated = illumination * window.reshape(shape)

    # D-08: normalise_psf (seeds.py:16-22) returns a zero-sum array
    # unchanged and without error, so without this guard a user-chosen
    # slit_width narrow enough to destroy the illumination energy could
    # produce an all-zero PSF seed that flows into blind-RL estimation
    # looking like a valid one. Compare against the ungated illumination's
    # own sum (a relative statement), not a fixed constant.
    original_sum = float(illumination.sum())
    epsilon = max(float(np.finfo(np.float32).eps), original_sum * 1e-7)
    if float(gated.sum()) < epsilon:
        raise ValueError(
            f"slit_width={fwhm!r} is too narrow to capture positive illumination "
            f"energy along axis {axis} (extent={size * pixel_size!r}); the ASLM "
            "slit is a static gate centered on the gate axis midpoint, assumed "
            "perfectly synchronized to the beam waist, so widen slit_width "
            "rather than adjusting timing"
        )
    return gated


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
    polar_deg: float = 90.0,
    azimuthal_deg: float = 0.0,
    slit_width: float | None = None,
    slit_axis: int | None = None,
    slit_width_px: int | None = None,
) -> np.ndarray:
    """Create a single-detection, light-sheet, or ASLM blind-estimation seed PSF.

    ASLM mode multiplies the illumination PSF, in its pre-rotation frame, by a
    static Gaussian slit gate centered on the geometric midpoint of the gate
    axis. The gate is a fixed spatial taper, not a time-resolved simulation:
    the rolling shutter is assumed to be perfectly synchronized with the
    swept beam waist, so the illuminated slit always sits exactly at the
    waist. Timing jitter, shutter/beam desynchronization, and sweep-velocity
    error are therefore not modelled and are explicitly out of scope for
    this milestone.
    """
    if psf_mode not in ("single", "light_sheet", "aslm"):
        raise ValueError(
            f"Unsupported psf_mode={psf_mode!r}; expected one of 'single', 'light_sheet', 'aslm'"
        )

    # D-07/T-07-02: validate before generating any PSF, extended here to the
    # non-finite-angle check -- a NaN/inf polar_deg or azimuthal_deg must fail
    # loudly rather than flow through normalise_psf's nan_to_num into an
    # all-zero seed.
    direction: np.ndarray | None = None
    if psf_mode != "single":
        direction = _spherical_direction(polar_deg, azimuthal_deg)

    gate_axis: int | None = None
    slit_fwhm: float | None = None
    if psf_mode == "aslm":  # D-07: validate before generating any PSF
        gate_axis = _resolve_slit_axis(direction, slit_axis)
        slit_fwhm = _resolve_slit_fwhm(slit_width, slit_width_px, dxy)

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
    if psf_mode == "aslm":
        illumination = _apply_aslm_slit_gate(illumination, gate_axis, slit_fwhm, dxy, dz)
    rotated = rotate_illumination(
        illumination, polar_deg=polar_deg, azimuthal_deg=azimuthal_deg, dxy=dxy, dz=dz
    )
    return normalise_psf(detection * rotated)
