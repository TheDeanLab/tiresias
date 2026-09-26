"""Theoretical PSF seed generation."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import psfmodels as pm
from scipy.ndimage import affine_transform, zoom
from scipy.signal import fftconvolve
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
    focus_offset: float = 0.0,
) -> np.ndarray:
    """Generate a normalized 3-D PSF seed with ``psfmodels.make_psf``.

    ``focus_offset`` shifts every sampled Z plane by a constant, in the same
    units as ``dz`` (D-10); 0.0 (the default) keeps every existing caller
    bit-identical.
    """
    del illumination_na
    if not math.isfinite(focus_offset):
        raise ValueError(f"focus_offset must be finite, got {focus_offset!r}")
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
    if focus_offset != 0.0:
        # D-10: plane i is sampled at defocus (i - (psf_size_z - 1) / 2) * dz
        # + focus_offset -- the same psfmodels.tot_psf x_offset convention
        # (_centered_zv's own pz shift, generalized off of the integer-z
        # path). psfmodels emits a UserWarning "dz is ignored" whenever dz
        # is passed alongside a z sequence, so drop it here.
        lim = (psf_size_z - 1) * dz / 2
        requested_kwargs["z"] = np.linspace(-lim + focus_offset, lim + focus_offset, psf_size_z)
        del requested_kwargs["dz"]
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


def _resolve_slit_fwhm(
    slit_width: float | None, slit_width_px: int | None, dz: float
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
    # D-17: this conversion uses dz, the sample spacing of the integration
    # axis (pre-rotation axis 0) -- overrides Phase 1 D-09, which used dxy
    # unconditionally regardless of which axis the gate acted on.
    # test_aslm_slit_width_px_converts_via_dz (plan 08.1-01) pins this.
    return slit_width_px * dz


def _gaussian_slit_window(size: int, fwhm: float, pixel_size: float) -> np.ndarray | None:
    """Build a geometric-midpoint-centered Gaussian taper, or None to skip the gate."""
    # D-18: below one pixel_size sample, the window would resolve to a
    # spike narrower than the simulated grid can represent -- fall back to
    # no convolution instead (the waist-limited limit, identical to
    # light_sheet). D-15: no full-extent shortcut -- a very wide slit still
    # builds a real (near-uniform) window here and converges to the
    # swept-average sheet through the convolution itself, rather than
    # returning None early for an exact light_sheet reduction.
    if fwhm < pixel_size:
        return None
    sigma_px = (fwhm / pixel_size) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    center = (size - 1) / 2.0  # D-05: geometric midpoint, matches psfmodels' centered beam waist
    idx = np.arange(size, dtype=np.float64)
    window = np.exp(-0.5 * ((idx - center) / sigma_px) ** 2)
    return window.astype(np.float32)


def _apply_aslm_slit_gate(
    illumination: np.ndarray, fwhm: float, dz: float
) -> np.ndarray:
    """Convolve the pre-rotation illumination with a sweep-integrated slit window.

    D-01/D-02: the rolling shutter is assumed perfectly synchronized to the
    swept beam waist, so the effective illumination at any point along the
    beam's propagation axis (pre-rotation axis 0) is the raw illumination
    convolved with the Gaussian slit window -- not a static per-axis
    multiply. This runs before rotation, so it stays correct for any
    (polar_deg, azimuthal_deg) direction with no axis snapping.
    """
    n = illumination.shape[0]
    # The odd 2n-1 kernel length puts the centre tap exactly at index n - 1
    # for both odd and even n, so a same-mode convolution introduces no
    # shift, and the waist-offset integral spans the whole simulated window
    # (D-07).
    window = _gaussian_slit_window(2 * n - 1, fwhm, dz)
    if window is None:
        gated = illumination  # D-18: below one dz sample, the waist-limited limit
    else:
        # Beam energy beyond the simulated window is treated as zero
        # (same-mode convolution, zero-padded edges); the clip removes tiny
        # negative values introduced by FFT round-off.
        gated = np.clip(
            fftconvolve(illumination, window.reshape(-1, 1, 1), mode="same", axes=0),
            0.0,
            None,
        ).astype(np.float32)

    # D-18: normalise_psf (seeds.py:16-22) returns a zero-sum array
    # unchanged and without error, so without this guard a degenerate
    # illumination input or a user-chosen slit_width could produce an
    # all-zero PSF seed that flows into blind-RL estimation looking like a
    # valid one. This guard covers the no-convolution fallback path too.
    # Compare against the ungated illumination's own sum (a relative
    # statement), not a fixed constant.
    original_sum = float(np.abs(illumination).sum(dtype=np.float64))
    epsilon = max(float(np.finfo(np.float32).eps), original_sum * 1e-7)
    gated_sum = float(gated.sum(dtype=np.float64))
    if not np.isfinite(gated).all() or gated_sum <= epsilon:
        raise ValueError(
            f"ASLM slit-integrated illumination has no positive finite energy "
            f"(slit_width={fwhm!r}, dz={dz!r}); the sweep-integrated model "
            "assumes the rolling shutter is perfectly synchronized to the "
            "swept beam waist, so check the illumination parameters and "
            "slit_width rather than timing"
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
    slit_width_px: int | None = None,
    emitter_offset: float = 0.0,
) -> np.ndarray:
    """Create a single-detection, light-sheet, or ASLM blind-estimation seed PSF.

    ASLM mode convolves the illumination PSF, in its pre-rotation frame, with
    a Gaussian slit window (FWHM ``slit_width``) along the beam's
    propagation axis (pre-rotation axis 0), before rotation. This is the
    sweep-integrated effective illumination seen by a rolling shutter
    assumed to be perfectly synchronized with the swept beam waist: the
    illuminated slit always sits exactly at the waist as it sweeps across
    the whole simulated window.

    A slit narrower than one dz sample applies no convolution -- the
    waist-limited limit, identical to ``light_sheet``. A very wide slit
    converges to the swept-average sheet rather than returning an exact
    ``light_sheet`` seed. Timing jitter, shutter/beam desynchronization, and
    sweep-velocity error are not modelled and are explicitly out of scope
    for this milestone.

    ``emitter_offset`` (D-10) is the emitter's offset from the illumination
    waist along the beam propagation direction, in the same units as
    ``dxy``/``dz``. A positive value moves the waist to a smaller
    pre-rotation axis-0 index, matching ``psfmodels.tot_psf``'s ``x_offset``
    convention. It is applied in the pre-rotation frame, so it follows the
    true beam direction (D-02) rather than a camera-snapped axis. ``aslm``
    is invariant to it (D-07): the swept waist tracks the emitter's slit
    across the whole simulated window under the perfectly synchronized
    rolling-shutter assumption, so the effective illumination relative to
    the emitter does not depend on where the emitter sits. ``single`` has no
    illumination beam and requires 0.0.
    """
    if psf_mode not in ("single", "light_sheet", "aslm"):
        raise ValueError(
            f"Unsupported psf_mode={psf_mode!r}; expected one of 'single', 'light_sheet', 'aslm'"
        )

    # D-10: validate before generating any PSF, mirroring the existing
    # angle-validation pattern below.
    if not math.isfinite(emitter_offset):
        raise ValueError(f"emitter_offset must be finite, got {emitter_offset!r}")
    if psf_mode == "single" and emitter_offset != 0.0:
        raise ValueError(
            "emitter_offset applies only to psf_mode='light_sheet' or 'aslm' "
            f"(single has no illumination beam), got emitter_offset={emitter_offset!r}"
        )

    # D-07/T-07-02: validate before generating any PSF, extended here to the
    # non-finite-angle check -- a NaN/inf polar_deg or azimuthal_deg must fail
    # loudly rather than flow through normalise_psf's nan_to_num into an
    # all-zero seed. The direction vector itself is no longer consumed here
    # (D-16 retired the gate-axis override resolver that used it); the call
    # remains purely for its validation side effect.
    if psf_mode != "single":
        _spherical_direction(polar_deg, azimuthal_deg)

    slit_fwhm: float | None = None
    if psf_mode == "aslm":  # D-07: validate before generating any PSF
        slit_fwhm = _resolve_slit_fwhm(slit_width, slit_width_px, dz)

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
        # D-10: only light_sheet forwards the off-waist emitter offset. D-07:
        # the ASLM waist is swept over the whole simulated window with the
        # rolling-shutter slit perfectly synchronized to the emitter's row,
        # so the effective illumination relative to the emitter does not
        # depend on where the emitter sits -- the ASLM seed is therefore
        # intentionally invariant to emitter_offset, and always forwards 0.0.
        focus_offset=emitter_offset if psf_mode == "light_sheet" else 0.0,
    )
    if psf_mode == "aslm":
        illumination = _apply_aslm_slit_gate(illumination, slit_fwhm, dz)
    rotated = rotate_illumination(
        illumination, polar_deg=polar_deg, azimuthal_deg=azimuthal_deg, dxy=dxy, dz=dz
    )
    return normalise_psf(detection * rotated)
