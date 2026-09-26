"""System-PSF axial FWHM measurement vs FOV position (light-sheet and ASLM).

This module measures the system-PSF axial FWHM (detection x effective
illumination) vs FOV position for the static light sheet and ASLM
(D-05/D-06). It deliberately calls `tiresias.seeds.generate_psf_seed`, the
inverse of `beam_profile.py`'s avoidance of that function, because the
system PSF -- detection times illumination, exactly what a user's blind-RL
seed looks like -- is exactly the quantity wanted here (D-10). This replaces
`simulate/gated_beam_profile.py`'s physically wrong self-gating measurement
(SC2): that module gated the illumination on the axis it then measured,
capping the reported width at the gate itself.
"""

from __future__ import annotations

import warnings

import numpy as np

from tiresias.seeds import generate_psf_seed, generate_theoretical_psf

__all__ = [
    "measure_light_sheet_system_fwhm_profile",
    "measure_aslm_system_fwhm_profile",
    "measure_detection_dof",
    "measure_sheet_thickness",
]


def _half_max_width(profile: np.ndarray, peak_index: int, spacing: float) -> float | None:
    """Return the interpolated half-max full width around `peak_index`.

    The outward first-below-half-max walk with linear interpolation, the
    same algorithm as the crossing loop in `beam_profile.py` and
    `examples/slit_width_sweep.py`'s `axial_fwhm`. This is a private copy
    per Phase 8 D-04 (no shared cross-module crossing helper). Returns None
    when the peak is <= 0 or either side never drops below half max.
    """
    peak_value = profile[peak_index]
    if peak_value <= 0:
        return None
    half_max = peak_value / 2.0

    def _crossing(indices: np.ndarray) -> float | None:
        values = profile[indices]
        below = np.where(values < half_max)[0]
        if below.size == 0:
            return None
        edge = below[0]
        if edge == 0:
            return None
        i0, i1 = indices[edge - 1], indices[edge]
        v0, v1 = profile[i0], profile[i1]
        frac = (half_max - v0) / (v1 - v0)
        return i0 + frac * (i1 - i0)

    left_indices = np.arange(peak_index, -1, -1)
    right_indices = np.arange(peak_index, profile.size)

    left_crossing = _crossing(left_indices)
    right_crossing = _crossing(right_indices)
    if left_crossing is None or right_crossing is None:
        return None
    return (right_crossing - left_crossing) * spacing


def _axial_fwhm(seed: np.ndarray, dz: float) -> float | None:
    """Return the axial (Z) FWHM of a (Z, Y, X) PSF seed, through its global peak."""
    peak_z, peak_y, peak_x = np.unravel_index(np.argmax(seed), seed.shape)
    profile = seed[:, peak_y, peak_x].astype(np.float64)
    return _half_max_width(profile, peak_z, dz)


def _validate_common(caller: str, values: dict) -> None:
    """Raise ValueError naming every missing/non-positive parameter, before any generation."""
    missing = [name for name, value in values.items() if value is None or value <= 0]
    if missing:
        raise ValueError(
            f"Missing or non-positive parameter(s) for {caller}: " + ", ".join(missing)
        )


def _validate_positions(caller: str, positions_um) -> np.ndarray:
    """Return `positions_um` as a validated, non-empty, 1-D, finite float64 array."""
    positions = np.asarray(positions_um, dtype=np.float64)
    if positions.ndim != 1 or positions.size == 0 or not np.all(np.isfinite(positions)):
        raise ValueError(
            f"{caller}: positions_um must be a non-empty 1-D sequence of finite "
            f"values, got {positions_um!r}"
        )
    return positions


def _measure_profile(
    caller: str, psf_mode: str, positions: np.ndarray, seed_kwargs: dict
) -> np.ndarray:
    """Measure one system-PSF axial FWHM per FOV position in `positions`.

    Each position is a real `generate_psf_seed` call with `emitter_offset=p`
    -- no short-circuiting to a single call, so ASLM's D-07 flatness is
    observed, not assumed. A None result stays NaN; never clamped or
    extrapolated.
    """
    dz = seed_kwargs["dz"]
    fwhm_um = np.full(positions.shape, np.nan, dtype=np.float64)
    failed: list[float] = []
    for i, p in enumerate(positions):
        seed = generate_psf_seed(psf_mode=psf_mode, emitter_offset=float(p), **seed_kwargs)
        value = _axial_fwhm(seed, dz)
        if value is None:
            # Round for display only -- positions_um itself stays exact.
            failed.append(round(float(p), 6))
            continue
        fwhm_um[i] = value
    if failed:
        warnings.warn(
            f"{caller}: no clean half-max axial crossing at position(s) (um): {failed!r}",
            stacklevel=3,
        )
    return fwhm_um


def measure_light_sheet_system_fwhm_profile(
    *,
    positions_um,
    detection_na: float,
    illumination_na: float,
    wavelength: float,
    ni: float,
    ns: float,
    ni0: float | None,
    dxy: float,
    dz: float,
    psf_size_z: int = 61,
    psf_size_xy: int = 128,
    polar_deg: float = 90.0,
    azimuthal_deg: float = 0.0,
    tg: float | None = None,
    tg0: float | None = None,
    ng: float | None = None,
    ng0: float | None = None,
    ti0: float | None = None,
    oversample_factor: int = 3,
    psf_model: str = "vectorial",
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the light-sheet system-PSF axial FWHM at each FOV position.

    An emitter at FOV position `p` sits off the illumination waist by `p`
    along the beam's propagation direction (D-06) -- each position is
    composed through `generate_psf_seed(psf_mode="light_sheet",
    emitter_offset=p, ...)` (D-10), never through psfmodels' own SPIM
    total-system-PSF function or a local re-composition. The reported
    quantity is the system-PSF (detection
    x rotated illumination) axial FWHM through the seed's global 3-D peak
    (D-05), matching the axial convention in `examples/slit_width_sweep.py`.

    `ni0` has no default: every caller must choose explicitly whether to
    pass a value or `None` (RESEARCH Pitfall 1).

    Returns `(positions_um, fwhm_um)`, both float64 and of equal length.
    Unmeasurable positions are NaN with a single named `UserWarning`.
    """
    _validate_common(
        "measure_light_sheet_system_fwhm_profile",
        {
            "detection_na": detection_na,
            "illumination_na": illumination_na,
            "wavelength": wavelength,
            "ni": ni,
            "ns": ns,
            "dxy": dxy,
            "dz": dz,
            "psf_size_z": psf_size_z,
            "psf_size_xy": psf_size_xy,
        },
    )
    positions = _validate_positions("measure_light_sheet_system_fwhm_profile", positions_um)
    seed_kwargs = dict(
        na=detection_na,
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
        background=0.0,
        polar_deg=polar_deg,
        azimuthal_deg=azimuthal_deg,
    )
    fwhm_um = _measure_profile(
        "measure_light_sheet_system_fwhm_profile", "light_sheet", positions, seed_kwargs
    )
    return positions.copy(), fwhm_um


def measure_aslm_system_fwhm_profile(
    *,
    positions_um,
    detection_na: float,
    illumination_na: float,
    wavelength: float,
    ni: float,
    ns: float,
    ni0: float | None,
    dxy: float,
    dz: float,
    slit_width: float,
    psf_size_z: int = 61,
    psf_size_xy: int = 128,
    polar_deg: float = 90.0,
    azimuthal_deg: float = 0.0,
    tg: float | None = None,
    tg0: float | None = None,
    ng: float | None = None,
    ng0: float | None = None,
    ti0: float | None = None,
    oversample_factor: int = 3,
    psf_model: str = "vectorial",
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the ASLM system-PSF axial FWHM at each FOV position.

    D-07: the rolling shutter is assumed perfectly synchronized to the swept
    beam waist, so the ASLM waist sweeps the whole simulated window -- there
    is no sweep-range parameter, and the curve is expected to be flat except
    for window-edge truncation. Each position is a real
    `generate_psf_seed(psf_mode="aslm", emitter_offset=p, slit_width=...)`
    call; the flatness is observed here, not assumed or short-circuited to a
    single call.

    Returns `(positions_um, fwhm_um)`, both float64 and of equal length.
    """
    _validate_common(
        "measure_aslm_system_fwhm_profile",
        {
            "detection_na": detection_na,
            "illumination_na": illumination_na,
            "wavelength": wavelength,
            "ni": ni,
            "ns": ns,
            "dxy": dxy,
            "dz": dz,
            "psf_size_z": psf_size_z,
            "psf_size_xy": psf_size_xy,
        },
    )
    if slit_width is None or slit_width <= 0:
        raise ValueError(
            f"measure_aslm_system_fwhm_profile: slit_width must be > 0, got {slit_width!r}"
        )
    positions = _validate_positions("measure_aslm_system_fwhm_profile", positions_um)
    seed_kwargs = dict(
        na=detection_na,
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
        background=0.0,
        polar_deg=polar_deg,
        azimuthal_deg=azimuthal_deg,
        slit_width=slit_width,
    )
    fwhm_um = _measure_profile(
        "measure_aslm_system_fwhm_profile", "aslm", positions, seed_kwargs
    )
    return positions.copy(), fwhm_um


def measure_detection_dof(
    *,
    detection_na: float,
    wavelength: float,
    ni: float,
    ns: float,
    ni0: float | None,
    dxy: float,
    dz: float,
    psf_size_z: int = 61,
    psf_size_xy: int = 128,
    oversample_factor: int = 3,
    psf_model: str = "vectorial",
) -> float:
    """Measure the detection-only depth of focus (D-12): axial FWHM of `psf_mode="single"`.

    A measured reference, never a closed-form formula -- Phase 9 found the
    paraxial formulas off by 2-32x on these PSFs (09-CONTEXT.md D-33).
    """
    _validate_common(
        "measure_detection_dof",
        {
            "detection_na": detection_na,
            "wavelength": wavelength,
            "ni": ni,
            "ns": ns,
            "dxy": dxy,
            "dz": dz,
            "psf_size_z": psf_size_z,
            "psf_size_xy": psf_size_xy,
        },
    )
    seed = generate_psf_seed(
        psf_mode="single",
        na=detection_na,
        detection_na=detection_na,
        illumination_na=None,
        wavelength=wavelength,
        ni=ni,
        ns=ns,
        ni0=ni0,
        tg=None,
        tg0=None,
        ng=None,
        ng0=None,
        ti0=None,
        oversample_factor=oversample_factor,
        psf_model=psf_model,
        dxy=dxy,
        dz=dz,
        psf_size_z=psf_size_z,
        psf_size_xy=psf_size_xy,
        background=0.0,
    )
    value = _axial_fwhm(seed, dz)
    if value is None:
        raise ValueError(
            "measure_detection_dof: no clean half-max axial crossing was found "
            "for the detection-only PSF"
        )
    return value


def measure_sheet_thickness(
    *,
    illumination_na: float,
    wavelength: float,
    ni: float,
    ns: float,
    ni0: float | None,
    dxy: float,
    dz: float,
    psf_size_z: int = 61,
    psf_size_xy: int = 128,
    oversample_factor: int = 3,
    psf_model: str = "vectorial",
) -> float:
    """Measure the illumination sheet thickness (D-12): waist FWHM of the illumination arm.

    A measured reference, never a closed-form formula. The width is measured
    along pre-rotation axis 2 (X) through the illumination-only PSF's global
    peak -- the axis the legacy broadside rotation maps onto Z. This value
    is physically comparable to the seed's Z scale only when `dz == dxy`,
    because of the legacy `rot90` relabel (ROT-04, out of scope here).
    """
    _validate_common(
        "measure_sheet_thickness",
        {
            "illumination_na": illumination_na,
            "wavelength": wavelength,
            "ni": ni,
            "ns": ns,
            "dxy": dxy,
            "dz": dz,
            "psf_size_z": psf_size_z,
            "psf_size_xy": psf_size_xy,
        },
    )
    psf = generate_theoretical_psf(
        detection_na=illumination_na,
        illumination_na=illumination_na,
        wavelength=wavelength,
        ni=ni,
        ns=ns,
        ni0=ni0,
        dxy=dxy,
        dz=dz,
        psf_size_z=psf_size_z,
        psf_size_xy=psf_size_xy,
    )
    peak_z, peak_y, peak_x = np.unravel_index(np.argmax(psf), psf.shape)
    profile = psf[peak_z, peak_y, :].astype(np.float64)
    value = _half_max_width(profile, peak_x, dxy)
    if value is None:
        raise ValueError(
            "measure_sheet_thickness: no clean half-max crossing was found "
            "for the illumination-only PSF"
        )
    return value
