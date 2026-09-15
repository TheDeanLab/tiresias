"""Position-dependent illumination beam-width measurement.

Analysis of an already-generated illumination PSF array (Y-axis FWHM per Z
position), kept separate from `seeds.py` (which scopes itself to PSF
*generation*) -- this module is the natural home for Phase 6's future
Rayleigh-range locator too.
"""

from __future__ import annotations

import warnings

import numpy as np

from .seeds import generate_theoretical_psf

__all__ = ["measure_beam_width_profile"]


def measure_beam_width_profile(
    *,
    illumination_na: float,
    wavelength: float,
    ni: float,
    ns: float,
    dxy: float,
    dz: float,
    psf_size_z: int = 61,
    psf_size_xy: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the illumination beam's transverse (Y) FWHM at every Z position.

    Returns `(positions_um, widths_um)`: two parallel float64 arrays, each of
    length `psf_size_z`. `positions_um[i] == i * dz`, in micrometres,
    strictly ascending. `widths_um[i]` is the interpolated half-max Y-profile
    width measured at `positions_um[i]`, or NaN when no clean half-max
    crossing exists at that position.
    """
    # ASVS V5: validate this function's OWN parameters before spending time
    # on PSF generation. illumination_na is validated HERE and nowhere else
    # -- generate_theoretical_psf deletes its own illumination_na argument on
    # entry (seeds.py:60) and never validates it, so this is the only guard
    # that will ever catch it. psf_size_z/psf_size_xy are validated here
    # because the downstream failure for a non-positive size is an opaque
    # `ValueError: zero-size array to reduction operation maximum which has
    # no identity` raised from inside NumPy, naming neither the parameter
    # nor the caller (measured during planning against psf_size_z=0 and
    # psf_size_xy=0). Shape copied from seeds.py:62-74 so the codebase has
    # one validation idiom, not two.
    required_values = {
        "illumination_na": illumination_na,
        "wavelength": wavelength,
        "ni": ni,
        "ns": ns,
        "dxy": dxy,
        "dz": dz,
        "psf_size_z": psf_size_z,
        "psf_size_xy": psf_size_xy,
    }
    missing = [name for name, value in required_values.items() if value is None or value <= 0]
    if missing:
        raise ValueError(
            "Missing or non-positive parameter(s) for measure_beam_width_profile: "
            + ", ".join(missing)
        )

    # D-04: generate the illumination-arm PSF internally by plugging
    # illumination_na into detection_na -- the same trick generate_psf_seed
    # already uses for its own illumination branch (seeds.py:332-336).
    # Deliberately never generate_psf_seed(): that returns detection times
    # rotated illumination, which would mix the detection PSF's own axial
    # extent into this measurement and destroy the NA-dependence Phase 5
    # must demonstrate.
    psf = generate_theoretical_psf(
        detection_na=illumination_na,
        illumination_na=illumination_na,
        wavelength=wavelength,
        ni=ni,
        ns=ns,
        dxy=dxy,
        dz=dz,
        psf_size_z=psf_size_z,
        psf_size_xy=psf_size_xy,
    )
    # psf shape is (psf_size_z, psf_size_xy, psf_size_xy) in (Z, Y, X) order.

    # D-03: fixed global peak, not re-located per Z slice -- empirically
    # verified during planning that psfmodels' illumination PSF has zero
    # lateral (Y/X) peak drift across Z slices near focus, but the
    # per-slice argmax DOES drift on far-from-focus slices whose on-axis
    # intensity has collapsed and whose slice maximum is a diffraction
    # sidelobe (measured at illumination_na=0.6, psf_size_z=21: per-slice
    # (y, x) argmax took four distinct values across the 21 slices). Fixing
    # the peak globally avoids tracking sidelobes instead of beam centre.
    _peak_z, peak_y, peak_x = np.unravel_index(np.argmax(psf), psf.shape)

    # D-08: positions in physical micrometres, voxel_index * dz, so Phase 6
    # and Phase 8 need no caller-side conversion.
    positions_um = np.arange(psf_size_z, dtype=np.float64) * dz
    widths_um = np.full(psf_size_z, np.nan, dtype=np.float64)
    failed_positions: list[float] = []

    # D-06: dense, one measurement per Z voxel, no stride or step parameter
    # -- the volumes are modest and Phase 6 needs whatever neighbouring
    # points it asks for.
    #
    # Far-from-focus slices legitimately produce large, non-monotonic
    # widths because the Y profile there carries diffraction sidelobe
    # structure rather than a clean single lobe (measured during planning:
    # at illumination_na=0.4 the widths span 0.73 um to 8.48 um across the
    # 61-slice window). That is in-scope, correct-per-spec output for
    # Phase 5, not a defect to chase; cleaning the curve is Phase 6/8
    # territory.
    for z in range(psf_size_z):
        # D-01/D-02: Z is the propagation axis the position sweeps along,
        # and the reported width is the transverse FWHM measured along Y at
        # fixed peak_x. Y and X are equivalent for this rotationally
        # symmetric on-axis PSF model; Y is the convention chosen to mirror
        # the existing single-profile-axis FWHM helper.
        profile = psf[z, :, peak_x].astype(np.float64)
        peak_value = profile[peak_y]
        if peak_value <= 0:
            failed_positions.append(positions_um[z])
            continue
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

        left_indices = np.arange(peak_y, -1, -1)
        right_indices = np.arange(peak_y, profile.size)
        left_crossing = _crossing(left_indices)
        right_crossing = _crossing(right_indices)
        if left_crossing is None or right_crossing is None:
            failed_positions.append(positions_um[z])
            continue
        widths_um[z] = (right_crossing - left_crossing) * dxy

    # D-09: seam for 05-02's named warnings.warn() contract. NaN alone
    # already satisfies "not silently wrong"; the warning is additive and
    # lands in the next plan.

    return positions_um, widths_um
