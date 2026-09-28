"""Position-dependent illumination beam-width measurement.

Analysis of an already-generated illumination PSF array (Y-axis FWHM per Z
position), kept separate from `seeds.py` (which scopes itself to PSF
*generation*). The Rayleigh-range locator built on top of this module's
output lives in the sibling module `rayleigh_range.py`; both sit outside the
installed `tiresias` package (D-01/D-02).
"""

from __future__ import annotations

import warnings

import numpy as np

# D-04: simulate/ is an external consumer of the installed tiresias package
# rather than a duplicate of its PSF-generation logic, so this import is
# absolute (tiresias.seeds), not relative -- simulate/ has no seeds sibling.
from tiresias.seeds import generate_theoretical_psf

__all__ = ["measure_beam_width_profile"]

# D-14: private, not exported (D-14) -- __all__ above stays unchanged. This
# module's own copy of the rtol=1e-6 float32-noise precedent (Phase 8 D-04
# forbade a cross-module helper for this crossing loop); the self-gated
# gated_beam_profile.py module that used to carry a sibling copy was retired
# in plan 08.1-08 (commit 71789e1) as physically wrong (SC2). It is costly to
# change: this exact value is baked into the 09-01 ring-free MEAS-04 fixture
# verification and every ringed-profile expectation in this module's own
# tests (simulate/tests/test_beam_profile.py), so a different value would
# require recapturing and re-verifying all of them.
_LOBE_MIN_RTOL = 1e-6


def _bound_to_first_local_minimum(
    profile: np.ndarray, indices: np.ndarray, peak_value: float
) -> tuple[np.ndarray, bool]:
    """Truncate an outward walk at the first flanking local minimum of the lobe."""
    # D-02: threshold is computed once per call from the CURRENT module
    # global (read at call time, never captured as a default-argument
    # value) so tests can patch _LOBE_MIN_RTOL and see the effect -- this is
    # the fixed per-slice reference D-02 requires, not a per-step
    # recomputation relative to a drifting local value.
    threshold = _LOBE_MIN_RTOL * peak_value
    for i in range(1, indices.size - 1):
        if (
            profile[indices[i - 1]] > profile[indices[i]]
            and profile[indices[i + 1]] - profile[indices[i]] > threshold
        ):
            # Truncation is inclusive of the minimum sample: when the
            # minimum itself is the first below-half-max sample, excluding
            # it would drop a valid crossing.
            return indices[: i + 1], True
    # The last index can never be a minimum -- no following sample confirms
    # a rise -- so a walk that reaches the array edge without finding one
    # falls back to the full, unmodified array. This MEAS-01 array-edge
    # fallback is what keeps ring-free profiles bit-identical (MEAS-04):
    # _crossing() then receives exactly the array it received before this
    # phase. There is deliberately no global scan for a deeper minimum
    # further out (D-03, and REQUIREMENTS.md Out of Scope).
    return indices, False


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
    failed_no_local_minimum: list[float] = []
    failed_lobe_too_narrow: list[float] = []

    # D-06: dense, one measurement per Z voxel, no stride or step parameter
    # -- the volumes are modest and Phase 6 needs whatever neighbouring
    # points it asks for.
    #
    # Far-from-focus slices legitimately produce large, non-monotonic
    # widths because the Y profile there carries diffraction sidelobe
    # structure rather than a clean single lobe (measured during planning:
    # at illumination_na=0.4 the widths span 0.73 um to 8.48 um across the
    # 61-slice window). Phase 9 bounds the search to the central lobe, so a
    # ring-merged reading becomes an honest NaN gap tagged "central lobe too
    # narrow" instead of silently including ring structure in the measured
    # width -- no smoothing, clamping, or extrapolation is applied.
    for z in range(psf_size_z):
        # D-01/D-02: Z is the propagation axis the position sweeps along,
        # and the reported width is the transverse FWHM measured along Y at
        # fixed peak_x. Y and X are equivalent for this rotationally
        # symmetric on-axis PSF model; Y is the convention chosen to mirror
        # the existing single-profile-axis FWHM helper.
        profile = psf[z, :, peak_x].astype(np.float64)
        peak_value = profile[peak_y]
        if peak_value <= 0:
            # D-15: no lobe to bound at all -- more fundamental than a
            # bounded-but-too-narrow lobe (D-05 priority), so this goes to
            # the no-local-minimum group. Round for display only --
            # positions_um itself stays exact (D-08); float64
            # multiplication of index * dz can produce artifacts like
            # 3 * 0.3 == 0.8999999999999999, which would silently fail to
            # name "0.9" in the warning text below.
            failed_no_local_minimum.append(round(float(positions_um[z]), 6))
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
        # MEAS-01/02/03: bound each side's outward walk to the first
        # flanking local minimum (rtol-tolerant) before crossing search --
        # _crossing itself is untouched, so a bounded finite width is
        # always bit-identical to the pre-fix width (the bounded index
        # array is always a prefix of the unbounded one).
        left_bounded, left_found_minimum = _bound_to_first_local_minimum(
            profile, left_indices, peak_value
        )
        right_bounded, right_found_minimum = _bound_to_first_local_minimum(
            profile, right_indices, peak_value
        )
        left_crossing = _crossing(left_bounded)
        right_crossing = _crossing(right_bounded)
        if left_crossing is None or right_crossing is None:
            # D-05: priority to the more fundamental failure. A side that
            # failed to cross AND never found a bounding local minimum
            # means there was no lobe boundary to search within at all --
            # tag the whole position "no local minimum before array edge".
            # Otherwise both sides that failed to cross did find a
            # bounding minimum, so the lobe itself is genuinely too narrow
            # to reach half-max -- "central lobe too narrow to reach
            # half-max".
            no_minimum_on_a_failed_side = (
                left_crossing is None and not left_found_minimum
            ) or (right_crossing is None and not right_found_minimum)
            # Round for display only -- positions_um itself stays exact
            # (D-08); float64 multiplication of index * dz can produce
            # artifacts like 3 * 0.3 == 0.8999999999999999, which would
            # silently fail to name "0.9" in the warning text below.
            position_um = round(float(positions_um[z]), 6)
            if no_minimum_on_a_failed_side:
                failed_no_local_minimum.append(position_um)
            else:
                failed_lobe_too_narrow.append(position_um)
            continue
        widths_um[z] = (right_crossing - left_crossing) * dxy

    # D-09: name every failed position, not just the first -- measured
    # during planning at illumination_na=0.6, psf_size_z=21, psf_size_xy=16,
    # index 0 measures cleanly while indices 1 through 5 do not, so the
    # NaN block is neither contiguous with an array edge nor inferable from
    # the array's shape. A caller who only saw "5 positions failed" or
    # "first failure at 0.3 um" could not reconstruct which entries to
    # skip. The array itself stays untouched -- no clamping to the lateral
    # extent, no extrapolation from neighbours, no zero fill, and no
    # dropping of failed entries (which would break the D-07/D-08
    # parallel-array length contract Phase 6 depends on); this warning is
    # purely additive surfacing, now grouped by reason (D-04/MEAS-05).
    message_parts = []
    if failed_no_local_minimum:
        message_parts.append(
            f"no local minimum before array edge at (um): {failed_no_local_minimum!r}"
        )
    if failed_lobe_too_narrow:
        message_parts.append(
            "central lobe too narrow to reach half-max at (um): "
            f"{failed_lobe_too_narrow!r}"
        )
    if message_parts:
        warnings.warn(
            "measure_beam_width_profile: " + "; ".join(message_parts),
            stacklevel=2,
        )

    return positions_um, widths_um
