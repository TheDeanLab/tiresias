"""Position-dependent illumination beam-width measurement under ASLM slit gating.

Extends `beam_profile.py`'s illumination-arm half-max measurement (Phase 5) to
a slit-gated illumination PSF, mirroring the Gaussian rolling-shutter taper
`generate_psf_seed`'s `psf_mode="aslm"` path already applies in production
(`tiresias.seeds._apply_aslm_slit_gate`/`_gaussian_slit_window`). Kept as a
sibling module to `beam_profile.py`, outside the installed `tiresias` package
(D-01/D-02), per D-04/D-05.
"""

from __future__ import annotations

import warnings

import numpy as np

# D-04: simulate/ is an external consumer of the installed tiresias package,
# so this import is absolute (tiresias.seeds), matching beam_profile.py's own
# convention. The second name, `_apply_aslm_slit_gate`, reaches past the
# leading-underscore "module-internal" convention deliberately: this module
# needs bit-for-bit parity with the gate the production aslm path applies,
# and no public equivalent exists. This is a documented one-off (08-RESEARCH
# Pitfall 6) -- it is not licence to import other seeds.py internals.
from tiresias.seeds import generate_theoretical_psf, _apply_aslm_slit_gate

__all__ = ["measure_gated_beam_width_profile"]

# D-04, 08-RESEARCH.md Pitfall 1: axis 1 is the transverse (Y) axis this
# measurement reports width along, in the pre-rotation frame this module
# (like beam_profile.py) never rotates out of. It is the only axis where the
# gate changes the answer: gating axis 0 or axis 2 multiplies each Z-slice's
# Y-profile by a per-slice-uniform scalar, which cancels out of the half-max
# crossing test and leaves the measured widths equal to the ungated ones
# within rtol=1e-6 (independently measured: max relative deviation 6.5e-08
# for axis 0 and 5.3e-08 for axis 2, against 6.4e-01 for axis 1, at
# illumination_na=0.4, psf_size_z=61, slit_width=2.0).
#
# This module must not call the rotated-frame gate-axis resolver in
# seeds.py, the spherical-direction helper, or the rotation routine: this
# module never rotates, so the production rotated-frame gate axis is the
# wrong question here.
GATE_AXIS: int = 1

# D-14: private (D-14), not exported -- __all__ below stays unchanged. Defined
# separately here and again in simulate/beam_profile.py rather than shared
# (Phase 8 D-04 forbids a cross-module helper for this duplicated crossing
# loop). The value matches the rtol=1e-6 float32-noise precedent already
# established in test_gate_axis_is_load_bearing (this file's own test
# module). It is costly to change: this exact value is baked into the 09-01
# ring-free MEAS-04 fixture verification and every ringed-profile
# expectation in this phase's tests, so a different value would require
# recapturing and re-verifying both.
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


def _measure_widths_from_array(
    psf: np.ndarray, dxy: float, positions_um: np.ndarray
) -> np.ndarray:
    """Apply the Phase 5 half-max interpolation loop to an already-prepared array.

    Mirrors `beam_profile.py::measure_beam_width_profile`'s fixed-global-peak,
    per-Z half-max-crossing algorithm, now including the independently
    duplicated central-lobe bound, so the public function here and Phase 5's
    original share one implementation in spirit (this is the documented
    duplication D-04 calls for, pinned by a dedicated regression test rather
    than refactored into a shared private helper across modules).
    """
    psf_size_z = psf.shape[0]

    # D-03 (Phase 5): fixed global peak, not re-located per Z slice --
    # psfmodels' illumination PSF has zero near-focus lateral drift, but a
    # per-slice argmax drifts onto sidelobes far from focus.
    _peak_z, peak_y, peak_x = np.unravel_index(np.argmax(psf), psf.shape)

    widths_um = np.full(psf_size_z, np.nan, dtype=np.float64)
    failed_no_local_minimum: list[float] = []
    failed_lobe_too_narrow: list[float] = []

    for z in range(psf_size_z):
        profile = psf[z, :, peak_x].astype(np.float64)
        peak_value = profile[peak_y]
        if peak_value <= 0:
            # D-15: no lobe to bound at all -- more fundamental than a
            # bounded-but-too-narrow lobe (D-05 priority), so this goes to
            # the no-local-minimum group. Round for display only --
            # positions_um itself stays exact; float64 multiplication of
            # index * dz can produce artifacts like
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
            position_um = round(float(positions_um[z]), 6)
            if no_minimum_on_a_failed_side:
                failed_no_local_minimum.append(position_um)
            else:
                failed_lobe_too_narrow.append(position_um)
            continue
        widths_um[z] = (right_crossing - left_crossing) * dxy

    # D-09 (Phase 5)/D-04 (Phase 9): name every failed position, not just
    # the first, and group by reason -- purely additive surfacing, the
    # array itself stays untouched (no clamping, no extrapolation, no
    # dropping of failed entries).
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
            "measure_gated_beam_width_profile: " + "; ".join(message_parts),
            stacklevel=2,
        )

    return widths_um


def measure_gated_beam_width_profile(
    *,
    illumination_na: float,
    wavelength: float,
    ni: float,
    ns: float,
    dxy: float,
    dz: float,
    slit_width: float,
    psf_size_z: int = 61,
    psf_size_xy: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the ASLM slit-gated illumination beam's transverse (Y) FWHM at every Z position.

    Returns `(positions_um, widths_um)`: two parallel float64 arrays, each of
    length `psf_size_z`, identical in contract to
    `beam_profile.measure_beam_width_profile`. `positions_um[i] == i * dz`,
    in micrometres, strictly ascending, starting at exactly 0.0. `widths_um[i]`
    is the interpolated half-max Y-profile width of the gated illumination PSF
    measured at `positions_um[i]`, or NaN when no clean half-max crossing
    exists at that position. There is deliberately no caller-facing axis
    parameter -- see GATE_AXIS.
    """
    # ASVS V5: validate this function's OWN parameters before spending time
    # on PSF generation, mirroring beam_profile.py's required_values/missing
    # guard, extended with slit_width. A malformed call fails in
    # milliseconds instead of after an ~8 s, ~372 MB allocation at this
    # phase's psf_size_z=1335 window.
    required_values = {
        "illumination_na": illumination_na,
        "wavelength": wavelength,
        "ni": ni,
        "ns": ns,
        "dxy": dxy,
        "dz": dz,
        "slit_width": slit_width,
        "psf_size_z": psf_size_z,
        "psf_size_xy": psf_size_xy,
    }
    missing = [name for name, value in required_values.items() if value is None or value <= 0]
    if missing:
        raise ValueError(
            "Missing or non-positive parameter(s) for measure_gated_beam_width_profile: "
            + ", ".join(missing)
        )

    # D-04: generate the illumination-arm PSF internally by plugging
    # illumination_na into detection_na -- the same trick
    # measure_beam_width_profile and generate_psf_seed's own illumination
    # branch use. Deliberately never generate_psf_seed(): that returns
    # detection times rotated illumination, which would mix the detection
    # PSF's own axial extent into this measurement.
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

    gated = _apply_aslm_slit_gate(psf, GATE_AXIS, slit_width, dxy, dz)

    positions_um = np.arange(psf_size_z, dtype=np.float64) * dz
    widths_um = _measure_widths_from_array(gated, dxy, positions_um)

    return positions_um, widths_um
