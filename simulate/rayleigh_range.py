"""Locate the Rayleigh-range FOV boundary from a measured beam-width profile.

Consumes the `(positions_um, widths_um)` pair `measure_beam_width_profile`
returns and generates no PSF of its own.
"""

from __future__ import annotations

import numpy as np

__all__ = ["locate_rayleigh_range"]


def locate_rayleigh_range(
    positions_um: np.ndarray, widths_um: np.ndarray
) -> tuple[float, float, float]:
    """Locate the beam waist and the two sqrt(2)x-waist Rayleigh boundaries.

    Returns `(waist_position_um, left_position_um, right_position_um)`, all
    in micrometres. `left` is toward decreasing Z, `right` toward increasing
    Z.

    Raises `ValueError` when either outward walk reaches an array edge
    without finding the sqrt(2)x-waist crossing.
    """
    # D-05: three positions only, no waist width. D-05 mandates the waist
    # position plus both side positions; the waist width is one
    # np.nanmin(widths_um) call away for any caller that already holds the
    # profile, and appending a width to a tuple of positions makes the
    # unpack order a foot-gun for Phase 8. A plain tuple is used rather than
    # a NamedTuple or dataclass because neither idiom appears anywhere else
    # in this codebase.

    positions_um = np.asarray(positions_um, dtype=np.float64)
    widths_um = np.asarray(widths_um, dtype=np.float64)

    # ASVS V5: validate both array parameters before any waist search runs.
    # Three sequential stages, not one flat list, because later checks are
    # undefined on inputs the earlier ones reject -- you cannot meaningfully
    # diff a 2-D array, and you cannot pair-index arrays of different
    # lengths. Within each stage every offender is collected and named in
    # one message, the same idiom `beam_profile.py` already uses.

    # Stage A -- shape.
    problems: list[str] = []
    if positions_um.ndim != 1:
        problems.append(f"positions_um must be 1-D, got ndim={positions_um.ndim}")
    if widths_um.ndim != 1:
        problems.append(f"widths_um must be 1-D, got ndim={widths_um.ndim}")
    if problems:
        raise ValueError("locate_rayleigh_range: " + "; ".join(problems))

    # Stage B -- pairing.
    problems = []
    if positions_um.shape != widths_um.shape:
        problems.append(
            "positions_um and widths_um must have the same length, got "
            f"positions_um.size={positions_um.size}, widths_um.size={widths_um.size}"
        )
    if positions_um.size == 0:
        problems.append("positions_um and widths_um must not be empty")
    if problems:
        raise ValueError("locate_rayleigh_range: " + "; ".join(problems))

    # Stage C -- content.
    problems = []
    if not np.isfinite(positions_um).all():
        problems.append("positions_um must contain only finite values")
    # This is the contract measure_beam_width_profile already guarantees
    # (positions_um[i] == i * dz), asserted here because the outward walks
    # read positions by index and a scrambled or duplicated position array
    # would produce a boundary that is arithmetically valid and physically
    # meaningless. A length-1 array trivially satisfies this: np.diff of it
    # is empty and np.all([]) is True, which is intended -- a single-sample
    # profile is well-formed input.
    elif not np.all(np.diff(positions_um) > 0):
        problems.append("positions_um must be strictly ascending")

    measured = ~np.isnan(widths_um)
    if not measured.any():
        # FOV-02: RESEARCH Pitfall 1 -- without this guard, the
        # np.nanargmin call below raises NumPy's own
        # `ValueError: All-NaN slice encountered`, which names neither the
        # parameter nor the caller (reproduced during planning).
        #
        # This message is deliberately kept SEPARATE from the array-too-short
        # message Task 2 adds (RESEARCH Open Question 1), because "increase
        # psf_size_z" is actively wrong advice for a profile in which
        # nothing was measurable at any position -- a longer axial array
        # yields more NaN, not a crossing. Phase 5's own warning already
        # named every unmeasurable position, so the caller has the detail;
        # what this message adds is the correct remedy: the lateral
        # measurement window, not the axial array.
        problems.append(
            "widths_um is entirely NaN -- every position is unmeasurable, so no "
            "waist can be located; increase psf_size_xy or illumination_na when "
            "generating the beam-width profile"
        )
    else:
        bad = measured & (~np.isfinite(widths_um) | (widths_um <= 0.0))
        if bad.any():
            # The sqrt(2) threshold is computed from the waist width, so a
            # zero waist makes the threshold zero and the first sample
            # examined on each side satisfies the crossing test immediately,
            # returning a boundary equal to the waist position -- a
            # plausible-looking number with no physical meaning.
            offending = [
                round(float(v), 6) for v in positions_um[bad]
            ]
            problems.append(
                "widths_um must be positive and finite at every measured "
                f"position, offending position(s) (um): {offending!r}"
            )

    # NaN among otherwise valid widths is NOT rejected -- it is a
    # documented, legitimate Phase 5 output for an unmeasurable position,
    # and D-08 requires the walk to skip it.
    if problems:
        raise ValueError("locate_rayleigh_range: " + "; ".join(problems))

    # D-06: the waist is the NaN-ignoring minimum of the measured profile,
    # chosen over any fitted minimum so the number stays grounded in what
    # was actually simulated.
    waist_index = int(np.nanargmin(widths_um))
    waist_width_um = float(widths_um[waist_index])
    waist_position_um = float(positions_um[waist_index])

    # D-06: this project's Rayleigh definition -- the FOV boundary is where
    # the beam width has grown sqrt(2)x from its waist.
    threshold_um = float(np.sqrt(2.0)) * waist_width_um

    # D-07: first-crossing-wins search, not a global search for the
    # cleanest crossing -- no smoothing, no fitting, and no second pass
    # looking further out.
    def _walk_outward(order: range) -> float | None:
        previous: int | None = None
        for index in order:
            if np.isnan(widths_um[index]):
                # D-08: a data gap does not stop the outward search, and a
                # skipped sample is never used as an interpolation
                # endpoint.
                continue
            if widths_um[index] >= threshold_um:
                if previous is None:
                    return None
                v0, v1 = widths_um[previous], widths_um[index]
                p0, p1 = positions_um[previous], positions_um[index]
                # Denominator cannot be zero: v1 >= threshold_um > v0 by
                # construction.
                frac = (threshold_um - v0) / (v1 - v0)
                # D-06: sub-voxel interpolation the roadmap's stability
                # criterion requires -- deliberately NOT snapped to a
                # sampled position.
                return float(p0 + frac * (p1 - p0))
            previous = index
        # D-09/D-10: reaching the array edge with no crossing -- whether
        # because the widths never grew enough or because the remaining
        # samples were all NaN -- is the array-too-small signal 06-02 turns
        # into a ValueError.
        return None

    # Both ranges start AT the waist index, which is guaranteed non-NaN and
    # guaranteed below the threshold, so `previous` is always populated on
    # the first iteration of a non-empty walk.
    left_position_um = _walk_outward(range(waist_index, -1, -1))
    right_position_um = _walk_outward(range(waist_index, positions_um.size))

    unreachable: list[str] = []
    if left_position_um is None:
        # Round for display only -- positions_um itself stays exact.
        # RESEARCH Pitfall 2, mirroring beam_profile.py's identical idiom:
        # float64 multiplication of index * dz can produce artifacts like
        # 3 * 0.3 == 0.8999999999999999, and without rounding this message
        # would name that artifact instead of "0.9" or "1.8".
        unreachable.append(
            f"Z={round(float(positions_um[0]), 6)} um (walking toward decreasing Z)"
        )
    if right_position_um is None:
        unreachable.append(
            f"Z={round(float(positions_um[-1]), 6)} um (walking toward increasing Z)"
        )

    if unreachable:
        # D-09/D-10: this single branch serves both decisions -- running out
        # of samples and a NaN run reaching the edge are indistinguishable
        # for this purpose, and `_walk_outward` deliberately returns the
        # same None for both.
        #
        # FOV-02: deliberately NOT done here -- no clamping to
        # positions_um[0]/positions_um[-1], no linear extrapolation past the
        # sampled range, no mirroring of the side that did succeed, and no
        # NaN return. A caller who receives a number from this function has
        # a number the simulation actually contains. At illumination_na=0.30
        # with the project's default 61-slice window the true crossing lies
        # outside the array on both sides, and clamping would report an
        # 18 um FOV extent that is an artifact of the window size rather
        # than of the optics -- which is exactly the Phase 8 figure this
        # guard protects.
        raise ValueError(
            "locate_rayleigh_range: Rayleigh-range crossing not found before "
            "array edge at " + ", ".join(unreachable) +
            "; increase psf_size_z (or dz) when generating the beam-width "
            "profile."
        )

    # D-05: the two sides are reported separately and are never averaged or
    # collapsed into one number here -- choosing symmetric-average versus
    # full-range is Phase 8's call.
    return waist_position_um, left_position_um, right_position_um
