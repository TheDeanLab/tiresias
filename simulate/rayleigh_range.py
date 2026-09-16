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
    """
    # D-05: three positions only, no waist width. D-05 mandates the waist
    # position plus both side positions; the waist width is one
    # np.nanmin(widths_um) call away for any caller that already holds the
    # profile, and appending a width to a tuple of positions makes the
    # unpack order a foot-gun for Phase 8. A plain tuple is used rather than
    # a NamedTuple or dataclass because neither idiom appears anywhere else
    # in this codebase.

    # FOV-02: 06-02 Task 1 inserts the full validation block at this seam.
    positions_um = np.asarray(positions_um, dtype=np.float64)
    widths_um = np.asarray(widths_um, dtype=np.float64)

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

    # D-09: 06-02 Task 2 adds the ValueError for a None side here.
    # D-05: the two sides are reported separately and are never averaged or
    # collapsed into one number here -- choosing symmetric-average versus
    # full-range is Phase 8's call.
    return waist_position_um, left_position_um, right_position_um
