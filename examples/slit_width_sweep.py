# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "numpy>=1.24",
#   "scipy>=1.10",
#   "tifffile>=2024.0",
#   "psfmodels>=0.3",
#   "matplotlib>=3.8",
#   "tiresias",
# ]
#
# [tool.uv.sources]
# tiresias = { path = "..", editable = true }
#
# [tool.uv]
# override-dependencies = ["cupy-cuda11x; python_version < '0'"]
# ///
# requires-python is deliberately narrower than pyproject.toml's ">=3.10": the
# only environment on this machine with a working psfmodels wheel is CPython
# 3.12; an unbounded floor lets uv provision a newer interpreter with no
# prebuilt psfmodels wheel, forcing an sdist build that needs MSVC (the exact
# blocker STATE.md records from Phase 1).
#
# This header is byte-identical to examples/light_sheet_vs_aslm.py's metadata
# block, per D-02 ("both scripts use this identical mechanism"). Plan 04-02
# proved empirically, against a real uv (0.12.13) with a clean cache, that
# [[tool.uv.dependency-metadata]] is NOT honored inside a script's own PEP 723
# block -- a real run still resolved and downloaded cupy-cuda11x. This
# override-dependencies entry restates cupy-cuda11x behind an environment
# marker that can never be satisfied, removing it from resolution while
# keeping D-01's [tool.uv.sources] mechanism and the zero-extra-flag
# invocation intact. See .planning/phases/04-pep-723-example-scripts/
# 04-02-SUMMARY.md for the full remediation-ladder writeup.
"""Sweep ASLM slit_width and report/plot the resulting axial FWHM (EX-03)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tiresias import generate_psf_seed

# D-07: realistic optical parameters reused verbatim from docs/usage.md's
# Python API examples (identical to examples/light_sheet_vs_aslm.py's COMMON;
# duplicated here per the single-file PEP 723 philosophy -- no shared helper
# module between the two scripts).
COMMON = {
    "na": 1.0,
    "detection_na": 1.0,
    "illumination_na": 0.2,
    "wavelength": 0.561,
    "ni": 1.33,
    "ns": 1.33,
    "ni0": None,
    "tg": None,
    "tg0": None,
    "ng": None,
    "ng0": None,
    "ti0": None,
    "oversample_factor": 3,
    "psf_model": "vectorial",
    "dxy": 0.108,
    "dz": 0.300,
    "psf_size_z": 61,
    "psf_size_xy": 128,
    "background": 0.0,
    "light_sheet_angle": 90.0,
}

# The full extent of the pre-rotation gate axis. Computed from COMMON, never
# hardcoded, so the two stay in lockstep if a demo parameter is ever edited.
# For light_sheet_angle=90.0 the gate axis resolves to axis 2 (X), whose pixel
# pitch is dxy (see seeds.py::_resolve_slit_axis) -- axis 0 would use dz
# instead. At slit_width == FULL_EXTENT, _gaussian_slit_window (seeds.py:213-
# 222) returns None and the gate is skipped entirely, so the aslm seed is
# bit-identical to the light_sheet seed (ASLM-05).
FULL_EXTENT = COMMON["psf_size_xy"] * COMMON["dxy"]  # 13.824 um

# D-08: 7 points, roughly log-spaced with a factor-of-two ladder, from a
# small physically-meaningful width up to FULL_EXTENT (the ASLM-05
# equivalence anchor) as the final, deliberately exact element.
SLIT_WIDTHS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, FULL_EXTENT)


def axial_fwhm(psf: np.ndarray, dz: float) -> float | None:
    """Return the axial (Z) FWHM of a (Z, Y, X) PSF seed in physical units.

    Locates the true peak with `np.unravel_index(np.argmax(psf), psf.shape)`
    rather than `shape // 2` -- for the D-07 parameters the peak sits at
    (Y, X) = (63, 63), not the geometric centre (64, 64), and the peak Z
    index itself migrates across this sweep (D-09, RESEARCH.md Pitfall 3).
    Finds the half-maximum crossing on each side of the peak by walking
    outward and linearly interpolating between the last above-half-maximum
    sample and the first below-half-maximum one. Returns None, never raises,
    when either side never drops below half maximum inside the volume, so a
    caller can report the point instead of dying.
    """
    peak_z, peak_y, peak_x = np.unravel_index(np.argmax(psf), psf.shape)
    profile = psf[:, peak_y, peak_x].astype(np.float64)
    peak_value = profile[peak_z]
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

    left_indices = np.arange(peak_z, -1, -1)  # peak -> start, descending
    right_indices = np.arange(peak_z, profile.size)  # peak -> end, ascending

    left_crossing = _crossing(left_indices)
    right_crossing = _crossing(right_indices)
    if left_crossing is None or right_crossing is None:
        return None
    return (right_crossing - left_crossing) * dz


def run_sweep() -> list[tuple[float, float | None]]:
    """Generate an aslm seed for each declared slit_width and measure its axial FWHM.

    Iterates SLIT_WIDTHS in declared order and calls generate_psf_seed for
    every point -- no local re-implementation of PSF generation, rotation, or
    slit gating. The declared sequence is the presentation order: this
    function is the single ordered source both print_table and
    build_sweep_figure consume, so neither re-derives or re-sorts it.
    """
    results: list[tuple[float, float | None]] = []
    for width in SLIT_WIDTHS:
        seed = generate_psf_seed(psf_mode="aslm", slit_width=width, **COMMON)
        fwhm = axial_fwhm(seed, COMMON["dz"])
        results.append((width, fwhm))
    return results


def print_table(results: list[tuple[float, float | None]], reference_fwhm: float) -> None:
    """Print the D-06 slit_width/FWHM table, in the same ascending order as SLIT_WIDTHS.

    The declared SLIT_WIDTHS sequence is the presentation order (the ordering
    contract): rows are printed exactly as run_sweep() returns them, with no
    sorting step that could reorder equal or near-equal points or introduce
    duplicates. The final row -- always the FULL_EXTENT point -- is marked as
    the ASLM-05 full-extent / light_sheet-equivalent anchor.
    """
    print(f"{'slit_width (um)':>16}  {'axial FWHM (um)':>16}  note")
    last_index = len(results) - 1
    for index, (width, fwhm) in enumerate(results):
        fwhm_str = f"{fwhm:>16.4f}" if fwhm is not None else f"{'n/a':>16}"
        note = "<- full-extent / light_sheet-equivalent anchor" if index == last_index else ""
        print(f"{width:>16.4f}  {fwhm_str}  {note}")
    print(
        f"{'light_sheet reference':>16}  {reference_fwhm:>16.4f}  "
        "(plain light_sheet, no slit gate)"
    )


def build_sweep_figure(
    results: list[tuple[float, float | None]], reference_fwhm: float
) -> plt.Figure:
    """Plot measured axial FWHM against slit_width, with a light_sheet reference line.

    Title and narrative are derived from what this run actually produced --
    the tradeoff direction is read off the computed first/last FWHM values,
    never asserted from a prior expectation (see this plan's
    must_haves.prohibitions).
    """
    widths = [width for width, _ in results]
    fwhms = [fwhm for _, fwhm in results]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(widths, fwhms, marker="o", linestyle="-", label="measured axial FWHM")
    ax.axhline(
        reference_fwhm,
        color="gray",
        linestyle="--",
        label=f"light_sheet reference ({reference_fwhm:.4f} um)",
    )
    ax.set_xlabel("slit_width (um)")
    ax.set_ylabel("axial FWHM (um)")
    ax.legend()

    first_width, first_fwhm = widths[0], fwhms[0]
    last_width, last_fwhm = widths[-1], fwhms[-1]
    if last_fwhm > first_fwhm:
        direction = "widens (axial resolution degrades) as slit_width increases"
    elif last_fwhm < first_fwhm:
        direction = "narrows (axial resolution improves) as slit_width increases"
    else:
        direction = "stays constant across the sweep"
    ax.set_title(
        "ASLM axial FWHM vs slit_width\n"
        f"Measured: FWHM {direction}\n"
        f"({first_fwhm:.4f} um at slit_width={first_width:.2f} um -> "
        f"{last_fwhm:.4f} um at slit_width={last_width:.2f} um)"
    )
    fig.tight_layout()
    return fig


def main() -> None:
    """Measure the light_sheet reference, run the sweep, print the table, and save the plot."""
    reference_seed = generate_psf_seed(psf_mode="light_sheet", **COMMON)
    reference_fwhm = axial_fwhm(reference_seed, COMMON["dz"])
    if reference_fwhm is None:
        raise RuntimeError("light_sheet reference seed produced no measurable axial FWHM")

    results = run_sweep()
    print_table(results, reference_fwhm)

    # ASLM-05 anchor check: at slit_width == FULL_EXTENT the gate is skipped
    # entirely (seeds.py::_gaussian_slit_window returns None), so the aslm
    # seed is bit-identical to the light_sheet seed and their FWHMs must be
    # exactly equal -- not merely close. An approximate comparison here would
    # hide a real regression in the gate-skip path, so this never raises on
    # mismatch: it prints the discrepancy and continues so the reader still
    # gets the table and plot.
    _anchor_width, anchor_fwhm = results[-1]
    if anchor_fwhm == reference_fwhm:
        print(
            f"anchor check: full-extent aslm FWHM ({anchor_fwhm:.4f} um) "
            f"matches light_sheet reference ({reference_fwhm:.4f} um) -- exact match"
        )
    else:
        print(
            f"anchor check: full-extent aslm FWHM ({anchor_fwhm!r} um) "
            f"does NOT match light_sheet reference ({reference_fwhm!r} um) -- mismatch"
        )

    fig = build_sweep_figure(results, reference_fwhm)
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "slit_width_sweep.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
