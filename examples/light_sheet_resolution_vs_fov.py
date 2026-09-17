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
"""Plot static light-sheet axial resolution vs. position across a fixed FOV window (EX-04)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# simulate/ is a sibling of examples/, not a declared PEP 723 dependency --
# a script run through `uv run` only gets its own directory prepended to
# sys.path, not the repo root, so the repo root must be inserted explicitly
# before `import simulate` can resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import simulate

# D-01: detection NA held fixed across the whole illumination_na sweep and
# named in the figure, identical to examples/aslm_resolution_vs_fov.py's own
# detection NA so the two scripts' figures are directly comparable (roadmap
# Success Criterion 2). Deliberately NOT threaded into the measurement call
# below -- MEAS-01 locks this measurement to the pre-rotation illumination
# PSF, and feeding the detection arm in would mix the detection PSF's own
# axial extent into the result and destroy the NA dependence this plot
# exists to show (the same warning appears at simulate/beam_profile.py lines
# 71-77).
DETECTION_NA = 1.0

# D-07: five points spanning the validated 0.1-0.45 band, identical to
# examples/aslm_resolution_vs_fov.py's own NA_SWEEP so the two scripts share
# the same x axis (roadmap Success Criterion 2). The floor, 0.10, exercises
# roadmap Success Criterion 5's lowest-sweep-point requirement. The ceiling
# is capped at 0.45 because at illumination NA 0.50 and above the located
# waist becomes a numerical artifact even at this window size (measured
# Rayleigh extents of 0.014, 0.502 and 1.231 um at NA 0.50, 0.55 and 0.60).
NA_SWEEP: tuple[float, ...] = (0.10, 0.15, 0.25, 0.35, 0.45)

# Duplicated rather than imported from a shared module on purpose: the
# project's single-file PEP 723 philosophy forbids a shared helper module
# between example scripts (examples/slit_width_sweep.py's own D-07 comment
# makes this explicit). Plan 08-04 commits a test that pins this script's
# NA_SWEEP/DETECTION_NA/COMMON to examples/aslm_resolution_vs_fov.py's own
# values so the duplication cannot silently drift.
COMMON = {
    "wavelength": 0.561,
    "ni": 1.33,
    "ns": 1.33,
    "dxy": 0.108,
    # D-01: the fixed +-200 um window at the project's standard dz=0.300
    # needs 1335 slices -- (1335 - 1) * 0.300 == 400.2 um total extent, i.e.
    # +-200.1 um once re-centred. Measured cost is about 8.3 s and 372 MB
    # per PSF generation on CPU, so there is no reason to coarsen dz instead.
    "dz": 0.300,
    "psf_size_z": 1335,
    "psf_size_xy": 128,
}


def run_sweep(
    nas: tuple[float, ...] = NA_SWEEP,
) -> list[tuple[float, np.ndarray, np.ndarray, tuple[float, float, float] | None]]:
    """Measure the light-sheet beam-width profile at each swept NA and annotate its Rayleigh boundary.

    Returns a list of `(na, centered_um, widths_um, rayleigh)` quadruples in
    the requested order. No slit gate applies to a static light sheet
    (D-04): `centered_um`/`widths_um` come straight from the shipped Phase 5
    `simulate.measure_beam_width_profile`, called unmodified. `rayleigh` is
    either a `(waist_um, left_um, right_um)` triple on the same centred axis
    as the curve (D-02, annotation only -- it never redefines the fixed
    window D-01 sets), or None when the Phase 6 locator raised `ValueError`
    for this NA (roadmap Success Criterion 5). A rejected NA keeps its
    ordered slot in the returned list and the sweep continues with the
    remaining points.
    """
    results: list[tuple[float, np.ndarray, np.ndarray, tuple[float, float, float] | None]] = []
    for na in nas:
        positions_um, widths_um = simulate.measure_beam_width_profile(illumination_na=na, **COMMON)
        offset = positions_um[-1] / 2.0
        centered_um = positions_um - offset
        try:
            waist_um, left_um, right_um = simulate.locate_rayleigh_range(positions_um, widths_um)
        except ValueError as exc:
            print(f"skipped Rayleigh annotation for illumination_na={na!r}: {exc}")
            rayleigh = None
        else:
            # Subtract the same offset used for centered_um so the
            # annotation lands on the same centred axis as the curve.
            rayleigh = (waist_um - offset, left_um - offset, right_um - offset)
        results.append((na, centered_um, widths_um, rayleigh))
    return results


def print_table(
    results: list[tuple[float, np.ndarray, np.ndarray, tuple[float, float, float] | None]],
) -> None:
    """Print one diagnostic row per swept NA, including the D-02 Rayleigh annotation.

    Statistics are computed with `np.nanmin`/`np.nanmax`/`np.isfinite(...).sum()`
    so the legitimately unmeasurable NaN positions are excluded from the
    statistics rather than propagating into every column. A row whose width
    profile has no finite entry at all, or whose Rayleigh annotation was
    rejected (Success Criterion 5), renders the affected cells with an
    explicit SKIPPED marker -- never as a number and never omitted. An empty
    `results` still prints the header and a plain no-points line rather than
    raising.

    This table is a diagnostic summary only: the position-resolved figure
    below is the deliverable, and the max/min ratio and Rayleigh-extent
    columns here do not replace it.
    """
    half_extent = (COMMON["psf_size_z"] - 1) * COMMON["dz"] / 2.0
    print(
        f"illumination NA sweep -- detection_na={DETECTION_NA:.2f}, "
        f"window=+-{half_extent:.1f} um"
    )
    print(
        f"{'na':>6}  {'finite/total':>14}  {'min width (um)':>16}  "
        f"{'max width (um)':>16}  {'max/min ratio':>14}  "
        f"{'rayleigh extent (um)':>21}  {'in window':>10}"
    )
    if not results:
        print("  (no sweep points)")
    else:
        for na, _centered_um, widths_um, rayleigh in results:
            total = widths_um.size
            finite_count = int(np.isfinite(widths_um).sum())
            counts = f"{finite_count}/{total}"
            if finite_count == 0:
                min_str = max_str = ratio_str = "SKIPPED"
            else:
                min_width = np.nanmin(widths_um)
                max_width = np.nanmax(widths_um)
                ratio = max_width / min_width
                min_str = f"{min_width:.4f}"
                max_str = f"{max_width:.4f}"
                ratio_str = f"{ratio:.4f}"
            if rayleigh is None:
                extent_str = "SKIPPED"
                in_window_str = "SKIPPED"
            else:
                _waist_um, left_um, right_um = rayleigh
                extent_str = f"{(right_um - left_um):.4f}"
                inside = -half_extent <= left_um <= half_extent and -half_extent <= right_um <= half_extent
                in_window_str = "yes" if inside else "no"
            print(
                f"{na:>6.2f}  {counts:>14}  {min_str:>16}  "
                f"{max_str:>16}  {ratio_str:>14}  "
                f"{extent_str:>21}  {in_window_str:>10}"
            )
    print(
        "note: this table is a diagnostic summary -- the position-resolved "
        "figure is the deliverable, not the ratio/extent columns above"
    )


def build_sweep_figure(
    results: list[tuple[float, np.ndarray, np.ndarray, tuple[float, float, float] | None]],
) -> plt.Figure | None:
    """Overlay one measured light-sheet axial-resolution-vs-position curve per swept NA.

    Returns None when no result holds any finite width -- the caller must
    handle that return rather than assume a Figure. An individual NA whose
    `widths_um` is entirely NaN is skipped (no invisible line, no misleading
    legend entry), but every other NA's curve is kept and plotted exactly as
    measured, NaN entries included: matplotlib renders NaN as a break in the
    line, which is the honest rendering of the far-field positions where the
    profile never drops to half maximum. These gaps are never filled,
    interpolated across, clipped, or hidden by narrowing the plotted x
    range.

    For each NA whose Rayleigh annotation succeeded (D-02), the left/right
    boundary is drawn as a thin dotted vertical line in the same colour as
    that NA's curve -- but only when the boundary position falls inside the
    fixed plotted window (D-01); a boundary that falls outside the window is
    never drawn off-canvas, and is instead named in the title so the reader
    learns the Rayleigh point left the window rather than silently seeing
    nothing. The Rayleigh annotation never rescales or redefines the plotted
    window -- `ax.set_xlim` is always pinned to the fixed half-extent
    computed from COMMON, independent of any measured Rayleigh position.
    Exactly one proxy legend entry describes the dotted lines, regardless of
    how many NA values were annotated. Any statement about how the curves
    behave is computed from `results`, never written as fixed text.
    """
    plottable = [
        (na, centered_um, widths_um, rayleigh)
        for na, centered_um, widths_um, rayleigh in results
        if np.isfinite(widths_um).any()
    ]
    if not plottable:
        print("no sweep point produced a measurable width -- skipping figure")
        return None

    half_extent = (COMMON["psf_size_z"] - 1) * COMMON["dz"] / 2.0

    fig, ax = plt.subplots(figsize=(9, 6))
    rayleigh_label_used = False
    out_of_window_notes: list[str] = []
    for na, centered_um, widths_um, rayleigh in plottable:
        (line,) = ax.plot(centered_um, widths_um, label=f"illumination_na={na:.2f}")
        color = line.get_color()
        if rayleigh is None:
            continue
        _waist_um, left_um, right_um = rayleigh
        for side_name, position_um in (("left", left_um), ("right", right_um)):
            if -half_extent <= position_um <= half_extent:
                ax.axvline(
                    position_um,
                    color=color,
                    linestyle=":",
                    alpha=0.4,
                    label=None if rayleigh_label_used else "Rayleigh boundary (sqrt(2)x waist width)",
                )
                rayleigh_label_used = True
            else:
                out_of_window_notes.append(
                    f"NA={na:.2f} {side_name} Rayleigh boundary ({position_um:.1f} um) left the window"
                )

    ax.set_xlabel("position along beam propagation axis (um)")
    ax.set_ylabel("illumination-limited axial resolution, transverse FWHM (um)")
    ax.set_xlim(-half_extent, half_extent)
    title_second_line = f"detection_na={DETECTION_NA:.2f}, window=+-{half_extent:.1f} um"
    if out_of_window_notes:
        title_second_line += " (" + "; ".join(out_of_window_notes) + ")"
    ax.set_title(
        "Static light-sheet axial resolution vs. position across the fixed window\n" + title_second_line
    )
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    """Run the sweep, print the table, build the figure, and save it under examples/output/."""
    results = run_sweep()
    print_table(results)
    fig = build_sweep_figure(results)
    if fig is None:
        raise SystemExit("no sweep point produced a measurable width -- nothing to plot or save")

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "light_sheet_resolution_vs_fov.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
