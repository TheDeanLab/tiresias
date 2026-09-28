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
"""Plot static light-sheet system-PSF axial FWHM vs. position across a fixed FOV window (EX-04, 8.1 D-09).

Notes / limitations (printed to stdout by main(), kept out of the figure
itself for publication -- user-directed deviation, plan 08.1-13):
    - Residual (deferred): illumination is simulated as a 3-D pencil beam,
      not a y-integrated light sheet, so off-waist light-sheet axial FWHM
      shows pencil-beam Fresnel structure beyond about 1-2 Rayleigh ranges
      rather than a smooth rise; the ASLM curve is flat by construction
      under perfect shutter/beam synchronization.
    - Sampling: dz == dxy with ni0 == ni.
"""

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

# 8.1 D-05: detection NA held fixed across the whole illumination_na sweep
# and named in the figure, identical to examples/aslm_resolution_vs_fov.py's
# own detection NA so the two scripts' figures are directly comparable
# (roadmap Success Criterion 2). It IS now threaded into the measurement
# below -- the reported quantity is the system-PSF (detection x effective
# illumination) axial FWHM, not an illumination-only width (D-06 replaces
# the retired self-gated measurement's illumination-only scope).
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
# makes this explicit). A test pins this script's NA_SWEEP/DETECTION_NA/
# COMMON/FOV_POSITIONS_UM to examples/aslm_resolution_vs_fov.py's own values
# so the duplication cannot silently drift.
#
# 8.1 D-05/D-09: ni0 equals ni to avoid the psfmodels default-ni0 aberration
# (RESEARCH Pitfall 1); dz equals dxy because the legacy rot90 fast path
# relabels pre-rotation axes without resampling instead of resampling them
# (ROT-04); psf_size_z=61 is the per-position window, because the detection
# PSF confines the system-PSF axial profile to about the depth of focus, so
# a much larger window (like the retired gate's +-200.1 um) is unnecessary
# here.
COMMON = {
    "wavelength": 0.561,
    "ni": 1.33,
    "ns": 1.33,
    "ni0": 1.33,
    "dxy": 0.108,
    "dz": 0.108,
    "psf_size_z": 61,
    "psf_size_xy": 128,
}

# 8.1 D-05/D-09: the FOV positions at which the system-PSF axial FWHM is
# measured, one real generate_psf_seed(emitter_offset=p) call per position
# (via simulate.measure_light_sheet_system_fwhm_profile). Identical to
# examples/aslm_resolution_vs_fov.py's own FOV_POSITIONS_UM so the two
# scripts share one x axis. +-50 um (not the retired gate's +-200.1 um)
# because beyond about 1-2 Rayleigh ranges this light-sheet curve becomes
# erratic (pencil-beam Fresnel structure) and at |p| >= 100 um it falls
# below its own waist value -- measured 0.494 um at |p|=100 vs 0.579 um at
# the waist (illumination_na=0.45). 51 points at 2 um steps, containing
# exactly 0.0. Measured cost is about 0.45 s per position, so a full sweep
# over NA_SWEEP takes about 2-3 minutes.
FOV_POSITIONS_UM: tuple[float, ...] = tuple(float(v) for v in np.linspace(-50.0, 50.0, 51))

# D-06: the Rayleigh-range annotation is illumination-only (unchanged from
# the shipped Phase 6 `simulate.measure_beam_width_profile` /
# `simulate.locate_rayleigh_range` pair) and needs a much larger window than
# the DOF-confined system-PSF measurement above to find the sqrt(2)x-waist
# crossing -- the project's standard dz=0.300 with 1335 slices spans
# +-200.1 um once re-centred, the same window the retired gate used to ship.
# `measure_beam_width_profile` cannot take ni0 (RESEARCH Pitfall 1 does not
# apply to it), so this annotation carries psfmodels' default-ni0 axial
# offset of about 0.7 um -- an approximate marker, not a precise boundary.
RAYLEIGH_WINDOW = {"dz": 0.300, "psf_size_z": 1335}

# D-19/RA-6-disclose: pencil-beam residual disclosure, duplicated per the
# single-file PEP 723 philosophy (no shared helper module between example
# scripts) rather than imported from a sibling script. Deferred to a later
# seeds.py phase (deferred-items.md DEF-SEEDS-2/DEF-SEEDS-5); tracked
# permanently by tests/test_example_simulation_audit.py.
PENCIL_BEAM_NOTE = (
    "Residual (deferred): illumination is simulated as a 3-D pencil beam, "
    "not a y-integrated light sheet, so off-waist light-sheet axial FWHM "
    "shows pencil-beam Fresnel structure beyond about 1-2 Rayleigh ranges "
    "rather than a smooth rise; the ASLM curve is flat by construction "
    "under perfect shutter/beam synchronization. Sampling: dz == dxy with "
    "ni0 == ni."
)


def run_sweep(
    nas: tuple[float, ...] = NA_SWEEP,
    positions_um: tuple[float, ...] | None = None,
) -> list[tuple[float, np.ndarray, np.ndarray, tuple[float, float, float] | None]]:
    """Measure the light-sheet system-PSF axial FWHM at each swept NA and annotate its Rayleigh boundary.

    Returns a list of `(na, positions_um, fwhm_um, rayleigh)` quadruples in
    the requested order. The curve itself comes from
    `simulate.measure_light_sheet_system_fwhm_profile(positions_um=positions,
    detection_na=DETECTION_NA, illumination_na=na, **COMMON)` -- an emitter
    at FOV position p sits off the illumination waist by p along
    propagation (D-06). `positions_um` defaults to the module-level
    `FOV_POSITIONS_UM`, read at call time, so
    `examples/resolution_vs_fov_by_na.py`'s own `run_sweep()` call and this
    module's regression tests both see the current value; the returned
    positions are already centred on the waist home, so no re-centring
    offset applies to the curve.

    `rayleigh` is separately measured through the illumination-only
    `simulate.measure_beam_width_profile(**RAYLEIGH_WINDOW)` (D-06 keeps
    this function for annotation only) over its own, much larger window --
    it is either a `(waist_um, left_um, right_um)` triple re-centred by that
    window's own offset (`positions_um[-1] / 2`) onto the same axis as the
    curve, or None when the Phase 6 locator raised `ValueError` for this NA
    (roadmap Success Criterion 5). A rejected NA keeps its ordered slot in
    the returned list and the sweep continues with the remaining points.
    """
    if positions_um is None:
        positions_um = FOV_POSITIONS_UM
    positions = np.asarray(positions_um, dtype=np.float64)
    results: list[tuple[float, np.ndarray, np.ndarray, tuple[float, float, float] | None]] = []
    for na in nas:
        centered_um, fwhm_um = simulate.measure_light_sheet_system_fwhm_profile(
            positions_um=positions,
            detection_na=DETECTION_NA,
            illumination_na=na,
            **COMMON,
        )

        rayleigh_positions_um, rayleigh_widths_um = simulate.measure_beam_width_profile(
            illumination_na=na,
            wavelength=COMMON["wavelength"],
            ni=COMMON["ni"],
            ns=COMMON["ns"],
            dxy=COMMON["dxy"],
            psf_size_xy=COMMON["psf_size_xy"],
            **RAYLEIGH_WINDOW,
        )
        offset = rayleigh_positions_um[-1] / 2.0
        try:
            waist_um, left_um, right_um = simulate.locate_rayleigh_range(
                rayleigh_positions_um, rayleigh_widths_um
            )
        except ValueError as exc:
            print(f"skipped Rayleigh annotation for illumination_na={na!r}: {exc}")
            rayleigh = None
        else:
            # Subtract the same offset used to centre the Rayleigh window so
            # the annotation lands on the same centred axis as the curve.
            rayleigh = (waist_um - offset, left_um - offset, right_um - offset)
        results.append((na, centered_um, fwhm_um, rayleigh))
    return results


def print_table(
    results: list[tuple[float, np.ndarray, np.ndarray, tuple[float, float, float] | None]],
) -> None:
    """Print one diagnostic row per swept NA, including the D-06 Rayleigh annotation.

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
    half_extent = max(abs(v) for v in FOV_POSITIONS_UM)
    print(
        f"illumination NA sweep -- detection_na={DETECTION_NA:.2f}, "
        f"FOV=+-{half_extent:.1f} um"
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
    """Overlay one measured light-sheet system-PSF axial FWHM curve per swept NA.

    Returns None when no result holds any finite width -- the caller must
    handle that return rather than assume a Figure. An individual NA whose
    `widths_um` is entirely NaN is skipped (no invisible line, no misleading
    legend entry), but every other NA's curve is kept and plotted exactly as
    measured, NaN entries included: matplotlib renders NaN as a break in the
    line. The system-PSF measurement is DOF-confined (D-09), so gaps are not
    expected across this +-50 um FOV window -- but any NaN that does occur
    is still plotted as an honest break, never filled, interpolated across,
    clipped, or hidden by narrowing the plotted x range.

    For each NA whose Rayleigh annotation succeeded (D-06), the left/right
    boundary is drawn as a thin dotted vertical line in the same colour as
    that NA's curve -- but only when the boundary position falls inside the
    fixed plotted window (roadmap Success Criterion 2); a boundary that
    falls outside the window is never drawn off-canvas, and is instead named
    in the title so the reader learns the Rayleigh point left the window
    rather than silently seeing nothing. The Rayleigh annotation never
    rescales or redefines the plotted window -- `ax.set_xlim` is always
    pinned to the fixed half-extent computed from `FOV_POSITIONS_UM`,
    independent of any measured Rayleigh position. Exactly one proxy legend
    entry describes the dotted lines, regardless of how many NA values were
    annotated. Any statement about how the curves behave is computed from
    `results`, never written as fixed text.
    """
    plottable = [
        (na, centered_um, widths_um, rayleigh)
        for na, centered_um, widths_um, rayleigh in results
        if np.isfinite(widths_um).any()
    ]
    if not plottable:
        print("no sweep point produced a measurable width -- skipping figure")
        return None

    half_extent = max(abs(v) for v in FOV_POSITIONS_UM)

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
    ax.set_ylabel("system-PSF axial FWHM (um)")
    ax.set_xlim(-half_extent, half_extent)
    title_second_line = f"detection_na={DETECTION_NA:.2f}, FOV=+-{half_extent:.1f} um"
    if out_of_window_notes:
        title_second_line += " (" + "; ".join(out_of_window_notes) + ")"
    ax.set_title(
        "Static light-sheet system-PSF axial FWHM vs. FOV position\n" + title_second_line
    )
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    """Run the sweep, print the table, build the figure, and save it under examples/output/."""
    results = run_sweep()
    print_table(results)
    # User-directed deviation (plan 08.1-13): the pencil-beam disclosure used
    # to render as a fig.text footnote; it is now printed to stdout (and kept
    # in the module docstring's Notes / limitations section) instead, so the
    # figure itself stays clean for publication.
    print(PENCIL_BEAM_NOTE)
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
