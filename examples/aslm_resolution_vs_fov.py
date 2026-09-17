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
"""Plot ASLM axial resolution vs. position across a fixed FOV window (EX-05)."""

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

# D-06: detection NA held fixed across the whole illumination_na sweep and
# named in the figure, matching examples/slit_width_sweep.py's own detection
# NA. It is deliberately NOT threaded into the measurement call below --
# MEAS-01 locks this measurement to the pre-rotation illumination PSF, and
# feeding the detection arm in would mix the detection PSF's own axial
# extent into the result and destroy the NA dependence this plot exists to
# show (the same warning appears at simulate/beam_profile.py lines 71-77).
DETECTION_NA = 1.0

# D-06: fixed across the whole sweep -- slit_width is not the swept variable
# in this phase; examples/slit_width_sweep.py already owns that axis.
# Matches examples/light_sheet_vs_aslm.py's existing SLIT_WIDTH precedent.
SLIT_WIDTH = 2.0

# D-07: five points spanning the validated 0.1-0.45 band. The floor, 0.10,
# exercises roadmap Success Criterion 5's lowest-sweep-point requirement and
# is the validated band's own floor. The ceiling is capped at 0.45 because at
# illumination NA 0.50 and above the located waist becomes a numerical
# artifact even at this window size (measured Rayleigh extents of 0.014,
# 0.502 and 1.231 um at NA 0.50, 0.55 and 0.60). Five points (not the full
# eight-point band) keep the whole run under about 25 s of PSF generation
# (measured per-point: 1.7 s at NA 0.10 rising to 10.3 s at NA 0.45) while
# still spanning the band end to end.
NA_SWEEP: tuple[float, ...] = (0.10, 0.15, 0.25, 0.35, 0.45)

COMMON = {
    "wavelength": 0.561,
    "ni": 1.33,
    "ns": 1.33,
    "dxy": 0.108,
    "dz": 0.300,
    # D-01: the fixed +-200 um window at the project's standard dz=0.300
    # needs 1335 slices -- (1335 - 1) * 0.300 == 400.2 um total extent, i.e.
    # +-200.1 um once re-centred. Measured cost is about 8.3 s and 372 MB
    # per PSF generation on CPU, so there is no reason to coarsen dz instead.
    "psf_size_z": 1335,
    "psf_size_xy": 128,
}


def run_sweep(
    nas: tuple[float, ...] = NA_SWEEP,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Measure the gated beam-width profile at each swept illumination NA.

    Returns a list of `(na, centered_um, widths_um)` triples in the
    requested order. `centered_um` re-centres `positions_um` on the array's
    own geometric midpoint -- `positions_um[0]` here is exactly 0.0 and
    needs no half-voxel offset, unlike examples/light_sheet_vs_aslm.py's
    display-coordinate formula, which carries an offset appropriate to a
    rotated combined seed volume.
    """
    results: list[tuple[float, np.ndarray, np.ndarray]] = []
    for na in nas:
        positions_um, widths_um = simulate.measure_gated_beam_width_profile(
            illumination_na=na, slit_width=SLIT_WIDTH, **COMMON
        )
        centered_um = positions_um - positions_um[-1] / 2.0
        results.append((na, centered_um, widths_um))
    return results


def print_table(results: list[tuple[float, np.ndarray, np.ndarray]]) -> None:
    """Print one diagnostic row per swept NA: finite count, min/max width, max/min ratio.

    Statistics are computed with `np.nanmin`/`np.nanmax`/`np.isfinite(...).sum()`
    so the legitimately unmeasurable NaN positions are excluded from the
    statistics rather than propagating into every column. A row with no
    finite entry at all renders its numeric cells as an explicit SKIPPED
    marker -- never as a number and never omitted -- matching
    examples/slit_width_sweep.py::print_table's discipline. An empty
    `results` still prints the header and a plain no-points line rather than
    raising on an empty sequence.

    This table is a diagnostic summary only: the position-resolved figure
    below is the deliverable, and the max/min ratio column here does not
    replace it -- no Rayleigh-range annotation applies to ASLM (D-03).
    """
    half_extent = (COMMON["psf_size_z"] - 1) * COMMON["dz"] / 2.0
    print(
        f"illumination NA sweep -- slit_width={SLIT_WIDTH:.2f} um, "
        f"detection_na={DETECTION_NA:.2f}, window=+-{half_extent:.1f} um"
    )
    print(
        f"{'na':>6}  {'finite/total':>14}  {'min width (um)':>16}  "
        f"{'max width (um)':>16}  {'max/min ratio':>14}"
    )
    if not results:
        print("  (no sweep points)")
    else:
        for na, _centered_um, widths_um in results:
            total = widths_um.size
            finite_count = int(np.isfinite(widths_um).sum())
            counts = f"{finite_count}/{total}"
            if finite_count == 0:
                print(
                    f"{na:>6.2f}  {counts:>14}  {'SKIPPED':>16}  "
                    f"{'SKIPPED':>16}  {'SKIPPED':>14}"
                )
                continue
            min_width = np.nanmin(widths_um)
            max_width = np.nanmax(widths_um)
            ratio = max_width / min_width
            print(
                f"{na:>6.2f}  {counts:>14}  {min_width:>16.4f}  "
                f"{max_width:>16.4f}  {ratio:>14.4f}"
            )
    print(
        "note: this table is a diagnostic summary -- the position-resolved "
        "figure is the deliverable, not the ratio column above"
    )


def build_sweep_figure(
    results: list[tuple[float, np.ndarray, np.ndarray]],
) -> plt.Figure | None:
    """Plot measured ASLM axial resolution vs. position, one line per NA.

    Returns None when no result holds any finite width. Widths are plotted
    exactly as measured: matplotlib renders NaN as a gap, which is the
    honest rendering of the far-field positions where no half-max crossing
    exists (roughly 30-40 percent of this window) -- these gaps are not
    filled, interpolated across, or clipped. The title states only the
    fixed parameters this run used; it does not assert a degradation
    direction this run did not measure.
    """
    if not any(np.isfinite(widths_um).any() for _na, _centered_um, widths_um in results):
        print("no sweep point produced a measurable width -- skipping figure")
        return None

    fig, ax = plt.subplots(figsize=(9, 6))
    for na, centered_um, widths_um in results:
        ax.plot(centered_um, widths_um, label=f"illumination_na={na:.2f}")
    ax.set_xlabel("position along beam axis (um)")
    ax.set_ylabel("illumination-limited axial resolution, transverse FWHM (um)")
    ax.legend()
    half_extent = (COMMON["psf_size_z"] - 1) * COMMON["dz"] / 2.0
    ax.set_title(
        "ASLM axial resolution vs. position\n"
        f"detection_na={DETECTION_NA:.2f}, slit_width={SLIT_WIDTH:.2f} um, "
        f"window=+-{half_extent:.1f} um"
    )
    fig.tight_layout()
    return fig


def main() -> None:
    """Run the sweep, build the figure, and save it under examples/output/."""
    results = run_sweep()
    print_table(results)
    fig = build_sweep_figure(results)
    if fig is None:
        raise SystemExit("no sweep point produced a measurable width -- nothing to plot or save")

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "aslm_resolution_vs_fov.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
