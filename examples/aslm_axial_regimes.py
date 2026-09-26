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
# This header is byte-identical to examples/slit_width_sweep.py's metadata
# block (D-19; tests/test_aslm_axial_regimes_example.py pins this). Plan 04-02
# proved empirically, against a real uv (0.12.13) with a clean cache, that
# [[tool.uv.dependency-metadata]] is NOT honored inside a script's own PEP 723
# block -- a real run still resolved and downloaded cupy-cuda11x. This
# override-dependencies entry restates cupy-cuda11x behind an environment
# marker that can never be satisfied, removing it from resolution while
# keeping D-01's [tool.uv.sources] mechanism and the zero-extra-flag
# invocation intact. See .planning/phases/04-pep-723-example-scripts/
# 04-02-SUMMARY.md for the full remediation-ladder writeup.
"""Plot system-PSF axial FWHM against illumination NA to show the sheet-vs-DOF regimes (D-11)."""

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

# Detection NA held fixed across the whole illumination_na sweep, matching
# the D-12 regime measurements recorded in
# .planning/phases/08.1-aslm-slit-model-rework/08.1-REGIME-MEASUREMENTS.md.
DETECTION_NA = 1.1

# Fixed across the whole sweep -- the swept parameter here is illumination
# NA, not slit_width (see the ILLUMINATION_NA_SWEEP comment below for why).
# Matches examples/slit_width_sweep.py's and examples/light_sheet_vs_aslm.py's
# existing SLIT_WIDTH precedent.
SLIT_WIDTH = 2.0

# Claude's discretion (08.1-CONTEXT.md): this figure sweeps illumination NA,
# not slit_width. Plan 08.1-04's D-13 measurement showed slit width moves the
# centre FWHM by less than 2 percent, while illumination NA moves the sheet
# thickness across the depth of focus (DOF) -- the axis that actually crosses
# between the two physical regimes this figure exists to show.
ILLUMINATION_NA_SWEEP: tuple[float, ...] = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)

# ni0 equals ni to avoid the psfmodels default-ni0 aberration (RESEARCH
# Pitfall 1); dz equals dxy because the legacy rot90 fast path relabels
# pre-rotation axes without resampling instead of resampling them (ROT-04).
# The 65-sample transverse window holds the NA 0.10 waist (the widest,
# hardest-to-contain sheet in this sweep) without truncation, matching the
# window plan 08.1-01's regime measurements used.
COMMON = {
    "wavelength": 0.561,
    "ni": 1.33,
    "ns": 1.33,
    "ni0": 1.33,
    "dxy": 0.1,
    "dz": 0.1,
    "psf_size_z": 65,
    "psf_size_xy": 65,
}


def run_sweep(
    nas: tuple[float, ...] = ILLUMINATION_NA_SWEEP,
) -> tuple[float, list[tuple[float, float, float, float]]]:
    """Measure the DOF reference and, per illumination NA, the sheet/system FWHMs.

    `dof_um` (D-12) is measured once through
    `simulate.measure_detection_dof(detection_na=DETECTION_NA, **COMMON)` --
    the axial FWHM of the detection-only PSF (`psf_mode="single"`), never a
    closed-form formula. Per swept NA, `sheet_um` (D-12) is the measured
    illumination waist FWHM via `simulate.measure_sheet_thickness`;
    `light_sheet_um` and `aslm_um` are the system-PSF axial FWHM at FOV
    position 0.0 via `simulate.measure_light_sheet_system_fwhm_profile` and
    `simulate.measure_aslm_system_fwhm_profile` respectively -- every
    reference and every plotted point is measured through `simulate`, never
    computed from a formula (D-12).

    Returns `(dof_um, [(na, sheet_um, light_sheet_um, aslm_um), ...])`, one
    tuple per requested NA in the given order.
    """
    dof_um = simulate.measure_detection_dof(detection_na=DETECTION_NA, **COMMON)
    results: list[tuple[float, float, float, float]] = []
    for na in nas:
        sheet_um = simulate.measure_sheet_thickness(illumination_na=na, **COMMON)
        _positions, light_sheet_fwhm = simulate.measure_light_sheet_system_fwhm_profile(
            positions_um=(0.0,),
            detection_na=DETECTION_NA,
            illumination_na=na,
            **COMMON,
        )
        _positions, aslm_fwhm = simulate.measure_aslm_system_fwhm_profile(
            positions_um=(0.0,),
            detection_na=DETECTION_NA,
            illumination_na=na,
            slit_width=SLIT_WIDTH,
            **COMMON,
        )
        results.append((na, sheet_um, float(light_sheet_fwhm[0]), float(aslm_fwhm[0])))
    return dof_um, results


def print_table(dof_um: float, results: list[tuple[float, float, float, float]]) -> None:
    """Print one diagnostic row per swept NA, with the regime label computed from the data.

    The "sheet > DOF" / "sheet < DOF" label is a direct comparison of the
    measured `sheet_um` against the measured `dof_um` for that row -- never a
    fixed assumption about where the crossover falls.
    """
    print(
        f"illumination NA sweep -- detection_na={DETECTION_NA:.2f}, "
        f"slit_width={SLIT_WIDTH:.2f} um, DOF={dof_um:.4f} um"
    )
    print(
        f"{'na':>6}  {'sheet (um)':>12}  {'light_sheet (um)':>18}  "
        f"{'aslm (um)':>12}  regime"
    )
    for na, sheet_um, light_sheet_um, aslm_um in results:
        regime = "sheet > DOF" if sheet_um > dof_um else "sheet < DOF"
        print(
            f"{na:>6.2f}  {sheet_um:>12.4f}  {light_sheet_um:>18.4f}  "
            f"{aslm_um:>12.4f}  {regime}"
        )


def build_regime_figure(
    dof_um: float, results: list[tuple[float, float, float, float]]
) -> plt.Figure:
    """Plot axial FWHM vs illumination NA, with the measured DOF and sheet-thickness references.

    The crossover between the two regimes is located from the data itself --
    the midpoint between the last swept NA with `sheet_um > dof_um` and the
    first with `sheet_um < dof_um` -- and used only to place the regime
    shading/labels, never asserted or hard-coded. Nothing in the title states
    a fixed outcome; only the measured detection_na and slit_width are named.
    """
    nas = [na for na, _sheet, _ls, _aslm in results]
    sheets = [sheet for _na, sheet, _ls, _aslm in results]
    light_sheets = [ls for _na, _sheet, ls, _aslm in results]
    aslms = [aslm for _na, _sheet, _ls, aslm in results]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        nas, light_sheets, marker="o", linestyle="-",
        label="light-sheet system FWHM (at waist)",
    )
    ax.plot(nas, aslms, marker="s", linestyle="-", label="ASLM system FWHM")
    ax.plot(
        nas, sheets, linestyle="--", color="tab:green",
        label="sheet thickness (measured waist FWHM)",
    )
    ax.axhline(
        dof_um, color="gray", linestyle=":",
        label="DOF (measured, psf_mode='single')",
    )

    # Locate the crossover from the measured data -- the last NA where the
    # sheet is still thicker than the DOF, and the first where it is thinner.
    thick_nas = [na for na, sheet, _ls, _aslm in results if sheet > dof_um]
    thin_nas = [na for na, sheet, _ls, _aslm in results if sheet < dof_um]
    if thick_nas and thin_nas and nas:
        crossover = (max(thick_nas) + min(thin_nas)) / 2.0
        y_top = max(sheets + light_sheets + aslms + [dof_um])
        ax.axvline(crossover, color="black", linestyle="-.", alpha=0.4)
        ax.text(
            (min(nas) + crossover) / 2.0, y_top * 0.95,
            "sheet > DOF", ha="center", va="top",
        )
        ax.text(
            (crossover + max(nas)) / 2.0, y_top * 0.95,
            "sheet < DOF", ha="center", va="top",
        )

    ax.set_xlabel("illumination NA")
    ax.set_ylabel("axial FWHM (um)")
    ax.legend()
    ax.set_title(
        "System-PSF axial FWHM vs illumination NA\n"
        f"detection_na={DETECTION_NA:.2f}, slit_width={SLIT_WIDTH:.2f} um"
    )
    fig.tight_layout()
    return fig


def main() -> None:
    """Run the sweep, print the table, build the figure, and save it under examples/output/."""
    dof_um, results = run_sweep()
    print_table(dof_um, results)
    fig = build_regime_figure(dof_um, results)

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "aslm_axial_regimes.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
