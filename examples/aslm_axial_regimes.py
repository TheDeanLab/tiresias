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
"""Plot system-PSF axial FWHM against illumination NA to show the sheet-vs-DOF regimes (D-11).

Notes / limitations (printed to stdout by main(), kept out of the figure
itself for publication -- gap closure, plan 08.1-12):
    - Residual (deferred): illumination is a 3-D pencil beam, not a
      y-integrated light sheet; lateral (Y) widths are optimistic and
      off-waist light_sheet profiles carry pencil-beam Fresnel structure.
    - Sampling: cubic voxels (dz == dxy), ni0 == ni. The illumination NA
      0.70 sheet spans only about 4 voxels at 0.1 um (no z supersampling).
    - At the waist, ASLM matches the static light sheet; their difference
      is off-waist (see light_sheet_vs_aslm.png).
    - Crossover convention: the measured sheet == DOF crossover (about 0.34
      at this script's parameters) differs from research's paraxial
      0.19-0.25 estimate because that estimate uses the Gaussian-beam
      1/e^2 NA convention, whose waist is thinner at equal NA than the
      hard-aperture NA convention this script (and psfmodels) use.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    LogLocator,
    MultipleLocator,
    NullFormatter,
    NullLocator,
)

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
# between the two physical regimes this figure exists to show. Extended down
# to 0.05 (gap closure, plan 08.1-12) so the low-NA end visibly sits on the
# DOF plateau (light_sheet/DOF = 0.990 at NA 0.05, per the planner's
# re-measurement); the high end (0.70) shows the sheet-limited approach,
# where the system FWHM tracks the shrinking sheet instead.
ILLUMINATION_NA_SWEEP: tuple[float, ...] = (
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
)

# Cubic voxels (dz == dxy) are required by simulate/'s _require_cubic_voxels
# guard (halt finding 1, enforced structurally since plan 08.1-11) -- the
# legacy rot90 fast path relabels pre-rotation axes without resampling,
# which is exact only for cubic voxels. ni0 equals ni to avoid psfmodels'
# own default design immersion index (1.515, oil), which would otherwise
# introduce a spherical-aberration focal shift (halt finding 2). The odd
# 129-sample (12.9 um) transverse window holds the NA_ill 0.05 waist (sheet
# FWHM about 5.8 um) with margin, and its odd sample count puts the focal
# plane exactly on a sample rather than between two samples.
COMMON = {
    "wavelength": 0.561,
    "ni": 1.33,
    "ns": 1.33,
    "ni0": 1.33,
    "dxy": 0.1,
    "dz": 0.1,
    "psf_size_z": 129,
    "psf_size_xy": 129,
}

# D-19 disclosure (research Q1, deferred-items.md DEF-SEEDS-2): psfmodels'
# illumination arm is a 3-D pencil beam, not integrated over the transverse
# (Y) axis the way a true light sheet / DSLM / ASLM sweep physically is.
# Identical wording to examples/light_sheet_vs_aslm.py's PENCIL_BEAM_NOTE, so
# the same residual is disclosed consistently across both figures.
PENCIL_BEAM_NOTE = (
    "Residual (deferred): illumination is a 3-D pencil beam, not a "
    "y-integrated light sheet; lateral (Y) widths are optimistic and "
    "off-waist light_sheet profiles carry pencil-beam Fresnel structure."
)

# Research Q4 sampling disclosure: the NA_ill 0.70 sheet (about 0.43 um at
# this COMMON) is sampled by only about 4 voxels at dz = 0.1 um, with no z
# supersampling (deferred-items.md DEF-SEEDS-3).
SAMPLING_NOTE = (
    "Sampling: cubic voxels (dz == dxy), ni0 == ni. The NA_ill 0.70 sheet "
    "spans only about 4 voxels at 0.1 um (no z supersampling)."
)

# ASLM equals the static light sheet at the waist (plan 08.1-09 measured a
# 0.072 relative L2 difference); their difference is off-waist, which this
# figure does not plot -- see examples/light_sheet_vs_aslm.py instead.
WAIST_NOTE = (
    "At the waist ASLM matches the static light sheet; their difference is "
    "off-waist (see light_sheet_vs_aslm.png)."
)

# Gap closure (user-requested publication cleanup, plan 08.1-12): the
# NA-convention explanation for why the measured crossover differs from
# research's paraxial estimate, previously rendered into the suptitle. Kept
# as disclosed content (module docstring + stdout), just moved out of the
# figure itself so the figure reads as a clean, publication-style plot.
CROSSOVER_CONVENTION_NOTE = (
    "Crossover convention: the measured sheet == DOF crossover here differs "
    "from research's paraxial 0.19-0.25 estimate because that estimate uses "
    "the Gaussian-beam 1/e^2 NA convention, whose waist is thinner at equal "
    "NA than the hard-aperture NA convention this script (and psfmodels) use."
)

# Gap closure (user display-fix round 1, plan 08.1-12 tracer gate): matplotlib's
# default log-axis formatter renders ticks in scientific notation (e.g.
# "6x10^0"), which is unreadable at a glance. Every log axis in this figure
# uses an explicit locator plus this plain-number formatter instead.
_PLAIN_NUMBER_FORMATTER = FuncFormatter(lambda value, _pos: f"{value:g}")


def _use_plain_number_ticks(axis, locator) -> None:
    """Apply `locator` plus plain-decimal tick labels to one figure axis.

    Disables the default minor ticks/labels too, since matplotlib's log-axis
    minor ticks would otherwise still render in scientific notation between
    the major ticks this function places.
    """
    axis.set_major_locator(locator)
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(_PLAIN_NUMBER_FORMATTER)
    axis.set_minor_formatter(NullFormatter())


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


def locate_crossover_na(
    dof_um: float, results: list[tuple[float, float, float, float]]
) -> float | None:
    """Return the illumination NA where the measured sheet thickness equals dof_um.

    The crossover is measured, never assumed: this walks `results` (each
    `(na, sheet_um, light_sheet_um, aslm_um)`, in the order given) looking
    for the first adjacent pair where `sheet_um - dof_um` goes from positive
    to non-positive, then linearly interpolates the NA at which `sheet_um`
    would equal `dof_um` between that pair. Returns None when no such sign
    change occurs anywhere in the sweep (every sheet stays on the same side
    of dof_um).
    """
    # Why the measured crossover (~0.34 at this script's parameters) differs
    # from research's paraxial 0.19-0.25 estimate (08.1-DEEP-RESEARCH-
    # FINDINGS.md Q3): research's estimate uses the Gaussian-beam waist FWHM
    # under the 1/e^2 NA convention. psfmodels' illumination arm is a
    # uniformly filled circular pupil, whose waist FWHM is wider at equal NA
    # under the hard-aperture NA convention this script (and psfmodels) use.
    # That width ratio is about 1.37x at equal NA (measured: predicted
    # 2.86/0.954/0.477 um vs measured 2.888/0.968/0.495 um at NA 0.1/0.3/0.6),
    # so a paraxial crossover near 0.25 maps to about 0.25 * 1.37 = 0.34 under
    # this script's convention -- consistent with the measured value below.
    # D-12 forbids closed-form references in the figure itself and the
    # stripped-source scan forbids the token "sqrt" outside comments, so this
    # derivation stays here as a comment, never as executable code or in a
    # docstring.
    previous_na: float | None = None
    previous_diff: float | None = None
    for na, sheet_um, _light_sheet_um, _aslm_um in results:
        diff = sheet_um - dof_um
        if previous_diff is not None and previous_diff > 0 and diff <= 0:
            frac = previous_diff / (previous_diff - diff)
            return previous_na + frac * (na - previous_na)
        previous_na, previous_diff = na, diff
    return None


def print_table(dof_um: float, results: list[tuple[float, float, float, float]]) -> None:
    """Print one diagnostic row per swept NA, plus the measured crossover NA.

    The "sheet > DOF" / "sheet < DOF" label is a direct comparison of the
    measured `sheet_um` against the measured `dof_um` for that row -- never a
    fixed assumption about where the crossover falls. The "ls/DOF" and
    "ls/sheet" columns are the light-sheet system FWHM's ratio to each
    reference, the same normalisation the regime panel of
    `build_regime_figure` plots. The trailing crossover line comes from
    `locate_crossover_na`, never a hard-coded value.
    """
    print(
        f"illumination NA sweep -- detection_na={DETECTION_NA:.2f}, "
        f"slit_width={SLIT_WIDTH:.2f} um, DOF={dof_um:.4f} um"
    )
    print(
        f"{'na':>6}  {'sheet (um)':>12}  {'light_sheet (um)':>18}  "
        f"{'aslm (um)':>12}  {'ls/DOF':>8}  {'ls/sheet':>10}  regime"
    )
    for na, sheet_um, light_sheet_um, aslm_um in results:
        regime = "sheet > DOF" if sheet_um > dof_um else "sheet < DOF"
        ls_dof_ratio = light_sheet_um / dof_um
        ls_sheet_ratio = light_sheet_um / sheet_um
        print(
            f"{na:>6.2f}  {sheet_um:>12.4f}  {light_sheet_um:>18.4f}  "
            f"{aslm_um:>12.4f}  {ls_dof_ratio:>8.3f}  {ls_sheet_ratio:>10.3f}  {regime}"
        )
    crossover = locate_crossover_na(dof_um, results)
    if crossover is not None:
        print(f"measured crossover NA_ill: {crossover:.3f}")
    else:
        print("measured crossover NA_ill: none within the sweep")


def build_regime_figure(
    dof_um: float, results: list[tuple[float, float, float, float]]
) -> plt.Figure:
    """Plot the two-panel D-11 regime figure: absolute FWHM and a normalised regime plot.

    The left panel (`ax_abs`) plots axial FWHM vs illumination NA on a log x
    axis, with the measured DOF and sheet-thickness references and the
    measured crossover (`locate_crossover_na`) marked. The right panel
    (`ax_norm`) plots the normalised ratio `system / dof_um` against
    `sheet / dof_um` on log-log axes, so the "tracks DOF when sheet > DOF,
    tracks sheet when sheet < DOF" claim is readable from the two limiting
    asymptotes (`y = 1`, DOF-limited; `y = x`, sheet-limited) rather than
    from the marker positions alone. The crossover is located from the data
    itself and used only to annotate the figure, never asserted or
    hard-coded (D-12). Nothing in the suptitle states a fixed outcome; only
    the measured detection_na, slit_width and crossover are named.
    """
    nas = [na for na, _sheet, _ls, _aslm in results]
    sheets = [sheet for _na, sheet, _ls, _aslm in results]
    light_sheets = [ls for _na, _sheet, ls, _aslm in results]
    aslms = [aslm for _na, _sheet, _ls, aslm in results]

    crossover = locate_crossover_na(dof_um, results)

    fig, (ax_abs, ax_norm) = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)

    # --- Left panel: absolute axial FWHM vs illumination NA (log x). ---
    ax_abs.plot(
        nas, light_sheets, marker="o", linestyle="-",
        label="light-sheet system FWHM (at waist)",
    )
    ax_abs.plot(nas, aslms, marker="s", linestyle="-", label="ASLM system FWHM")
    ax_abs.plot(
        nas, sheets, linestyle="--", color="tab:green",
        label="sheet thickness (measured waist FWHM)",
    )
    ax_abs.axhline(
        dof_um, color="gray", linestyle=":",
        label="detection DOF (measured)",
    )
    ax_abs.set_xscale("log")
    ax_abs.set_yscale("log")

    # Gap closure (user display-fix round 1): a linear y axis let the
    # sheet-thickness curve at NA 0.05 (about 5.8 um) stretch the axis so far
    # that the system-FWHM points (0.4-0.9 um) collapsed to a near-flat line
    # at the bottom -- the same complaint that rejected the 08.1-07 figure.
    # The log y axis, with limits derived from the plotted data (never
    # hard-coded), keeps both the DOF plateau and the sheet-limited approach
    # visible on the same panel.
    y_values_abs = sheets + light_sheets + aslms + [dof_um]
    y_lo_abs = min(y_values_abs) / 1.15
    y_hi_abs = max(y_values_abs) * 1.15
    ax_abs.set_ylim(y_lo_abs, y_hi_abs)

    _use_plain_number_ticks(
        ax_abs.xaxis, FixedLocator((0.05, 0.1, 0.2, 0.3, 0.5, 0.7))
    )
    _use_plain_number_ticks(
        ax_abs.yaxis, LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0))
    )

    if crossover is not None:
        # Labels are placed 90% of the way up the log-scaled y range (in log
        # space, not linear), so they stay inside the axes and readable
        # regardless of how wide the data-derived y_lo_abs/y_hi_abs span is.
        label_y = 10.0 ** (0.1 * np.log10(y_lo_abs) + 0.9 * np.log10(y_hi_abs))
        ax_abs.axvline(
            crossover, color="black", linestyle="-.",
            label=f"measured crossover (illumination NA = {crossover:.2f})",
        )
        if min(nas) < crossover:
            ax_abs.text(
                (min(nas) * crossover) ** 0.5, label_y,
                "sheet > DOF", ha="center", va="top",
            )
        if crossover < max(nas):
            ax_abs.text(
                (crossover * max(nas)) ** 0.5, label_y,
                "sheet < DOF", ha="center", va="top",
            )

    ax_abs.set_xlabel("illumination NA")
    ax_abs.set_ylabel("axial FWHM (um)")
    ax_abs.legend()

    # --- Right panel: normalised regime plot, system/DOF vs sheet/DOF. ---
    sheet_ratios = [sheet / dof_um for sheet in sheets]
    light_sheet_ratios = [ls / dof_um for ls in light_sheets]
    aslm_ratios = [aslm / dof_um for aslm in aslms]

    ax_norm.plot(
        sheet_ratios, light_sheet_ratios, marker="o", linestyle="-",
        label="light-sheet system FWHM (at waist)",
    )
    ax_norm.plot(
        sheet_ratios, aslm_ratios, marker="s", linestyle="-",
        label="ASLM system FWHM",
    )
    ax_norm.set_xscale("log")
    ax_norm.set_yscale("log")

    x_min = min(sheet_ratios)
    x_max = max(sheet_ratios)
    ax_norm.axhline(1.0, color="gray", linestyle=":", label="DOF-limited (system = DOF)")
    ax_norm.plot(
        [x_min, x_max], [x_min, x_max], color="tab:red", linestyle="--",
        label="sheet-limited (system = sheet)",
    )
    ax_norm.axvline(1.0, color="black", linestyle="-.", label="sheet = DOF")

    # Gap closure (user display-fix round 2): the sheet-limited diagonal
    # spans the whole x range, so autoscaling on it alone pushed the y axis
    # up to about 6.6 -- far past where any system-FWHM ratio point actually
    # sits. Clip y to the plotted ratio data (plus the y = 1 DOF-limited
    # reference) with a small margin, derived from the data rather than
    # hard-coded, and let the diagonal run off the top instead of dictating
    # the axis.
    y_values_norm = light_sheet_ratios + aslm_ratios + [1.0]
    y_lo_norm = min(y_values_norm) / 1.15
    y_hi_norm = max(y_values_norm) * 1.15
    ax_norm.set_ylim(y_lo_norm, y_hi_norm)

    _use_plain_number_ticks(
        ax_norm.xaxis, LogLocator(base=10.0, subs=(1.0, 2.0, 5.0))
    )
    _use_plain_number_ticks(ax_norm.yaxis, MultipleLocator(0.2))

    ax_norm.set_xlabel("sheet thickness / DOF")
    ax_norm.set_ylabel("system axial FWHM / DOF")
    ax_norm.legend()

    # Gap closure (user-requested publication cleanup, plan 08.1-12): the
    # suptitle is now a clean, publication-style title plus one parameter
    # line. The measured crossover value is still shown, through the
    # ax_abs axvline legend entry above -- it is never dropped, only moved
    # out of the suptitle's prose. The pencil-beam, sampling, waist and
    # crossover-convention disclosures move to the module docstring and are
    # printed to stdout by main(), per the user's request to keep the
    # figure itself free of the citation/blurb while not losing the
    # disclosures themselves.
    fig.suptitle(
        "System-PSF axial FWHM vs illumination NA\n"
        f"detection NA = {DETECTION_NA:.2f}, slit width = {SLIT_WIDTH:.2f} um",
        fontsize=11,
    )
    return fig


def main() -> None:
    """Run the sweep, print the table and disclosures, build the figure, and save it under examples/output/."""
    dof_um, results = run_sweep()
    print_table(dof_um, results)
    # Gap closure (user-requested publication cleanup, plan 08.1-12): these
    # disclosures used to render into the figure's suptitle; they are now
    # printed to stdout (and kept in the module docstring's Notes /
    # limitations section) instead, so the figure itself stays clean.
    print(PENCIL_BEAM_NOTE)
    print(SAMPLING_NOTE)
    print(WAIST_NOTE)
    print(CROSSOVER_CONVENTION_NOTE)
    fig = build_regime_figure(dof_um, results)

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "aslm_axial_regimes.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
