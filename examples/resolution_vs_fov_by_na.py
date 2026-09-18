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
"""Pair the shipped light-sheet and ASLM resolution-vs-FOV curves, one page per illumination NA."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# D-06: paths are derived from this script's own resolved __file__, never
# from sys.argv, cwd or an environment variable (T-LRU-01) -- load_example_
# script never receives a caller-supplied path in production use, so there
# is no route by which an arbitrary file gets executed. This also means the
# script never needs to touch sys.path itself: each loaded script already
# inserts the repo root using its own resolved __file__ before importing
# simulate, and this script never imports simulate directly.
LIGHT_SHEET_SCRIPT = Path(__file__).resolve().parent / "light_sheet_resolution_vs_fov.py"
ASLM_SCRIPT = Path(__file__).resolve().parent / "aslm_resolution_vs_fov.py"

OUTPUT_NAME = "resolution_vs_fov_by_na.pdf"

# D-03: axis labels copied verbatim from both shipped figures so the paired
# page reads identically to the two per-script PNGs.
X_LABEL = "position along beam propagation axis (um)"
Y_LABEL = "illumination-limited axial resolution, transverse FWHM (um)"

LIGHT_SHEET_LABEL = "static light sheet"
ASLM_LABEL_TEMPLATE = "ASLM (slit_width={:.2f} um)"


def load_example_script(path: Path) -> ModuleType:
    """Load a shipped example script as a module without triggering its __main__ behavior.

    Safe: both shipped scripts guard `main()` behind `if __name__ ==
    "__main__":`, so exec_module only defines module-level names. This is
    also the mechanism D-06 chose over a shared helper module, which the
    project's single-file PEP 723 philosophy forbids.
    """
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair_sweeps(
    light_sheet: ModuleType, aslm: ModuleType
) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Pair each script's own run_sweep() output by NA, never re-deriving the sweep (D-05).

    Returns one `(na, ls_centered, ls_widths, aslm_centered, aslm_widths)`
    entry per value of `light_sheet.NA_SWEEP`, in that order.
    """
    if light_sheet.NA_SWEEP != aslm.NA_SWEEP:
        raise ValueError(
            "NA_SWEEP disagrees between the two shipped scripts: "
            f"light_sheet={light_sheet.NA_SWEEP!r} aslm={aslm.NA_SWEEP!r}"
        )
    if light_sheet.COMMON != aslm.COMMON:
        raise ValueError(
            "COMMON disagrees between the two shipped scripts: "
            f"light_sheet={light_sheet.COMMON!r} aslm={aslm.COMMON!r}"
        )

    # D-05: reuse each script's own run_sweep() at its committed defaults --
    # this is the whole numerical-identity guarantee. Indexed positionally
    # because the light-sheet list yields quadruples (na, centered, widths,
    # rayleigh) and the ASLM list yields triples (na, centered, widths).
    light_sheet_results = light_sheet.run_sweep()
    aslm_results = aslm.run_sweep()

    light_sheet_by_na = {result[0]: result for result in light_sheet_results}
    aslm_by_na = {result[0]: result for result in aslm_results}

    missing_from_aslm = set(light_sheet_by_na) - set(aslm_by_na)
    missing_from_light_sheet = set(aslm_by_na) - set(light_sheet_by_na)
    if missing_from_aslm or missing_from_light_sheet:
        raise ValueError(
            "NA values are not paired between the two sweeps: "
            f"missing_from_aslm={sorted(missing_from_aslm)!r} "
            f"missing_from_light_sheet={sorted(missing_from_light_sheet)!r}"
        )

    paired: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for na in light_sheet.NA_SWEEP:
        ls_result = light_sheet_by_na[na]
        aslm_result = aslm_by_na[na]
        paired.append((na, ls_result[1], ls_result[2], aslm_result[1], aslm_result[2]))
    return paired


def build_na_page(
    na: float,
    ls_centered: np.ndarray,
    ls_widths: np.ndarray,
    aslm_centered: np.ndarray,
    aslm_widths: np.ndarray,
    *,
    half_extent: float,
    detection_na: float,
    slit_width: float,
) -> plt.Figure:
    """Build one page pairing the light-sheet and ASLM curves for a single illumination NA (D-01).

    Both curves are always plotted, so the legend always carries exactly two
    entries, even when one curve holds no finite point. Widths are plotted
    exactly as measured, NaN entries included: matplotlib renders those as
    honest breaks in the line, and those gaps are never closed.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ls_centered, ls_widths, label=LIGHT_SHEET_LABEL)
    ax.plot(aslm_centered, aslm_widths, label=ASLM_LABEL_TEMPLATE.format(slit_width))

    no_data_notes = []
    if not np.isfinite(ls_widths).any():
        no_data_notes.append(LIGHT_SHEET_LABEL)
    if not np.isfinite(aslm_widths).any():
        no_data_notes.append("ASLM")
    if no_data_notes:
        ax.text(
            0.5,
            0.5,
            f"no measurable width for: {', '.join(no_data_notes)}",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_xlim(-half_extent, half_extent)
    ax.legend()
    # Deliberately left autoscaled per page (not shared across pages): both
    # curves on a page share one y scale, which is what the pairing needs;
    # forcing one scale across all five pages would flatten the low-NA pages
    # against the high-NA range.
    ax.set_title(
        f"Resolution vs. FOV at illumination_na={na:.2f}\n"
        f"detection_na={detection_na:.2f}, slit_width={slit_width:.2f} um, "
        f"window=+-{half_extent:.1f} um"
    )
    fig.tight_layout()
    return fig


def write_by_na_pdf(figures: list[plt.Figure], output_path: Path) -> int:
    """Write each figure as one PDF page (D-07) and return the resulting page count."""
    with PdfPages(output_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)
        return pdf.get_pagecount()


def main() -> None:
    """Load both shipped scripts, pair their sweeps, and write the combined by-NA PDF."""
    light_sheet = load_example_script(LIGHT_SHEET_SCRIPT)
    aslm = load_example_script(ASLM_SCRIPT)

    paired = pair_sweeps(light_sheet, aslm)
    half_extent = (light_sheet.COMMON["psf_size_z"] - 1) * light_sheet.COMMON["dz"] / 2.0

    for na, ls_centered, ls_widths, aslm_centered, aslm_widths in paired:
        ls_finite = int(np.isfinite(ls_widths).sum())
        aslm_finite = int(np.isfinite(aslm_widths).sum())
        print(
            f"illumination_na={na:.2f}: light_sheet finite={ls_finite}/{ls_widths.size}, "
            f"aslm finite={aslm_finite}/{aslm_widths.size}"
        )

    figures = [
        build_na_page(
            na,
            ls_centered,
            ls_widths,
            aslm_centered,
            aslm_widths,
            half_extent=half_extent,
            detection_na=light_sheet.DETECTION_NA,
            slit_width=aslm.SLIT_WIDTH,
        )
        for na, ls_centered, ls_widths, aslm_centered, aslm_widths in paired
    ]

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAME
    page_count = write_by_na_pdf(figures, output_path)
    print(f"wrote {output_path} ({page_count} pages)")


if __name__ == "__main__":
    main()
