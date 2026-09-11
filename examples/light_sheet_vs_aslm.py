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
# Remediation rung 2 (plan 04-02 Task 1): [[tool.uv.dependency-metadata]] is
# NOT honored inside a script's own PEP 723 block on this uv version (0.12.13)
# -- a real run still resolved and downloaded cupy-cuda11x. This
# override-dependencies entry restates cupy-cuda11x behind an environment
# marker that can never be satisfied, removing it from resolution while
# keeping D-01's [tool.uv.sources] mechanism and the zero-extra-flag
# invocation intact.
"""Contrast static light_sheet and ASLM PSF seed generation via generate_psf_seed()."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import tiresias
from tiresias import generate_psf_seed

# D-07: realistic optical parameters reused verbatim from docs/usage.md's
# Python API examples, for consistency with the existing documentation.
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
# aslm demo slit width, matching docs/usage.md's own ASLM example.
SLIT_WIDTH = 2.0


def describe_environment() -> None:
    """Print the resolved tiresias source location and cupy availability."""
    print(f"tiresias: {tiresias.__file__}")
    try:
        # Presence check only -- never import cupy itself.
        cupy_present = importlib.util.find_spec("cupy") is not None
    except Exception:
        cupy_present = True
    print(f"cupy present in this environment: {'yes' if cupy_present else 'no'}")


def summarise_mode(label: str, seed: np.ndarray, gate: str) -> None:
    """Print one shape/energy/gate summary line for a generated PSF seed."""
    energy = float(seed.sum())
    print(f"{label} shape={seed.shape} energy={energy:.6f} gate={gate}")


def build_comparison_figure(
    seed_light_sheet: np.ndarray,
    seed_aslm: np.ndarray,
    common: dict,
    slit_width: float,
) -> plt.Figure:
    """Assemble the D-04 comparison figure: XZ and YZ MIP panels for both modes.

    Every panel is plotted in physical micrometres (derived from `dz`/`dxy`),
    centered on zero, with a shared per-row intensity scale so the two modes
    are directly, honestly comparable.
    """
    dz = common["dz"]
    dxy = common["dxy"]
    psf_size_z = common["psf_size_z"]
    psf_size_xy = common["psf_size_xy"]
    z_extent_um = psf_size_z * dz
    lateral_extent_um = psf_size_xy * dxy

    # XZ MIP: max over Y (axis=1) -> (Z, X). YZ MIP: max over X (axis=2) -> (Z, Y).
    xz_light_sheet = seed_light_sheet.max(axis=1)
    xz_aslm = seed_aslm.max(axis=1)
    yz_light_sheet = seed_light_sheet.max(axis=2)
    yz_aslm = seed_aslm.max(axis=2)

    # Extent centered on zero (left, right, bottom, top) with origin='upper':
    # index 0 (top row) -> -z_extent/2, last row -> +z_extent/2. Both modes
    # and both projections share this same lateral/axial physical framing so
    # the gate axis midpoint annotated below lands exactly at Z=0.
    xz_extent = (
        -lateral_extent_um / 2,
        lateral_extent_um / 2,
        z_extent_um / 2,
        -z_extent_um / 2,
    )
    yz_extent = xz_extent  # same physical Z/lateral spans for both projections

    xz_vmax = float(max(xz_light_sheet.max(), xz_aslm.max()))
    yz_vmax = float(max(yz_light_sheet.max(), yz_aslm.max()))
    # Per 04-02: a linear scale makes light_sheet and aslm visually
    # indistinguishable for these locked D-07 parameters -- the gate's effect
    # lives in low-intensity off-waist structure ~3 orders of magnitude below
    # peak. Log scale, shared per row across both modes, makes it visible.
    xz_norm = LogNorm(vmin=xz_vmax * 1e-3, vmax=xz_vmax)
    yz_norm = LogNorm(vmin=yz_vmax * 1e-3, vmax=yz_vmax)

    # constrained_layout (not tight_layout): tight_layout does not account for
    # the per-row colorbars added below and produces overlapping panels/text.
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)

    row_specs = (
        (axes[0, 0], axes[0, 1], xz_light_sheet, xz_aslm, xz_extent, xz_norm, "XZ", "X"),
        (axes[1, 0], axes[1, 1], yz_light_sheet, yz_aslm, yz_extent, yz_norm, "YZ", "Y"),
    )

    half_width = slit_width / 2.0
    for ax_ls, ax_al, panel_ls, panel_al, extent, norm, proj_label, lateral_label in row_specs:
        # aspect="equal" (not "auto"): dz/dxy ~= 2.78, so a square-aspect,
        # index-based panel would visually stretch the axial direction by
        # nearly 3x relative to the stated micrometre units -- "equal"
        # matches the displayed geometry to the physical extent.
        ax_ls.imshow(panel_ls, aspect="equal", norm=norm, extent=extent)
        ax_ls.set_title(f"light_sheet {proj_label} MIP")
        im_al = ax_al.imshow(panel_al, aspect="equal", norm=norm, extent=extent)
        ax_al.set_title(f"aslm {proj_label} MIP")

        for ax in (ax_ls, ax_al):
            ax.set_xlabel(f"{lateral_label} (um)")
            ax.set_ylabel("Z (um)")

        cbar = fig.colorbar(im_al, ax=[ax_ls, ax_al], shrink=0.85, pad=0.02)
        cbar.set_label("normalised intensity (fraction of total energy)")

        # Slit-gate footprint annotation, aslm panel only. The gate is
        # applied to axis 2 (X) in the illumination's *pre-rotation* frame;
        # with light_sheet_angle=90.0 the illumination is then rotated into
        # the Z/X plane, so in the saved seed the narrowing appears along Z
        # -- the vertical axis shared by both the XZ and YZ panels here.
        for z_val in (-half_width, half_width):
            ax_al.axhline(z_val, color="white", linestyle="--", linewidth=1.0)
        ax_al.text(
            extent[0] * 0.9,
            -half_width,
            f"slit_width={slit_width:.2f} um",
            color="white",
            fontsize=8,
            ha="left",
            va="bottom",
        )

    fig.suptitle(
        f"light_sheet vs aslm PSF seed comparison (aslm slit_width={slit_width:.2f} um)"
    )
    return fig


def main() -> None:
    """Generate light_sheet and aslm PSF seeds, print summaries, and save a comparison figure."""
    describe_environment()

    seed_light_sheet = generate_psf_seed(psf_mode="light_sheet", **COMMON)
    seed_aslm = generate_psf_seed(psf_mode="aslm", slit_width=SLIT_WIDTH, **COMMON)

    # light_sheet_angle=90.0 (D-07) resolves gate axis 2 (X); see
    # seeds.py::_resolve_slit_axis.
    gate_axis = 2
    full_extent = COMMON["psf_size_xy"] * COMMON["dxy"]
    summarise_mode("light_sheet", seed_light_sheet, "none")
    summarise_mode(
        "aslm",
        seed_aslm,
        f"slit_width={SLIT_WIDTH}um axis={gate_axis}(X) extent={full_extent:.3f}um",
    )

    fig = build_comparison_figure(seed_light_sheet, seed_aslm, COMMON, SLIT_WIDTH)

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "light_sheet_vs_aslm.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
