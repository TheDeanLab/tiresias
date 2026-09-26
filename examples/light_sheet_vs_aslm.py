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

# D-07: optical parameters originate from docs/usage.md's Python API examples,
# for baseline consistency with the existing documentation -- except the two
# aperture entries below, which are deliberately raised for this demo and no
# longer track the docs.
COMMON = {
    "na": 1.0,
    "detection_na": 1.1,
    "illumination_na": 0.6,
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
    "polar_deg": 90.0,
    "azimuthal_deg": 0.0,
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
        cupy_present = False  # probe failed; treat as "not confirmed present"
    print(f"cupy present in this environment: {'yes' if cupy_present else 'no'}")


def summarise_mode(label: str, seed: np.ndarray, gate: str) -> None:
    """Print one shape/energy/gate summary line for a generated PSF seed."""
    energy = float(seed.sum())
    print(f"{label} shape={seed.shape} energy={energy:.6f} gate={gate}")


def axial_profile(seed: np.ndarray) -> np.ndarray:
    """Return the Z intensity profile of a (Z, Y, X) seed through its true peak.

    Locates the global peak with `np.unravel_index(np.argmax(seed), seed.shape)`
    rather than `shape // 2` -- for the D-07 parameters the true peak sits at
    (Y, X) = (63, 64), not the geometric centre (64, 64), and the peak Z index
    itself migrates with `slit_width` (see RESEARCH.md Common Pitfalls #3).
    """
    peak_z, peak_y, peak_x = np.unravel_index(np.argmax(seed), seed.shape)
    return seed[:, peak_y, peak_x].astype(np.float64)


def lateral_fwhm(psf: np.ndarray, dxy: float) -> float:
    """Return the lateral (X) half-max width of a (Z, Y, X) PSF seed through its true peak.

    Locates the global peak the same way `axial_profile` does -- via
    `np.unravel_index(np.argmax(...))`, never a per-slice search, since a
    per-slice argmax drifts onto sidelobes far from focus. Walks outward
    along X from that peak row and linearly interpolates the half-maximum
    crossing on each side, using the identical first-crossing interpolation
    as `slit_width_sweep.py`'s `axial_fwhm` (duplicated here per the
    single-file PEP 723 philosophy -- no shared helper module between the
    two scripts). Raises ValueError, rather than returning a sentinel, when
    the peak is non-positive or a crossing is not found within the volume --
    `main()` needs a definite float to build the D-14 core radius from.
    """
    peak_z, peak_y, peak_x = np.unravel_index(np.argmax(psf), psf.shape)
    profile = psf[peak_z, peak_y, :].astype(np.float64)
    peak_value = profile[peak_x]
    if peak_value <= 0:
        raise ValueError("lateral_fwhm: peak value is not positive, cannot measure half-max width")
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

    left_indices = np.arange(peak_x, -1, -1)  # peak -> start, descending
    right_indices = np.arange(peak_x, profile.size)  # peak -> end, ascending

    left_crossing = _crossing(left_indices)
    right_crossing = _crossing(right_indices)
    if left_crossing is None or right_crossing is None:
        raise ValueError("lateral_fwhm: half-max crossing not found within the volume")
    return (right_crossing - left_crossing) * dxy


def outside_core_fraction(seed: np.ndarray, r_core_um: float, dxy: float) -> np.ndarray:
    """Per Z plane, the lateral energy fraction outside a fixed detection-core disc.

    The core is a disc of radius `r_core_um` centred on the seed's GLOBAL
    peak lateral position (`np.unravel_index(np.argmax(seed), seed.shape)`)
    -- fixed for every plane, never re-located per slice, matching
    `axial_profile`'s fixed-peak convention. Mirrors the D-14 committed
    test's metric (`tests/test_seeds.py::
    test_aslm_removes_out_of_focus_energy_at_every_z_beyond_dof`):
    `1 - E_core / E_plane` per Z. A plane carrying zero total energy maps to
    0.0 (there is no out-of-focus energy to report there), rather than the
    test's own convention of 1.0 -- this panel's purpose is showing where
    energy is lost, not flagging degenerate planes.
    """
    _, peak_y, peak_x = np.unravel_index(np.argmax(seed), seed.shape)
    yy, xx = np.meshgrid(
        np.arange(seed.shape[1]), np.arange(seed.shape[2]), indexing="ij"
    )
    core_mask = np.hypot((yy - peak_y) * dxy, (xx - peak_x) * dxy) <= r_core_um
    e_plane = seed.sum(axis=(1, 2), dtype=np.float64)
    e_core = (seed * core_mask[None, :, :]).sum(axis=(1, 2), dtype=np.float64)
    fraction_out = np.zeros_like(e_plane)
    positive = e_plane > 0
    fraction_out[positive] = 1.0 - (e_core[positive] / e_plane[positive])
    return fraction_out


def build_comparison_figure(
    seed_light_sheet: np.ndarray,
    seed_aslm: np.ndarray,
    common: dict,
    slit_width: float,
    *,
    r_core_um: float,
) -> plt.Figure:
    """Assemble the D-04/D-14 comparison figure: MIP panels, axial profile, and outside-core panel.

    Every panel is plotted in physical micrometres (derived from `dz`/`dxy`),
    centered on zero, with a shared per-row intensity scale so the two modes
    are directly, honestly comparable. `r_core_um` sets the fixed detection-
    core radius for the bottom D-14 outside-core-fraction panel (see
    `outside_core_fraction`).
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
    # indistinguishable for these locked D-07 parameters. D-19: the
    # sweep-integrated slit removes low-intensity off-waist energy across
    # the whole PSF (D-14), not only its axial tail, and that removal lives
    # mostly in structure ~3 orders of magnitude below peak. Log scale,
    # shared per row across both modes, makes it visible.
    xz_norm = LogNorm(vmin=xz_vmax * 1e-3, vmax=xz_vmax)
    yz_norm = LogNorm(vmin=yz_vmax * 1e-3, vmax=yz_vmax)

    # constrained_layout (not tight_layout): tight_layout does not account for
    # the per-row colorbars added below and produces overlapping panels/text.
    # 4-row gridspec: rows 0-1 are the 2x2 MIP grid, row 2 spans both columns
    # for the axial intensity-profile overlay, row 3 spans both columns for
    # the D-14 outside-core-fraction panel added below.
    fig = plt.figure(figsize=(11, 17), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 0.8, 0.8])
    axes = np.array(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ]
    )

    row_specs = (
        (axes[0, 0], axes[0, 1], xz_light_sheet, xz_aslm, xz_extent, xz_norm, "XZ", "X"),
        (axes[1, 0], axes[1, 1], yz_light_sheet, yz_aslm, yz_extent, yz_norm, "YZ", "Y"),
    )

    for ax_ls, ax_al, panel_ls, panel_al, extent, norm, proj_label, lateral_label in row_specs:
        # aspect="equal" (not "auto"): dz/dxy ~= 2.78, so a square-aspect,
        # index-based panel would visually stretch the axial direction by
        # nearly 3x relative to the stated micrometre units -- "equal"
        # matches the displayed geometry to the physical extent.
        ax_ls.imshow(panel_ls, aspect="equal", norm=norm, extent=extent)
        ax_ls.set_title(f"light_sheet {proj_label} MIP")
        im_al = ax_al.imshow(panel_al, aspect="equal", norm=norm, extent=extent)
        # D-19: the slit integrates along the pre-rotation beam-propagation
        # axis (D-02) for every direction -- there is no fixed camera-axis
        # band to draw here (the retired D-15/D-19 Z-band footprint gate is
        # gone). The slit note lives in the title (not an image overlay):
        # aspect="equal" shrinks the rendered image box within the subplot
        # to keep physical units equal, by an amount that differs slightly
        # per row, so an in-image text anchor can land outside the
        # actually-rendered (and clipped) box in one row but not the other.
        # A title is never clipped to that box.
        ax_al.set_title(
            f"aslm {proj_label} MIP\n"
            f"slit_width={slit_width:.2f} um, integrated along propagation "
            "(X at this direction)",
            fontsize=9,
        )

        for ax in (ax_ls, ax_al):
            ax.set_xlabel(f"{lateral_label} (um)")
            ax.set_ylabel("Z (um)")

        cbar = fig.colorbar(im_al, ax=[ax_ls, ax_al], shrink=0.85, pad=0.02)
        cbar.set_label("normalised intensity (fraction of total energy)")

    # Axial intensity-profile overlay: makes the gate's effect readable
    # without the reader computing anything (D-04 <specifics>). Z coordinate
    # centered on zero via pixel centers, the same "centered on zero"
    # convention the projection panels above use for their extent.
    profile_light_sheet = axial_profile(seed_light_sheet)
    profile_aslm = axial_profile(seed_aslm)
    z_um = -z_extent_um / 2 + (np.arange(psf_size_z) + 0.5) * dz

    ax_profile = fig.add_subplot(gs[2, :])
    # Each curve normalised to its own peak -- the plot compares *shape*, not
    # the energy redistribution the gate causes, so say so in the axis label
    # rather than presenting these as raw normalised intensities.
    ax_profile.plot(
        z_um,
        profile_light_sheet / profile_light_sheet.max(),
        marker="o",
        markersize=3,
        label="light_sheet",
    )
    ax_profile.plot(
        z_um,
        profile_aslm / profile_aslm.max(),
        marker="o",
        markersize=3,
        label=f"aslm (slit_width={slit_width:.2f} um)",
    )
    ax_profile.axhline(0.5, color="gray", linestyle="--", linewidth=1.0)
    ax_profile.set_xlabel("Z (um)")
    ax_profile.set_ylabel("intensity, each curve peak-normalised to 1.0")
    ax_profile.set_title("Axial (Z) intensity profile through each seed's true peak")
    # Zoom to where the curves actually carry visible structure -- the full
    # +-9.15 um volume extent (matching the projection panels above) squeezes
    # the half-max crossing separation the reader needs to see into a few
    # pixels of screen space. Bound derived from where either curve clears 2%
    # of its own peak, plus a fixed margin, rather than a hardcoded window.
    visible = (
        (profile_light_sheet / profile_light_sheet.max() >= 0.02)
        | (profile_aslm / profile_aslm.max() >= 0.02)
    )
    visible_z = z_um[visible]
    margin = 2.0 * dz
    ax_profile.set_xlim(visible_z.min() - margin, visible_z.max() + margin)
    ax_profile.legend()

    # D-14 outside-core-fraction panel: makes the "energy removed at every Z"
    # claim decidable by eye, since at these locked parameters the MIP panels
    # above differ only subtly (planner measurement: margins -0.0006 to
    # -0.011 across 36 tested planes). Same z_um centring as the axial
    # profile above.
    fraction_out_light_sheet = outside_core_fraction(seed_light_sheet, r_core_um, dxy)
    fraction_out_aslm = outside_core_fraction(seed_aslm, r_core_um, dxy)

    ax_fraction = fig.add_subplot(gs[3, :])
    ax_fraction.plot(z_um, fraction_out_light_sheet, marker="o", markersize=3, label="light_sheet")
    ax_fraction.plot(
        z_um,
        fraction_out_aslm,
        marker="o",
        markersize=3,
        label=f"aslm (slit_width={slit_width:.2f} um)",
    )
    ax_fraction.set_xlabel("Z (um)")
    ax_fraction.set_ylabel("fraction of plane energy outside core")
    ax_fraction.set_title(
        f"Lateral energy fraction outside the detection core (r={r_core_um:.4f} um) vs Z"
    )
    ax_fraction.set_xlim(visible_z.min() - margin, visible_z.max() + margin)
    ax_fraction.legend()

    fig.suptitle(
        f"light_sheet vs aslm PSF seed comparison (aslm slit_width={slit_width:.2f} um)"
    )
    return fig


def main() -> None:
    """Generate light_sheet and aslm PSF seeds, print summaries, and save a comparison figure."""
    describe_environment()

    seed_light_sheet = generate_psf_seed(psf_mode="light_sheet", **COMMON)
    seed_aslm = generate_psf_seed(psf_mode="aslm", slit_width=SLIT_WIDTH, **COMMON)
    # D-14: seed_single supplies the fixed core radius (via lateral_fwhm)
    # that the committed D-14 test also uses -- the core is defined from the
    # unslit detection-only PSF, not from either compared mode.
    seed_single = generate_psf_seed(psf_mode="single", **COMMON)
    dxy = COMMON["dxy"]
    r_core_um = lateral_fwhm(seed_single, dxy)

    # D-19: the slit integrates along the pre-rotation propagation axis
    # (D-02) for every direction -- there is no per-axis gate resolution
    # anymore, and no full-extent equivalence shortcut (D-15).
    propagation_window_um = COMMON["psf_size_z"] * COMMON["dz"]
    summarise_mode("light_sheet", seed_light_sheet, "none")
    summarise_mode(
        "aslm",
        seed_aslm,
        f"slit_width={SLIT_WIDTH}um sweep-integrated along propagation, "
        f"window={propagation_window_um:.3f}um",
    )

    fig = build_comparison_figure(
        seed_light_sheet, seed_aslm, COMMON, SLIT_WIDTH, r_core_um=r_core_um
    )

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "light_sheet_vs_aslm.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
