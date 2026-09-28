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
"""Contrast static light_sheet and ASLM PSF seed generation via generate_psf_seed().

Notes / limitations (printed to stdout by main(), kept out of the figure
itself for publication -- gap closure, plan 08.1-12):
    - Residual (deferred): illumination is a 3-D pencil beam, not a
      y-integrated light sheet; lateral (Y) widths are optimistic and
      off-waist light_sheet profiles carry pencil-beam Fresnel structure.
    - Sampling: cubic voxels (dz == dxy), ni0 == ni. slit_width is the FWHM
      of a Gaussian window (D-03); Dean et al. 2015 used a rectangular
      slit W = 2*xR, and a Gaussian with the same second moment has FWHM
      ~0.68*W.
"""

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
#
# Plan 08.1-09 (halt route steps 1-2, research Q5) corrected three values that
# produced an inaccurate figure:
#   - "dz" == "dxy": the legacy cardinal-direction rotation path
#     (`_legacy_rot90_rotation`) relabels pre-rotation axes onto the display
#     axes with no resampling. That relabel is exact only for cubic voxels
#     (halt finding 1, ROT-04) -- the seeds.py fix for anisotropic sampling
#     is deferred (deferred-items.md DEF-SEEDS-1).
#   - "psf_size_z" == "psf_size_xy" (a cube window): after the relabel,
#     pre-rotation axis 0 (the "psf_size_z" samples) lands on the display X
#     axis. An equal size fills X edge-to-edge with no zero-padded white
#     bands (the artifact the user flagged in the previous figure). The odd
#     size (129) puts the focal plane exactly on a sample, rather than
#     between two samples at a 63.5-style half-integer centre -- the planner
#     probe showed the argmax landing off-centre at an even size. The 13.9 um
#     window (129 * 0.108 um) spans many Rayleigh ranges at illumination_na
#     0.6 and still comfortably holds the OFF_WAIST_UM offset below.
#   - "ni0" == "ni" (1.33): "ni0=None" falls back to psfmodels' own default
#     design immersion index (1.515, oil), producing a spherical-aberration
#     focal shift in these water-immersion demos (halt finding 2).
COMMON = {
    "na": 1.0,
    "detection_na": 1.1,
    "illumination_na": 0.6,
    "wavelength": 0.561,
    "ni": 1.33,
    "ns": 1.33,
    "ni0": 1.33,
    "tg": None,
    "tg0": None,
    "ng": None,
    "ng0": None,
    "ti0": None,
    "oversample_factor": 3,
    "psf_model": "vectorial",
    "dxy": 0.108,
    "dz": 0.108,
    "psf_size_z": 129,
    "psf_size_xy": 129,
    "background": 0.0,
    "polar_deg": 90.0,
    "azimuthal_deg": 0.0,
}
# aslm demo slit width, matching docs/usage.md's own ASLM example.
SLIT_WIDTH = 2.0

# The emitter's offset from the illumination waist, along the propagation
# direction, for the off-waist light_sheet column added by plan 08.1-09
# (halt route step 3). Roughly ten paraxial Rayleigh ranges at
# illumination_na 0.6 (research Q3 table) -- a sanity scale only, not a
# derivation. Fixed by this plan and never re-tuned after viewing the figure
# (T-08.1-18); it stays well inside the COMMON window's +-6.97 um half-extent.
OFF_WAIST_UM = 5.0

# Display-only zoom for every MIP panel below (halt-gate fix requested after
# the tracer checkpoint): set_xlim/set_ylim crops each panel's *view* to this
# physical +-half-width, in micrometres -- the simulated window itself stays
# the full COMMON psf_size_z/psf_size_xy cube, nothing here changes what is
# simulated. At the per-panel LogNorm floor this fix also introduces (1e-4 of
# each panel's own peak), the raw +-6.97 um simulated extent buries the
# in-focus core and the off-waist spread the user asked to see inside a
# handful of screen pixels.
MIP_CROP_HALF_WIDTH_UM = 4.0

# One fixed colour per curve, shared by every line panel below (the axial
# profile, the energy-sectioning panel, and the D-14 outside-core panel).
# Plan 08.1-09's second tracer round flagged that the off-waist light_sheet
# curve rendered green in one panel and orange in another -- matplotlib's
# default colour cycle assigns colours by plot ORDER within each axes, and
# the two panels plotted their curves in different orders. An explicit
# per-curve colour map removes that dependency on plot order.
CURVE_COLORS = {
    "light_sheet": "tab:blue",
    "light_sheet_off_waist": "tab:orange",
    "aslm": "tab:green",
}

# D-19 disclosure (research Q1, deferred-items.md DEF-SEEDS-2): psfmodels'
# illumination arm is a 3-D pencil beam, not integrated over the transverse
# (Y) axis the way a true light sheet / DSLM / ASLM sweep physically is.
# Lateral (Y) widths reported anywhere in this figure are therefore
# optimistic, and the off-waist light_sheet profiles carry pencil-beam
# Fresnel structure instead of a smooth widening -- deferred-items.md
# DEF-SEEDS-5, and the reason this figure's off-waist comparison uses the
# energy-sectioning metric (in_focus_energy_fraction), not the peak-line
# axial FWHM, which is erratic under this residual.
PENCIL_BEAM_NOTE = (
    "Residual (deferred): illumination is a 3-D pencil beam, not a "
    "y-integrated light sheet; lateral (Y) widths are optimistic and "
    "off-waist light_sheet profiles carry pencil-beam Fresnel structure."
)

# D-19 disclosure (research Q2): the sampling invariants this script enforces
# (_require_simulation_sampling) and the Gaussian-vs-Dean slit mapping (D-03),
# so a reader can trace slit_width back to Dean et al. 2015's rectangular
# convention without re-deriving it. A Gaussian window with the same second
# moment as Dean's rectangular W has sigma = W/sqrt(12) ~= 0.289*W, whose
# FWHM (2*sqrt(2*ln(2))*sigma) is ~0.68*W.
SAMPLING_AND_SLIT_NOTE = (
    "Sampling: cubic voxels (dz == dxy), ni0 == ni. slit_width is the FWHM "
    "of a Gaussian window (D-03); Dean et al. 2015 used a rectangular slit "
    "W = 2*xR, and a Gaussian with the same second moment has FWHM ~0.68*W."
)


def _require_simulation_sampling(common: dict) -> None:
    """Reject anisotropic sampling or a mismatched immersion index before simulating.

    The legacy cardinal-direction rotation path (`_legacy_rot90_rotation`) is
    an exact pre-rotation-axis relabel only for cubic voxels (halt finding 1,
    ROT-04); a mismatched `ni0` reintroduces the spherical-aberration focal
    shift documented as halt finding 2. This is the figure-layer guard for
    halt route step 4 -- the `simulate/` package's own guard is plan
    08.1-11's responsibility, not this script's.
    """
    dz = common["dz"]
    dxy = common["dxy"]
    if dz != dxy:
        raise ValueError(
            f"_require_simulation_sampling: dz ({dz}) != dxy ({dxy}); the legacy rot90 "
            "rotation path is an exact axis relabel only for cubic voxels (halt finding 1)"
        )
    ni0 = common["ni0"]
    ni = common["ni"]
    if ni0 != ni:
        raise ValueError(
            f"_require_simulation_sampling: ni0 ({ni0}) != ni ({ni}); a mismatched ni0 "
            "reintroduces spherical aberration via psfmodels' own default design index "
            "(halt finding 2)"
        )


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
    rather than `shape // 2` -- RA-9 pins that peak within one sample of the
    array centre for these cubic, ni0==ni parameters, but the exact index can
    still differ by a sample between light_sheet and aslm seeds.
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


def axial_fwhm(psf: np.ndarray, dz: float) -> float:
    """Return the axial (Z) half-max width of a (Z, Y, X) PSF seed through its true peak.

    Mirrors `lateral_fwhm`'s first-crossing linear-interpolation convention
    (duplicated here rather than shared, per the single-file PEP 723
    philosophy), walked outward along Z instead of X from the seed's global
    peak (`np.unravel_index(np.argmax(...))`). `main()` measures this once
    on `seed_single` to get the DOF (depth of field) window that
    `in_focus_energy_fraction` and the energy-sectioning panel both use.
    Raises ValueError, rather than returning a sentinel, when the peak is
    non-positive or a crossing is not found within the volume.
    """
    peak_z, peak_y, peak_x = np.unravel_index(np.argmax(psf), psf.shape)
    profile = psf[:, peak_y, peak_x].astype(np.float64)
    peak_value = profile[peak_z]
    if peak_value <= 0:
        raise ValueError("axial_fwhm: peak value is not positive, cannot measure half-max width")
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
        raise ValueError("axial_fwhm: half-max crossing not found within the volume")
    return (right_crossing - left_crossing) * dz


def plane_energy_profile(seed: np.ndarray) -> np.ndarray:
    """Return the per-Z-plane energy fraction of a (Z, Y, X) seed, summing to 1.0.

    `seed.sum(axis=(1, 2), dtype=np.float64)` divided by the seed's total
    energy -- the raw per-plane energy vector `in_focus_energy_fraction`
    below windows and sums. Raises ValueError when the seed's total energy
    is not positive (there is nothing to normalise a fraction against).
    """
    plane_energy = seed.sum(axis=(1, 2), dtype=np.float64)
    total = float(plane_energy.sum())
    if total <= 0:
        raise ValueError("plane_energy_profile: seed total energy is not positive")
    return plane_energy / total


def in_focus_energy_fraction(seed: np.ndarray, half_window_um: float, dz: float) -> float:
    """Return the fraction of `seed`'s energy within `half_window_um` of the array centre plane.

    Measures optical sectioning around the emitter's focal plane: the sum of
    `plane_energy_profile` over planes whose offset from the centre plane
    index `(psf_size_z - 1) / 2` is at most `half_window_um`. Uses the array
    CENTRE, not each seed's own argmax plane, because RA-9 pins every seed's
    peak within one sample of that centre under this script's cubic,
    ni0==ni COMMON -- a fixed reference keeps the light_sheet-vs-aslm
    comparison honest even where the peak drifts by a sample between modes.
    Under the pencil-beam residual (deferred-items.md DEF-SEEDS-5) this
    fraction falls monotonically off-waist where the peak-line axial FWHM
    does not.
    """
    profile = plane_energy_profile(seed)
    centre = (seed.shape[0] - 1) / 2.0
    offsets_um = (np.arange(seed.shape[0]) - centre) * dz
    in_window = np.abs(offsets_um) <= half_window_um
    return float(profile[in_window].sum())


def detection_core_radius_um(seed_single: np.ndarray, dxy: float) -> float:
    """Return the D-14 detection-core radius: half the detection lateral FWHM.

    Resolves the 08.1-06 open item -- that plan's `r_core_um` was the full
    lateral FWHM (a diameter), not a radius -- to the committed D-14 test's
    definition (`tests/test_seeds.py::
    test_aslm_removes_out_of_focus_energy_at_every_z_beyond_dof`:
    `r_core = _half_max_width(...) / 2.0`).
    """
    return lateral_fwhm(seed_single, dxy) / 2.0


def measured_illumination_cross_section_fwhm_um(
    seed_light_sheet_off_waist: np.ndarray, seed_single: np.ndarray, dxy: float
) -> float:
    """Return the rotated illumination beam's lateral half-max width at the emitter's row.

    `generate_psf_seed` composes `light_sheet = normalise(detection *
    rotated_illumination)` (src/tiresias/seeds.py), and `seed_single =
    normalise(detection)` is the same `detection` factor up to a positive
    scalar that cancels out of any ratio. `seed_light_sheet_off_waist /
    seed_single`, evaluated along the detection peak's own (Z, Y) row,
    therefore recovers the rotated illumination cross section -- up to an
    irrelevant positive scale -- at the plane where the fixed emitter
    actually sits: it is the illumination *waist* that `emitter_offset` moves
    off that plane, not the emitter itself (see `generate_psf_seed`'s
    docstring). This is a diagnostic figure annotation (the display fix
    requested after the Task 1 tracer checkpoint, computed from these two
    already-generated seeds) -- it is not the D-14 core radius or the RA-6
    sectioning metric, and no other code depends on it being accurate away
    from this one row.
    """
    peak_z, peak_y, _ = np.unravel_index(np.argmax(seed_single), seed_single.shape)
    detection_row = seed_single[peak_z, peak_y, :].astype(np.float64)
    illuminated_row = seed_light_sheet_off_waist[peak_z, peak_y, :].astype(np.float64)
    epsilon = max(float(np.finfo(np.float64).eps), float(detection_row.max()) * 1e-6)
    ratio = illuminated_row / np.clip(detection_row, epsilon, None)

    peak_x = int(np.argmax(ratio))
    peak_value = ratio[peak_x]
    if peak_value <= 0:
        raise ValueError(
            "measured_illumination_cross_section_fwhm_um: recovered illumination ratio "
            "peak is not positive, cannot measure half-max width"
        )
    half_max = peak_value / 2.0

    def _crossing(indices: np.ndarray) -> float | None:
        values = ratio[indices]
        below = np.where(values < half_max)[0]
        if below.size == 0:
            return None
        edge = below[0]
        if edge == 0:
            return None
        i0, i1 = indices[edge - 1], indices[edge]
        v0, v1 = ratio[i0], ratio[i1]
        frac = (half_max - v0) / (v1 - v0)
        return i0 + frac * (i1 - i0)

    left_indices = np.arange(peak_x, -1, -1)  # peak -> start, descending
    right_indices = np.arange(peak_x, ratio.size)  # peak -> end, ascending

    left_crossing = _crossing(left_indices)
    right_crossing = _crossing(right_indices)
    if left_crossing is None or right_crossing is None:
        raise ValueError(
            "measured_illumination_cross_section_fwhm_um: half-max crossing not found "
            "within the volume"
        )
    return (right_crossing - left_crossing) * dxy


def build_comparison_figure(
    seed_light_sheet: np.ndarray,
    seed_light_sheet_off_waist: np.ndarray,
    seed_aslm: np.ndarray,
    common: dict,
    slit_width: float,
    *,
    off_waist_um: float,
    r_core_um: float,
    illumination_fwhm_um: float,
    dof_um: float,
) -> plt.Figure:
    """Assemble the light_sheet-vs-aslm comparison figure: MIPs, axial profile, sectioning, and outside-core panels.

    Every MIP panel is plotted in physical micrometres (derived from
    `dz`/`dxy`), centered on zero, cropped to `MIP_CROP_HALF_WIDTH_UM` on
    every axis. The three MIP columns are: light_sheet at the waist,
    light_sheet with the emitter `off_waist_um` off-waist (halt route step
    3), and aslm -- aslm gets only one column because it is bit-identical
    for every `emitter_offset` (D-07). A detection-only (no illumination
    gating) column was added as a post-tracer-checkpoint deviation and
    removed again per the user's second tracer round ("I don't think the
    detection needs to be included in the final sample") -- the
    sheet-thickness measurement it enabled is kept as the
    `illumination_fwhm_um` annotation below, computed from `seed_single` by
    the caller, not plotted as its own panel. Each panel is normalised to
    its *own* peak (not a shared per-row max) with a fixed LogNorm floor, so
    the off-waist and waist tails are both visible at once instead of one
    clipping the other; `illumination_fwhm_um` (see
    `measured_illumination_cross_section_fwhm_um`) annotates the off-waist
    column with the measured illumination thickness at the emitter's plane.
    `r_core_um` (see `detection_core_radius_um`) sets the fixed
    detection-core radius for the bottom D-14 outside-core-fraction panel
    (see `outside_core_fraction`). `dof_um` (see `axial_fwhm`) sets the
    depth-of-field half-window for the energy-sectioning panel (RA-6, see
    `in_focus_energy_fraction`). Every line panel below plots its three
    curves in the same `CURVE_COLORS` order, so a curve's colour is
    identical across every panel it appears in.
    """
    dz = common["dz"]
    dxy = common["dxy"]
    psf_size_z = common["psf_size_z"]
    psf_size_xy = common["psf_size_xy"]
    z_extent_um = psf_size_z * dz
    lateral_extent_um = psf_size_xy * dxy

    # XZ MIP: max over Y (axis=1) -> (Z, X). YZ MIP: max over X (axis=2) -> (Z, Y).
    xz_light_sheet = seed_light_sheet.max(axis=1)
    xz_light_sheet_off_waist = seed_light_sheet_off_waist.max(axis=1)
    xz_aslm = seed_aslm.max(axis=1)
    yz_light_sheet = seed_light_sheet.max(axis=2)
    yz_light_sheet_off_waist = seed_light_sheet_off_waist.max(axis=2)
    yz_aslm = seed_aslm.max(axis=2)

    # Extent centered on zero (left, right, bottom, top) with origin='upper':
    # index 0 (top row) -> -z_extent/2, last row -> +z_extent/2. All four
    # columns and both projections share this same lateral/axial physical
    # framing so the focal plane annotated below lands exactly at Z=0.
    xz_extent = (
        -lateral_extent_um / 2,
        lateral_extent_um / 2,
        z_extent_um / 2,
        -z_extent_um / 2,
    )
    yz_extent = xz_extent  # same physical Z/lateral spans for both projections

    # Each MIP panel is normalised to its OWN peak (not a shared per-row max)
    # with a fixed LogNorm floor. The prior single shared-max LogNorm made
    # the off-waist and waist/aslm panels differ by roughly an order of
    # magnitude in peak brightness (off-waist spreads the same total energy
    # over a much larger volume), so that shared scale clipped everything
    # but the brightest panel's tails into the LogNorm floor -- exactly the
    # "energy looks gone, everything else is a small dot" artifact the user
    # flagged. Because every panel is independently peak-normalised, the
    # shared colourbar below is labelled "intensity / panel peak", not an
    # absolute radiometric scale.
    panel_norm = LogNorm(vmin=1e-4, vmax=1.0)

    def _peak_normalised(panel: np.ndarray) -> np.ndarray:
        peak = float(panel.max())
        if peak <= 0:
            peak = float(np.finfo(np.float64).eps)
        return panel / peak

    # constrained_layout (not tight_layout): tight_layout does not account for
    # the per-row colorbars added below and produces overlapping panels/text.
    # 5-row gridspec: rows 0-1 are the 2x3 MIP grid (light_sheet waist,
    # light_sheet off-waist, aslm), row 2 spans all three columns for the
    # axial intensity-profile overlay, row 3 (new, Task 2) spans all three
    # columns for the plane-energy optical-sectioning panel (RA-6), and row 4
    # spans all three columns for the D-14 outside-core-fraction panel
    # (moved down from row 3 to make room for row 3's new panel).
    fig = plt.figure(figsize=(15, 20), constrained_layout=True)
    gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 0.8, 0.8, 0.8])
    axes = np.array(
        [
            [fig.add_subplot(gs[0, col]) for col in range(3)],
            [fig.add_subplot(gs[1, col]) for col in range(3)],
        ]
    )

    row_specs = (
        (
            axes[0, :],
            (xz_light_sheet, xz_light_sheet_off_waist, xz_aslm),
            xz_extent,
            "XZ",
            "X",
        ),
        (
            axes[1, :],
            (yz_light_sheet, yz_light_sheet_off_waist, yz_aslm),
            yz_extent,
            "YZ",
            "Y",
        ),
    )

    for ax_row, panels, extent, proj_label, lateral_label in row_specs:
        ax_waist, ax_off_waist, ax_aslm = ax_row
        panel_waist, panel_off_waist, panel_aslm = panels

        # aspect="equal" (not "auto"): voxels are now cubic (dz == dxy), so
        # an equal-aspect, index-based panel already matches the displayed
        # geometry to the physical micrometre extent -- unlike the prior
        # anisotropic sampling, no visual stretch correction is needed here.
        ax_waist.imshow(
            _peak_normalised(panel_waist), aspect="equal", norm=panel_norm, extent=extent
        )
        ax_waist.set_title(f"light sheet {proj_label} MIP\nemitter at the waist", fontsize=8)

        ax_off_waist.imshow(
            _peak_normalised(panel_off_waist), aspect="equal", norm=panel_norm, extent=extent
        )
        ax_off_waist.set_title(
            f"light sheet {proj_label} MIP\nemitter {off_waist_um:.1f} um off-waist\n"
            f"sheet FWHM {illumination_fwhm_um:.1f} um at this position",
            fontsize=8,
        )

        im_aslm = ax_aslm.imshow(
            _peak_normalised(panel_aslm), aspect="equal", norm=panel_norm, extent=extent
        )
        # Three short lines, not two long ones: a single long second line
        # ran off the right edge of the figure at fontsize 9 -- the same
        # class of title-clipping issue 08.1-06 fixed
        # for the in-image annotation this title text replaced.
        ax_aslm.set_title(
            f"ASLM {proj_label} MIP\n"
            f"slit width = {slit_width:.2f} um, integrated along propagation\n"
            "(X at this direction), identical at every FOV position",
            fontsize=8,
        )

        for ax in (ax_waist, ax_off_waist, ax_aslm):
            ax.set_xlabel(f"{lateral_label} (um)")
            ax.set_ylabel("Z (um)")
            # Display-only crop (MIP_CROP_HALF_WIDTH_UM) -- the simulated
            # window itself is unchanged (still the full COMMON psf_size
            # cube); only the plotted view is zoomed, so the in-focus core
            # and the off-waist spread are both visible at once instead of
            # squeezed into a few pixels of the full +-6.97 um extent.
            ax.set_xlim(-MIP_CROP_HALF_WIDTH_UM, MIP_CROP_HALF_WIDTH_UM)
            ax.set_ylim(MIP_CROP_HALF_WIDTH_UM, -MIP_CROP_HALF_WIDTH_UM)

        cbar = fig.colorbar(im_aslm, ax=list(ax_row), shrink=0.85, pad=0.02)
        cbar.set_label("intensity / panel peak (log)")

    # Axial intensity-profile overlay: makes the waist / off-waist / aslm
    # difference readable without the reader computing anything. Z
    # coordinate centered on zero via pixel centers, the same "centered on
    # zero" convention the projection panels above use for their extent.
    profile_light_sheet = axial_profile(seed_light_sheet)
    profile_light_sheet_off_waist = axial_profile(seed_light_sheet_off_waist)
    profile_aslm = axial_profile(seed_aslm)
    z_um = -z_extent_um / 2 + (np.arange(psf_size_z) + 0.5) * dz

    ax_profile = fig.add_subplot(gs[2, :])
    # Each curve normalised to its own peak -- the plot compares *shape*, not
    # the energy redistribution the gate causes, so say so in the axis label
    # rather than presenting these as raw normalised intensities. Explicit
    # `color=CURVE_COLORS[...]` keeps this curve's colour identical to its
    # appearance in the sectioning and outside-core panels below.
    ax_profile.plot(
        z_um,
        profile_light_sheet / profile_light_sheet.max(),
        marker="o",
        markersize=3,
        color=CURVE_COLORS["light_sheet"],
        label="light sheet",
    )
    ax_profile.plot(
        z_um,
        profile_light_sheet_off_waist / profile_light_sheet_off_waist.max(),
        marker="o",
        markersize=3,
        color=CURVE_COLORS["light_sheet_off_waist"],
        label=f"light sheet ({off_waist_um:.1f} um off-waist)",
    )
    ax_profile.plot(
        z_um,
        profile_aslm / profile_aslm.max(),
        marker="o",
        markersize=3,
        color=CURVE_COLORS["aslm"],
        label=f"ASLM (slit width = {slit_width:.2f} um)",
    )
    ax_profile.axhline(0.5, color="gray", linestyle="--", linewidth=1.0)
    ax_profile.set_xlabel("Z (um)")
    ax_profile.set_ylabel("intensity, each curve peak-normalised to 1.0")
    ax_profile.set_title("Axial (Z) intensity profile through each seed's true peak")
    # Zoom to where the curves actually carry visible structure -- the full
    # volume extent (matching the projection panels above) squeezes the
    # half-max crossing separation the reader needs to see into a few pixels
    # of screen space. Bound derived from where any curve clears 2% of its
    # own peak, plus a fixed margin, rather than a hardcoded window.
    visible = (
        (profile_light_sheet / profile_light_sheet.max() >= 0.02)
        | (profile_light_sheet_off_waist / profile_light_sheet_off_waist.max() >= 0.02)
        | (profile_aslm / profile_aslm.max() >= 0.02)
    )
    visible_z = z_um[visible]
    margin = 2.0 * dz
    ax_profile.set_xlim(visible_z.min() - margin, visible_z.max() + margin)
    ax_profile.legend()

    # Plane-energy optical-sectioning panel (row 3, new in Task 2): RA-6.
    # log y axis (energy spans orders of magnitude away from focus); dashed
    # vertical lines mark the +-DOF window that `in_focus_energy_fraction`
    # sums over; each legend label carries its own measured in-focus
    # fraction, computed from the data, never hardcoded. This metric stays
    # monotonic off-waist under the pencil-beam residual where the peak-line
    # axial FWHM in the panel above does not (deferred-items.md DEF-SEEDS-5).
    energy_light_sheet = plane_energy_profile(seed_light_sheet)
    energy_light_sheet_off_waist = plane_energy_profile(seed_light_sheet_off_waist)
    energy_aslm = plane_energy_profile(seed_aslm)

    fraction_light_sheet = in_focus_energy_fraction(seed_light_sheet, dof_um, dz)
    fraction_light_sheet_off_waist = in_focus_energy_fraction(
        seed_light_sheet_off_waist, dof_um, dz
    )
    fraction_aslm = in_focus_energy_fraction(seed_aslm, dof_um, dz)

    ax_sectioning = fig.add_subplot(gs[3, :])
    ax_sectioning.semilogy(
        z_um,
        energy_light_sheet,
        marker="o",
        markersize=3,
        color=CURVE_COLORS["light_sheet"],
        label=f"light sheet (in-focus energy {fraction_light_sheet:.3f})",
    )
    ax_sectioning.semilogy(
        z_um,
        energy_light_sheet_off_waist,
        marker="o",
        markersize=3,
        color=CURVE_COLORS["light_sheet_off_waist"],
        label=(
            f"light sheet ({off_waist_um:.1f} um off-waist) "
            f"(in-focus energy {fraction_light_sheet_off_waist:.3f})"
        ),
    )
    ax_sectioning.semilogy(
        z_um,
        energy_aslm,
        marker="o",
        markersize=3,
        color=CURVE_COLORS["aslm"],
        label=(
            f"ASLM (slit width = {slit_width:.2f} um) "
            f"(in-focus energy {fraction_aslm:.3f})"
        ),
    )
    ax_sectioning.axvline(-dof_um, color="gray", linestyle="--", linewidth=1.0)
    ax_sectioning.axvline(dof_um, color="gray", linestyle="--", linewidth=1.0)
    ax_sectioning.set_xlabel("Z (um)")
    ax_sectioning.set_ylabel("plane energy fraction (log)")
    ax_sectioning.set_title(
        f"Plane energy vs Z (optical sectioning); dashed lines at +-DOF ({dof_um:.4f} um)"
    )
    ax_sectioning.set_xlim(visible_z.min() - margin, visible_z.max() + margin)
    ax_sectioning.legend(fontsize=8)

    # D-14 outside-core-fraction panel (row 4, moved down from row 3 to make
    # room for the sectioning panel above): makes the "energy removed
    # off-waist" claim decidable by eye. Same z_um centring as the axial
    # profile above. r_core_um is now `detection_core_radius_um`'s half-FWHM
    # radius (the 08.1-06 open item), stated explicitly in the title.
    fraction_out_light_sheet = outside_core_fraction(seed_light_sheet, r_core_um, dxy)
    fraction_out_light_sheet_off_waist = outside_core_fraction(
        seed_light_sheet_off_waist, r_core_um, dxy
    )
    fraction_out_aslm = outside_core_fraction(seed_aslm, r_core_um, dxy)

    ax_fraction = fig.add_subplot(gs[4, :])
    ax_fraction.plot(
        z_um,
        fraction_out_light_sheet,
        marker="o",
        markersize=3,
        color=CURVE_COLORS["light_sheet"],
        label="light sheet",
    )
    ax_fraction.plot(
        z_um,
        fraction_out_light_sheet_off_waist,
        marker="o",
        markersize=3,
        color=CURVE_COLORS["light_sheet_off_waist"],
        label=f"light sheet ({off_waist_um:.1f} um off-waist)",
    )
    ax_fraction.plot(
        z_um,
        fraction_out_aslm,
        marker="o",
        markersize=3,
        color=CURVE_COLORS["aslm"],
        label=f"ASLM (slit width = {slit_width:.2f} um)",
    )
    ax_fraction.set_xlabel("Z (um)")
    ax_fraction.set_ylabel("fraction of plane energy outside core")
    ax_fraction.set_title(
        f"Lateral energy fraction outside the detection core "
        f"(core radius = {r_core_um:.4f} um, half the detection lateral FWHM) vs Z"
    )
    ax_fraction.set_xlim(visible_z.min() - margin, visible_z.max() + margin)
    ax_fraction.legend(fontsize=8)

    # Gap closure (user-requested publication cleanup, plan 08.1-12): the
    # suptitle is now a clean, publication-style title with no citation or
    # disclosure blurb. The pencil-beam and sampling/slit-mapping
    # disclosures move to the module docstring and are printed to stdout by
    # main() instead -- never dropped, just moved out of the figure.
    fig.suptitle(
        f"Light sheet vs ASLM PSF seed comparison (ASLM slit width = {slit_width:.2f} um)",
        fontsize=11,
    )
    return fig


def main() -> None:
    """Generate light_sheet (waist/off-waist) and aslm PSF seeds, print summaries, and save a comparison figure."""
    _require_simulation_sampling(COMMON)
    describe_environment()

    seed_light_sheet = generate_psf_seed(psf_mode="light_sheet", **COMMON)
    seed_light_sheet_off_waist = generate_psf_seed(
        psf_mode="light_sheet", emitter_offset=OFF_WAIST_UM, **COMMON
    )
    seed_aslm = generate_psf_seed(psf_mode="aslm", slit_width=SLIT_WIDTH, **COMMON)
    # D-14/OPEN-06b: seed_single supplies the fixed core radius (via
    # detection_core_radius_um, half the detection lateral FWHM -- the
    # committed D-14 test's definition) and the DOF (via axial_fwhm) that
    # in_focus_energy_fraction and the sectioning panel both use. The core
    # and DOF are defined from the unslit detection-only PSF, not from
    # either compared mode.
    seed_single = generate_psf_seed(psf_mode="single", **COMMON)
    dxy = COMMON["dxy"]
    dz = COMMON["dz"]
    r_core_um = detection_core_radius_um(seed_single, dxy)
    dof_um = axial_fwhm(seed_single, dz)
    # Display-fix annotation (post-tracer-checkpoint deviation): measured,
    # never hardcoded, from the two already-generated seeds -- see
    # measured_illumination_cross_section_fwhm_um's docstring.
    illumination_fwhm_um = measured_illumination_cross_section_fwhm_um(
        seed_light_sheet_off_waist, seed_single, dxy
    )

    # RA-6: measured, never hardcoded, in-focus energy fractions for the
    # printed summary and the SUMMARY.md record.
    fraction_light_sheet = in_focus_energy_fraction(seed_light_sheet, dof_um, dz)
    fraction_light_sheet_off_waist = in_focus_energy_fraction(
        seed_light_sheet_off_waist, dof_um, dz
    )
    fraction_aslm = in_focus_energy_fraction(seed_aslm, dof_um, dz)

    # D-19: the slit integrates along the pre-rotation propagation axis
    # (D-02) for every direction -- there is no per-axis gate resolution
    # anymore, and no full-extent equivalence shortcut (D-15).
    propagation_window_um = COMMON["psf_size_z"] * COMMON["dz"]
    summarise_mode("light_sheet", seed_light_sheet, "none")
    summarise_mode(
        "light_sheet_off_waist",
        seed_light_sheet_off_waist,
        f"none, emitter {OFF_WAIST_UM} um off-waist",
    )
    summarise_mode(
        "aslm",
        seed_aslm,
        f"slit_width={SLIT_WIDTH}um sweep-integrated along propagation, "
        f"window={propagation_window_um:.3f}um",
    )
    print(
        "measured illumination cross-section FWHM at the emitter's plane "
        f"({OFF_WAIST_UM:.1f} um off-waist): {illumination_fwhm_um:.3f} um"
    )
    print(f"detection DOF (axial FWHM of seed_single): {dof_um:.4f} um")
    print(f"D-14 detection core radius (half the detection lateral FWHM): {r_core_um:.4f} um")
    print(
        "in-focus energy fraction (RA-6): "
        f"light_sheet waist={fraction_light_sheet:.3f}, "
        f"light_sheet {OFF_WAIST_UM:.1f}um off-waist={fraction_light_sheet_off_waist:.3f}, "
        f"aslm={fraction_aslm:.3f}"
    )
    # Gap closure (user-requested publication cleanup, plan 08.1-12): these
    # disclosures used to render into the figure's suptitle; they are now
    # printed to stdout (and kept in the module docstring's Notes /
    # limitations section) instead, so the figure itself stays clean.
    print(PENCIL_BEAM_NOTE)
    print(SAMPLING_AND_SLIT_NOTE)

    fig = build_comparison_figure(
        seed_light_sheet,
        seed_light_sheet_off_waist,
        seed_aslm,
        COMMON,
        SLIT_WIDTH,
        off_waist_um=OFF_WAIST_UM,
        r_core_um=r_core_um,
        illumination_fwhm_um=illumination_fwhm_um,
        dof_um=dof_um,
    )

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "light_sheet_vs_aslm.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
