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

# D-07: optical parameters originate from docs/usage.md's Python API examples,
# for baseline consistency with the existing documentation -- except the two
# aperture entries below, which are deliberately raised for this demo and no
# longer track the docs (identical to examples/light_sheet_vs_aslm.py's
# COMMON; duplicated here per the single-file PEP 723 philosophy -- no shared
# helper module between the two scripts. Plan 08.1-13's audit test pins the
# two COMMONs equal).
#
# Halt route steps 1-2 (research Q3/Q5): three values corrected from the
# pre-08.1-10 sweep, which was measured on anisotropic (dz=0.3) sampling
# through the legacy rot90 relabel and an ni0=None aberration -- producing a
# spurious non-monotonic trend (08.1-06 open item OPEN-06a) that research Q3
# identifies as a bug signal, not physics, since axial FWHM vs ASLM slit
# width must be monotonic non-decreasing:
#   - "dz" == "dxy": the legacy cardinal-direction rotation path
#     (`_legacy_rot90_rotation`) relabels pre-rotation axes onto the display
#     axes with no resampling. That relabel is exact only for cubic voxels
#     (halt finding 1, ROT-04) -- the seeds.py fix for anisotropic sampling
#     is deferred (deferred-items.md DEF-SEEDS-1).
#   - "psf_size_z" == "psf_size_xy" (a cube window): after the relabel,
#     pre-rotation axis 0 (the "psf_size_z" samples) lands on the display X
#     axis. An equal size fills X edge-to-edge with no zero-padded white
#     bands. The odd size (129) puts the focal plane exactly on a sample,
#     rather than between two samples at a half-integer centre.
#   - "ni0" == "ni" (1.33): "ni0=None" falls back to psfmodels' own default
#     design immersion index (1.515, oil), producing a spherical-aberration
#     focal shift in this water-immersion demo (halt finding 2).
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

# The slit window integrates along the pre-rotation propagation axis,
# sampled at dz (D-02, D-17). PROPAGATION_EXTENT is the full simulated
# propagation window -- a slit at least this wide averages over essentially
# the whole simulated sweep. It no longer reduces to light_sheet (D-15).
PROPAGATION_EXTENT = COMMON["psf_size_z"] * COMMON["dz"]  # 13.932 um

# D-13/D-18: 7 points, from a sub-dz width (below one dz==0.108um sample,
# exercising the D-18 waist-limited fallback) up through the full
# propagation extent.
SLIT_WIDTHS: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, PROPAGATION_EXTENT)

# D-15/D-13: effectively uniform across the whole propagation window, far
# beyond any physical slit -- gives the swept-average sheet reference, the
# wide-slit limit a very wide slit converges to. Replaces the old exact
# light_sheet equivalence anchor (D-15 retires that shortcut).
SWEPT_AVERAGE_SLIT_WIDTH = 1.0e6

# D-19 disclosure (research Q1, deferred-items.md DEF-SEEDS-2): psfmodels'
# illumination arm is a 3-D pencil beam, not integrated over the transverse
# (Y) axis the way a true light sheet / DSLM / ASLM sweep physically is.
# Identical content to examples/light_sheet_vs_aslm.py's PENCIL_BEAM_NOTE,
# duplicated here per the single-file PEP 723 philosophy.
PENCIL_BEAM_NOTE = (
    "Residual (deferred): illumination is a 3-D pencil beam, not a "
    "y-integrated light sheet; lateral (Y) widths are optimistic and "
    "off-waist light_sheet profiles carry pencil-beam Fresnel structure."
)

# D-19 disclosure (research Q5): the sampling invariants this script enforces
# (_require_simulation_sampling).
SAMPLING_NOTE = "Sampling: cubic voxels (dz == dxy), ni0 == ni."

# D-03/D-19 disclosure (research Q2): the Gaussian-vs-Dean slit mapping, so a
# reader can trace slit_width back to Dean et al. 2015's rectangular
# convention without re-deriving it. A Gaussian window with the same second
# moment as Dean's rectangular W has sigma = W/sqrt(12) ~= 0.289*W, whose
# FWHM (2*sqrt(2*ln(2))*sigma) is ~0.68*W.
SLIT_MAPPING_NOTE = (
    "slit_width is the FWHM of a Gaussian window (D-03); Dean et al. 2015 "
    "used a rectangular slit W = 2*xR, and a Gaussian with the same second "
    "moment has FWHM ~0.68*W."
)


def _require_simulation_sampling(common: dict) -> None:
    """Reject anisotropic sampling or a mismatched immersion index before simulating.

    The legacy cardinal-direction rotation path (`_legacy_rot90_rotation`) is
    an exact pre-rotation-axis relabel only for cubic voxels (halt finding 1,
    ROT-04); a mismatched `ni0` reintroduces the spherical-aberration focal
    shift documented as halt finding 2. Mirrors
    examples/light_sheet_vs_aslm.py's guard of the same name (plan 08.1-09) --
    duplicated here per the single-file PEP 723 philosophy.
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


def axial_fwhm(psf: np.ndarray, dz: float) -> float | None:
    """Return the axial (Z) FWHM of a (Z, Y, X) PSF seed in physical units.

    Locates the true peak with `np.unravel_index(np.argmax(psf), psf.shape)`
    rather than `shape // 2` -- for the D-07 parameters the peak sits at
    (Y, X) = (63, 64), not the geometric centre (64, 64), and the peak Z
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


def run_sweep(widths: tuple[float, ...] = SLIT_WIDTHS) -> list[tuple[float, float | None]]:
    """Generate an aslm seed for each requested slit_width and measure its axial FWHM.

    Iterates `widths` (defaulting to the committed SLIT_WIDTHS) in the given
    order and calls generate_psf_seed for every point -- no local
    re-implementation of PSF generation, rotation, or slit gating. The
    requested sequence is the presentation order: this function is the
    single ordered source both print_table and build_sweep_figure consume,
    so neither re-derives or re-sorts it.

    A slit_width the library rejects raises ValueError from
    generate_psf_seed -- only for a non-positive width, or when the
    slit-integrated illumination has no positive finite energy (D-18); the
    old too-narrow-slit error is gone, replaced by the D-18 waist-limited
    fallback (below one dz sample, no convolution runs at all). That
    ValueError is caught here specifically -- not via a bare except, since it
    is the library's own contract for exactly those cases and anything
    broader would hide a real defect -- the point is recorded as None and the
    sweep continues with the remaining points. A point whose axial_fwhm comes
    back None (the profile never dropped to half maximum inside the volume)
    is likewise recorded as None. Either way the point keeps its ordered slot
    in the returned list -- it is never dropped.
    """
    results: list[tuple[float, float | None]] = []
    for width in widths:
        try:
            seed = generate_psf_seed(psf_mode="aslm", slit_width=width, **COMMON)
        except ValueError as exc:
            print(f"skipped slit_width={width!r}: rejected by generate_psf_seed: {exc}")
            results.append((width, None))
            continue
        fwhm = axial_fwhm(seed, COMMON["dz"])
        if fwhm is None:
            print(
                f"skipped slit_width={width!r}: axial_fwhm could not be measured "
                "(the profile never dropped to half maximum within the volume)"
            )
        results.append((width, fwhm))
    return results


def print_table(
    results: list[tuple[float, float | None]],
    waist_limited_fwhm: float,
    swept_average_fwhm: float,
) -> None:
    """Print the D-13 slit_width/FWHM table, in the same order run_sweep() returned.

    Rows are printed exactly as run_sweep() returns them, with no sorting
    step that could reorder equal or near-equal points or introduce
    duplicates. A None entry -- a rejected slit_width or an unmeasurable FWHM
    -- renders as an explicit "SKIPPED" marker, never as a number and never
    omitted; the run_sweep() log above already named the offending width and
    reproduced the library's own reason. An empty result list still prints
    the header and reference lines rather than raising -- there is no row
    index to look up on an empty sequence. Both measured reference limits
    (D-13/D-15/D-18) are printed after the sweep rows: the waist-limited
    (light_sheet) value the narrow end approaches, and the swept-average
    sheet value the wide end approaches.
    """
    print(f"{'slit_width (um)':>16}  {'axial FWHM (um)':>16}  note")
    if not results:
        print("  (no sweep points -- every requested slit_width was skipped; see log above)")
    else:
        last_index = len(results) - 1
        for index, (width, fwhm) in enumerate(results):
            if fwhm is None:
                fwhm_str = f"{'SKIPPED':>16}"
                note = "<- skipped: rejected slit_width or unmeasurable FWHM, see log above"
            elif index == 0:
                fwhm_str = f"{fwhm:>16.4f}"
                note = "<- sub-dz slit: waist-limited, no convolution (D-18)"
            elif index == last_index:
                fwhm_str = f"{fwhm:>16.4f}"
                note = "<- slit spans the whole propagation window"
            else:
                fwhm_str = f"{fwhm:>16.4f}"
                note = ""
            print(f"{width:>16.4f}  {fwhm_str}  {note}")
    print(
        f"{'waist-limited (light_sheet)':>16}  {waist_limited_fwhm:>16.4f}  "
        "(narrow-slit reference, D-18)"
    )
    print(
        f"{'swept-average sheet':>16}  {swept_average_fwhm:>16.4f}  "
        "(wide-slit reference, D-13/D-15)"
    )


def build_sweep_figure(
    results: list[tuple[float, float | None]],
    waist_limited_fwhm: float,
    swept_average_fwhm: float,
) -> plt.Figure | None:
    """Plot measured axial FWHM against slit_width, with the two D-13 reference lines.

    Only measured (non-None) points are plotted -- a skipped point is never
    rendered as data. Returns None, rather than raising from unpacking an
    empty sequence, when no point in `results` was measurable; the caller
    must handle that return instead of assuming a Figure.

    Title and narrative are derived from what this run actually produced --
    the tradeoff direction and the monotonicity claim are both read off the
    computed measured FWHM values, never asserted from a prior expectation
    (see this plan's must_haves.prohibitions / threat T-08.1-14).
    """
    measured = [(width, fwhm) for width, fwhm in results if fwhm is not None]
    if not measured:
        print("no sweep point produced a measurable FWHM -- skipping figure")
        return None

    widths = [width for width, _ in measured]
    fwhms = [fwhm for _, fwhm in measured]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(widths, fwhms, marker="o", linestyle="-", label="measured axial FWHM")
    ax.axhline(
        waist_limited_fwhm,
        color="gray",
        linestyle="--",
        label=f"waist-limited (light_sheet) reference ({waist_limited_fwhm:.4f} um)",
    )
    ax.axhline(
        swept_average_fwhm,
        color="tab:red",
        linestyle=":",
        label=f"swept-average sheet reference ({swept_average_fwhm:.4f} um)",
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

    # D-13: monotonic non-decreasing trend, computed from the measured
    # sequence with a small tolerance for float round-off -- never asserted.
    monotonic = all(
        fwhms[i + 1] >= fwhms[i] - 1e-6 for i in range(len(fwhms) - 1)
    )
    monotonic_str = "yes" if monotonic else "no"

    ax.set_title(
        "ASLM axial FWHM vs slit_width\n"
        f"Measured: FWHM {direction}\n"
        f"monotonic non-decreasing: {monotonic_str}\n"
        f"({first_fwhm:.4f} um at slit_width={first_width:.2f} um -> "
        f"{last_fwhm:.4f} um at slit_width={last_width:.2f} um)"
    )
    # D-19: one combined footnote carrying the pencil-beam, sampling and
    # Dean slit-mapping disclosures. rect=(0, 0.1, 1, 1) reserves the bottom
    # 10% of the figure for it so tight_layout does not overlap the axes --
    # the same clipping class 08.1-06/08.1-09 fixed for other in-figure text.
    fig.text(
        0.5,
        0.01,
        f"{PENCIL_BEAM_NOTE} {SAMPLING_NOTE} {SLIT_MAPPING_NOTE}",
        ha="center",
        va="bottom",
        fontsize=7,
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


def main() -> None:
    """Measure the two D-13 reference limits, run the sweep, print the table, and save the plot."""
    _require_simulation_sampling(COMMON)
    waist_limited_seed = generate_psf_seed(psf_mode="light_sheet", **COMMON)
    waist_limited_fwhm = axial_fwhm(waist_limited_seed, COMMON["dz"])
    if waist_limited_fwhm is None:
        raise RuntimeError("light_sheet reference seed produced no measurable axial FWHM")

    # D-13/D-15: the wide-slit limit, measured through the production aslm
    # path at an effectively uniform slit_width rather than asserted.
    swept_average_seed = generate_psf_seed(
        psf_mode="aslm", slit_width=SWEPT_AVERAGE_SLIT_WIDTH, **COMMON
    )
    swept_average_fwhm = axial_fwhm(swept_average_seed, COMMON["dz"])
    if swept_average_fwhm is None:
        raise RuntimeError("swept-average aslm seed produced no measurable axial FWHM")

    results = run_sweep()
    print_table(results, waist_limited_fwhm, swept_average_fwhm)

    if all(fwhm is None for _, fwhm in results):
        # Degenerate-sweep guard: every point was rejected or unmeasurable.
        # There is no anchor to check and nothing to plot, so report that
        # plainly and exit non-zero rather than raising from an empty
        # unpack further down -- a sweep that measured nothing has not done
        # the job this script exists to do.
        print("no sweep point produced a measurable FWHM -- nothing to plot or save")
        raise SystemExit(1)

    # D-18 narrow anchor: the first sweep point (SLIT_WIDTHS[0], below one dz
    # sample) applies no convolution at all, so its FWHM must exactly equal
    # the waist-limited (light_sheet) reference -- not merely close. An
    # approximate comparison here would hide a real regression in the D-18
    # fallback path, so this never raises on mismatch: it prints the
    # discrepancy and continues so the reader still gets the table and plot.
    _narrow_width, narrow_fwhm = results[0]
    if narrow_fwhm is None:
        print(
            "narrow anchor skipped: the sub-dz slit_width point itself "
            "was skipped, see log above -- no aslm FWHM to compare"
        )
    elif narrow_fwhm == waist_limited_fwhm:
        print(
            f"narrow anchor: exact match (sub-dz aslm FWHM {narrow_fwhm:.4f} um "
            f"== waist-limited reference {waist_limited_fwhm:.4f} um)"
        )
    else:
        print(
            f"narrow anchor: mismatch (sub-dz aslm FWHM {narrow_fwhm!r} um "
            f"!= waist-limited reference {waist_limited_fwhm!r} um)"
        )

    # D-13 wide-limit report: how close the widest simulated slit_width comes
    # to the true swept-average limit (SWEPT_AVERAGE_SLIT_WIDTH), reported
    # as a plain measured difference rather than asserted equality.
    _wide_width, wide_fwhm = results[-1]
    if wide_fwhm is None:
        print(
            "wide-limit report skipped: the widest slit_width point itself "
            "was skipped, see log above -- no aslm FWHM to compare"
        )
    else:
        wide_diff = abs(wide_fwhm - swept_average_fwhm)
        print(
            f"wide-limit report: widest simulated aslm FWHM ({wide_fwhm:.4f} um) "
            f"differs from the swept-average reference ({swept_average_fwhm:.4f} um) "
            f"by {wide_diff:.4f} um"
        )

    # D-13 trend line: monotonic non-decreasing, computed from the measured
    # sequence (skipping None entries) with a small float-round-off
    # tolerance -- never asserted from a prior expectation.
    measured_fwhms = [fwhm for _, fwhm in results if fwhm is not None]
    monotonic = all(
        measured_fwhms[i + 1] >= measured_fwhms[i] - 1e-6
        for i in range(len(measured_fwhms) - 1)
    )
    print(f"monotonic non-decreasing: {'yes' if monotonic else 'no'}")

    fig = build_sweep_figure(results, waist_limited_fwhm, swept_average_fwhm)
    if fig is None:
        # Unreachable given the all-None guard above (at least one point is
        # measurable at this point), but never assume a Figure without
        # checking -- build_sweep_figure's contract is to return None when
        # nothing was measurable.
        print("no sweep point produced a measurable FWHM -- nothing to plot or save")
        raise SystemExit(1)
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "slit_width_sweep.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
