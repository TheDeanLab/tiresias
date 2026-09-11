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

    # XZ maximum-intensity projection (max over Y, axis=1) -> (Z, X) array,
    # sharing one intensity scale across both panels for an honest comparison.
    xz_light_sheet = seed_light_sheet.max(axis=1)
    xz_aslm = seed_aslm.max(axis=1)
    vmax = float(max(xz_light_sheet.max(), xz_aslm.max()))
    # The slit gate's effect on this seed is concentrated in low-intensity
    # structure away from the on-axis peak (the light_sheet's natural
    # off-waist widening that ASLM's uniform gate suppresses) -- a linear
    # scale washes that structure out entirely. Log scale, shared across both
    # panels, is what actually makes the narrower aslm extent visible.
    log_norm = LogNorm(vmin=vmax * 1e-3, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, panel, title in (
        (axes[0], xz_light_sheet, "light_sheet"),
        (axes[1], xz_aslm, "aslm"),
    ):
        ax.imshow(panel, aspect="auto", norm=log_norm)
        ax.set_title(title)
    fig.tight_layout()

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "light_sheet_vs_aslm.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
