"""Regression tests for examples/light_sheet_vs_aslm.py (halt route steps 1-3, plan 08.1-09).

Pins the packaging contract shared by every PEP 723 example script in this
repo (identical header, scoped imports, no import side effects), the
corrected COMMON sampling (cubic voxels, matched immersion, odd cube
window), RA-9 (detection/light-sheet peaks sit at the array centre under
that sampling), the three-column comparison figure (light_sheet waist,
light_sheet off-waist, aslm), the D-14/RA-6 energy-sectioning helpers
(Task 2), and the pencil-beam/sampling/slit-mapping disclosures (D-19).
"""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_RELATIVE = "examples/light_sheet_vs_aslm.py"
_REFERENCE_SCRIPT = "examples/slit_width_sweep.py"
_SCRIPT_PATH = _ROOT / _SCRIPT_RELATIVE

# Line 17 is the metadata block's closing `# ///` marker; everything after it
# is per-script explanatory prose that is deliberately not compared here --
# only the machine-meaningful metadata block must match.
_HEADER_LINES = 17

_IMPORT_ALLOWLIST = {"__future__", "importlib", "pathlib", "matplotlib", "numpy", "tiresias"}


def _load(relative_path: str):
    """Load a script as a module without triggering its __main__ behavior.

    Safe: the script guards `main()` behind `if __name__ == "__main__":`, so
    `exec_module` only defines module-level names -- the same idiom
    tests/test_aslm_axial_regimes_example.py already uses.
    """
    path = _ROOT / relative_path
    spec = importlib.util.spec_from_file_location(
        relative_path.replace("/", "_").replace(".py", ""), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _gaussian_blob(shape: tuple[int, int, int], sigma: float) -> np.ndarray:
    """Return a strictly-positive float32 Gaussian blob centred in `shape`.

    A small positive floor (never exactly zero) keeps every plane's total
    energy positive, satisfying `lateral_fwhm`/`outside_core_fraction`'s
    positivity requirements on this synthetic figure-building test.
    """
    z = np.arange(shape[0])
    y = np.arange(shape[1])
    x = np.arange(shape[2])
    cz, cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0, (shape[2] - 1) / 2.0
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    blob = np.exp(
        -(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))
    )
    return (blob + 1e-6).astype(np.float32)


class Pep723PackagingTests(unittest.TestCase):
    def test_pep723_header_matches_the_shipped_reference(self):
        reference_lines = (_ROOT / _REFERENCE_SCRIPT).read_text(encoding="utf-8").splitlines()[
            :_HEADER_LINES
        ]
        script_lines = _SCRIPT_PATH.read_text(encoding="utf-8").splitlines()[:_HEADER_LINES]
        if script_lines != reference_lines:
            diverging = [
                f"line {index + 1}: {actual!r} != {expected!r}"
                for index, (actual, expected) in enumerate(zip(script_lines, reference_lines))
                if actual != expected
            ]
            self.fail(
                f"{_SCRIPT_RELATIVE}: PEP 723 header diverges from "
                f"{_REFERENCE_SCRIPT}:\n" + "\n".join(diverging)
            )

    def test_script_imports_only_the_expected_modules(self):
        roots = _imported_roots(_SCRIPT_PATH)
        extra = roots - _IMPORT_ALLOWLIST
        self.assertEqual(
            extra, set(), f"{_SCRIPT_RELATIVE} imports module(s) outside the allowlist: {extra}"
        )

    def test_importing_the_script_has_no_side_effects(self):
        output_dir = _ROOT / "examples" / "output"
        before = set(output_dir.glob("*")) if output_dir.exists() else set()
        _load(_SCRIPT_RELATIVE)
        after = set(output_dir.glob("*")) if output_dir.exists() else set()
        self.assertEqual(before, after, f"{_SCRIPT_RELATIVE} wrote to examples/output/ on import")
        self.assertEqual(plt.get_fignums(), [], f"{_SCRIPT_RELATIVE} opened a figure on import")


class SamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load(_SCRIPT_RELATIVE)

    def test_common_uses_cubic_voxels_matched_immersion_and_an_odd_cube_window(self):
        common = self.module.COMMON
        self.assertEqual(common["dz"], common["dxy"])
        self.assertEqual(common["ni0"], common["ni"])
        self.assertEqual(common["psf_size_z"], common["psf_size_xy"])
        self.assertEqual(common["psf_size_z"] % 2, 1)
        off_waist_um = self.module.OFF_WAIST_UM
        self.assertGreater(off_waist_um, 0)
        self.assertLess(off_waist_um, common["psf_size_z"] * common["dz"] / 2)

    def test_detection_and_light_sheet_peaks_sit_at_the_array_centre(self):
        # RA-9: with cubic sampling and ni0==ni, the single and light_sheet
        # seed argmax each sit within one sample of the array centre on
        # every axis.
        common = self.module.COMMON
        psf_size_z = common["psf_size_z"]
        psf_size_xy = common["psf_size_xy"]
        centre = ((psf_size_z - 1) // 2, (psf_size_xy - 1) // 2, (psf_size_xy - 1) // 2)

        seed_single = self.module.generate_psf_seed(psf_mode="single", **common)
        seed_light_sheet = self.module.generate_psf_seed(psf_mode="light_sheet", **common)

        for seed in (seed_single, seed_light_sheet):
            peak = np.unravel_index(np.argmax(seed), seed.shape)
            for axis_peak, axis_centre in zip(peak, centre):
                self.assertLessEqual(abs(axis_peak - axis_centre), 1)


class ComparisonFigureTests(unittest.TestCase):
    def test_figure_has_three_mip_columns_including_the_off_waist_light_sheet(self):
        # A detection-only (widefield) fourth column was added as a
        # post-tracer-checkpoint deviation and then removed again per the
        # user's second tracer round -- the grid is back to three columns
        # (light_sheet waist, light_sheet off-waist, aslm), and no title is
        # expected to mention "detection only" any more.
        module = _load(_SCRIPT_RELATIVE)
        common = dict(module.COMMON)
        common["psf_size_z"] = 9
        common["psf_size_xy"] = 9

        seed_light_sheet = _gaussian_blob((9, 9, 9), sigma=1.5)
        seed_light_sheet_off_waist = _gaussian_blob((9, 9, 9), sigma=1.5)
        seed_aslm = _gaussian_blob((9, 9, 9), sigma=1.5)

        fig = module.build_comparison_figure(
            seed_light_sheet,
            seed_light_sheet_off_waist,
            seed_aslm,
            common,
            2.0,
            off_waist_um=5.0,
            r_core_um=0.2,
            illumination_fwhm_um=3.1,
            dof_um=0.3,
        )
        try:
            image_axes = [ax for ax in fig.axes if len(ax.images) > 0]
            self.assertGreaterEqual(len(image_axes), 6)

            titles = [ax.get_title() for ax in fig.axes]
            self.assertTrue(
                any("off-waist" in title for title in titles),
                f"no axes title mentions 'off-waist': {titles}",
            )
        finally:
            plt.close(fig)
        self.assertEqual(plt.get_fignums(), [])


class PlaneEnergyProfileTests(unittest.TestCase):
    def test_plane_energy_profile_is_normalised_and_rejects_empty_seeds(self):
        module = _load(_SCRIPT_RELATIVE)
        seed = np.zeros((5, 3, 3), dtype=np.float64)
        seed[2, 1, 1] = 4.0
        seed[0, 0, 0] = 1.0

        profile = module.plane_energy_profile(seed)

        self.assertEqual(profile.dtype, np.float64)
        self.assertEqual(profile.shape, (5,))
        self.assertLess(abs(float(profile.sum()) - 1.0), 1e-12)

        with self.assertRaises(ValueError):
            module.plane_energy_profile(np.zeros((5, 3, 3), dtype=np.float64))


class InFocusEnergyFractionTests(unittest.TestCase):
    def test_in_focus_energy_fraction_counts_only_planes_inside_the_window(self):
        module = _load(_SCRIPT_RELATIVE)
        dz = 0.1

        # All energy in the centre plane (index 2 of 5) -> fraction 1.0 even
        # with a zero-width window, since offset 0 is always <= half_window.
        seed_centre_only = np.zeros((5, 3, 3), dtype=np.float64)
        seed_centre_only[2, 1, 1] = 1.0
        self.assertAlmostEqual(
            module.in_focus_energy_fraction(seed_centre_only, half_window_um=0.0, dz=dz),
            1.0,
        )

        # Energy split between the centre plane (index 2) and an edge plane
        # (index 0, offset 2*dz from centre). A half-window narrower than
        # 2*dz excludes the edge plane, so only the centre plane's energy
        # counts as in-focus.
        seed_split = np.zeros((5, 3, 3), dtype=np.float64)
        seed_split[2, 1, 1] = 0.6
        seed_split[0, 1, 1] = 0.4
        fraction = module.in_focus_energy_fraction(seed_split, half_window_um=0.5 * dz, dz=dz)
        self.assertAlmostEqual(fraction, 0.6)


class DetectionCoreRadiusTests(unittest.TestCase):
    def test_detection_core_radius_is_half_the_lateral_fwhm(self):
        module = _load(_SCRIPT_RELATIVE)
        seed = _gaussian_blob((9, 9, 9), sigma=1.5)
        dxy = 0.108

        expected = module.lateral_fwhm(seed, dxy) / 2.0
        self.assertAlmostEqual(module.detection_core_radius_um(seed, dxy), expected)


class SectioningAgainstRealSeedsTests(unittest.TestCase):
    """RA-6: real seeds generated from the script's own COMMON (not synthetic)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load(_SCRIPT_RELATIVE)
        common = cls.module.COMMON
        cls.dz = common["dz"]
        cls.seed_single = cls.module.generate_psf_seed(psf_mode="single", **common)
        cls.seed_light_sheet = cls.module.generate_psf_seed(psf_mode="light_sheet", **common)
        cls.seed_light_sheet_off_waist = cls.module.generate_psf_seed(
            psf_mode="light_sheet", emitter_offset=cls.module.OFF_WAIST_UM, **common
        )
        cls.seed_aslm = cls.module.generate_psf_seed(
            psf_mode="aslm", slit_width=cls.module.SLIT_WIDTH, **common
        )
        cls.seed_aslm_off_waist = cls.module.generate_psf_seed(
            psf_mode="aslm",
            slit_width=cls.module.SLIT_WIDTH,
            emitter_offset=cls.module.OFF_WAIST_UM,
            **common,
        )
        cls.dof_um = cls.module.axial_fwhm(cls.seed_single, cls.dz)

    def test_off_waist_light_sheet_loses_in_focus_energy_that_aslm_keeps(self):
        module = self.module
        fraction_waist = module.in_focus_energy_fraction(
            self.seed_light_sheet, self.dof_um, self.dz
        )
        fraction_off_waist = module.in_focus_energy_fraction(
            self.seed_light_sheet_off_waist, self.dof_um, self.dz
        )
        fraction_aslm = module.in_focus_energy_fraction(self.seed_aslm, self.dof_um, self.dz)

        # RA-6: off-waist light_sheet loses at least 0.1 more in-focus energy
        # than aslm, and is strictly below the waist light_sheet value.
        self.assertLessEqual(fraction_off_waist, fraction_aslm - 0.1)
        self.assertLess(fraction_off_waist, fraction_waist)

        # D-07: aslm is bit-identical for every emitter_offset.
        np.testing.assert_array_equal(self.seed_aslm_off_waist, self.seed_aslm)


class DisclosureTests(unittest.TestCase):
    def test_figure_discloses_the_pencil_beam_residual_sampling_and_slit_mapping(self):
        module = _load(_SCRIPT_RELATIVE)
        common = dict(module.COMMON)
        common["psf_size_z"] = 9
        common["psf_size_xy"] = 9

        seed_light_sheet = _gaussian_blob((9, 9, 9), sigma=1.5)
        seed_light_sheet_off_waist = _gaussian_blob((9, 9, 9), sigma=1.5)
        seed_aslm = _gaussian_blob((9, 9, 9), sigma=1.5)

        fig = module.build_comparison_figure(
            seed_light_sheet,
            seed_light_sheet_off_waist,
            seed_aslm,
            common,
            2.0,
            off_waist_um=5.0,
            r_core_um=0.2,
            illumination_fwhm_um=3.1,
            dof_um=0.3,
        )
        try:
            texts = [fig.get_suptitle() or ""]
            texts += [t.get_text() for t in fig.texts]
            texts += [ax.get_title() for ax in fig.axes]
            combined = "\n".join(texts)
            self.assertIn("pencil beam", combined)
            self.assertIn("dz == dxy", combined)
            self.assertIn("ni0 == ni", combined)
            self.assertIn("Dean", combined)
        finally:
            plt.close(fig)
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
