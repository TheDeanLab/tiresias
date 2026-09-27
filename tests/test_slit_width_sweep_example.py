"""Regression tests for examples/slit_width_sweep.py (halt route steps 1-2, plan 08.1-10).

Pins the corrected COMMON sampling (cubic voxels, matched immersion, odd cube
window -- halt route steps 1-2, resolving 08.1-06's open item OPEN-06a) and
the RA-3/RA-4/RA-5 sweep-trend bracketing, plus (Task 2) the pencil-beam,
sampling and Dean slit-mapping disclosures (D-03/D-19) rendered on the sweep
figure.
"""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_RELATIVE = "examples/slit_width_sweep.py"
_SCRIPT_PATH = _ROOT / _SCRIPT_RELATIVE

_IMPORT_ALLOWLIST = {"__future__", "pathlib", "matplotlib", "numpy", "tiresias"}


def _load(relative_path: str):
    """Load a script as a module without triggering its __main__ behavior.

    Safe: the script guards `main()` behind `if __name__ == "__main__":`, so
    `exec_module` only defines module-level names -- the same idiom
    tests/test_light_sheet_vs_aslm_example.py already uses.
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


class Pep723PackagingTests(unittest.TestCase):
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
        module = self.module
        common = module.COMMON
        self.assertEqual(common["dz"], common["dxy"])
        self.assertEqual(common["ni0"], common["ni"])
        self.assertEqual(common["psf_size_z"], common["psf_size_xy"])
        self.assertEqual(common["psf_size_z"] % 2, 1)
        self.assertLess(module.SLIT_WIDTHS[0], common["dz"])
        self.assertEqual(module.SLIT_WIDTHS[-1], module.PROPAGATION_EXTENT)


class SweepTrendTests(unittest.TestCase):
    """RA-3/RA-4/RA-5: monotonic non-decreasing, bracketed by the two measured limits."""

    def setUp(self) -> None:
        self.module = _load(_SCRIPT_RELATIVE)

    def test_sweep_is_monotonic_and_bracketed_by_the_two_limits(self):
        module = self.module
        common = module.COMMON
        dz = common["dz"]

        widths = (module.SLIT_WIDTHS[0], 2.0, 8.0, module.PROPAGATION_EXTENT)
        results = module.run_sweep(widths)
        fwhms = [fwhm for _, fwhm in results]

        self.assertTrue(
            all(fwhm is not None for fwhm in fwhms), f"unmeasurable FWHM in {results}"
        )

        for i in range(len(fwhms) - 1):
            self.assertGreaterEqual(fwhms[i + 1], fwhms[i] - 1e-6)

        waist_seed = module.generate_psf_seed(psf_mode="light_sheet", **common)
        waist = module.axial_fwhm(waist_seed, dz)
        self.assertIsNotNone(waist)
        self.assertEqual(fwhms[0], waist)

        swept_seed = module.generate_psf_seed(
            psf_mode="aslm", slit_width=module.SWEPT_AVERAGE_SLIT_WIDTH, **common
        )
        swept = module.axial_fwhm(swept_seed, dz)
        self.assertIsNotNone(swept)

        self.assertLessEqual(waist, fwhms[-1])
        self.assertLessEqual(fwhms[-1], swept + 1e-6)
        self.assertGreater(swept, waist)


class DisclosureTests(unittest.TestCase):
    """D-03/D-19: the figure states the pencil-beam, sampling and slit-mapping residuals."""

    def test_figure_discloses_the_pencil_beam_residual_sampling_and_slit_mapping(self):
        module = _load(_SCRIPT_RELATIVE)
        results = [(0.1, 1.0), (2.0, 1.1)]
        fig = module.build_sweep_figure(results, 1.0, 1.2)
        self.assertIsNotNone(fig)
        try:
            texts = [fig.get_suptitle() or ""]
            texts += [t.get_text() for t in fig.texts]
            texts += [ax.get_title() for ax in fig.axes]
            combined = "\n".join(texts)
            self.assertIn("pencil beam", combined)
            self.assertIn("dz == dxy", combined)
            self.assertIn("ni0 == ni", combined)
            self.assertIn("Dean", combined)
            self.assertIn("W = 2*xR", combined)
        finally:
            plt.close(fig)
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
