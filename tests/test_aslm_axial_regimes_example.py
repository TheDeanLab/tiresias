"""Regression tests for examples/aslm_axial_regimes.py (D-11, plan 08.1-07).

Pins three things as permanent regression gates, not one-shot migration
checks. First, the packaging contract shared by every PEP 723 example script
in this repo (identical header, scoped imports, no import side effects).
Second, the D-11 regime-ordering claim: at illumination NA 0.10 the sheet is
thicker than the DOF and the measured system-PSF FWHMs sit closer to the DOF;
at NA 0.60 the sheet is thinner than the DOF and the measured FWHMs sit
closer to the sheet thickness. Third, the D-12 measurement discipline: every
reference line comes from `simulate.measure_*`, never a closed-form formula.
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
_SCRIPT_RELATIVE = "examples/aslm_axial_regimes.py"
_REFERENCE_SCRIPT = "examples/slit_width_sweep.py"
_SCRIPT_PATH = _ROOT / _SCRIPT_RELATIVE

# Line 17 is the metadata block's closing `# ///` marker; everything after it
# is per-script explanatory prose that is deliberately not compared here --
# only the machine-meaningful metadata block must match, per D-19.
_HEADER_LINES = 17

_IMPORT_ALLOWLIST = {"__future__", "pathlib", "sys", "numpy", "matplotlib", "simulate"}


def _load(relative_path: str):
    """Load a script as a module without triggering its __main__ behavior.

    Safe: the script guards `main()` behind `if __name__ == "__main__":`, so
    `exec_module` only defines module-level names -- the same idiom
    tests/test_resolution_vs_fov_examples.py already uses for the two
    system-PSF example scripts.
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


def _called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                names.add(func.attr)
            elif isinstance(func, ast.Name):
                names.add(func.id)
    return names


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


class RegimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load(_SCRIPT_RELATIVE)

    def test_run_sweep_shows_both_regimes_qualitatively(self):
        """RA-7 (gap closure, plan 08.1-12 Task 2): no exact-number regime
        tolerance is asserted (halt route step 5) -- only the ordering,
        limiting-reference and crossover-bracket claims that hold regardless
        of the exact measured values.
        """
        dof_um, results = self.module.run_sweep(nas=(0.10, 0.30, 0.40, 0.60))
        self.assertEqual(len(results), 4)

        sheets = [sheet for _na, sheet, _ls, _aslm in results]
        for earlier, later in zip(sheets, sheets[1:]):
            self.assertGreater(earlier, later, sheets)

        # D-11: sheet(0.10) > DOF > sheet(0.60).
        self.assertGreater(sheets[0], dof_um)
        self.assertGreater(dof_um, sheets[-1])

        for na, sheet_um, light_sheet_um, aslm_um in results:
            reference_min = min(sheet_um, dof_um)
            reference_max = max(sheet_um, dof_um)
            for mode, f in (("light_sheet", light_sheet_um), ("aslm", aslm_um)):
                with self.subTest(na=na, mode=mode):
                    self.assertLessEqual(f, reference_min + 1e-9)
                    self.assertLess(abs(f - reference_min), abs(f - reference_max))

        crossover = self.module.locate_crossover_na(dof_um, results)
        self.assertIsNotNone(crossover)
        self.assertGreater(crossover, 0.10)
        self.assertLess(crossover, 0.60)

    def test_locate_crossover_na_interpolates_the_measured_sign_change(self):
        """Synthetic sign-change data: the crossover is a linear interpolation
        between the last sheet-above-DOF row and the first sheet-at-or-below-
        DOF row, never a search over the raw NA grid.
        """
        dof_um = 1.0
        results = [
            (0.1, 2.0, 0.0, 0.0),
            (0.3, 1.5, 0.0, 0.0),
            (0.5, 0.5, 0.0, 0.0),
        ]
        crossover = self.module.locate_crossover_na(dof_um, results)
        self.assertAlmostEqual(crossover, 0.4, places=12)

        no_crossing_results = [
            (0.1, 2.0, 0.0, 0.0),
            (0.3, 1.8, 0.0, 0.0),
            (0.5, 1.5, 0.0, 0.0),
        ]
        self.assertIsNone(self.module.locate_crossover_na(dof_um, no_crossing_results))

    def test_figure_draws_measured_dof_and_sheet_reference_lines(self):
        dof_um = 1.0
        results = [
            (0.10, 3.0, 0.95, 0.95),
            (0.60, 0.5, 0.45, 0.45),
        ]
        fig = self.module.build_regime_figure(dof_um, results)

        # Gap closure (user display-fix round 1): the left panel's y axis
        # must be log-scaled so the sheet-thickness curve at low NA no
        # longer stretches the system-FWHM points into a flat line.
        self.assertEqual(fig.axes[0].get_yscale(), "log")

        abs_legend_texts = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        self.assertTrue(any("DOF" in text for text in abs_legend_texts), abs_legend_texts)
        self.assertTrue(
            any("sheet thickness" in text for text in abs_legend_texts), abs_legend_texts
        )

        # Gap closure (publication cleanup, plan 08.1-12): the measured
        # crossover value is still shown -- through this legend entry --
        # even though it moved out of the suptitle prose.
        self.assertTrue(
            any("crossover" in text for text in abs_legend_texts), abs_legend_texts
        )

        norm_legend_texts = [text.get_text() for text in fig.axes[1].get_legend().get_texts()]
        self.assertTrue(
            any("DOF-limited" in text for text in norm_legend_texts), norm_legend_texts
        )
        self.assertTrue(
            any("sheet-limited" in text for text in norm_legend_texts), norm_legend_texts
        )

        # Gap closure (publication cleanup, plan 08.1-12): the citation and
        # disclosure blurb are out of the figure text entirely -- moved to
        # the module docstring and printed to stdout by main() instead.
        figure_texts = [fig.get_suptitle() or ""]
        figure_texts += [t.get_text() for t in fig.texts]
        figure_texts += [ax.get_title() for ax in fig.axes]
        figure_texts += abs_legend_texts + norm_legend_texts
        combined_figure_text = "\n".join(figure_texts)
        self.assertNotIn("Residual", combined_figure_text)
        self.assertNotIn("Dean", combined_figure_text)
        self.assertNotIn("(deferred)", combined_figure_text)

        # The disclosure content itself must survive somewhere: the module
        # docstring's Notes / limitations section.
        docstring = self.module.__doc__ or ""
        self.assertIn("pencil beam", docstring)
        self.assertIn("Sampling", docstring)
        self.assertIn("crossover", docstring)

        plt.close(fig)
        self.assertEqual(plt.get_fignums(), [])

    def test_script_derives_references_by_measurement(self):
        called = _called_names(_SCRIPT_PATH)
        self.assertIn("measure_detection_dof", called)
        self.assertIn("measure_sheet_thickness", called)

        source = _SCRIPT_PATH.read_text(encoding="utf-8")
        stripped_lines = [
            line for line in source.splitlines() if not line.strip().startswith("#")
        ]
        stripped_source = "\n".join(stripped_lines)
        self.assertNotIn("sqrt", stripped_source)


if __name__ == "__main__":
    unittest.main()
