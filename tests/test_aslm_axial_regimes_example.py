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

    def test_run_sweep_reproduces_both_regimes(self):
        dof_um, results = self.module.run_sweep(nas=(0.10, 0.60))
        self.assertEqual(len(results), 2)
        (na_thick, sheet_thick, ls_thick, aslm_thick), (
            na_thin,
            sheet_thin,
            ls_thin,
            aslm_thin,
        ) = results
        self.assertEqual(na_thick, 0.10)
        self.assertEqual(na_thin, 0.60)

        # D-11: sheet(0.10) > DOF > sheet(0.60).
        self.assertGreater(sheet_thick, dof_um)
        self.assertGreater(dof_um, sheet_thin)

        # D-11: at NA 0.10 (sheet thicker than DOF), the measured system FWHM
        # sits closer to the DOF than to the sheet thickness.
        for f in (ls_thick, aslm_thick):
            self.assertLess(abs(f - dof_um), abs(f - sheet_thick))

        # D-11: at NA 0.60 (sheet thinner than DOF), the measured system FWHM
        # sits closer to the sheet thickness than to the DOF.
        for f in (ls_thin, aslm_thin):
            self.assertLess(abs(f - sheet_thin), abs(f - dof_um))

    def test_figure_draws_measured_dof_and_sheet_reference_lines(self):
        dof_um = 1.0
        results = [
            (0.10, 3.0, 1.05, 1.05),
            (0.60, 0.5, 0.55, 0.55),
        ]
        fig = self.module.build_regime_figure(dof_um, results)
        legend_texts = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        self.assertTrue(any("DOF" in text for text in legend_texts), legend_texts)
        self.assertTrue(any("sheet thickness" in text for text in legend_texts), legend_texts)

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
