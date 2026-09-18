"""Regression tests for examples/resolution_vs_fov_by_na.py.

A new file, not an extension of tests/test_resolution_vs_fov_examples.py:
that Phase 8 test file's `_NEW_SCRIPTS` tuple, its import allowlist, and its
run_sweep arity assumptions describe the two shipped scripts only, and
widening them would re-open a phase that is already verified and signed off
(D-04). This file pins four things about the new third script instead.
First, the packaging contract (D-06/D-07): the same PEP 723 metadata block
the two shipped scripts already use. Second, import and call discipline: no
import outside a narrow allowlist, and no call to either shipped
profile-measurement function or any gap-filling routine. Third, the pairing
contract (D-05): `pair_sweeps` reuses each script's own `run_sweep()` output
verbatim, keyed by NA, tolerating the two scripts' differing return arities,
and raising when the two sweeps disagree. Fourth, page structure (D-01,
D-02, D-03) and multi-page PDF output (D-07).
"""

from __future__ import annotations

import ast
import importlib.util
import tempfile
import types
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_RELATIVE_PATH = "examples/resolution_vs_fov_by_na.py"
_REFERENCE_SCRIPT = "examples/slit_width_sweep.py"

# Line 17 is the metadata block's closing `# ///` marker; lines 18 onward are
# per-script explanatory prose that is deliberately not compared here -- only
# the machine-meaningful metadata block must match.
_HEADER_LINES = 17

# Copied from tests/test_resolution_vs_fov_examples.py rather than imported:
# coupling this new gate to a shipped test's private surface would make the
# new gate brittle to unrelated changes in an already-signed-off test file.
_IMPORT_ALLOWLIST = {"__future__", "importlib", "pathlib", "types", "matplotlib", "numpy"}
_FORBIDDEN_GAP_FILL_NAMES = {
    "nan_to_num",
    "interp",
    "fillna",
    "dropna",
    "masked_invalid",
    "ffill",
    "bfill",
}
_FORBIDDEN_REDERIVATION_NAMES = {
    "measure_beam_width_profile",
    "measure_gated_beam_width_profile",
}


def _load(relative_path: str):
    """Load a script as a module without triggering its __main__ behavior.

    Safe: examples/resolution_vs_fov_by_na.py guards `main()` behind
    `if __name__ == "__main__":`, so exec_module only defines module-level
    names -- the same idiom tests/test_resolution_vs_fov_examples.py uses
    for the two shipped scripts.
    """
    path = _ROOT / relative_path
    spec = importlib.util.spec_from_file_location(
        relative_path.replace("/", "_").replace(".py", ""), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_toml(lines: list[str]) -> str:
    """Recover the PEP 723 TOML body from a script's leading comment block.

    Drops the opening/closing `# ///` marker lines, then strips a leading
    `#` plus at most one following space from each remaining line.
    """
    body = lines[1:-1]
    toml_lines = []
    for line in body:
        stripped = line[1:] if line.startswith("#") else line
        if stripped.startswith(" "):
            stripped = stripped[1:]
        toml_lines.append(stripped)
    return "\n".join(toml_lines)


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


def _make_light_sheet_stub(
    na_sweep: tuple[float, ...], common: dict, results: dict
) -> types.SimpleNamespace:
    """Build a stub light-sheet module whose run_sweep() returns quadruples."""

    def run_sweep(nas=na_sweep):
        return [(na, *results[na], None) for na in nas]

    return types.SimpleNamespace(
        NA_SWEEP=na_sweep,
        COMMON=common,
        DETECTION_NA=1.0,
        run_sweep=run_sweep,
    )


def _make_aslm_stub(na_sweep: tuple[float, ...], common: dict, results: dict) -> types.SimpleNamespace:
    """Build a stub ASLM module whose run_sweep() returns triples."""

    def run_sweep(nas=na_sweep):
        return [(na, *results[na]) for na in nas]

    return types.SimpleNamespace(
        NA_SWEEP=na_sweep,
        COMMON=common,
        DETECTION_NA=1.0,
        SLIT_WIDTH=2.0,
        run_sweep=run_sweep,
    )


class Pep723PackagingTests(unittest.TestCase):
    """D-06/D-07: the new script follows the shipped PEP 723 packaging contract."""

    def test_pep723_header_matches_the_shipped_reference(self):
        reference_lines = (_ROOT / _REFERENCE_SCRIPT).read_text(encoding="utf-8").splitlines()[
            :_HEADER_LINES
        ]
        script_lines = (_ROOT / _SCRIPT_RELATIVE_PATH).read_text(encoding="utf-8").splitlines()[
            :_HEADER_LINES
        ]
        self.assertEqual(script_lines, reference_lines)

    def test_pep723_metadata_parses_and_scopes_dependencies(self):
        import sys

        if sys.version_info < (3, 11):
            self.skipTest("tomllib arrived in Python 3.11")
        import tomllib

        lines = (_ROOT / _SCRIPT_RELATIVE_PATH).read_text(encoding="utf-8").splitlines()[
            :_HEADER_LINES
        ]
        metadata = tomllib.loads(_extract_toml(lines))

        self.assertEqual(metadata["requires-python"], ">=3.10,<3.13")
        self.assertEqual(
            metadata["dependencies"],
            [
                "numpy>=1.24",
                "scipy>=1.10",
                "tifffile>=2024.0",
                "psfmodels>=0.3",
                "matplotlib>=3.8",
                "tiresias",
            ],
        )
        self.assertEqual(
            metadata["tool"]["uv"]["sources"]["tiresias"],
            {"path": "..", "editable": True},
        )
        override = metadata["tool"]["uv"]["override-dependencies"]
        self.assertEqual(len(override), 1)
        self.assertIn("cupy-cuda11x", override[0])
        self.assertIn("python_version < '0'", override[0])

    def test_output_name_constant(self):
        module = _load(_SCRIPT_RELATIVE_PATH)
        self.assertEqual(module.OUTPUT_NAME, "resolution_vs_fov_by_na.pdf")


class ImportAndCallDisciplineTests(unittest.TestCase):
    """Import discipline and no re-derivation (D-05)."""

    def test_script_imports_only_the_expected_modules(self):
        roots = _imported_roots(_ROOT / _SCRIPT_RELATIVE_PATH)
        extra = roots - _IMPORT_ALLOWLIST
        self.assertEqual(extra, set(), f"imports outside the allowlist: {extra}")

    def test_script_never_re_derives_the_sweep(self):
        calls = _called_names(_ROOT / _SCRIPT_RELATIVE_PATH)
        offenders = calls & _FORBIDDEN_REDERIVATION_NAMES
        self.assertEqual(offenders, set(), f"calls profile-measurement function(s): {offenders}")
        self.assertIn("run_sweep", calls, "the script never calls run_sweep")

    def test_script_never_fills_gaps(self):
        calls = _called_names(_ROOT / _SCRIPT_RELATIVE_PATH)
        offenders = calls & _FORBIDDEN_GAP_FILL_NAMES
        self.assertEqual(offenders, set(), f"calls gap-filling routine(s): {offenders}")

    def test_importing_the_script_has_no_side_effects(self):
        output_dir = _ROOT / "examples" / "output"
        before = set(output_dir.glob("*")) if output_dir.exists() else set()
        _load(_SCRIPT_RELATIVE_PATH)
        after = set(output_dir.glob("*")) if output_dir.exists() else set()
        self.assertEqual(before, after, "importing the script wrote to examples/output/")
        self.assertEqual(plt.get_fignums(), [], "importing the script opened a figure")


class PairSweepsTests(unittest.TestCase):
    """D-05: pair_sweeps reuses each stub's own run_sweep() output verbatim."""

    def setUp(self):
        self.module = _load(_SCRIPT_RELATIVE_PATH)
        self.na_sweep = (0.10, 0.25, 0.45)
        self.common = {"psf_size_z": 61, "dz": 0.3}

    def test_pairs_one_entry_per_na_in_order(self):
        ls_results = {
            0.10: (np.array([-1.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0])),
            0.25: (np.array([-1.0, 0.0, 1.0]), np.array([4.0, 5.0, 6.0])),
            0.45: (np.array([-1.0, 0.0, 1.0]), np.array([7.0, 8.0, 9.0])),
        }
        aslm_results = {
            0.10: (np.array([-2.0, 0.0, 2.0]), np.array([9.0, 8.0, 7.0])),
            0.25: (np.array([-2.0, 0.0, 2.0]), np.array([6.0, 5.0, 4.0])),
            0.45: (np.array([-2.0, 0.0, 2.0]), np.array([3.0, 2.0, 1.0])),
        }
        light_sheet = _make_light_sheet_stub(self.na_sweep, self.common, ls_results)
        aslm = _make_aslm_stub(self.na_sweep, self.common, aslm_results)

        paired = self.module.pair_sweeps(light_sheet, aslm)

        self.assertEqual(len(paired), 3)
        for expected_na, entry in zip(self.na_sweep, paired):
            na, ls_centered, ls_widths, aslm_centered, aslm_widths = entry
            self.assertEqual(na, expected_na)
            np.testing.assert_array_equal(ls_centered, ls_results[na][0])
            np.testing.assert_array_equal(ls_widths, ls_results[na][1])
            np.testing.assert_array_equal(aslm_centered, aslm_results[na][0])
            np.testing.assert_array_equal(aslm_widths, aslm_results[na][1])

    def test_all_nan_widths_still_produce_an_entry(self):
        # D-02: nothing dropped, even when one curve is entirely unmeasurable.
        nan_arr = np.full(3, np.nan)
        ls_results = {na: (np.array([-1.0, 0.0, 1.0]), nan_arr.copy()) for na in self.na_sweep}
        aslm_results = {na: (np.array([-2.0, 0.0, 2.0]), np.array([1.0, 2.0, 3.0])) for na in self.na_sweep}
        light_sheet = _make_light_sheet_stub(self.na_sweep, self.common, ls_results)
        aslm = _make_aslm_stub(self.na_sweep, self.common, aslm_results)

        paired = self.module.pair_sweeps(light_sheet, aslm)

        self.assertEqual(len(paired), 3)
        for entry in paired:
            _na, _ls_centered, ls_widths, _aslm_centered, _aslm_widths = entry
            self.assertTrue(np.isnan(ls_widths).all())

    def test_raises_when_na_sweep_disagrees(self):
        common = self.common
        results = {na: (np.array([0.0]), np.array([1.0])) for na in (0.10, 0.25, 0.45, 0.50)}
        light_sheet = _make_light_sheet_stub((0.10, 0.25, 0.45), common, results)
        aslm = _make_aslm_stub((0.10, 0.25, 0.50), common, results)

        with self.assertRaises(ValueError):
            self.module.pair_sweeps(light_sheet, aslm)

    def test_raises_when_common_disagrees(self):
        results = {na: (np.array([0.0]), np.array([1.0])) for na in self.na_sweep}
        light_sheet = _make_light_sheet_stub(self.na_sweep, {"psf_size_z": 61, "dz": 0.3}, results)
        aslm = _make_aslm_stub(self.na_sweep, {"psf_size_z": 121, "dz": 0.3}, results)

        with self.assertRaises(ValueError):
            self.module.pair_sweeps(light_sheet, aslm)


class BuildNaPageTests(unittest.TestCase):
    """D-01/D-02/D-03: page structure and content."""

    def setUp(self):
        self.module = _load(_SCRIPT_RELATIVE_PATH)

    def test_page_labels_title_legend_and_xlim(self):
        fig = self.module.build_na_page(
            0.25,
            np.array([-1.0, 0.0, 1.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([-1.0, 0.0, 1.0]),
            np.array([3.0, 2.0, 1.0]),
            half_extent=200.1,
            detection_na=1.0,
            slit_width=2.0,
        )
        try:
            ax = fig.axes[0]
            self.assertEqual(ax.get_xlabel(), self.module.X_LABEL)
            self.assertEqual(ax.get_ylabel(), self.module.Y_LABEL)
            self.assertIn("0.25", ax.get_title())
            legend = ax.get_legend()
            self.assertIsNotNone(legend)
            self.assertEqual(len(legend.get_texts()), 2)
            xlim = ax.get_xlim()
            self.assertAlmostEqual(xlim[0], -200.1)
            self.assertAlmostEqual(xlim[1], 200.1)
        finally:
            plt.close(fig)

    def test_all_nan_curve_still_produces_a_figure_with_two_legend_entries(self):
        # D-02: nothing dropped, even at page-build time.
        fig = self.module.build_na_page(
            0.10,
            np.array([-1.0, 0.0, 1.0]),
            np.full(3, np.nan),
            np.array([-1.0, 0.0, 1.0]),
            np.array([3.0, 2.0, 1.0]),
            half_extent=200.1,
            detection_na=1.0,
            slit_width=2.0,
        )
        try:
            self.assertIsNotNone(fig)
            legend = fig.axes[0].get_legend()
            self.assertEqual(len(legend.get_texts()), 2)
        finally:
            plt.close(fig)


class WriteByNaPdfTests(unittest.TestCase):
    """D-01/D-07: multi-page PDF output."""

    def test_writes_five_pages_and_returns_five(self):
        module = _load(_SCRIPT_RELATIVE_PATH)
        figures = []
        for i in range(5):
            fig, ax = plt.subplots()
            ax.plot([0, 1], [i, i + 1])
            figures.append(fig)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "combined.pdf"
            page_count = module.write_by_na_pdf(figures, output_path)

            self.assertEqual(page_count, 5)
            self.assertTrue(output_path.is_file())
            self.assertEqual(output_path.read_bytes()[:5], b"%PDF-")


if __name__ == "__main__":
    unittest.main()
