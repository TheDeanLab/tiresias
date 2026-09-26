"""Regression tests for the Phase 8 / 8.1 resolution-vs-FOV example scripts.

Pins EX-04, EX-05, EX-06, and 8.1 D-09 as permanent regression gates, not
one-shot migration checks. Three things are locked here. First, the
packaging contract (EX-06): both new scripts carry the same PEP 723
metadata block the two shipped example scripts already use -- dependencies
scoped to what they actually import, local `tiresias` resolved via
`[tool.uv.sources]`, and no GPU array library in resolution. Second,
comparability (roadmap Success Criterion 2): the two scripts declare one
identical NA sweep, one identical fixed detection NA, one identical
COMMON block, and one identical FOV grid, so the duplicated constant blocks
the single-file PEP 723 philosophy requires cannot silently drift apart.
Third, the milestone's numerical claim: measured through the scripts' own
`run_sweep` wiring, the ASLM system-PSF axial FWHM curve is flatter than the
light-sheet system-PSF axial FWHM curve at the same illumination NA.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import simulate

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on the CI 3.10 leg
    tomllib = None

_ROOT = Path(__file__).resolve().parents[1]
_NEW_SCRIPTS = (
    "examples/light_sheet_resolution_vs_fov.py",
    "examples/aslm_resolution_vs_fov.py",
)
_REFERENCE_SCRIPT = "examples/slit_width_sweep.py"

# Line 17 is the metadata block's closing `# ///` marker; lines 18 onward are
# per-script explanatory prose (e.g. the requires-python rationale, the
# uv override-dependencies remediation story) that is deliberately not
# compared here -- only the machine-meaningful metadata block must match.
_HEADER_LINES = 17

# The GPU-free import guarantee, stated as an allowlist: any future import of
# a GPU array library (cupy, torch, jax, ...) fails this test automatically.
_IMPORT_ALLOWLIST = {"__future__", "pathlib", "sys", "numpy", "matplotlib", "simulate"}

# Gap-filling/interpolating/NaN-replacing routines that would manufacture
# data the simulation never produced. The system-PSF measurement is
# DOF-confined (8.1 D-09), so gaps are not expected across the shipped
# +-50 um FOV window; matplotlib still renders any NaN entry as an honest
# break in the line rather than silently smoothing over it.
_FORBIDDEN_GAP_FILL_NAMES = {
    "nan_to_num",
    "interp",
    "fillna",
    "dropna",
    "masked_invalid",
    "ffill",
    "bfill",
}

# 8.1 D-09: neither rewired script may call the retired self-gated function
# any more -- the system-PSF functions below replace it.
_FORBIDDEN_RETIRED_NAMES = {"measure_gated_beam_width_profile"}


def _load(relative_path: str):
    """Load a script as a module without triggering its __main__ behavior.

    Safe: both new scripts guard `main()` behind
    `if __name__ == "__main__":`, so exec_module only defines module-level
    names -- the same idiom tests/test_no_legacy_rotation_surface.py already
    uses for the two pre-existing example scripts.
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


class Pep723PackagingTests(unittest.TestCase):
    """EX-06: both new scripts follow the shipped PEP 723 packaging contract."""

    def test_pep723_header_matches_the_shipped_reference(self):
        reference_lines = (_ROOT / _REFERENCE_SCRIPT).read_text(encoding="utf-8").splitlines()[
            :_HEADER_LINES
        ]
        for relative_path in _NEW_SCRIPTS:
            with self.subTest(script=relative_path):
                script_lines = (_ROOT / relative_path).read_text(encoding="utf-8").splitlines()[
                    :_HEADER_LINES
                ]
                if script_lines != reference_lines:
                    diverging = [
                        f"line {index + 1}: {actual!r} != {expected!r}"
                        for index, (actual, expected) in enumerate(
                            zip(script_lines, reference_lines)
                        )
                        if actual != expected
                    ]
                    self.fail(
                        f"{relative_path}: PEP 723 header diverges from "
                        f"{_REFERENCE_SCRIPT}:\n" + "\n".join(diverging)
                    )

    @unittest.skipUnless(sys.version_info >= (3, 11), "tomllib arrived in Python 3.11")
    def test_pep723_metadata_parses_and_scopes_dependencies(self):
        for relative_path in _NEW_SCRIPTS:
            with self.subTest(script=relative_path):
                lines = (_ROOT / relative_path).read_text(encoding="utf-8").splitlines()[
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
                # Not a text search for "cupy-cuda11x" -- the override entry
                # legitimately names it; assert on the parsed structure (an
                # environment marker that can never be satisfied) instead.
                override = metadata["tool"]["uv"]["override-dependencies"]
                self.assertEqual(len(override), 1)
                self.assertIn("cupy-cuda11x", override[0])
                self.assertIn("python_version < '0'", override[0])

    def test_scripts_import_only_the_expected_modules(self):
        for relative_path in _NEW_SCRIPTS:
            with self.subTest(script=relative_path):
                roots = _imported_roots(_ROOT / relative_path)
                extra = roots - _IMPORT_ALLOWLIST
                self.assertEqual(
                    extra,
                    set(),
                    f"{relative_path} imports module(s) outside the allowlist: {extra}",
                )

    def test_importing_a_script_has_no_side_effects(self):
        output_dir = _ROOT / "examples" / "output"
        for relative_path in _NEW_SCRIPTS:
            with self.subTest(script=relative_path):
                before = set(output_dir.glob("*")) if output_dir.exists() else set()
                _load(relative_path)
                after = set(output_dir.glob("*")) if output_dir.exists() else set()
                self.assertEqual(before, after, f"{relative_path} wrote to examples/output/ on import")
                self.assertEqual(plt.get_fignums(), [], f"{relative_path} opened a figure on import")


class SharedSweepTests(unittest.TestCase):
    """Roadmap Success Criterion 2 / D-05 / D-06 / D-09: one shared axis, not two that can drift."""

    def test_both_scripts_share_one_sweep_one_window_and_one_fov(self):
        aslm = _load("examples/aslm_resolution_vs_fov.py")
        light_sheet = _load("examples/light_sheet_resolution_vs_fov.py")

        # Defends the single-file PEP 723 philosophy's cost: the constant
        # block must be duplicated rather than shared between scripts, so
        # this equality check is the only thing standing between that
        # duplication and a silent divergence that would make the two
        # figures incomparable without any error.
        self.assertEqual(aslm.NA_SWEEP, light_sheet.NA_SWEEP)
        self.assertEqual(aslm.DETECTION_NA, light_sheet.DETECTION_NA)
        self.assertEqual(aslm.COMMON, light_sheet.COMMON)
        self.assertEqual(aslm.FOV_POSITIONS_UM, light_sheet.FOV_POSITIONS_UM)

        self.assertIn(0.0, aslm.FOV_POSITIONS_UM)

        fov = aslm.FOV_POSITIONS_UM
        for i, value in enumerate(fov):
            self.assertEqual(value, -fov[-1 - i])

        self.assertEqual(aslm.COMMON["ni0"], aslm.COMMON["ni"])
        self.assertEqual(aslm.COMMON["dz"], aslm.COMMON["dxy"])

        for na in aslm.NA_SWEEP:
            self.assertGreaterEqual(na, 0.10)
            self.assertLessEqual(na, 0.45)


class MilestoneClaimTests(unittest.TestCase):
    """The milestone's numerical claim and the unmeasurable-gap honesty rule."""

    def setUp(self) -> None:
        self.aslm = _load("examples/aslm_resolution_vs_fov.py")
        self.light_sheet = _load("examples/light_sheet_resolution_vs_fov.py")

    def test_aslm_profile_is_flatter_than_light_sheet_at_the_same_na(self):
        # RAYLEIGH_WINDOW's psf_size_z is patched from 1335 to 61 so the
        # light-sheet script's Rayleigh annotation (a much larger window
        # than the main COMMON sweep) costs a fraction of a second instead
        # of ~10s per NA; the patch is reverted by mock.patch.dict's
        # context-manager exit. Only 4 positions_um are measured
        # (0.0/5.0/10.0/20.0 um), the same points the planner measured
        # during planning (0.579/0.672/0.602/1.022 um at NA=0.45, ratio
        # 1.77 for light_sheet vs about 1.0 for ASLM). Asserted against the
        # 0.75 factor rather than the measured ratios themselves, so a
        # psfmodels version bump does not make this test brittle.
        for na in (0.40, 0.45):
            with self.subTest(na=na):
                with mock.patch.dict(self.light_sheet.RAYLEIGH_WINDOW, {"psf_size_z": 61}):
                    # Call each script's OWN run_sweep, not the simulate
                    # functions directly -- the point is to prove the claim
                    # survives the scripts' own wiring. The two return
                    # arities differ: ASLM yields a triple, light-sheet a
                    # quadruple.
                    _na_a, _positions_a, widths_a = self.aslm.run_sweep(
                        nas=(na,), positions_um=(0.0, 5.0, 10.0, 20.0)
                    )[0]
                    _na_l, _positions_l, widths_l, _rayleigh = self.light_sheet.run_sweep(
                        nas=(na,), positions_um=(0.0, 5.0, 10.0, 20.0)
                    )[0]

                ratio_aslm = np.nanmax(widths_a) / np.nanmin(widths_a)
                ratio_light_sheet = np.nanmax(widths_l) / np.nanmin(widths_l)
                self.assertLess(ratio_aslm, 0.75 * ratio_light_sheet)

    def test_scripts_do_not_fill_the_unmeasurable_gaps(self):
        # Defends: the system-PSF measurement is DOF-confined, so gaps are
        # not expected across the shipped +-50 um FOV window; matplotlib
        # already renders any NaN entry as an honest break in the line, and
        # filling one would manufacture data the simulation never produced.
        for relative_path in _NEW_SCRIPTS:
            with self.subTest(script=relative_path):
                offenders = _called_names(_ROOT / relative_path) & _FORBIDDEN_GAP_FILL_NAMES
                self.assertEqual(
                    offenders,
                    set(),
                    f"{relative_path} calls gap-filling routine(s): {offenders}",
                )

    def test_scripts_measure_through_the_system_psf_functions(self):
        # 8.1 D-09: neither script may re-derive its curve through the
        # retired self-gated function; each must call its own new
        # system-PSF measurement function (and, for light_sheet, keep the
        # untouched illumination-only Rayleigh annotation, D-06).
        aslm_calls = _called_names(_ROOT / "examples/aslm_resolution_vs_fov.py")
        light_sheet_calls = _called_names(_ROOT / "examples/light_sheet_resolution_vs_fov.py")

        self.assertIn("measure_aslm_system_fwhm_profile", aslm_calls)
        self.assertIn("measure_light_sheet_system_fwhm_profile", light_sheet_calls)
        self.assertIn("measure_beam_width_profile", light_sheet_calls)

        self.assertEqual(aslm_calls & _FORBIDDEN_RETIRED_NAMES, set())
        self.assertEqual(light_sheet_calls & _FORBIDDEN_RETIRED_NAMES, set())

    def test_a_rejected_rayleigh_point_does_not_stop_the_light_sheet_sweep(self):
        # This test must NEVER assert that the real locate_rayleigh_range
        # raises: at the shipped Rayleigh window every NA from 0.10 to 0.45
        # was measured to resolve cleanly, so the handler below is a
        # defensive branch that may never trigger in production -- roadmap
        # Success Criterion 5's wording is either/or. This test proves the
        # skip-and-continue handler itself works, via a synthetic forced
        # failure, not that the real function fires it.
        def _always_raise(*_args, **_kwargs):
            raise ValueError("forced for test: synthetic Rayleigh rejection")

        with mock.patch.dict(
            self.light_sheet.RAYLEIGH_WINDOW, {"psf_size_z": 61}
        ), mock.patch.object(simulate, "locate_rayleigh_range", side_effect=_always_raise):
            results = self.light_sheet.run_sweep(nas=(0.25, 0.35), positions_um=(0.0,))

        self.assertEqual(len(results), 2)
        for _na, _centered_um, widths_um, rayleigh in results:
            self.assertTrue(np.isfinite(widths_um).any())
            self.assertIsNone(rayleigh)


if __name__ == "__main__":
    unittest.main()
