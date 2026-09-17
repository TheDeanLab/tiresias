"""Regression tests for the Phase 8 resolution-vs-FOV example scripts.

Pins EX-04, EX-05, and EX-06 as permanent regression gates, not one-shot
migration checks. Three things are locked here. First, the packaging
contract (EX-06): both new scripts carry the same PEP 723 metadata block the
two shipped example scripts already use -- dependencies scoped to what they
actually import, local `tiresias` resolved via `[tool.uv.sources]`, and no
GPU array library in resolution. Second, comparability (roadmap Success
Criterion 2): the two scripts declare one identical NA sweep, one identical
fixed detection NA, and one identical window, so the duplicated constant
blocks the single-file PEP 723 philosophy requires cannot silently drift
apart. Third, the milestone's numerical claim: measured through the scripts'
own `run_sweep` wiring, the ASLM profile is flatter than the light-sheet
profile at the same illumination NA.
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
# data the simulation never produced. Roughly 30-40 percent of the shipped
# +-200.1 um window is legitimately unmeasurable at every NA (08-RESEARCH.md
# Pitfall 5); matplotlib already renders those NaN entries as honest breaks
# in the line.
_FORBIDDEN_GAP_FILL_NAMES = {
    "nan_to_num",
    "interp",
    "fillna",
    "dropna",
    "masked_invalid",
    "ffill",
    "bfill",
}


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
    """Roadmap Success Criterion 2 / D-01 / D-06: one shared axis, not two that can drift."""

    def test_both_scripts_share_one_sweep_and_one_window(self):
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

        for na in aslm.NA_SWEEP:
            self.assertGreaterEqual(na, 0.10)
            self.assertLessEqual(na, 0.45)

        self.assertEqual(aslm.COMMON["psf_size_z"], 1335)
        self.assertEqual(aslm.COMMON["dz"], 0.300)
        half_window = (aslm.COMMON["psf_size_z"] - 1) * aslm.COMMON["dz"] / 2.0
        self.assertAlmostEqual(half_window, 200.1, places=6)


class MilestoneClaimTests(unittest.TestCase):
    """The milestone's numerical claim and the unmeasurable-gap honesty rule."""

    def setUp(self) -> None:
        self.aslm = _load("examples/aslm_resolution_vs_fov.py")
        self.light_sheet = _load("examples/light_sheet_resolution_vs_fov.py")

    def test_aslm_profile_is_flatter_than_light_sheet_at_the_same_na(self):
        # psf_size_z patched to 61 so this test costs a fraction of a second
        # per NA instead of ~10s at the shipped window; the patch is
        # reverted by mock.patch.dict's context-manager exit. Only NA 0.40
        # and 0.45 are used: at this reduced window the +-9.15 um span is
        # far too narrow to reach the divergent far field at low NA, so the
        # ungated-vs-gated ratio collapses toward 1.0 there and the
        # assertion would be vacuous (measured ungated-to-gated relative
        # ratios at this reduced window: 1.000 at NA 0.10, 0.999 at 0.15,
        # 0.990 at 0.25, 0.737 at 0.35, 0.523 at 0.40, 0.618 at 0.45). The
        # full +-200.1 um window the scripts actually ship shows the effect
        # across the whole band; this test trades that coverage for speed
        # and so must stay at the NA values where the effect is
        # unambiguous at 61 slices. Asserted against the 0.75 factor rather
        # than the measured ratios themselves, so a psfmodels version bump
        # does not make this test brittle.
        for na in (0.40, 0.45):
            with self.subTest(na=na):
                with mock.patch.dict(self.aslm.COMMON, {"psf_size_z": 61}), mock.patch.dict(
                    self.light_sheet.COMMON, {"psf_size_z": 61}
                ):
                    # Call each script's OWN run_sweep, not the simulate
                    # functions directly -- the point is to prove the claim
                    # survives the scripts' own wiring, including the gate
                    # the ASLM script applies through
                    # measure_gated_beam_width_profile. The two return
                    # arities differ: ASLM yields a triple, light-sheet a
                    # quadruple.
                    _na_a, _centered_a, widths_a = self.aslm.run_sweep(nas=(na,))[0]
                    _na_l, _centered_l, widths_l, _rayleigh = self.light_sheet.run_sweep(
                        nas=(na,)
                    )[0]

                ratio_aslm = np.nanmax(widths_a) / np.nanmin(widths_a)
                ratio_light_sheet = np.nanmax(widths_l) / np.nanmin(widths_l)
                self.assertLess(ratio_aslm, 0.75 * ratio_light_sheet)

    def test_scripts_do_not_fill_the_unmeasurable_gaps(self):
        # Defends: roughly 30-40 percent of the shipped +-200.1 um window is
        # legitimately unmeasurable at every NA; matplotlib already renders
        # those NaN entries as honest breaks in the line, and filling them
        # would manufacture data the simulation never produced.
        for relative_path in _NEW_SCRIPTS:
            with self.subTest(script=relative_path):
                offenders = _called_names(_ROOT / relative_path) & _FORBIDDEN_GAP_FILL_NAMES
                self.assertEqual(
                    offenders,
                    set(),
                    f"{relative_path} calls gap-filling routine(s): {offenders}",
                )

    def test_a_rejected_rayleigh_point_does_not_stop_the_light_sheet_sweep(self):
        # This test must NEVER assert that the real locate_rayleigh_range
        # raises: at the shipped +-200.1 um window every NA from 0.10 to
        # 0.45 was measured to resolve cleanly, so the handler below is a
        # defensive branch that may never trigger in production -- roadmap
        # Success Criterion 5's wording is either/or. This test proves the
        # skip-and-continue handler itself works, via a synthetic forced
        # failure, not that the real function fires it.
        def _always_raise(*_args, **_kwargs):
            raise ValueError("forced for test: synthetic Rayleigh rejection")

        with mock.patch.dict(self.light_sheet.COMMON, {"psf_size_z": 61}), mock.patch.object(
            simulate, "locate_rayleigh_range", side_effect=_always_raise
        ):
            results = self.light_sheet.run_sweep(nas=(0.25, 0.35))

        self.assertEqual(len(results), 2)
        for _na, _centered_um, widths_um, rayleigh in results:
            self.assertTrue(np.isfinite(widths_um).any())
            self.assertIsNone(rayleigh)


if __name__ == "__main__":
    unittest.main()
