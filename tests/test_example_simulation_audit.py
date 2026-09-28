"""Permanent cross-example simulation audit (HALT-2-audit, HALT-4-examples, plan 08.1-13).

Discharges 08.1-EXECUTION-HALT.md's route step 2 ("audit all examples/ and
simulate/ callers") and step 4 at the example/figure layer, for every
git-tracked `examples/*.py` script -- not just the scripts individually
reworked by plans 08.1-09/10/12. A new or edited example that reintroduces
anisotropic sampling (`dz != dxy`), a mismatched immersion index
(`ni0 != ni`), a literal `ni0=None`, or drops the pencil-beam disclosure is
caught here rather than discovered later as a figure that quietly misleads
readers (RA-6-disclose, D-05, D-09).
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]

# The retired gate's public name is built by concatenation so this file
# never contains the literal -- mirrors the mandatory-needle-construction
# idiom in tests/test_no_legacy_rotation_surface.py.
_RETIRED_GATED_MEASUREMENT_NAME = "measure_gated" + "_beam_width_profile"

# Every git-tracked example script that defines a COMMON dict, named
# explicitly so a new COMMON-bearing script is audited by name rather than
# silently picked up (or silently missed) by this set. Add new scripts here
# deliberately when they gain a COMMON block.
_SCRIPTS_WITH_COMMON = {
    "aslm_axial_regimes.py",
    "aslm_resolution_vs_fov.py",
    "light_sheet_resolution_vs_fov.py",
    "light_sheet_vs_aslm.py",
    "slit_width_sweep.py",
}


def _tracked_example_scripts() -> list[str]:
    """Return git-tracked examples/*.py relative paths, sorted.

    Raises FileNotFoundError if git is unavailable -- callers convert that
    into self.skipTest so the audit degrades gracefully outside a git
    checkout rather than silently reporting nothing.
    """
    result = subprocess.run(
        ["git", "ls-files", "--", "examples"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(
        line for line in result.stdout.splitlines() if line and line.endswith(".py")
    )


def _load(relative_path: str, *, unique_name: str):
    """Load a script as a module without triggering its __main__ behavior.

    Safe: every tracked example script guards `main()` behind
    `if __name__ == "__main__":`, so exec_module only defines module-level
    names -- the same idiom tests/test_light_sheet_vs_aslm_example.py and
    tests/test_no_legacy_rotation_surface.py already use. `unique_name` keeps
    each module's sys.modules-free spec independent when loading several
    scripts in the same test.
    """
    path = _ROOT / relative_path
    spec = importlib.util.spec_from_file_location(unique_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strip_comments(source: str) -> str:
    """Drop every line whose first non-blank character is '#'."""
    kept = []
    for line in source.splitlines():
        if line.strip().startswith("#"):
            continue
        kept.append(line)
    return "\n".join(kept)


class CrossExampleSamplingAuditTests(unittest.TestCase):
    def test_every_example_common_uses_cubic_voxels_and_matched_immersion(self):
        try:
            tracked = _tracked_example_scripts()
        except FileNotFoundError:
            self.skipTest("git is unavailable in this environment; the scan needs a git checkout")
            return

        scripts_with_common: set[str] = set()
        for index, relative_path in enumerate(tracked):
            with self.subTest(script=relative_path):
                module = _load(relative_path, unique_name=f"audit_common_{index}")
                common = getattr(module, "COMMON", None)
                if common is None:
                    continue
                scripts_with_common.add(Path(relative_path).name)
                self.assertEqual(
                    common["dz"],
                    common["dxy"],
                    f"{relative_path}: COMMON['dz'] != COMMON['dxy']",
                )
                self.assertEqual(
                    common["ni0"],
                    common["ni"],
                    f"{relative_path}: COMMON['ni0'] != COMMON['ni']",
                )

        # examples/light_sheet_resolution_vs_fov.py's RAYLEIGH_WINDOW
        # (dz=0.300, illumination-only, unrotated -- D-06) is exempt by
        # construction: this audit reads only each module's COMMON dict, and
        # RAYLEIGH_WINDOW is a separate, smaller dict passed only to the
        # untouched simulate.measure_beam_width_profile annotation call.
        self.assertEqual(
            scripts_with_common,
            _SCRIPTS_WITH_COMMON,
            "Set of examples/*.py scripts defining COMMON changed -- add the "
            "new script to _SCRIPTS_WITH_COMMON deliberately after auditing "
            "its sampling.",
        )

    def test_no_example_passes_ni0_none(self):
        try:
            tracked = _tracked_example_scripts()
        except FileNotFoundError:
            self.skipTest("git is unavailable in this environment; the scan needs a git checkout")
            return

        for relative_path in tracked:
            with self.subTest(script=relative_path):
                path = _ROOT / relative_path
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                offenders: list[str] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        for keyword in node.keywords:
                            if (
                                keyword.arg == "ni0"
                                and isinstance(keyword.value, ast.Constant)
                                and keyword.value.value is None
                            ):
                                offenders.append(f"line {node.lineno}: ni0=None keyword call")
                    elif isinstance(node, ast.Dict):
                        for key, value in zip(node.keys, node.values):
                            if (
                                isinstance(key, ast.Constant)
                                and key.value == "ni0"
                                and isinstance(value, ast.Constant)
                                and value.value is None
                            ):
                                offenders.append(f"line {key.lineno}: \"ni0\": None dict entry")
                self.assertEqual(
                    offenders,
                    [],
                    f"{relative_path} passes ni0=None:\n" + "\n".join(offenders),
                )

    def test_the_two_fixed_na_scripts_share_one_common(self):
        light_sheet_vs_aslm = _load(
            "examples/light_sheet_vs_aslm.py", unique_name="audit_common_ls_vs_aslm"
        )
        slit_width_sweep = _load(
            "examples/slit_width_sweep.py", unique_name="audit_common_slit_sweep"
        )
        self.assertEqual(light_sheet_vs_aslm.COMMON, slit_width_sweep.COMMON)

    def test_every_simulating_example_discloses_the_pencil_beam_residual(self):
        try:
            tracked = _tracked_example_scripts()
        except FileNotFoundError:
            self.skipTest("git is unavailable in this environment; the scan needs a git checkout")
            return

        for relative_path in tracked:
            with self.subTest(script=relative_path):
                source = (_ROOT / relative_path).read_text(encoding="utf-8")
                if "generate_psf_seed(" not in source and "simulate.measure_" not in source:
                    continue
                stripped = _strip_comments(source)
                self.assertIn(
                    "pencil beam",
                    stripped,
                    f"{relative_path} simulates but never discloses the pencil-beam residual",
                )

    def test_no_example_names_the_retired_gated_measurement(self):
        try:
            tracked = _tracked_example_scripts()
        except FileNotFoundError:
            self.skipTest("git is unavailable in this environment; the scan needs a git checkout")
            return

        offenders = []
        for relative_path in tracked:
            source = (_ROOT / relative_path).read_text(encoding="utf-8")
            if _RETIRED_GATED_MEASUREMENT_NAME in source:
                offenders.append(relative_path)
        self.assertEqual(
            offenders,
            [],
            f"Retired gated-measurement name found in: {offenders}",
        )


class FovFigureDisclosureTests(unittest.TestCase):
    """Pencil-beam disclosure on the three FOV scripts lives in prose, not the figure (RA-6-disclose).

    User-directed deviation from the plan's original must_have (a rendered
    fig.text footnote on every FOV figure): the user chose "stop, redo
    without footnote" and applied the same publication rule plan 08.1-12
    already used for the other three example figures -- disclosures belong
    in the module docstring's "Notes / limitations" section and are printed
    to stdout by main(), never rendered into the figure itself. This class
    is therefore a regression/characterization layer over the corrected
    implementation (workflow.tdd_mode is not enabled in
    .planning/config.json), asserting the inverse of what the plan's Task 2
    <behavior> block originally specified: the figure stays footnote-free,
    while the constant and the docstring still carry the disclosure.
    """

    def test_fov_figures_carry_the_pencil_beam_footnote(self):
        with self.subTest(builder="aslm_resolution_vs_fov.build_sweep_figure"):
            module = _load(
                "examples/aslm_resolution_vs_fov.py", unique_name="audit_footnote_aslm_fov"
            )
            fig = module.build_sweep_figure(
                [(0.25, np.array([-1.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0]))]
            )
            try:
                self.assertIsNotNone(fig)
                combined = "\n".join(t.get_text() for t in fig.texts)
                self.assertNotIn("pencil beam", combined)
                self.assertIn("pencil beam", module.PENCIL_BEAM_NOTE)
                self.assertIn("pencil beam", module.__doc__)
            finally:
                plt.close(fig)

        with self.subTest(builder="light_sheet_resolution_vs_fov.build_sweep_figure"):
            module = _load(
                "examples/light_sheet_resolution_vs_fov.py",
                unique_name="audit_footnote_ls_fov",
            )
            fig = module.build_sweep_figure(
                [(0.25, np.array([-1.0, 0.0, 1.0]), np.array([1.2, 1.0, 1.2]), None)]
            )
            try:
                self.assertIsNotNone(fig)
                combined = "\n".join(t.get_text() for t in fig.texts)
                self.assertNotIn("pencil beam", combined)
                self.assertIn("pencil beam", module.PENCIL_BEAM_NOTE)
                self.assertIn("pencil beam", module.__doc__)
            finally:
                plt.close(fig)

        with self.subTest(builder="resolution_vs_fov_by_na.build_na_page"):
            module = _load(
                "examples/resolution_vs_fov_by_na.py", unique_name="audit_footnote_by_na"
            )
            fig = module.build_na_page(
                0.25,
                np.array([-1.0, 0.0, 1.0]),
                np.array([1.0, 2.0, 3.0]),
                np.array([-1.0, 0.0, 1.0]),
                np.array([3.0, 2.0, 1.0]),
                half_extent=50.0,
                detection_na=1.0,
                slit_width=2.0,
            )
            try:
                self.assertIsNotNone(fig)
                combined = "\n".join(t.get_text() for t in fig.texts)
                self.assertNotIn("pencil beam", combined)
                self.assertIn("pencil beam", module.PENCIL_BEAM_NOTE)
                self.assertIn("pencil beam", module.__doc__)
            finally:
                plt.close(fig)

        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
