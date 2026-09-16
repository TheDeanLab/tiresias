"""Prove the removed single-angle rotation surface never reappears (ROT-06).

`generate_psf_seed()`'s single-angle rotation parameter and its matching CLI
flag were fully replaced by the two-parameter `(polar_deg, azimuthal_deg)`
spherical direction surface in plan 07-02, and every human-facing reference
(`docs/usage.md`, `README.md`, and both example scripts) was migrated in plan
07-04. The two tests below are permanent regression gates, not one-shot
migration checks: they discharge ROT-06 by proving the removed parameter name
and the removed CLI flag appear nowhere in tracked source, and that both
shipped example scripts still match the current `generate_psf_seed` signature.
"""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import unittest
from pathlib import Path

from tiresias.seeds import generate_psf_seed

# Mandatory needle construction: both search strings are built by
# concatenating fragments so that this file -- itself inside the scanned
# tree -- never contains either complete literal. Do NOT "simplify" this
# into a single string constant; doing so would make the assertions below
# permanently, silently unsatisfiable (this file would always match its own
# needle).
_NEEDLE = "light_sheet" + "_angle"
_FLAG_NEEDLE = "--light-sheet" + "-angle"

_SCANNED_SUFFIXES = {".py", ".md", ".json", ".txt", ".toml", ".cfg", ".yaml", ".yml"}
_SCANNED_PATHS = ("src", "tests", "docs", "examples", "README.md")

_EXAMPLE_SCRIPTS = (
    "examples/light_sheet_vs_aslm.py",
    "examples/slit_width_sweep.py",
)


class LegacyRotationSurfaceTests(unittest.TestCase):
    def test_no_legacy_rotation_parameter_in_tracked_sources(self):
        root = Path(__file__).resolve().parents[1]
        try:
            result = subprocess.run(
                ["git", "ls-files", "--", *_SCANNED_PATHS],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError:
            self.skipTest("git is unavailable in this environment; the scan needs a git checkout")
            return

        tracked_files = [line for line in result.stdout.splitlines() if line]
        offenders: list[str] = []
        for relative_path in tracked_files:
            path = root / relative_path
            if path.suffix not in _SCANNED_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if _NEEDLE in line or _FLAG_NEEDLE in line:
                    offenders.append(f"{relative_path}:{line_number}: {line.strip()}")

        self.assertEqual(
            offenders,
            [],
            "Legacy rotation surface reference(s) found:\n" + "\n".join(offenders),
        )

    def test_example_scripts_use_the_current_seed_signature(self):
        root = Path(__file__).resolve().parents[1]
        seed_params = set(inspect.signature(generate_psf_seed).parameters)

        for relative_path in _EXAMPLE_SCRIPTS:
            with self.subTest(script=relative_path):
                path = root / relative_path
                spec = importlib.util.spec_from_file_location(
                    relative_path.replace("/", "_").replace(".py", ""), path
                )
                module = importlib.util.module_from_spec(spec)
                # Safe: both example scripts guard main() behind
                # `if __name__ == "__main__"`, so exec_module has no
                # side effects beyond defining module-level names.
                spec.loader.exec_module(module)

                common_keys = set(module.COMMON)
                # Checks the call contract only, not the rendered figure --
                # this is deliberately cheap: it would have caught a stale
                # kwarg in an example script's COMMON dict without running
                # the full multi-minute PSF sweep either script performs.
                self.assertTrue(
                    common_keys.issubset(seed_params),
                    f"{relative_path}: COMMON has keys not in generate_psf_seed's "
                    f"signature: {common_keys - seed_params}",
                )
                self.assertEqual(module.COMMON["polar_deg"], 90.0)
                self.assertEqual(module.COMMON["azimuthal_deg"], 0.0)


if __name__ == "__main__":
    unittest.main()
