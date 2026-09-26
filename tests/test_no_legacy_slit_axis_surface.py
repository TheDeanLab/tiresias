"""Prove the retired ASLM gate-axis override keyword and CLI flag never
reappear (D-16).

`generate_psf_seed()`'s explicit gate-axis override parameter and its
matching CLI flag were retired in plan 08.1-03: since plan 08.1-01, the
sweep-integrated ASLM slit gate always convolves along the pre-rotation
propagation axis (D-02), so a per-call axis override was a no-op kept only
for validation until this plan. The two tests below are permanent
regression gates, not one-shot migration checks, mirroring the Phase 7
tests/test_no_legacy_rotation_surface.py precedent (ROT-06): they discharge
D-16 by proving the removed parameter name and the removed CLI flag appear
nowhere in tracked source, and that both shipped CLI parsers and the seed
function signature agree the surface is gone.
"""

from __future__ import annotations

import inspect
import subprocess
import unittest
from pathlib import Path

from tiresias import cli
from tiresias.seeds import generate_psf_seed

# Mandatory needle construction: both search strings are built by
# concatenating fragments so that this file -- itself inside the scanned
# tree -- never contains either complete literal. Do NOT "simplify" this
# into a single string constant; doing so would make the assertions below
# permanently, silently unsatisfiable (this file would always match its own
# needle).
_NEEDLE = "slit" + "_axis"
_FLAG_NEEDLE = "--slit" + "-axis"

_SCANNED_SUFFIXES = {".py", ".md", ".json", ".txt", ".toml", ".cfg", ".yaml", ".yml"}
_SCANNED_PATHS = ("src", "tests", "docs", "examples", "README.md")

# tests/fixtures/legacy_rotation_baseline.json is a frozen 07-01 capture
# whose own _warning text says it MUST NEVER be regenerated, and that
# _warning text names the retired resolver this needle also matches. No
# production code reads its gate_axis key after this plan; editing the
# fixture to satisfy this scan would undermine the ROT-04 bit-identity
# guarantee the fixture exists to prove.
_ALLOWLISTED = frozenset({"tests/fixtures/legacy_rotation_baseline.json"})

# tests/test_seeds.py and tests/test_cli.py each carry exactly one D-16
# rejection test whose *name* must literally spell out the retired keyword
# and flag so `pytest -k` can select them (08.1-03-PLAN.md Task 2's
# acceptance criteria) -- an unavoidable, narrow exception to the
# "never spell it out" rule this scanner otherwise enforces everywhere
# else. Each of those two lines, and only those two lines, carries this
# marker so the scanner excuses just them, not the rest of either file.
_INTENTIONAL_TEST_NAME_MARKER = "retired-surface-test-name"


class RetiredSlitAxisSurfaceTests(unittest.TestCase):
    def test_no_retired_slit_axis_surface_in_tracked_sources(self):
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
            if relative_path in _ALLOWLISTED:
                continue
            path = root / relative_path
            if path.suffix not in _SCANNED_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if _INTENTIONAL_TEST_NAME_MARKER in line:
                    continue
                if _NEEDLE in line or _FLAG_NEEDLE in line:
                    offenders.append(f"{relative_path}:{line_number}: {line.strip()}")

        self.assertEqual(
            offenders,
            [],
            "Retired slit-axis surface reference(s) found:\n" + "\n".join(offenders),
        )

    def test_seed_signature_and_cli_parsers_lack_the_retired_surface(self):
        seed_params = set(inspect.signature(generate_psf_seed).parameters)
        self.assertNotIn(_NEEDLE, seed_params)

        minimal_argv = ["--image-path", "volume.tif", "--output-path", "out.tif"]
        for build_parser in (cli.build_estimate_psf_parser, cli.build_deconvolve_parser):
            namespace = build_parser().parse_args(minimal_argv)
            self.assertFalse(hasattr(namespace, _NEEDLE))


if __name__ == "__main__":
    unittest.main()
