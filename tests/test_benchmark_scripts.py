from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


class BenchmarkScriptTests(unittest.TestCase):
    def test_nextflow_launcher_defaults_to_scout_fast_candidate(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = (
            repo_root
            / "tests"
            / "fixtures"
            / "benchmark_scripts"
            / "benchmark_ch00_fft_engines_nextflow.sh"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bin_dir = tmp / "bin"
            bin_dir.mkdir()
            capture = tmp / "nextflow_args.txt"
            for name, body in {
                "module": "#!/usr/bin/env bash\nexit 0\n",
                "nextflow": (
                    "#!/usr/bin/env bash\n"
                    "printf '%s\\n' \"$@\" > \"$NEXTFLOW_CAPTURE\"\n"
                ),
            }.items():
                path = bin_dir / name
                path.write_text(body)
                path.chmod(path.stat().st_mode | stat.S_IXUSR)

            env = os.environ.copy()
            for key in list(env):
                if key.startswith("BASH_FUNC_module") or key.startswith("BASH_FUNC_nextflow"):
                    env.pop(key)
            env.update(
                {
                    "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
                    "NEXTFLOW_CAPTURE": str(capture),
                    "NXF_HOME": str(tmp / "nxf_home"),
                    "OUT_DIR": str(tmp / "out"),
                    "WORK_DIR": str(tmp / "work"),
                }
            )

            result = subprocess.run(
                [str(script)],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
            )

            args = capture.read_text().splitlines()

        self.assertEqual(args[args.index("--queue") + 1], "GPUp40")
        self.assertEqual(args[args.index("--run_matlab") + 1], "0")
        self.assertEqual(args[args.index("--gpu_power_limit_watts") + 1], "0")
        self.assertEqual(args[args.index("--matlab_queue") + 1], "super")
        self.assertEqual(args[args.index("--cupy_engine") + 1], "cupyx")
        self.assertEqual(args[args.index("--cupy_fast_engine") + 1], "scout")
        self.assertEqual(args[args.index("--cupy_n_iters") + 1], "10")
        self.assertEqual(args[args.index("--cupy_fast_n_iters") + 1], "10")
        self.assertEqual(args[args.index("--cupy_blind_max_tiles") + 1], "16")
        self.assertEqual(args[args.index("--cupy_fast_blind_max_tiles") + 1], "16")
        self.assertEqual(args[args.index("--cupy_blind_z_slices") + 1], "128")
        self.assertEqual(args[args.index("--cupy_fast_blind_z_slices") + 1], "128")
        self.assertEqual(args[args.index("--cupy_tile_selection_strategy") + 1], "spatial_snr_v1")
        self.assertEqual(args[args.index("--cupy_fast_tile_selection_strategy") + 1], "spatial_snr_v1")
        self.assertEqual(args[args.index("--cupy_fast_coarse_region_limit") + 1], "8")
        self.assertEqual(args[args.index("--cupy_fast_adaptive_scout_iters") + 1], "2")
        self.assertEqual(args[args.index("--cupy_fast_adaptive_keep_tiles") + 1], "4")
        self.assertEqual(args[args.index("--matlab_workers") + 1], "24")
        self.assertEqual(args[args.index("--matlab_threads") + 1], "1")
        self.assertIn("--matlab_psf_script", args)
        self.assertIn("deconvolution-gpu/workflow/scripts/psf_estimation.py", args[args.index("--matlab_psf_script") + 1])


if __name__ == "__main__":
    unittest.main()
