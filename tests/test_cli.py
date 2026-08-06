from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class CliTests(unittest.TestCase):
    def test_estimate_psf_cli_loads_calibrated_seed_without_optical_arguments(self):
        from tiresias import cli

        seed = np.ones((5, 7, 7), dtype=np.float32)
        seed /= seed.sum()

        with (
            mock.patch.object(cli, "load_psf_seed", return_value=seed) as load_seed,
            mock.patch.object(cli, "generate_theoretical_psf") as generate,
            mock.patch.object(cli, "estimate_psf_from_chunks", return_value=seed) as estimate,
            mock.patch.object(cli, "imwrite"),
        ):
            cli.estimate_psf_main(
                [
                    "--image-path",
                    "volume.tif",
                    "--output-path",
                    "estimated_psf.tif",
                    "--psf-seed-path",
                    "calibrated_psf.tif",
                    "--psf-size-z",
                    "5",
                    "--psf-size-xy",
                    "7",
                ]
            )

        load_seed.assert_called_once_with(Path("calibrated_psf.tif"), (5, 7, 7))
        generate.assert_not_called()
        self.assertIs(estimate.call_args.kwargs["psf_seed"], seed)

    def test_estimate_psf_cli_generates_seed_and_writes_estimated_psf(self):
        from tiresias import cli

        seed = np.ones((3, 3, 3), dtype=np.float32) / 27.0
        estimated = np.ones((3, 3, 3), dtype=np.float32) / 27.0

        with (
            mock.patch.object(cli, "generate_theoretical_psf", return_value=seed) as generate,
            mock.patch.object(cli, "estimate_psf_from_chunks", return_value=estimated) as estimate,
            mock.patch.object(cli, "imwrite") as imwrite,
        ):
            cli.estimate_psf_main(
                [
                    "--image-path",
                    "volume.tif",
                    "--output-path",
                    "estimated_psf.tif",
                    "--dxy",
                    "0.108",
                    "--dz",
                    "0.3",
                    "--wavelength",
                    "0.561",
                    "--detection-na",
                    "1.0",
                    "--ni",
                    "1.33",
                    "--ns",
                    "1.33",
                    "--n-iters",
                    "4",
                ]
            )

        self.assertEqual(generate.call_args.kwargs["dxy"], 0.108)
        self.assertEqual(estimate.call_args.kwargs["image_path"], Path("volume.tif"))
        self.assertEqual(estimate.call_args.kwargs["n_iters"], 4)
        self.assertEqual(estimate.call_args.kwargs["cupy_fft_engine"], "scout")
        imwrite.assert_called_once_with(Path("estimated_psf.tif"), estimated)

    def test_estimate_psf_cli_accepts_scout_options(self):
        from tiresias import cli

        seed = np.ones((3, 3, 3), dtype=np.float32) / 27.0

        with (
            mock.patch.object(cli, "generate_theoretical_psf", return_value=seed),
            mock.patch.object(cli, "estimate_psf_from_chunks", return_value=seed) as estimate,
            mock.patch.object(cli, "imwrite"),
        ):
            cli.estimate_psf_main(
                [
                    "--image-path",
                    "volume.tif",
                    "--output-path",
                    "estimated_psf.tif",
                    "--dxy",
                    "0.108",
                    "--dz",
                    "0.3",
                    "--wavelength",
                    "0.561",
                    "--detection-na",
                    "1.0",
                    "--ni",
                    "1.33",
                    "--ns",
                    "1.33",
                    "--cupy-fft-engine",
                    "scout",
                    "--adaptive-scout-iters",
                    "2",
                    "--adaptive-keep-tiles",
                    "6",
                ]
            )

        self.assertEqual(estimate.call_args.kwargs["cupy_fft_engine"], "scout")
        self.assertEqual(estimate.call_args.kwargs["adaptive_scout_iters"], 2)
        self.assertEqual(estimate.call_args.kwargs["adaptive_keep_tiles"], 6)

    def test_deconvolve_cli_reads_inputs_and_writes_restored_tiff(self):
        from tiresias import cli

        image = np.ones((3, 5, 5), dtype=np.float32)
        psf = np.ones((3, 3, 3), dtype=np.float32) / 27.0
        restored = image.copy()

        with (
            mock.patch.object(cli, "imread", side_effect=[image, psf]) as imread,
            mock.patch.object(cli, "deconvolve_with_cucim", return_value=restored) as deconvolve,
            mock.patch.object(cli, "imwrite") as imwrite,
        ):
            cli.deconvolve_main(
                [
                    "--image-path",
                    "volume.tif",
                    "--psf-path",
                    "estimated_psf.tif",
                    "--output-path",
                    "restored.tif",
                    "--n-iters",
                    "8",
                    "--device-id",
                    "1",
                ]
            )

        self.assertEqual(imread.call_args_list[0].args[0], Path("volume.tif"))
        self.assertEqual(imread.call_args_list[1].args[0], Path("estimated_psf.tif"))
        deconvolve.assert_called_once_with(image, psf, 8, device_id=1)
        imwrite.assert_called_once_with(Path("restored.tif"), restored)


if __name__ == "__main__":
    unittest.main()
