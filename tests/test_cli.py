from __future__ import annotations

import unittest
import warnings
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
            mock.patch.object(cli, "generate_psf_seed") as generate,
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
            mock.patch.object(cli, "generate_psf_seed", return_value=seed) as generate,
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
            mock.patch.object(cli, "generate_psf_seed", return_value=seed),
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

    def test_estimate_psf_cli_aslm_mode_generates_gated_seed_end_to_end(self):
        from tiresias import cli

        def _capture_seed(*, psf_seed, **kwargs):
            del kwargs
            return psf_seed

        common_argv = [
            "--image-path",
            "volume.tif",
            "--output-path",
            "estimated_psf.tif",
            "--detection-na",
            "1.0",
            "--illumination-na",
            "0.2",
            "--wavelength",
            "0.561",
            "--ni",
            "1.33",
            "--ns",
            "1.33",
            "--dxy",
            "0.108",
            "--dz",
            "0.3",
            "--oversample-factor",
            "1",
            "--psf-size-z",
            "15",
            "--psf-size-xy",
            "15",
        ]

        seeds = {}
        with mock.patch.object(cli, "imwrite"):
            for mode_args, key in (
                (["--psf-mode", "light_sheet"], "light_sheet"),
                (["--psf-mode", "aslm", "--slit-width", "0.4"], "aslm"),
            ):
                with mock.patch.object(
                    cli, "estimate_psf_from_chunks", side_effect=_capture_seed
                ) as estimate:
                    cli.estimate_psf_main(common_argv + mode_args)
                    seeds[key] = estimate.call_args.kwargs["psf_seed"]

        light_sheet = seeds["light_sheet"]
        aslm = seeds["aslm"]

        self.assertEqual(aslm.shape, (15, 15, 15))
        self.assertEqual(aslm.dtype, np.float32)
        self.assertAlmostEqual(float(aslm.sum(dtype=np.float64)), 1.0, delta=1e-5)
        self.assertFalse(np.allclose(aslm, light_sheet))

        def _axis0_variance(psf):
            marginal = psf.sum(axis=(1, 2), dtype=np.float64)
            idx = np.arange(psf.shape[0], dtype=np.float64)
            centroid = float((marginal * idx).sum() / marginal.sum())
            return float((marginal * (idx - centroid) ** 2).sum() / marginal.sum())

        self.assertLess(_axis0_variance(aslm), _axis0_variance(light_sheet))

    def test_estimate_psf_cli_passes_aslm_flags_through_to_generate_psf_seed(self):
        from tiresias import cli

        seed = np.ones((3, 3, 3), dtype=np.float32) / 27.0

        with (
            mock.patch.object(cli, "generate_psf_seed", return_value=seed) as generate,
            mock.patch.object(cli, "estimate_psf_from_chunks", return_value=seed),
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
                    "--psf-mode",
                    "aslm",
                    "--slit-width-px",
                    "4",
                    "--slit-axis",
                    "2",
                    "--light-sheet-angle",
                    "75.0",
                ]
            )

        self.assertEqual(generate.call_args.kwargs["psf_mode"], "aslm")
        self.assertEqual(generate.call_args.kwargs["slit_width_px"], 4)
        self.assertIsNone(generate.call_args.kwargs["slit_width"])
        self.assertEqual(generate.call_args.kwargs["slit_axis"], 2)
        self.assertEqual(generate.call_args.kwargs["light_sheet_angle"], 75.0)

    def test_estimate_psf_cli_defaults_preserve_single_mode_parameters(self):
        from tiresias import cli

        seed = np.ones((3, 3, 3), dtype=np.float32) / 27.0

        with (
            mock.patch.object(cli, "generate_psf_seed", return_value=seed) as generate,
            mock.patch.object(cli, "estimate_psf_from_chunks", return_value=seed),
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
                ]
            )

        self.assertEqual(generate.call_args.kwargs["psf_mode"], "single")
        self.assertEqual(generate.call_args.kwargs["light_sheet_angle"], 90.0)
        self.assertIsNone(generate.call_args.kwargs["slit_width"])
        self.assertIsNone(generate.call_args.kwargs["slit_width_px"])
        self.assertIsNone(generate.call_args.kwargs["slit_axis"])

        with (
            mock.patch.object(cli, "generate_psf_seed", return_value=seed) as generate,
            mock.patch.object(cli, "estimate_psf_from_chunks", return_value=seed),
            mock.patch.object(cli, "imwrite"),
        ):
            cli.estimate_psf_main(
                [
                    "--image-path",
                    "volume.tif",
                    "--output-path",
                    "estimated_psf.tif",
                    "--camera-pixel-size",
                    "6.5",
                    "--magnification",
                    "60",
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
                ]
            )

        self.assertEqual(generate.call_args.kwargs["dxy"], 6.5 / 60)

    def test_cli_module_exposes_no_direct_theoretical_psf_bypass(self):
        from tiresias import cli

        self.assertFalse(hasattr(cli, "generate_theoretical_psf"))

    def test_estimate_psf_cli_default_mode_output_is_bit_identical_to_pre_refactor(self):
        from tiresias import cli
        from tiresias.seeds import generate_theoretical_psf

        expected = generate_theoretical_psf(
            na=None,
            detection_na=1.0,
            illumination_na=None,
            wavelength=0.561,
            ni=1.33,
            ns=1.33,
            ni0=None,
            tg=None,
            tg0=None,
            ng=None,
            ng0=None,
            ti0=None,
            oversample_factor=3,
            psf_model="vectorial",
            dxy=0.108,
            dz=0.3,
            psf_size_z=61,
            psf_size_xy=128,
            background=0.0,
        )

        with (
            mock.patch.object(cli, "estimate_psf_from_chunks") as estimate,
            mock.patch.object(cli, "imwrite"),
        ):
            cli.estimate_psf_main(
                [
                    "--image-path",
                    "volume.tif",
                    "--output-path",
                    "estimated_psf.tif",
                    "--detection-na",
                    "1.0",
                    "--wavelength",
                    "0.561",
                    "--ni",
                    "1.33",
                    "--ns",
                    "1.33",
                    "--dxy",
                    "0.108",
                    "--dz",
                    "0.3",
                ]
            )
            actual = estimate.call_args.kwargs["psf_seed"]

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, (61, 128, 128))

    def test_deconvolve_cli_generates_aslm_seed_when_psf_path_omitted(self):
        from tiresias import cli

        image = np.ones((3, 5, 5), dtype=np.float32)

        common_argv = [
            "--image-path",
            "volume.tif",
            "--output-path",
            "restored.tif",
            "--detection-na",
            "1.0",
            "--illumination-na",
            "0.2",
            "--wavelength",
            "0.561",
            "--ni",
            "1.33",
            "--ns",
            "1.33",
            "--dxy",
            "0.108",
            "--dz",
            "0.3",
            "--oversample-factor",
            "1",
            "--psf-size-z",
            "15",
            "--psf-size-xy",
            "15",
            "--n-iters",
            "8",
            "--device-id",
            "1",
        ]

        seeds = {}
        for mode_args, key in (
            (["--psf-mode", "light_sheet"], "light_sheet"),
            (["--psf-mode", "aslm", "--slit-width", "0.4"], "aslm"),
        ):
            with (
                mock.patch.object(cli, "imread", return_value=image) as imread,
                mock.patch.object(
                    cli, "deconvolve_with_cupy", side_effect=lambda image, psf, n_iters, **kwargs: psf
                ) as deconvolve,
                mock.patch.object(cli, "imwrite"),
            ):
                cli.deconvolve_main(common_argv + mode_args)
                self.assertEqual(imread.call_count, 1)
                self.assertEqual(imread.call_args.args[0], Path("volume.tif"))
                self.assertEqual(deconvolve.call_args.args[2], 8)
                self.assertEqual(deconvolve.call_args.kwargs["device_id"], 1)
                seeds[key] = deconvolve.call_args.args[1]

        light_sheet = seeds["light_sheet"]
        aslm = seeds["aslm"]

        self.assertEqual(aslm.shape, (15, 15, 15))
        self.assertEqual(aslm.dtype, np.float32)
        self.assertAlmostEqual(float(aslm.sum(dtype=np.float64)), 1.0, delta=1e-5)
        self.assertFalse(np.allclose(aslm, light_sheet))

        def _axis0_variance(psf):
            marginal = psf.sum(axis=(1, 2), dtype=np.float64)
            idx = np.arange(psf.shape[0], dtype=np.float64)
            centroid = float((marginal * idx).sum() / marginal.sum())
            return float((marginal * (idx - centroid) ** 2).sum() / marginal.sum())

        self.assertLess(_axis0_variance(aslm), _axis0_variance(light_sheet))

    def test_deconvolve_cli_reads_inputs_and_writes_restored_tiff(self):
        from tiresias import cli

        image = np.ones((3, 5, 5), dtype=np.float32)
        psf = np.ones((3, 3, 3), dtype=np.float32) / 27.0
        restored = image.copy()

        with (
            mock.patch.object(cli, "imread", side_effect=[image, psf]) as imread,
            mock.patch.object(cli, "deconvolve_with_cupy", return_value=restored) as deconvolve,
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

    def test_deconvolve_cli_psf_path_takes_precedence_over_generation_flags(self):
        from tiresias import cli

        image = np.ones((3, 5, 5), dtype=np.float32)
        psf = np.ones((3, 3, 3), dtype=np.float32) / 27.0
        restored = image.copy()

        with (
            mock.patch.object(cli, "imread", side_effect=[image, psf]) as imread,
            mock.patch.object(cli, "generate_psf_seed") as generate,
            mock.patch.object(cli, "deconvolve_with_cupy", return_value=restored) as deconvolve,
            mock.patch.object(cli, "imwrite"),
            warnings.catch_warnings(record=True) as recorded_warnings,
        ):
            warnings.simplefilter("always")
            cli.deconvolve_main(
                [
                    "--image-path",
                    "volume.tif",
                    "--psf-path",
                    "estimated_psf.tif",
                    "--output-path",
                    "restored.tif",
                    "--psf-mode",
                    "aslm",
                    "--slit-width",
                    "0.4",
                    "--detection-na",
                    "1.0",
                    "--wavelength",
                    "0.561",
                    "--ni",
                    "1.33",
                    "--ns",
                    "1.33",
                    "--dxy",
                    "0.108",
                    "--dz",
                    "0.3",
                    "--n-iters",
                    "8",
                    "--device-id",
                    "1",
                ]
            )

        generate.assert_not_called()
        self.assertEqual(imread.call_args_list[1].args[0], Path("estimated_psf.tif"))
        deconvolve.assert_called_once_with(image, psf, 8, device_id=1)
        self.assertEqual(recorded_warnings, [])

    def test_both_cli_parsers_expose_the_aslm_flag_surface(self):
        from tiresias import cli

        estimate_ns = cli.build_estimate_psf_parser().parse_args(
            ["--image-path", "volume.tif", "--output-path", "out.tif"]
        )
        deconvolve_ns = cli.build_deconvolve_parser().parse_args(
            ["--image-path", "volume.tif", "--output-path", "out.tif"]
        )

        for attr in ("psf_mode", "slit_width", "slit_width_px", "slit_axis", "light_sheet_angle"):
            with self.subTest(attr=attr):
                self.assertTrue(hasattr(estimate_ns, attr))
                self.assertTrue(hasattr(deconvolve_ns, attr))
                self.assertEqual(getattr(estimate_ns, attr), getattr(deconvolve_ns, attr))


if __name__ == "__main__":
    unittest.main()
