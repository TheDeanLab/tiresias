from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tiresias import seeds


class SeedTests(unittest.TestCase):
    def test_load_psf_seed_center_crops_and_normalizes_tiff(self):
        source = np.arange(7 * 9 * 11, dtype=np.float32).reshape(7, 9, 11)

        with mock.patch.object(seeds, "imread", return_value=source) as imread:
            psf = seeds.load_psf_seed(Path("calibrated_psf.tif"), (5, 5, 5))

        expected = source[1:6, 2:7, 3:8]
        expected = expected / expected.sum(dtype=np.float64)
        self.assertEqual(psf.shape, (5, 5, 5))
        self.assertEqual(psf.dtype, np.float32)
        np.testing.assert_allclose(psf, expected, rtol=1e-6, atol=1e-8)
        imread.assert_called_once_with(Path("calibrated_psf.tif"))

    def test_load_psf_seed_rejects_zero_energy(self):
        with mock.patch.object(
            seeds,
            "imread",
            return_value=np.zeros((3, 3, 3), dtype=np.float32),
        ):
            with self.assertRaisesRegex(ValueError, "no positive finite energy"):
                seeds.load_psf_seed("empty.tif", (3, 3, 3))

    def test_resolve_dxy_accepts_direct_pixel_size(self):
        self.assertEqual(seeds.resolve_dxy(0.108, 6.5, 60), 0.108)

    def test_resolve_dxy_computes_camera_pixel_size_over_magnification(self):
        self.assertAlmostEqual(seeds.resolve_dxy(None, 6.5, 60), 6.5 / 60)

    def test_resolve_dxy_requires_positive_source(self):
        with self.assertRaisesRegex(ValueError, "dxy must be > 0"):
            seeds.resolve_dxy(None, None, None)

    def test_generate_theoretical_psf_normalizes_psfmodels_output(self):
        raw = np.ones((3, 5, 5), dtype=np.float32) * 2.0
        with mock.patch.object(seeds.pm, "make_psf", return_value=raw) as make_psf:
            psf = seeds.generate_theoretical_psf(
                detection_na=1.0,
                wavelength=0.561,
                ni=1.33,
                ns=1.33,
                dxy=0.108,
                dz=0.3,
                psf_size_z=3,
                psf_size_xy=5,
            )

        self.assertEqual(psf.shape, (3, 5, 5))
        self.assertTrue(np.isclose(psf.sum(dtype=np.float64), 1.0))
        self.assertEqual(make_psf.call_args.kwargs["NA"], 1.0)

    def test_light_sheet_seed_multiplies_detection_by_rotated_illumination(self):
        detection = np.ones((3, 5, 5), dtype=np.float32)
        illumination = np.zeros((3, 5, 5), dtype=np.float32)
        illumination[:, :, 2] = 1.0

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            side_effect=[detection, illumination],
        ):
            psf = seeds.generate_psf_seed(
                psf_mode="light_sheet",
                na=1.0,
                detection_na=1.0,
                illumination_na=0.2,
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
                psf_size_z=3,
                psf_size_xy=5,
                background=0.0,
                light_sheet_angle=90.0,
            )

        self.assertEqual(psf.shape, detection.shape)
        self.assertTrue(np.isclose(psf.sum(dtype=np.float64), 1.0))
        self.assertGreater(float(psf.sum(axis=(1, 2)).max()), 0.0)

    def test_single_seed_returns_normalized_detection_psf(self):
        detection = np.arange(3 * 5 * 5, dtype=np.float32).reshape(3, 5, 5)
        expected = detection / detection.sum(dtype=np.float64)

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            return_value=detection,
        ) as generate_theoretical_psf:
            psf = seeds.generate_psf_seed(
                psf_mode="single",
                na=1.0,
                detection_na=1.0,
                illumination_na=0.2,
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
                psf_size_z=3,
                psf_size_xy=5,
                background=0.0,
                light_sheet_angle=90.0,
            )

        self.assertEqual(psf.dtype, np.float32)
        np.testing.assert_allclose(psf, expected, rtol=1e-6, atol=1e-8)
        self.assertEqual(generate_theoretical_psf.call_count, 1)

    def test_generate_psf_seed_rejects_unsupported_psf_mode(self):
        detection = np.ones((3, 5, 5), dtype=np.float32)

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            return_value=detection,
        ):
            with self.assertRaisesRegex(ValueError, "(?i)nsupported psf_mode"):
                seeds.generate_psf_seed(
                    psf_mode="not_a_real_mode",
                    na=1.0,
                    detection_na=1.0,
                    illumination_na=0.2,
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
                    psf_size_z=3,
                    psf_size_xy=5,
                    background=0.0,
                    light_sheet_angle=90.0,
                )

    def test_light_sheet_seed_at_non_right_angle(self):
        detection = np.ones((5, 5, 5), dtype=np.float32)
        illumination = np.zeros((5, 5, 5), dtype=np.float32)
        illumination[:, :, 2] = 1.0

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            side_effect=[detection, illumination],
        ):
            psf = seeds.generate_psf_seed(
                psf_mode="light_sheet",
                na=1.0,
                detection_na=1.0,
                illumination_na=0.2,
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
                psf_size_z=5,
                psf_size_xy=5,
                background=0.0,
                light_sheet_angle=45.0,
            )

        self.assertEqual(psf.shape, illumination.shape)
        self.assertEqual(psf.dtype, np.float32)
        self.assertTrue(np.isclose(psf.sum(dtype=np.float64), 1.0))

        right_angle_rotated = seeds._center_crop_or_pad(
            np.rot90(illumination, k=1, axes=(0, 2)), illumination.shape
        )
        right_angle_reference = seeds.normalise_psf(detection * right_angle_rotated)
        self.assertFalse(np.allclose(psf, right_angle_reference))

    def test_aslm_seed_real_numeric_end_to_end(self):
        common_kwargs = dict(
            na=1.0,
            detection_na=1.0,
            illumination_na=0.2,
            wavelength=0.561,
            ni=1.33,
            ns=1.33,
            ni0=None,
            tg=None,
            tg0=None,
            ng=None,
            ng0=None,
            ti0=None,
            oversample_factor=1,
            psf_model="vectorial",
            dxy=0.108,
            dz=0.3,
            psf_size_z=15,
            psf_size_xy=15,
            background=0.0,
            light_sheet_angle=90.0,
        )

        light_sheet = seeds.generate_psf_seed(psf_mode="light_sheet", **common_kwargs)
        aslm = seeds.generate_psf_seed(
            psf_mode="aslm", slit_width=0.4, **common_kwargs
        )

        self.assertEqual(aslm.shape, (15, 15, 15))
        self.assertEqual(aslm.dtype, np.float32)
        self.assertLess(abs(float(aslm.sum(dtype=np.float64)) - 1.0), 1e-5)
        self.assertFalse(np.allclose(aslm, light_sheet))

        def axis0_variance(psf):
            marginal = psf.sum(axis=(1, 2), dtype=np.float64)
            idx = np.arange(marginal.shape[0], dtype=np.float64)
            centroid = float((marginal * idx).sum() / marginal.sum())
            return float((marginal * (idx - centroid) ** 2).sum() / marginal.sum())

        self.assertLess(axis0_variance(aslm), axis0_variance(light_sheet))

    def test_psfmodels_centers_beam_waist_at_geometric_midpoint(self):
        illumination = seeds.generate_theoretical_psf(
            na=0.2,
            detection_na=0.2,
            illumination_na=0.2,
            wavelength=0.561,
            ni=1.33,
            ns=1.33,
            oversample_factor=1,
            psf_model="vectorial",
            dxy=0.108,
            dz=0.3,
            psf_size_z=15,
            psf_size_xy=15,
            background=0.0,
        )

        def centroid(marginal):
            idx = np.arange(marginal.shape[0], dtype=np.float64)
            return float((marginal * idx).sum() / marginal.sum(dtype=np.float64))

        axis0_marginal = illumination.sum(axis=(1, 2), dtype=np.float64)
        axis2_marginal = illumination.sum(axis=(0, 1), dtype=np.float64)

        self.assertLess(abs(centroid(axis0_marginal) - 7.0), 0.5)
        self.assertLess(abs(centroid(axis2_marginal) - 7.0), 0.5)

    def test_aslm_seed_multiplies_detection_by_gated_illumination(self):
        detection = np.ones((5, 5, 5), dtype=np.float32)
        illumination = np.ones((5, 5, 5), dtype=np.float32)

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            side_effect=[detection, illumination],
        ):
            psf = seeds.generate_psf_seed(
                psf_mode="aslm",
                na=1.0,
                detection_na=1.0,
                illumination_na=0.2,
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
                psf_size_z=5,
                psf_size_xy=5,
                background=0.0,
                light_sheet_angle=90.0,
                slit_width=0.216,
            )

        sigma = (0.216 / 0.108) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        idx = np.arange(5, dtype=np.float64)
        window = np.exp(-0.5 * ((idx - 2.0) / sigma) ** 2).astype(np.float32)
        gated = illumination * window.reshape(1, 1, 5)
        expected = seeds.normalise_psf(
            detection * seeds.rotate_illumination_psf(gated, 90.0)
        )

        self.assertEqual(psf.shape, (5, 5, 5))
        np.testing.assert_allclose(psf, expected, rtol=1e-6, atol=1e-8)
        self.assertTrue(np.isclose(psf.sum(dtype=np.float64), 1.0))

    def test_aslm_slit_axis_override_forces_axis(self):
        detection = np.ones((9, 9, 9), dtype=np.float32)
        illumination = np.ones((9, 9, 9), dtype=np.float32)
        captured = {}

        def _record(illumination_arg, angle):
            captured["gated"] = np.array(illumination_arg, copy=True)
            captured["angle"] = angle
            return illumination_arg

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            side_effect=[detection, illumination],
        ), mock.patch.object(seeds, "rotate_illumination_psf", side_effect=_record):
            seeds.generate_psf_seed(
                psf_mode="aslm",
                na=1.0,
                detection_na=1.0,
                illumination_na=0.2,
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
                psf_size_z=9,
                psf_size_xy=9,
                background=0.0,
                light_sheet_angle=90.0,
                slit_width=0.216,
                slit_axis=0,
            )

        axis0_profile = captured["gated"].sum(axis=(1, 2))
        axis2_profile = captured["gated"].sum(axis=(0, 1))
        self.assertEqual(int(np.argmax(axis0_profile)), 4)
        self.assertLess(axis0_profile[0], axis0_profile[4])
        self.assertTrue(np.allclose(axis2_profile, axis2_profile[0]))

    def test_aslm_rejects_invalid_slit_axis(self):
        detection = np.ones((9, 9, 9), dtype=np.float32)

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            return_value=detection,
        ) as generate_theoretical_psf:
            with self.assertRaisesRegex(ValueError, "slit_axis must be 0 or 2"):
                seeds.generate_psf_seed(
                    psf_mode="aslm",
                    na=1.0,
                    detection_na=1.0,
                    illumination_na=0.2,
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
                    psf_size_z=9,
                    psf_size_xy=9,
                    background=0.0,
                    light_sheet_angle=90.0,
                    slit_width=0.216,
                    slit_axis=1,
                )

        generate_theoretical_psf.assert_not_called()


if __name__ == "__main__":
    unittest.main()
