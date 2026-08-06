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


if __name__ == "__main__":
    unittest.main()
