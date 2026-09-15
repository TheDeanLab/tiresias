from __future__ import annotations

import unittest

import numpy as np


class BeamProfileTests(unittest.TestCase):
    def test_public_api_measures_a_real_illumination_psf_end_to_end(self):
        from tiresias import measure_beam_width_profile

        kwargs = dict(
            wavelength=0.561,
            ni=1.33,
            ns=1.33,
            dxy=0.108,
            dz=0.300,
            psf_size_z=61,
            psf_size_xy=128,
        )
        centres = []
        for illumination_na in (0.2, 0.4, 0.6):
            positions_um, widths_um = measure_beam_width_profile(
                illumination_na=illumination_na, **kwargs
            )

            self.assertEqual(positions_um.shape, (61,))
            self.assertEqual(widths_um.shape, (61,))
            self.assertEqual(positions_um.dtype, np.float64)
            self.assertEqual(widths_um.dtype, np.float64)
            np.testing.assert_allclose(
                positions_um, np.arange(61) * 0.300, rtol=0, atol=1e-12
            )

            centre = float(widths_um[30])
            band = 0.51 * 0.561 / illumination_na
            self.assertTrue(
                0.8 * band <= centre <= 1.5 * band,
                f"illumination_na={illumination_na!r}: centre={centre!r} not in "
                f"[{0.8 * band!r}, {1.5 * band!r}]",
            )
            centres.append(centre)

        self.assertGreater(centres[0], centres[1])
        self.assertGreater(centres[1], centres[2])


if __name__ == "__main__":
    unittest.main()
