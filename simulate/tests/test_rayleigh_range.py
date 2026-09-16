from __future__ import annotations

import unittest

import numpy as np

from simulate import locate_rayleigh_range, measure_beam_width_profile


class RayleighRangeTests(unittest.TestCase):
    def test_locates_the_rayleigh_boundary_of_a_real_measured_beam_profile_end_to_end(self):
        positions_um, widths_um = measure_beam_width_profile(
            illumination_na=0.4,
            wavelength=0.561,
            ni=1.33,
            ns=1.33,
            dxy=0.108,
            dz=0.300,
            psf_size_z=61,
            psf_size_xy=128,
        )
        self.assertTrue(np.isfinite(widths_um).all())

        waist_position_um, left_position_um, right_position_um = locate_rayleigh_range(
            positions_um, widths_um
        )

        self.assertAlmostEqual(waist_position_um, 10.5, delta=0.05)
        self.assertAlmostEqual(left_position_um, 3.700491745153643, delta=0.05)
        self.assertAlmostEqual(right_position_um, 15.933018294998002, delta=0.05)
        self.assertAlmostEqual(
            right_position_um - left_position_um, 12.2325, delta=0.1
        )

        self.assertLess(left_position_um, waist_position_um)
        self.assertLess(waist_position_um, right_position_um)

        # Load-bearing assertion for ROADMAP Success Criterion 1: proves the
        # returned positions genuinely sit where the measured width has
        # grown sqrt(2)x from the waist, rather than merely being plausible
        # numbers.
        threshold_um = np.sqrt(2.0) * float(np.min(widths_um))
        self.assertAlmostEqual(
            float(np.interp(left_position_um, positions_um, widths_um)),
            threshold_um,
            delta=1e-9,
        )
        self.assertAlmostEqual(
            float(np.interp(right_position_um, positions_um, widths_um)),
            threshold_um,
            delta=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
