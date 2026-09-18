"""Regression tests for simulate/gated_beam_profile.py.

Pins the single riskiest design decision in this module -- which pre-rotation
axis the ASLM slit gate must narrow along for `measure_gated_beam_width_profile`
to have any measurable effect. Per D-04 and 08-RESEARCH.md Pitfall 1: gating
along the axis the production `_resolve_slit_axis` would pick for the
project's default propagation direction (axis 2, X) is a numerical no-op for
this measurement convention -- a regression that reintroduced that resolver
here would still run, still plot, and still print no error while proving
nothing. `test_gate_axis_is_load_bearing` is the mechanical check for exactly
that failure mode.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from simulate import gated_beam_profile
from simulate.beam_profile import measure_beam_width_profile
from simulate.gated_beam_profile import (
    GATE_AXIS,
    _measure_widths_from_array,
    measure_gated_beam_width_profile,
)
from tiresias.seeds import generate_theoretical_psf, _apply_aslm_slit_gate

COMMON = dict(
    wavelength=0.561,
    ni=1.33,
    ns=1.33,
    dxy=0.108,
    dz=0.300,
    psf_size_xy=128,
)


class GatedBeamProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Generate the shared illumination_na=0.4, psf_size_z=61 PSF once and
        # reuse it across the axis tests so the suite stays under a few
        # seconds, rather than re-generating a PSF per test.
        cls.psf_size_z = 61
        cls.illumination_na = 0.4
        cls.slit_width = 2.0
        cls.psf = generate_theoretical_psf(
            detection_na=cls.illumination_na,
            illumination_na=cls.illumination_na,
            psf_size_z=cls.psf_size_z,
            **COMMON,
        )
        cls.positions_um = np.arange(cls.psf_size_z, dtype=np.float64) * COMMON["dz"]

    def test_gate_axis_is_load_bearing(self):
        # D-04, 08-RESEARCH.md Pitfall 1: gating along the axis the
        # production _resolve_slit_axis would pick for the project's default
        # propagation direction (axis 2, X) -- or axis 0 (Z) -- is a
        # numerical no-op for this Z-position/Y-width measurement
        # convention. Only axis 1 (Y) has any effect. A regression that
        # reintroduced the production rotated-frame resolver here would
        # still run, still plot, and still print no error while proving
        # nothing; this test is the mechanical guard against exactly that.
        ungated_widths = _measure_widths_from_array(self.psf, COMMON["dxy"], self.positions_um)

        gated_axis0 = _apply_aslm_slit_gate(
            self.psf, 0, self.slit_width, COMMON["dxy"], COMMON["dz"]
        )
        widths_axis0 = _measure_widths_from_array(gated_axis0, COMMON["dxy"], self.positions_um)

        gated_axis2 = _apply_aslm_slit_gate(
            self.psf, 2, self.slit_width, COMMON["dxy"], COMMON["dz"]
        )
        widths_axis2 = _measure_widths_from_array(gated_axis2, COMMON["dxy"], self.positions_um)

        gated_axis1 = _apply_aslm_slit_gate(
            self.psf, 1, self.slit_width, COMMON["dxy"], COMMON["dz"]
        )
        widths_axis1 = _measure_widths_from_array(gated_axis1, COMMON["dxy"], self.positions_um)

        # Axis 0 and axis 2 agree with ungated only to about 1e-07 relative
        # (the gate multiplies a float32 array) -- do not tighten this
        # tolerance.
        np.testing.assert_allclose(
            widths_axis0, ungated_widths, rtol=1e-6, equal_nan=True
        )
        np.testing.assert_allclose(
            widths_axis2, ungated_widths, rtol=1e-6, equal_nan=True
        )

        # Axis 1 diverges by six orders of magnitude more than axis 0/2's
        # float32 rounding noise. A tight expected value would pin a
        # psfmodels-version-sensitive number; a coarse absolute-deviation
        # floor still discriminates a real regression.
        max_abs_deviation = float(
            np.nanmax(np.abs(widths_axis1 - ungated_widths))
        )
        self.assertGreater(max_abs_deviation, 1.0, max_abs_deviation)

        # The NaN pattern (which positions are unmeasurable) is identical in
        # all four -- the gate changes measured width, not which positions
        # are measurable at all.
        ungated_nan = np.isnan(ungated_widths)
        for name, widths in (
            ("axis0", widths_axis0),
            ("axis1", widths_axis1),
            ("axis2", widths_axis2),
        ):
            with self.subTest(variant=name):
                self.assertTrue(np.array_equal(np.isnan(widths), ungated_nan))

    def test_public_function_uses_the_load_bearing_axis(self):
        self.assertEqual(GATE_AXIS, 1)

        import inspect

        params = inspect.signature(measure_gated_beam_width_profile).parameters
        self.assertNotIn("axis", params)
        self.assertNotIn("slit_axis", params)
        self.assertFalse(any("axis" in name for name in params))

        gated_axis1 = _apply_aslm_slit_gate(
            self.psf, 1, self.slit_width, COMMON["dxy"], COMMON["dz"]
        )
        expected_widths = _measure_widths_from_array(
            gated_axis1, COMMON["dxy"], self.positions_um
        )

        _positions_um, widths_um = measure_gated_beam_width_profile(
            illumination_na=self.illumination_na,
            slit_width=self.slit_width,
            psf_size_z=self.psf_size_z,
            **COMMON,
        )
        np.testing.assert_allclose(widths_um, expected_widths, rtol=0, atol=1e-12, equal_nan=True)

    def test_gating_narrows_the_profile_relative_to_ungated(self):
        _positions_um, gated_widths = measure_gated_beam_width_profile(
            illumination_na=self.illumination_na,
            slit_width=self.slit_width,
            psf_size_z=self.psf_size_z,
            **COMMON,
        )
        _positions_um_u, ungated_widths = measure_beam_width_profile(
            illumination_na=self.illumination_na,
            psf_size_z=self.psf_size_z,
            **COMMON,
        )
        gated_ratio = float(np.nanmax(gated_widths) / np.nanmin(gated_widths))
        ungated_ratio = float(np.nanmax(ungated_widths) / np.nanmin(ungated_widths))
        # Measured: 6.060 against 11.578.
        self.assertLess(gated_ratio, 0.75 * ungated_ratio, (gated_ratio, ungated_ratio))

    def test_duplicated_loop_matches_phase_five(self):
        for psf_size_z in (21, 61):
            with self.subTest(psf_size_z=psf_size_z):
                positions_um = np.arange(psf_size_z, dtype=np.float64) * COMMON["dz"]
                psf = generate_theoretical_psf(
                    detection_na=0.3,
                    illumination_na=0.3,
                    psf_size_z=psf_size_z,
                    **COMMON,
                )
                duplicated_widths = _measure_widths_from_array(psf, COMMON["dxy"], positions_um)
                _positions_um_u, phase5_widths = measure_beam_width_profile(
                    illumination_na=0.3, psf_size_z=psf_size_z, **COMMON
                )
                np.testing.assert_allclose(
                    duplicated_widths, phase5_widths, rtol=0, atol=1e-12, equal_nan=True
                )

    def test_returns_the_parallel_array_contract(self):
        for psf_size_z in (3, 9, 21):
            with self.subTest(psf_size_z=psf_size_z):
                positions_um, widths_um = measure_gated_beam_width_profile(
                    illumination_na=0.3,
                    slit_width=self.slit_width,
                    psf_size_z=psf_size_z,
                    **COMMON,
                )
                self.assertEqual(positions_um.dtype, np.float64)
                self.assertEqual(widths_um.dtype, np.float64)
                self.assertEqual(positions_um.shape, (psf_size_z,))
                self.assertEqual(widths_um.shape, (psf_size_z,))
                np.testing.assert_allclose(
                    positions_um, np.arange(psf_size_z) * COMMON["dz"], rtol=0, atol=1e-12
                )
                self.assertEqual(positions_um[0], 0.0)

    def test_rejects_missing_or_non_positive_parameters(self):
        good = dict(
            illumination_na=0.4,
            slit_width=2.0,
            psf_size_z=5,
            psf_size_xy=32,
            wavelength=0.561,
            ni=1.33,
            ns=1.33,
            dxy=0.108,
            dz=0.300,
        )
        for bad_value in (0, -1.0, None):
            with self.subTest(slit_width=bad_value):
                kwargs = dict(good, slit_width=bad_value)
                with mock.patch.object(
                    gated_beam_profile, "generate_theoretical_psf"
                ) as mocked_generate:
                    with self.assertRaisesRegex(
                        ValueError, "measure_gated_beam_width_profile"
                    ) as ctx:
                        measure_gated_beam_width_profile(**kwargs)
                    self.assertIn("slit_width", str(ctx.exception))
                    mocked_generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
