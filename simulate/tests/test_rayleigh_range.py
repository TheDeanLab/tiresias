from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path

import numpy as np

from simulate import locate_rayleigh_range, measure_beam_width_profile, rayleigh_range

# D-07-style shared realistic optical parameters, mirroring how
# simulate/tests/test_beam_profile.py holds its own COMMON, so no test
# re-types them.
COMMON = dict(
    wavelength=0.561,
    ni=1.33,
    ns=1.33,
    dxy=0.108,
    dz=0.300,
    psf_size_z=61,
    psf_size_xy=128,
)


def _measure(illumination_na: float) -> tuple[np.ndarray, np.ndarray]:
    return measure_beam_width_profile(illumination_na=illumination_na, **COMMON)


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


    def test_returns_three_plain_floats_ordered_left_waist_right(self):
        # D-05 and FOV-01: full detail, not a 2- or 4-tuple -- RESEARCH Open
        # Question 2 was resolved against also returning the waist width.
        positions_um, widths_um = _measure(0.4)
        result = locate_rayleigh_range(positions_um, widths_um)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

        waist, left, right = result
        for name, value in (("waist", waist), ("left", left), ("right", right)):
            with self.subTest(position=name):
                # `isinstance` is deliberately NOT used: np.float64 passes
                # an isinstance check against float and would let a numpy
                # scalar leak into the contract Phase 8 consumes.
                self.assertIs(type(value), float)

        self.assertLess(left, waist)
        self.assertLess(waist, right)

    def test_waist_is_a_sampled_position_while_the_boundaries_are_interpolated(self):
        # D-06 and the precision edge probe.
        positions_um, widths_um = _measure(0.4)
        waist, left, right = locate_rayleigh_range(positions_um, widths_um)

        self.assertTrue(np.any(positions_um == waist))
        self.assertFalse(np.any(positions_um == left))
        self.assertFalse(np.any(positions_um == right))

        dz = COMMON["dz"]
        for name, value in (("left", left), ("right", right)):
            with self.subTest(position=name):
                nearest_multiple = round(value / dz) * dz
                self.assertFalse(math.isclose(value, nearest_multiple, abs_tol=1e-6))

        self.assertAlmostEqual(left, 3.700491745153643, delta=0.05)
        self.assertAlmostEqual(right, 15.933018294998002, delta=0.05)

    def test_lower_illumination_na_yields_a_wider_reported_fov(self):
        # ROADMAP Success Criterion 2 -- the headline claim of this
        # milestone, and the one genuinely physics-backed integration test
        # in this phase.
        #
        # NA values at or below 0.30 are deliberately excluded from this
        # sweep because at these array parameters their sqrt(2) crossing
        # lies outside the 61-slice window -- that is 06-02's ValueError
        # case, measured during planning, not a defect in this test.
        expected_extents = {0.45: 9.4081, 0.40: 12.2325, 0.35: 16.2432}
        extents = []
        for illumination_na in (0.45, 0.40, 0.35):
            with self.subTest(illumination_na=illumination_na):
                positions_um, widths_um = _measure(illumination_na)
                _waist, left, right = locate_rayleigh_range(positions_um, widths_um)
                extent = right - left
                self.assertTrue(np.isfinite(extent))
                self.assertAlmostEqual(
                    extent, expected_extents[illumination_na], delta=0.1
                )
                extents.append(extent)

        self.assertLess(extents[0], extents[1])
        self.assertLess(extents[1], extents[2])
        self.assertGreater(extents[2] - extents[0], 5.0)

    def test_the_two_sides_are_reported_separately_and_never_collapsed(self):
        # D-05's anti-collapse contract: the real measured profile is
        # genuinely asymmetric about its waist, and D-05 exists so that
        # asymmetry survives into Phase 8's plots rather than being
        # averaged away here. An implementation that computed one side and
        # mirrored it, or that returned a symmetric half-extent about the
        # waist, fails this test.
        positions_um, widths_um = _measure(0.4)
        waist, left, right = locate_rayleigh_range(positions_um, widths_um)

        left_half = waist - left
        right_half = right - waist
        self.assertAlmostEqual(left_half, 6.799, delta=0.05)
        self.assertAlmostEqual(right_half, 5.433, delta=0.05)
        self.assertGreater(abs(left_half - right_half), 1.0)

    def test_rejects_malformed_position_and_width_arrays_before_searching_for_a_waist(self):
        # ASVS V5: every malformed (positions_um, widths_um) pair must raise
        # before any waist search runs, each naming the offending
        # parameter(s).
        cases = [
            ("2-D positions", np.zeros((2, 2)), np.zeros((2, 2)), ["positions_um", "widths_um"]),
            (
                "length mismatch",
                np.array([0.0, 0.3, 0.6, 0.9, 1.2]),
                np.array([1.0, 1.0, 1.0, 1.0]),
                ["positions_um", "widths_um"],
            ),
            ("both empty", np.array([]), np.array([]), ["positions_um", "widths_um"]),
            (
                "repeated position",
                np.array([0.0, 0.3, 0.3, 0.9]),
                np.array([1.0, 0.9, 1.0, 1.5]),
                ["positions_um"],
            ),
            (
                "descending step",
                np.array([0.0, 0.6, 0.3]),
                np.array([1.0, 0.9, 1.5]),
                ["positions_um"],
            ),
            (
                "NaN in positions",
                np.array([0.0, 0.3, np.nan]),
                np.array([1.0, 0.9, 1.5]),
                ["positions_um"],
            ),
            (
                "inf in positions",
                np.array([0.0, 0.3, np.inf]),
                np.array([1.0, 0.9, 1.5]),
                ["positions_um"],
            ),
            (
                "zero width",
                np.array([0.0, 0.3, 0.6]),
                np.array([1.0, 0.0, 1.5]),
                ["widths_um"],
            ),
            (
                "negative width",
                np.array([0.0, 0.3, 0.6]),
                np.array([1.0, -0.5, 1.5]),
                ["widths_um"],
            ),
            (
                "inf width",
                np.array([0.0, 0.3, 0.6]),
                np.array([1.0, np.inf, 1.5]),
                ["widths_um"],
            ),
        ]
        for label, positions_um, widths_um, must_contain in cases:
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, "locate_rayleigh_range: "):
                    locate_rayleigh_range(positions_um, widths_um)
                try:
                    locate_rayleigh_range(positions_um, widths_um)
                except ValueError as exc:
                    text = str(exc)
                    for token in must_contain:
                        self.assertIn(token, text)

    def test_names_every_offending_array_in_one_error(self):
        positions_um = np.array([0.0, 0.6, 0.3])
        widths_um = np.full(3, np.nan)
        with self.assertRaises(ValueError) as ctx:
            locate_rayleigh_range(positions_um, widths_um)
        text = str(ctx.exception)
        self.assertIn("positions_um", text)
        self.assertIn("widths_um", text)

    def test_an_entirely_unmeasurable_profile_names_the_lateral_window_not_the_axial_array(self):
        # This is the one input shape D-09/D-10 do not walk through -- giving
        # it the array-too-short remedy would send the caller to lengthen an
        # axis that is not the problem.
        positions_um = np.arange(9) * 0.3
        widths_um = np.full(9, np.nan)
        with self.assertRaises(ValueError) as ctx:
            locate_rayleigh_range(positions_um, widths_um)
        text = str(ctx.exception)
        self.assertIn("widths_um", text)
        self.assertIn("psf_size_xy", text)
        self.assertNotIn("psf_size_z", text)
        self.assertNotIn("All-NaN slice encountered", text)

    def test_nan_among_valid_widths_is_accepted_not_rejected(self):
        # Pins the boundary between "malformed input" and "legitimate data
        # gap" so a later hardening pass cannot tighten the guard into
        # rejecting D-08's own case.
        positions_um = np.arange(7, dtype=float) * 0.3
        widths_um = np.array([2.0, 1.4, 1.0, np.nan, 1.05, 1.5, 2.0])
        waist, left, right = locate_rayleigh_range(positions_um, widths_um)
        self.assertLess(left, waist)
        self.assertLess(waist, right)

    def test_a_profile_too_short_to_contain_the_crossing_raises_instead_of_clamping(self):
        # ROADMAP Success Criterion 3. At illumination_na=0.30 the waist
        # width is 0.97032 um so the threshold is 1.37224 um, and the
        # largest width anywhere in the 61-slice window stays below it --
        # while NA=0.35 at the same parameters does cross, which is why
        # 06-01's NA sweep starts at 0.35.
        positions_um, widths_um = _measure(0.30)
        self.assertTrue(np.isfinite(widths_um).all())

        with self.assertRaises(ValueError) as ctx:
            locate_rayleigh_range(positions_um, widths_um)
        text = str(ctx.exception)
        for token in ("Z=0.0 um", "Z=18.0 um", "psf_size_z", "dz"):
            self.assertIn(token, text)

    def test_the_named_edge_position_is_rounded_for_display_only(self):
        # RESEARCH Pitfall 2.
        positions_um = np.arange(7, dtype=float) * 0.3
        widths_um = np.array([1.2, 1.15, 1.1, 1.0, 1.1, 1.15, 1.2])

        with self.assertRaises(ValueError) as ctx:
            locate_rayleigh_range(positions_um, widths_um)
        text = str(ctx.exception)
        self.assertIn("Z=1.8 um", text)
        self.assertNotIn("1.7999999999999998", text)

        # The rounding lives in the message, never in the data.
        self.assertEqual(positions_um[6], 6 * 0.3)

    def test_only_the_side_that_failed_is_named(self):
        # D-09's per-side reporting.
        positions_um = np.arange(7, dtype=float) * 0.3
        widths_um = np.array([1.2, 1.15, 1.1, 1.0, 1.1, 1.5, 2.0])

        with self.assertRaises(ValueError) as ctx:
            locate_rayleigh_range(positions_um, widths_um)
        text = str(ctx.exception)
        self.assertIn("Z=0.0 um", text)
        self.assertNotIn("Z=1.8 um", text)
        self.assertNotIn("1.4356", text)

    def test_nothing_is_returned_when_a_walk_reaches_an_edge(self):
        # Mirrors
        # test_unmeasurable_positions_are_never_clamped_or_extrapolated in
        # simulate/tests/test_beam_profile.py.
        positions_um, widths_um = _measure(0.30)
        with self.assertRaises(ValueError) as ctx:
            result = locate_rayleigh_range(positions_um, widths_um)
            self.fail(f"unexpected return value {result!r}")
        self.assertIs(type(ctx.exception), ValueError)

        synthetic_positions_um = np.arange(7, dtype=float) * 0.3
        synthetic_widths_um = np.array([1.2, 1.15, 1.1, 1.0, 1.1, 1.15, 1.2])
        with self.assertRaises(ValueError) as ctx:
            result = locate_rayleigh_range(synthetic_positions_um, synthetic_widths_um)
            self.fail(f"unexpected return value {result!r}")
        self.assertIs(type(ctx.exception), ValueError)

    def test_module_imports_are_confined_to_numpy(self):
        # The structural half of the no-fitted-model prohibition (D-07).
        # This mechanically forbids pulling in a root-finder, a smoothing
        # filter, a curve-fitting routine, or the PSF generator itself,
        # without relying on any text search -- proof that this function is
        # a pure transform of the two arrays it is handed.
        source = Path(rayleigh_range.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)

        modules: set[str] = set()
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                modules.add(node.module)
                for alias in node.names:
                    names.add(alias.name)

        self.assertTrue(modules <= {"__future__", "numpy"}, modules)
        self.assertTrue(names <= {"annotations"}, names)

    def test_boundary_search_contains_no_fitted_or_smoothed_beam_model(self):
        # The textual half of the no-fitted-model prohibition. Stripping
        # '#' comment lines before scanning lets this executor's own
        # explanatory comments name what was rejected without invalidating
        # this gate. The square-root token is deliberately NOT on this
        # list -- unlike beam_profile.py's equivalent guard, this module
        # legitimately needs it for the sqrt(2) threshold D-06 defines.
        source = Path(rayleigh_range.__file__).read_text(encoding="utf-8")
        stripped_lines = [
            line for line in source.splitlines() if not line.strip().startswith("#")
        ]
        stripped_source = "\n".join(stripped_lines)

        # <!-- planner-discipline-allow: curve_fit -->
        # <!-- planner-discipline-allow: polyfit -->
        # <!-- planner-discipline-allow: optimize -->
        # <!-- planner-discipline-allow: interp1d -->
        # <!-- planner-discipline-allow: savgol -->
        # <!-- planner-discipline-allow: peak_widths -->
        # <!-- planner-discipline-allow: gaussian_filter -->
        forbidden_tokens = (
            "curve_fit",
            "polyfit",
            "optimize",
            "interp1d",
            "savgol",
            "peak_widths",
            "gaussian_filter",
        )
        for token in forbidden_tokens:
            with self.subTest(token=token):
                self.assertNotIn(token, stripped_source)

        self.assertTrue(stripped_source.strip())
        self.assertIn("nanargmin", stripped_source)


if __name__ == "__main__":
    unittest.main()
