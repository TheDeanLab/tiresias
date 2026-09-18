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


def _synthetic_quadratic_profile(step: float) -> tuple[np.ndarray, np.ndarray]:
    """A hand-authored convex width curve, not a measured beam profile.

    Waist sits at exactly z = 10.0 um with width exactly 1.0 um, chosen so
    the sqrt(2) threshold is exactly `np.sqrt(2.0)` and the analytic
    crossings are `10.0 -/+ 2 * np.sqrt(np.sqrt(2.0) - 1.0)` =
    8.712811... and 11.287189... um. Per RESEARCH Pitfall 5:
    `locate_rayleigh_range`'s inputs are already two arrays, so its
    interpolation behavior can be proven directly and deterministically on
    this synthetic profile without paying for a PSF generation per case --
    the one genuinely physics-backed test in this phase is 06-01's
    NA-vs-FOV sweep, which needs real optics.
    """
    positions_um = np.arange(round(20.0 / step) + 1, dtype=np.float64) * step
    widths_um = 1.0 + 0.25 * (positions_um - 10.0) ** 2
    return positions_um, widths_um


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

    def test_a_waist_at_the_first_index_fails_that_side_without_an_index_error(self):
        # RESEARCH Pitfall 4. Nothing in D-06 guarantees an interior waist,
        # so the walk toward the near edge examines exactly one sample; the
        # danger is an IndexError from indexing an empty walk rather than a
        # wrong number.
        positions_um = np.arange(21, dtype=np.float64) * 0.5
        widths_um = 1.0 + 0.25 * positions_um**2

        with self.assertRaises(ValueError) as ctx:
            locate_rayleigh_range(positions_um, widths_um)
        self.assertIs(type(ctx.exception), ValueError)
        text = str(ctx.exception)
        self.assertIn("Z=0.0 um", text)
        self.assertIn("decreasing Z", text)
        self.assertNotIn("Z=10.0 um", text)

    def test_a_waist_at_the_last_index_fails_that_side_without_an_index_error(self):
        # The mirror -- both directions are tested because the two walks are
        # separate code paths and a one-sided fix would pass the first test
        # alone.
        positions_um = np.arange(21, dtype=np.float64) * 0.5
        widths_um = 1.0 + 0.25 * (10.0 - positions_um) ** 2

        with self.assertRaises(ValueError) as ctx:
            locate_rayleigh_range(positions_um, widths_um)
        self.assertIs(type(ctx.exception), ValueError)
        text = str(ctx.exception)
        self.assertIn("Z=10.0 um", text)
        self.assertIn("increasing Z", text)
        self.assertNotIn("Z=0.0 um", text)

    def test_a_nan_run_reaching_the_edge_names_the_array_edge_not_the_last_measured_sample(self):
        # D-10, and the assertion that distinguishes this implementation
        # from one that reports the last clean sample. D-09's wording names
        # the array edge, and D-10 routes the NaN case through the identical
        # branch precisely because an all-NaN tail and a too-short array are
        # indistinguishable from inside this function.
        positions_um = np.arange(41, dtype=np.float64) * 0.5
        widths_um = 1.0 + 0.25 * (positions_um - 10.0) ** 2
        widths_um[23:] = np.nan

        with self.assertRaises(ValueError) as ctx:
            locate_rayleigh_range(positions_um, widths_um)
        text = str(ctx.exception)
        self.assertIn("Z=20.0 um", text)
        self.assertNotIn("Z=11.0 um", text)
        self.assertNotIn("Z=11.5 um", text)

    def test_a_single_sample_profile_is_well_formed_input_that_is_simply_too_short(self):
        # The empty-category boundary: a length-1 profile is legal,
        # well-formed input that simply cannot contain a crossing;
        # conflating it with malformed input would send the caller to fix
        # the wrong thing.
        positions_um = np.array([0.0])
        widths_um = np.array([1.0])

        with self.assertRaises(ValueError) as ctx:
            locate_rayleigh_range(positions_um, widths_um)
        text = str(ctx.exception)
        self.assertIn("Rayleigh-range crossing not found before array edge", text)
        self.assertIn("Z=0.0 um", text)
        self.assertNotIn("strictly ascending", text)
        self.assertNotIn("empty", text)

    def test_a_nan_immediately_before_the_crossing_is_skipped_not_used_as_a_bracket(self):
        # D-08, and the direct discriminator for RESEARCH Pitfall 3: NaN
        # comparisons evaluate False in NumPy, so a naive port of
        # beam_profile.py's _crossing() never registers the NaN as the
        # crossing and therefore *looks* correct, while still handing it to
        # the interpolation as the lower bracket endpoint -- producing a
        # NaN boundary that no other test in this phase would catch.
        positions_um, clean_widths_um = _synthetic_quadratic_profile(0.5)
        _waist, _left, clean_right = locate_rayleigh_range(positions_um, clean_widths_um)

        gapped_widths_um = clean_widths_um.copy()
        gapped_widths_um[22] = np.nan
        _waist, left, right = locate_rayleigh_range(positions_um, gapped_widths_um)

        self.assertTrue(np.isfinite(right))
        self.assertAlmostEqual(right, 11.203427124746190, places=9)
        # Proves the NaN genuinely widened the interpolation bracket rather
        # than being ignored outright.
        self.assertGreater(abs(right - clean_right), 1e-6)
        # The bracket spans the two nearest MEASURED samples (10.5, 11.5).
        # A copied _crossing() would instead use the raw index-minus-one
        # bracket, i.e. the NaN sample itself (position 11.0) as the lower
        # endpoint, making `frac` NaN and the reported boundary NaN -- a
        # case already excluded by the isfinite and pinned-value assertions
        # above. (A literal "not between 11.0 and 11.5" check on the VALUE
        # is not a usable discriminator here: (11.0, 11.5) is a strict
        # subset of the valid (10.5, 11.5) span, so the correct answer,
        # 11.2034..., necessarily falls inside it too.)
        self.assertTrue(10.5 < right < 11.5)
        # One side's gap must not perturb the other.
        self.assertAlmostEqual(left, 8.737258300203047, places=9)

    def test_a_nan_gap_is_skipped_on_the_decreasing_z_side_too(self):
        # The mirror of the test above -- the two walks are separate
        # invocations of `_walk_outward`, so a one-sided fix would pass the
        # first test alone.
        positions_um, clean_widths_um = _synthetic_quadratic_profile(0.5)
        _waist, _clean_left, clean_right = locate_rayleigh_range(positions_um, clean_widths_um)

        gapped_widths_um = clean_widths_um.copy()
        gapped_widths_um[18] = np.nan
        _waist, left, right = locate_rayleigh_range(positions_um, gapped_widths_um)

        self.assertTrue(np.isfinite(left))
        self.assertAlmostEqual(left, 8.796572875253810, places=9)
        self.assertAlmostEqual(right, clean_right, places=9)

    def test_a_multi_sample_nan_gap_is_skipped_the_same_way_as_a_single_one(self):
        # D-08's "continue walking outward" wording, extended to a
        # two-sample gap.
        positions_um, clean_widths_um = _synthetic_quadratic_profile(0.5)

        gapped_widths_um = clean_widths_um.copy()
        gapped_widths_um[[21, 22]] = np.nan
        _waist, _left, right = locate_rayleigh_range(positions_um, gapped_widths_um)

        self.assertTrue(np.isfinite(right))
        self.assertAlmostEqual(right, 11.104569499661586, places=9)
        # The reported boundary moves further from the analytic crossing as
        # the gap widens, which is the honest consequence of interpolating
        # across a real data gap -- the alternative, refusing to report
        # anything whenever a NaN appears, is what D-08 explicitly rejects.

    def test_the_first_crossing_wins_even_when_a_cleaner_one_lies_further_out(self):
        # D-07: first-crossing-wins was chosen over smoothing or fitting
        # precisely so no new closed-form approximation enters the code
        # path. The waist is index 5 (width 1.0); the increasing-Z side
        # crosses the threshold at index 6, dips back below at index 7,
        # then rises monotonically -- a global or best-crossing search
        # would report the later, cleaner crossing beyond index 7 instead.
        positions_um = np.arange(11, dtype=np.float64)
        widths_um = np.array(
            [3.0, 2.0, 1.6, 1.2, 1.05, 1.0, 1.5, 1.1, 2.0, 2.5, 3.0]
        )

        _waist, left, right = locate_rayleigh_range(positions_um, widths_um)

        self.assertAlmostEqual(right, 5.82842712474619, places=9)
        self.assertLess(right, 6.5)
        self.assertAlmostEqual(left, 2.4644660940672622, places=9)

    def test_a_sample_exactly_at_the_threshold_counts_as_the_crossing(self):
        # The adjacency edge probe. A strict `>` test would skip this
        # sample entirely and report a crossing further out, so an exact
        # 8.0 is the falsifiable signature of the `>=` rule D-06 specifies.
        positions_um = np.arange(11, dtype=np.float64)
        widths_um = np.array(
            [3.0, 2.0, 1.6, 1.2, 1.05, 1.0, 1.05, 1.2, float(np.sqrt(2.0)), 2.0, 3.0]
        )

        _waist, _left, right = locate_rayleigh_range(positions_um, widths_um)

        self.assertEqual(right, 8.0)

    def test_the_reported_boundary_is_stable_under_axial_sampling_refinement(self):
        # ROADMAP Success Criterion 4, proven on synthetic arrays with zero
        # PSF generations (RESEARCH Pitfall 5): `locate_rayleigh_range`
        # consumes two arrays rather than optical parameters, so this
        # criterion is provable without paying for a PSF generation per
        # sampling step -- unlike Phase 5's equivalent refinement test,
        # whose function's contract starts from optical parameters. These
        # values are pure float64 arithmetic over hand-authored arrays and
        # are therefore exactly reproducible, unlike the psfmodels-backed
        # numbers in 06-01, so the tight tolerances here are deliberate and
        # safe.
        #
        # This closed form describes the hand-authored TEST FIXTURE, not
        # the beam physics, and is not a model the implementation is
        # permitted to use -- 06-01's AST allowlist and token scan already
        # forbid any such model from reaching simulate/rayleigh_range.py.
        half_width_um = 2.0 * np.sqrt(np.sqrt(2.0) - 1.0)
        analytic_left = 10.0 - half_width_um
        analytic_right = 10.0 + half_width_um

        lefts: list[float] = []
        rights: list[float] = []
        extents: list[float] = []
        for step in (0.5, 0.25, 0.125):
            with self.subTest(step=step):
                positions_um, widths_um = _synthetic_quadratic_profile(step)
                waist, left, right = locate_rayleigh_range(positions_um, widths_um)
                # The step is chosen so z = 10.0 is always a sampled
                # position -- without that the waist itself would move
                # between runs and confound the measurement being refined
                # with the thing being measured (mirrors Phase 5's
                # constant-physical-window reasoning).
                self.assertEqual(waist, 10.0)
                self.assertEqual(float(np.min(widths_um)), 1.0)
                lefts.append(left)
                rights.append(right)
                extents.append(right - left)

        coarsest_voxel_um = 0.5
        quarter_voxel_um = 0.25 * coarsest_voxel_um
        # Planning-measured shift is about 0.0232 um on each side -- a
        # roughly five-fold margin under this quarter-voxel bar, so a
        # future regression that merely squeaks under it is still visibly
        # worse.
        self.assertLess(abs(lefts[0] - lefts[-1]), quarter_voxel_um)
        self.assertLess(abs(rights[0] - rights[-1]), quarter_voxel_um)

        # The stronger claim the roadmap's "not by a large jump" wording
        # implies: the absolute error against the analytic crossing
        # strictly decreases at each refinement, on both sides (about
        # 0.0244, 0.0029, 0.0012 um). A snapped-to-sample or otherwise
        # non-interpolating implementation would show errors that jump
        # around at the voxel scale instead of shrinking.
        left_errors = [abs(v - analytic_left) for v in lefts]
        right_errors = [abs(v - analytic_right) for v in rights]
        self.assertGreater(left_errors[0], left_errors[1])
        self.assertGreater(left_errors[1], left_errors[2])
        self.assertGreater(right_errors[0], right_errors[1])
        self.assertGreater(right_errors[1], right_errors[2])

        # The derived quantity Phase 8 will actually plot is stable too.
        for measured, expected in zip(extents, (2.525483, 2.568621, 2.571889)):
            self.assertAlmostEqual(measured, expected, places=5)
        analytic_extent = 2.0 * half_width_um
        for extent in extents:
            self.assertLess(abs(extent - analytic_extent), 0.06)
        self.assertLess(max(extents) - min(extents), quarter_voxel_um)

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
