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

import ast
import json
import unittest
import warnings
from pathlib import Path
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


def _load_ring_free_width_baseline():
    """Load the MEAS-04 regression fixture captured by plan 09-01 Task 1.

    Loaded by path, never by package import -- simulate/tests/fixtures/ has
    no __init__.py. This is test-support code, so duplicating the loader
    here (rather than importing it from test_beam_profile.py) does not
    violate Phase 8 D-04, which governs only the production loop.
    """
    fixture_path = Path(__file__).parent / "fixtures" / "ring_free_width_baseline.json"
    with fixture_path.open() as handle:
        return json.load(handle)


def _load_ringed_unbounded_baseline():
    """Load the D-08 ringed pre-fix widths_um fixture captured by plan 09-01 Task 2.

    Loaded by path, never by package import, mirroring
    _load_ring_free_width_baseline() above.
    """
    fixture_path = Path(__file__).parent / "fixtures" / "ringed_unbounded_baseline.json"
    with fixture_path.open() as handle:
        return json.load(handle)


def _synthetic_volume(*profiles: list[float]) -> np.ndarray:
    """Build a (len(profiles), 15, 15) float32 volume with each profile along Y at X=7."""
    volume = np.zeros((len(profiles), 15, 15), dtype=np.float32)
    for z, profile in enumerate(profiles):
        volume[z, :, 7] = profile
    return volume


# Synthetic profile constants (15 samples, peak 1.0 at index 7). Duplicated
# verbatim from test_beam_profile.py per the plan's instruction -- this
# module does not import them from that module, which may not exist yet in
# this parallel wave. Verified against a probe implementation during
# planning (09-03-PLAN.md <context>).
_Y_CLEAN = [0.0, 0.01, 0.02, 0.05, 0.1, 0.3, 0.7, 1.0, 0.7, 0.3, 0.1, 0.05, 0.02, 0.01, 0.0]
_Y_MERGED_RING = [0.0, 0.05, 0.1, 0.3, 0.7, 0.6, 0.8, 1.0, 0.8, 0.6, 0.7, 0.3, 0.1, 0.05, 0.0]
_Y_MIXED_SIDES = [0.6, 0.62, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 0.8, 0.6, 0.7, 0.3, 0.1, 0.05, 0.0]
_Y_EDGE_NO_MIN = [0.0, 0.01, 0.02, 0.05, 0.1, 0.3, 0.7, 1.0, 0.9, 0.8, 0.75, 0.7, 0.65, 0.62, 0.6]
_Y_WOBBLE_SUB_TOL = [
    0.0, 0.01, 0.02, 0.05, 0.1, 0.3, 0.7, 1.0, 0.95, 0.95 - 4e-7, 0.95, 0.3, 0.1, 0.05, 0.0,
]
_Y_DIP_ABOVE_TOL = [0.0, 0.01, 0.02, 0.05, 0.1, 0.3, 0.7, 1.0, 0.95, 0.94, 0.95, 0.3, 0.1, 0.05, 0.0]
_Y_PEAK_RELATIVE = [
    0.0, 0.01, 0.02, 0.05, 0.1, 0.3, 0.7, 1.0, 0.8, 0.6, 0.6 + 8e-7, 0.3, 0.1, 0.05, 0.0,
]
_Y_FIRST_MIN_WINS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.3, 0.7, 1.0, 0.8, 0.7, 0.75, 0.3, 0.1, 0.2, 0.15]
_Y_MIN_BELOW_HALF = [0.0, 0.01, 0.02, 0.05, 0.1, 0.3, 0.7, 1.0, 0.8, 0.4, 0.45, 0.1, 0.05, 0.02, 0.0]
_Y_ZERO = [0.0] * 15

COMMON = dict(
    wavelength=0.561,
    ni=1.33,
    ns=1.33,
    dxy=0.108,
    dz=0.300,
    psf_size_xy=128,
)


def _measure_gated_synthetic(volume: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Call _measure_widths_from_array directly and return (widths_um, messages).

    Unlike beam_profile.py's _measure_synthetic, no mock of PSF generation is
    needed here: _measure_widths_from_array takes an already-prepared array
    directly, with no internal PSF generation step of its own.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        widths_um = _measure_widths_from_array(
            volume, 0.108, np.arange(volume.shape[0], dtype=np.float64) * 0.300
        )
    return widths_um, [str(w.message) for w in caught]


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

    def test_ring_free_gated_widths_match_the_pre_fix_baseline_bit_for_bit(self):
        import warnings

        baseline = _load_ring_free_width_baseline()
        self.assertEqual(len(baseline["gated_beam_profile"]), 3)

        for key, entry in baseline["gated_beam_profile"].items():
            with self.subTest(combo=key):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    _positions_um, widths_um = measure_gated_beam_width_profile(
                        **entry["params"]
                    )

                # MEAS-04 demands bit-identical output against the frozen
                # pre-fix fixture -- allclose would tolerate a silent
                # regression that assert_array_equal catches.
                np.testing.assert_array_equal(
                    widths_um, np.array(entry["widths_um"], dtype=np.float64)
                )
                self.assertEqual(len(caught), 0)

    def test_gated_synthetic_merged_ring_is_tagged_central_lobe_too_narrow(self):
        # Pre-fix, this profile measures a finite ring-inclusive width with no
        # warning (the unbounded search walks straight past the shallow
        # merged ring). Bounding the search to the first flanking local
        # minimum stops it there instead, and the lobe never reaches
        # half-max within that bound.
        volume = _synthetic_volume(_Y_MERGED_RING, _Y_MERGED_RING, _Y_MERGED_RING)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            widths_um = _measure_widths_from_array(
                volume, 0.108, np.arange(3, dtype=np.float64) * 0.300
            )

        self.assertTrue(np.isnan(widths_um).all())
        self.assertEqual(len(caught), 1)
        self.assertEqual(
            str(caught[0].message),
            "measure_gated_beam_width_profile: central lobe too narrow to reach "
            "half-max at (um): [0.0, 0.3, 0.6]",
        )

    def test_real_ringed_gated_psf_bounding_only_blanks_ring_merged_positions(self):
        baseline = _load_ringed_unbounded_baseline()
        entry = baseline["ringed_widths"]["gated_beam_profile"]["na0.4_z61_slit2.0"]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            positions_um, bounded = measure_gated_beam_width_profile(**entry["params"])

        unbounded = np.array(entry["widths_um"], dtype=np.float64)
        finite = np.isfinite(bounded)

        # Bounding may only ever turn a finite reading into NaN, never
        # change a finite value.
        np.testing.assert_array_equal(bounded[finite], unbounded[finite])

        newly_nan = ~finite & np.isfinite(unbounded)
        # Planning measured 11/61 newly-NaN positions; deliberately not
        # pinned to that exact count so a psfmodels version bump does not
        # break this test -- only that bounding has a real, non-vacuous
        # effect.
        self.assertGreaterEqual(int(newly_nan.sum()), 1)

        self.assertEqual(len(caught), 1)
        text = str(caught[0].message)
        self.assertTrue(text.startswith("measure_gated_beam_width_profile: "))
        parts = text[len("measure_gated_beam_width_profile: ") :].split("; ")
        for part in parts:
            self.assertFalse(part.startswith("no local minimum before array edge"))
        narrow_prefix = "central lobe too narrow to reach half-max at (um): "
        narrow_part = next(p for p in parts if p.startswith(narrow_prefix))
        warned_positions = ast.literal_eval(narrow_part[len(narrow_prefix) :])
        expected_positions = {round(float(p), 6) for p in positions_um[newly_nan]}
        self.assertEqual(set(warned_positions), expected_positions)

        # First-minimum evidence: setUpClass's cls.psf was generated with the
        # same params (illumination_na=0.4, psf_size_z=61, COMMON), so gating
        # it here reproduces the exact array measure_gated_beam_width_profile
        # measured internally.
        gated = _apply_aslm_slit_gate(
            self.psf, GATE_AXIS, self.slit_width, COMMON["dxy"], COMMON["dz"]
        )
        peak_y, peak_x = np.unravel_index(np.argmax(gated), gated.shape)[1:]
        half_max_scale = 0.5
        newly_nan_indices = np.where(newly_nan)[0]
        for z in newly_nan_indices:
            profile = gated[z, :, peak_x].astype(np.float64)
            peak_value = profile[peak_y]
            half_max = peak_value * half_max_scale
            left_indices = np.arange(peak_y, -1, -1)
            right_indices = np.arange(peak_y, profile.size)
            left_bounded, left_found = gated_beam_profile._bound_to_first_local_minimum(
                profile, left_indices, peak_value
            )
            right_bounded, right_found = gated_beam_profile._bound_to_first_local_minimum(
                profile, right_indices, peak_value
            )
            found_and_above_half_max = (
                left_found and (profile[left_bounded] >= half_max).all()
            ) or (
                right_found and (profile[right_bounded] >= half_max).all()
            )
            self.assertTrue(
                found_and_above_half_max,
                f"z={z!r}: expected at least one side to find a bounding "
                "minimum before half-max",
            )

    def test_gated_synthetic_sub_tolerance_wobble_is_not_a_lobe_boundary(self):
        volume = _synthetic_volume(_Y_WOBBLE_SUB_TOL, _Y_WOBBLE_SUB_TOL, _Y_WOBBLE_SUB_TOL)
        widths_um, messages = _measure_gated_synthetic(volume)

        self.assertEqual(messages, [])
        for width in widths_um:
            self.assertAlmostEqual(float(width), 0.5607692, delta=1e-6)
        self.assertEqual(gated_beam_profile._LOBE_MIN_RTOL, 1e-6)

        # Non-vacuity guard: without this, the test could pass only because
        # float32 rounding erased the deliberate sub-tolerance wobble.
        p = volume[0, :, 7].astype(np.float64)
        self.assertTrue(p[8] > p[9] < p[10])
        self.assertLess(p[10] - p[9], gated_beam_profile._LOBE_MIN_RTOL * p[7])

    def test_gated_synthetic_lobe_tolerance_is_load_bearing(self):
        volume = _synthetic_volume(_Y_WOBBLE_SUB_TOL, _Y_WOBBLE_SUB_TOL, _Y_WOBBLE_SUB_TOL)
        with mock.patch.object(gated_beam_profile, "_LOBE_MIN_RTOL", 0.0):
            widths_um, messages = _measure_gated_synthetic(volume)

        self.assertTrue(np.isnan(widths_um).all())
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0],
            "measure_gated_beam_width_profile: central lobe too narrow to reach "
            "half-max at (um): [0.0, 0.3, 0.6]",
        )

    def test_gated_synthetic_tolerance_is_relative_to_the_peak_not_the_local_value(self):
        volume = _synthetic_volume(_Y_PEAK_RELATIVE, _Y_PEAK_RELATIVE, _Y_PEAK_RELATIVE)
        widths_um, messages = _measure_gated_synthetic(volume)

        self.assertEqual(messages, [])
        for width in widths_um:
            self.assertAlmostEqual(float(width), 0.522, delta=1e-6)

        p = volume[0, :, 7].astype(np.float64)
        self.assertLess(p[10] - p[9], 1e-6 * p[7])
        self.assertGreater(p[10] - p[9], 1e-6 * p[9])

    def test_gated_synthetic_above_tolerance_dip_bounds_the_lobe(self):
        volume = _synthetic_volume(_Y_DIP_ABOVE_TOL, _Y_DIP_ABOVE_TOL, _Y_DIP_ABOVE_TOL)
        widths_um, messages = _measure_gated_synthetic(volume)

        self.assertTrue(np.isnan(widths_um).all())
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0],
            "measure_gated_beam_width_profile: central lobe too narrow to reach "
            "half-max at (um): [0.0, 0.3, 0.6]",
        )

    def test_gated_synthetic_first_flanking_minimum_wins_over_a_deeper_one(self):
        # A deepest-minimum search would bound further out and measure a
        # different width; first-found wins instead, mirroring
        # rayleigh_range.py's D-07 first-crossing-wins precedent.
        volume = _synthetic_volume(_Y_FIRST_MIN_WINS, _Y_FIRST_MIN_WINS, _Y_FIRST_MIN_WINS)
        widths_um, messages = _measure_gated_synthetic(volume)

        self.assertTrue(np.isnan(widths_um).all())
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0],
            "measure_gated_beam_width_profile: central lobe too narrow to reach "
            "half-max at (um): [0.0, 0.3, 0.6]",
        )

    def test_gated_synthetic_minimum_below_half_max_is_kept_inclusive(self):
        volume = _synthetic_volume(_Y_MIN_BELOW_HALF, _Y_MIN_BELOW_HALF, _Y_MIN_BELOW_HALF)
        widths_um, messages = _measure_gated_synthetic(volume)

        self.assertEqual(messages, [])
        for width in widths_um:
            self.assertAlmostEqual(float(width), 0.351, delta=1e-6)

    def test_gated_synthetic_side_reaching_the_edge_without_a_minimum_is_tagged_no_local_minimum(
        self,
    ):
        volume = _synthetic_volume(_Y_EDGE_NO_MIN, _Y_EDGE_NO_MIN, _Y_EDGE_NO_MIN)
        widths_um, messages = _measure_gated_synthetic(volume)

        self.assertTrue(np.isnan(widths_um).all())
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0],
            "measure_gated_beam_width_profile: no local minimum before array edge "
            "at (um): [0.0, 0.3, 0.6]",
        )

    def test_gated_synthetic_mixed_side_failure_prefers_no_local_minimum(self):
        volume = _synthetic_volume(_Y_MIXED_SIDES, _Y_MIXED_SIDES, _Y_MIXED_SIDES)
        widths_um, messages = _measure_gated_synthetic(volume)

        self.assertTrue(np.isnan(widths_um).all())
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0],
            "measure_gated_beam_width_profile: no local minimum before array edge "
            "at (um): [0.0, 0.3, 0.6]",
        )
        self.assertNotIn("central lobe too narrow", messages[0])

    def test_gated_synthetic_zero_peak_slice_is_tagged_no_local_minimum(self):
        volume = _synthetic_volume(_Y_ZERO, _Y_CLEAN, _Y_CLEAN)
        widths_um, messages = _measure_gated_synthetic(volume)

        self.assertTrue(np.isnan(widths_um[0]))
        for width in widths_um[1:]:
            self.assertAlmostEqual(float(width), 0.324, delta=1e-6)
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0],
            "measure_gated_beam_width_profile: no local minimum before array edge "
            "at (um): [0.0]",
        )

    def test_gated_synthetic_peak_on_the_y_boundary_is_tagged_no_local_minimum(self):
        volume = np.zeros((3, 9, 9), dtype=np.float32)
        volume[:, 0, 4] = 1.0
        volume[:, 1, 4] = 0.1

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            widths_um = _measure_widths_from_array(
                volume, 0.108, np.arange(3, dtype=np.float64) * 0.300
            )
        messages = [str(w.message) for w in caught]

        self.assertTrue(np.isnan(widths_um).all())
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0],
            "measure_gated_beam_width_profile: no local minimum before array edge "
            "at (um): [0.0, 0.3, 0.6]",
        )

    def test_gated_synthetic_both_reasons_share_one_grouped_warning(self):
        volume = _synthetic_volume(_Y_MIXED_SIDES, _Y_MERGED_RING, _Y_CLEAN)
        widths_um, messages = _measure_gated_synthetic(volume)

        self.assertTrue(np.isnan(widths_um[0]))
        self.assertTrue(np.isnan(widths_um[1]))
        self.assertAlmostEqual(float(widths_um[2]), 0.324, delta=1e-6)
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0],
            "measure_gated_beam_width_profile: no local minimum before array edge at (um): [0.0]; central lobe too narrow to reach half-max at (um): [0.3]",
        )

    def test_gated_bounding_is_an_independent_copy_not_a_shared_helper(self):
        # D-14, Phase 8 D-04: this test must NOT reference beam_profile's
        # private names -- 09-02 may land after this plan in the same wave,
        # and this module's independence must not depend on that ordering.
        source = Path(gated_beam_profile.__file__).read_text(encoding="utf-8")
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

        allowed_modules = {"__future__", "warnings", "numpy", "tiresias.seeds"}
        allowed_names = {
            "annotations",
            "generate_theoretical_psf",
            "_apply_aslm_slit_gate",
        }
        self.assertTrue(modules <= allowed_modules, modules)
        self.assertTrue(names <= allowed_names, names)

        self.assertEqual(gated_beam_profile._LOBE_MIN_RTOL, 1e-6)
        self.assertEqual(
            gated_beam_profile._bound_to_first_local_minimum.__module__,
            "simulate.gated_beam_profile",
        )


if __name__ == "__main__":
    unittest.main()
