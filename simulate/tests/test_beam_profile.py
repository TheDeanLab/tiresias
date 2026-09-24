from __future__ import annotations

import ast
import json
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from simulate import beam_profile
from tiresias import seeds


def _load_ring_free_width_baseline():
    """Load the MEAS-04 regression fixture captured by plan 09-01 Task 1.

    Loaded by path, never by package import -- simulate/tests/fixtures/ has
    no __init__.py, mirroring tests/test_seeds.py::_load_legacy_rotation_baseline.
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


# D-07-style shared realistic optical parameters, mirroring how the example
# scripts hold a COMMON dict, so no test re-types them.
COMMON = dict(
    wavelength=0.561,
    ni=1.33,
    ns=1.33,
    dxy=0.108,
    dz=0.300,
    psf_size_xy=128,
)

# Synthetic profile constants (15 samples, peak 1.0 at index 7), verified
# against a probe implementation during planning (09-02-PLAN.md <context>).
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


def _measure_synthetic(volume: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Mock make_psf with volume, measure it, and return (widths_um, warning message strings)."""
    with mock.patch.object(seeds.pm, "make_psf", return_value=volume):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _positions_um, widths_um = beam_profile.measure_beam_width_profile(
                illumination_na=0.4,
                wavelength=0.561,
                ni=1.33,
                ns=1.33,
                dxy=0.108,
                dz=0.300,
                psf_size_z=volume.shape[0],
                psf_size_xy=volume.shape[1],
            )
    return widths_um, [str(w.message) for w in caught]


class BeamProfileTests(unittest.TestCase):
    def test_public_api_measures_a_real_illumination_psf_end_to_end(self):
        from simulate import measure_beam_width_profile

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

    def test_returns_two_parallel_float64_arrays_one_entry_per_z_voxel(self):
        from simulate.beam_profile import measure_beam_width_profile

        positions_um, widths_um = measure_beam_width_profile(
            illumination_na=0.3, psf_size_z=9, **COMMON
        )
        self.assertIsInstance(positions_um, np.ndarray)
        self.assertIsInstance(widths_um, np.ndarray)
        self.assertEqual(positions_um.shape, (9,))
        self.assertEqual(widths_um.shape, (9,))
        self.assertEqual(positions_um.dtype, np.float64)
        self.assertEqual(widths_um.dtype, np.float64)
        self.assertEqual(positions_um.shape, widths_um.shape)

        for psf_size_z in (3, 9, 21):
            with self.subTest(psf_size_z=psf_size_z):
                positions_n, widths_n = measure_beam_width_profile(
                    illumination_na=0.3, psf_size_z=psf_size_z, **COMMON
                )
                self.assertEqual(len(positions_n), psf_size_z)
                self.assertEqual(len(widths_n), psf_size_z)

    def test_positions_are_physical_micrometres_in_ascending_voxel_order(self):
        from simulate.beam_profile import measure_beam_width_profile

        for dz in (0.300, 0.150):
            with self.subTest(dz=dz):
                params = dict(COMMON)
                params["dz"] = dz
                positions_um, _widths_um = measure_beam_width_profile(
                    illumination_na=0.3, psf_size_z=9, **params
                )
                np.testing.assert_allclose(
                    positions_um, np.arange(9) * dz, rtol=0, atol=1e-12
                )
                self.assertTrue(np.all(np.diff(positions_um) > 0))
                self.assertEqual(positions_um[0], 0.0)

        # Prove positions are labels applied after measurement, not
        # measurement inputs: with the PSF backend mocked (so the measured
        # array is genuinely held fixed, since the mock ignores dz), halving
        # dz halves every position while leaving every width bit-identical.
        zz, yy, xx = np.meshgrid(
            np.arange(9), np.arange(9), np.arange(9), indexing="ij"
        )
        raw = np.exp(
            -0.5 * (((zz - 4) ** 2 + (yy - 4) ** 2 + (xx - 4) ** 2) / (1.5**2))
        ).astype(np.float32)

        with mock.patch.object(seeds.pm, "make_psf", return_value=raw):
            positions_a, widths_a = measure_beam_width_profile(
                illumination_na=0.3,
                wavelength=0.561,
                ni=1.33,
                ns=1.33,
                dxy=0.108,
                dz=0.300,
                psf_size_z=9,
                psf_size_xy=9,
            )
            positions_b, widths_b = measure_beam_width_profile(
                illumination_na=0.3,
                wavelength=0.561,
                ni=1.33,
                ns=1.33,
                dxy=0.108,
                dz=0.150,
                psf_size_z=9,
                psf_size_xy=9,
            )
        np.testing.assert_array_equal(widths_a, widths_b)
        np.testing.assert_allclose(positions_b, positions_a / 2, rtol=0, atol=1e-12)

        # The returned order already IS ascending-position order: a stable
        # sort by position is a no-op.
        np.testing.assert_array_equal(
            widths_a, widths_a[np.argsort(positions_a, kind="stable")]
        )

    def test_higher_illumination_na_tightens_measured_waist(self):
        from simulate.beam_profile import measure_beam_width_profile

        centres = []
        for illumination_na in (0.2, 0.3, 0.4, 0.5, 0.6):
            _positions_um, widths_um = measure_beam_width_profile(
                illumination_na=illumination_na, psf_size_z=9, **COMMON
            )
            centre = float(widths_um[4])
            self.assertTrue(np.isfinite(centre), f"illumination_na={illumination_na!r}")
            centres.append(centre)

        for earlier, later in zip(centres, centres[1:]):
            self.assertGreater(earlier, later, centres)

        self.assertLess(centres[-1], 0.5 * centres[0], centres)

    def test_generates_only_the_illumination_arm_psf_exactly_once(self):
        from simulate.beam_profile import measure_beam_width_profile

        raw = np.zeros((5, 9, 9), dtype=np.float32)
        raw[:, 4, 4] = 1.0
        raw[:, 3, 4] = 0.4
        raw[:, 5, 4] = 0.4

        with mock.patch.object(seeds.pm, "make_psf", return_value=raw) as make_psf:
            measure_beam_width_profile(
                illumination_na=0.3,
                wavelength=0.561,
                ni=1.33,
                ns=1.33,
                dxy=0.108,
                dz=0.300,
                psf_size_z=5,
                psf_size_xy=9,
            )

        self.assertEqual(make_psf.call_count, 1)
        self.assertEqual(make_psf.call_args.kwargs["NA"], 0.3)

    def test_module_imports_are_confined_to_numpy_warnings_and_the_psf_generator(self):
        source = Path(beam_profile.__file__).read_text(encoding="utf-8")
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
        allowed_names = {"annotations", "generate_theoretical_psf"}
        self.assertTrue(modules <= allowed_modules, modules)
        self.assertTrue(names <= allowed_names, names)

    def test_rejects_missing_or_non_positive_parameters_before_generating_a_psf(self):
        from simulate import measure_beam_width_profile

        good = dict(
            illumination_na=0.4,
            wavelength=0.561,
            ni=1.33,
            ns=1.33,
            dxy=0.108,
            dz=0.300,
            psf_size_z=5,
            psf_size_xy=32,
        )
        # Both -1 and 0 are covered because psf_size_z/psf_size_xy are
        # integers where zero is the interesting boundary, while the
        # optical parameters are floats where negative is the interesting
        # one -- the guard's `value <= 0` test treats them identically, so
        # this matrix exercises both boundary shapes for every parameter.
        for name in good:
            for bad in (None, 0, -1):
                with self.subTest(parameter=name, value=bad):
                    kwargs = dict(good, **{name: bad})
                    with mock.patch.object(seeds.pm, "make_psf") as make_psf:
                        with self.assertRaisesRegex(ValueError, name):
                            measure_beam_width_profile(**kwargs)
                        self.assertEqual(make_psf.call_count, 0)

    def test_names_every_offending_parameter_in_one_error(self):
        from simulate import measure_beam_width_profile

        good = dict(
            illumination_na=0.4,
            wavelength=0.561,
            ni=1.33,
            ns=1.33,
            dxy=0.108,
            dz=0.300,
            psf_size_z=5,
            psf_size_xy=32,
        )
        with self.assertRaises(ValueError) as ctx:
            measure_beam_width_profile(**dict(good, ni=None, dz=-0.5))
        message = str(ctx.exception)
        self.assertIn("ni", message)
        self.assertIn("dz", message)

    def test_unmeasurable_positions_are_nan_and_every_one_is_named_in_a_warning(self):
        from simulate import measure_beam_width_profile

        # Indices 1-5 NaN, index 0 and 6 finite -- measured during planning
        # at illumination_na=0.6, psf_size_z=21, psf_size_xy=16 (RESEARCH
        # Pitfall 3), proving the NaN block is not edge-anchored.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _positions_um, widths_um = measure_beam_width_profile(
                illumination_na=0.6,
                wavelength=0.561,
                ni=1.33,
                ns=1.33,
                dxy=0.108,
                dz=0.300,
                psf_size_z=21,
                psf_size_xy=16,
            )

        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, UserWarning))
        text = str(caught[0].message)
        for token in ("0.3", "0.6", "0.9", "1.2", "1.5"):
            self.assertIn(token, text)

        self.assertTrue(np.isnan(widths_um[[1, 2, 3, 4, 5]]).all())
        self.assertFalse(np.isnan(widths_um[0]))
        self.assertFalse(np.isnan(widths_um[6]))
        self.assertEqual(int(np.isfinite(widths_um).sum()), 16)

    def test_warning_position_count_tracks_the_lateral_window(self):
        from simulate import measure_beam_width_profile

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _positions_um, widths_um = measure_beam_width_profile(
                illumination_na=0.6,
                wavelength=0.561,
                ni=1.33,
                ns=1.33,
                dxy=0.108,
                dz=0.300,
                psf_size_z=21,
                psf_size_xy=8,
            )

        self.assertEqual(int(np.isnan(widths_um).sum()), 8)
        self.assertTrue(np.isnan(widths_um[:8]).all())
        self.assertEqual(len(caught), 1)
        text = str(caught[0].message)
        expected_positions = [round(i * 0.300, 4) for i in range(8)]
        for position in expected_positions:
            self.assertIn(str(position), text)

    def test_realistic_parameters_emit_no_warning_and_no_nan(self):
        from simulate import measure_beam_width_profile

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _positions_um, widths_um = measure_beam_width_profile(
                illumination_na=0.4,
                wavelength=0.561,
                ni=1.33,
                ns=1.33,
                dxy=0.108,
                dz=0.300,
                psf_size_z=9,
                psf_size_xy=128,
            )

        self.assertEqual(len(caught), 0)
        self.assertTrue(np.isfinite(widths_um).all())

    def test_unmeasurable_positions_are_never_clamped_or_extrapolated(self):
        from simulate import measure_beam_width_profile

        dxy = 0.108
        psf_size_xy = 16
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _positions_um, widths_um = measure_beam_width_profile(
                illumination_na=0.6,
                wavelength=0.561,
                ni=1.33,
                ns=1.33,
                dxy=dxy,
                dz=0.300,
                psf_size_z=21,
                psf_size_xy=psf_size_xy,
            )

        nan_mask = np.isnan(widths_um)
        self.assertTrue(nan_mask.any())
        finite = widths_um[~nan_mask]
        self.assertTrue((finite < psf_size_xy * dxy).all())

    def test_measurement_path_contains_no_closed_form_or_fitted_beam_model(self):
        # Stripping '#' comment lines before scanning lets this executor's
        # own explanatory prose name what was rejected without invalidating
        # this gate -- only src/tiresias/beam_profile.py is scanned, never
        # this test file, so writing the forbidden literals here cannot
        # self-invalidate the gate.
        source = Path(beam_profile.__file__).read_text(encoding="utf-8")
        stripped_lines = [
            line for line in source.splitlines() if not line.strip().startswith("#")
        ]
        stripped_source = "\n".join(stripped_lines)

        forbidden_tokens = (
            "curve_fit",
            "peak_widths",
            "polyfit",
            "optimize",
            "sqrt",
            "scipy",
            "generate_psf_seed",
        )
        for token in forbidden_tokens:
            with self.subTest(token=token):
                self.assertNotIn(token, stripped_source)

        self.assertTrue(stripped_source.strip())
        self.assertIn("generate_theoretical_psf", stripped_source)

    def test_width_measurement_is_subvoxel_accurate_under_lateral_refinement(self):
        from simulate import measure_beam_width_profile

        base = dict(
            illumination_na=0.4,
            wavelength=0.561,
            ni=1.33,
            ns=1.33,
            dz=0.300,
            psf_size_z=5,
        )
        # psf_size_xy doubles as dxy halves so the physical lateral window
        # (psf_size_xy * dxy) stays constant -- without this the refinement
        # would also shrink the window and confound sampling with
        # truncation. Measured during planning: coarse=0.73414,
        # fine=0.72962 (shift 0.0045 um), finer=0.72833 (shift 0.0058 um)
        # against a 0.108 um coarse voxel -- roughly a 5x margin under the
        # quarter-voxel assertion below.
        coarse = float(
            measure_beam_width_profile(dxy=0.108, psf_size_xy=128, **base)[1][2]
        )
        fine = float(
            measure_beam_width_profile(dxy=0.054, psf_size_xy=256, **base)[1][2]
        )
        finer = float(
            measure_beam_width_profile(dxy=0.027, psf_size_xy=512, **base)[1][2]
        )
        for name, value in (("coarse", coarse), ("fine", fine), ("finer", finer)):
            with self.subTest(name=name):
                self.assertTrue(np.isfinite(value))
        for name, value in (("fine", fine), ("finer", finer)):
            with self.subTest(name=name):
                self.assertLess(abs(coarse - value), 0.25 * 0.108)

    def test_measured_width_is_not_snapped_to_a_voxel_multiple(self):
        from simulate import measure_beam_width_profile

        base = dict(
            illumination_na=0.4,
            wavelength=0.561,
            ni=1.33,
            ns=1.33,
            dz=0.300,
            psf_size_z=5,
        )
        coarse = float(
            measure_beam_width_profile(dxy=0.108, psf_size_xy=128, **base)[1][2]
        )
        # A nearest-voxel threshold implementation would give an exact
        # integer or half-integer multiple of dxy; the interpolated
        # implementation gives about 6.7976, not within 1e-6 of an integer.
        ratio = coarse / 0.108
        self.assertGreater(abs(ratio - round(ratio)), 1e-6)

    def test_exact_half_max_plateau_resolves_to_the_outermost_sample(self):
        from simulate import measure_beam_width_profile

        # Global peak at (z, y, x) = (any, 4, 4); the Y profile through
        # x=4 has a two-sample exactly-half-max plateau on each side of the
        # peak: [0.0, 0.1, 0.5, 0.5, 1.0, 0.5, 0.5, 0.1, 0.0].
        raw = np.zeros((3, 9, 9), dtype=np.float32)
        raw[:, :, 4] = [0.0, 0.1, 0.5, 0.5, 1.0, 0.5, 0.5, 0.1, 0.0]

        with mock.patch.object(seeds.pm, "make_psf", return_value=raw):
            _positions_um, widths_um = measure_beam_width_profile(
                illumination_na=0.4,
                wavelength=0.561,
                ni=1.33,
                ns=1.33,
                dxy=0.108,
                dz=0.300,
                psf_size_z=3,
                psf_size_xy=9,
            )

        # The strict below-half-max test (`values < half_max`, not `<=`)
        # treats an exactly-half-max sample as still inside the beam, so
        # the left crossing lands on index 2 and the right on index 6 --
        # the OUTERMOST half-max samples, the widest reading. Under a
        # less-than-or-equal test the crossings would instead land at
        # indices 3 and 5 and the width would be 2 * dxy instead of
        # 4 * dxy, so this test discriminates the two conventions. Note
        # generate_theoretical_psf normalises by sum before this function
        # sees it (irrelevant here since the array is mocked directly, but
        # normalisation scales uniformly and leaves the half-max relation
        # exact in the real path too).
        expected = 4 * 0.108
        for width in widths_um:
            self.assertAlmostEqual(float(width), expected, places=9)

    def test_single_z_slice_returns_a_length_one_profile(self):
        from simulate import measure_beam_width_profile

        positions_um, widths_um = measure_beam_width_profile(
            illumination_na=0.4,
            wavelength=0.561,
            ni=1.33,
            ns=1.33,
            dxy=0.108,
            dz=0.300,
            psf_size_z=1,
            psf_size_xy=64,
        )
        # Smallest legal input -- distinct from psf_size_z=0, which Task
        # 1's guard rejects with a ValueError.
        self.assertEqual(positions_um.shape, (1,))
        self.assertEqual(widths_um.shape, (1,))
        self.assertEqual(positions_um[0], 0.0)
        self.assertTrue(np.isfinite(widths_um[0]))

    def test_peak_on_the_y_boundary_yields_nan_on_the_side_with_no_outward_sample(self):
        from simulate import measure_beam_width_profile

        # Global peak sits at y == 0 -- the left walk has no sample outside
        # the peak, so below[0] == 0 fires immediately and there is no
        # bracketing pair to interpolate between. NaN is the only honest
        # answer.
        raw = np.zeros((3, 9, 9), dtype=np.float32)
        raw[:, 0, 4] = 1.0
        raw[:, 1, 4] = 0.1

        with mock.patch.object(seeds.pm, "make_psf", return_value=raw):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _positions_um, widths_um = measure_beam_width_profile(
                    illumination_na=0.4,
                    wavelength=0.561,
                    ni=1.33,
                    ns=1.33,
                    dxy=0.108,
                    dz=0.300,
                    psf_size_z=3,
                    psf_size_xy=9,
                )

        self.assertTrue(np.isnan(widths_um).all())
        self.assertEqual(len(caught), 1)
        text = str(caught[0].message)
        expected_positions = [round(i * 0.300, 4) for i in range(3)]
        for position in expected_positions:
            self.assertIn(str(position), text)

    def test_ring_free_widths_match_the_pre_fix_baseline_bit_for_bit(self):
        from simulate import measure_beam_width_profile

        baseline = _load_ring_free_width_baseline()
        self.assertEqual(len(baseline["beam_profile"]), 14)

        for key, entry in baseline["beam_profile"].items():
            with self.subTest(combo=key):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    _positions_um, widths_um = measure_beam_width_profile(**entry["params"])

                # MEAS-04 demands bit-identical output against the frozen
                # pre-fix fixture -- allclose would tolerate a silent
                # regression that assert_array_equal catches.
                np.testing.assert_array_equal(
                    widths_um, np.array(entry["widths_um"], dtype=np.float64)
                )
                self.assertEqual(len(caught), 0)

    def test_synthetic_merged_ring_is_tagged_central_lobe_too_narrow(self):
        # Pre-fix, this profile measured 7 * dxy = 0.756 um (ring-inclusive:
        # the unbounded search walked straight past the shallow merged ring
        # at indices 5/9 to the deeper minimum further out). Bounding the
        # search to the first flanking local minimum stops it there instead,
        # and the lobe never reaches half-max within that bound.
        volume = _synthetic_volume(_Y_MERGED_RING, _Y_MERGED_RING, _Y_MERGED_RING)

        with mock.patch.object(seeds.pm, "make_psf", return_value=volume):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _positions_um, widths_um = beam_profile.measure_beam_width_profile(
                    illumination_na=0.4,
                    wavelength=0.561,
                    ni=1.33,
                    ns=1.33,
                    dxy=0.108,
                    dz=0.300,
                    psf_size_z=3,
                    psf_size_xy=15,
                )

        self.assertTrue(np.isnan(widths_um).all())
        self.assertEqual(len(caught), 1)
        self.assertEqual(
            str(caught[0].message),
            "measure_beam_width_profile: central lobe too narrow to reach "
            "half-max at (um): [0.0, 0.3, 0.6]",
        )

    def test_real_ringed_psf_bounding_only_blanks_ring_merged_positions(self):
        baseline = _load_ringed_unbounded_baseline()
        ringed = baseline["ringed_widths"]["beam_profile"]

        for key, entry in ringed.items():
            with self.subTest(combo=key):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    positions_um, bounded = beam_profile.measure_beam_width_profile(
                        **entry["params"]
                    )

                unbounded = np.array(entry["widths_um"], dtype=np.float64)
                finite = np.isfinite(bounded)

                # Bounding may only ever turn a finite reading into NaN,
                # never change a finite value (the bounded index array is
                # always a prefix of the unbounded one, and _crossing
                # returns the first below-half-max sample in either case).
                np.testing.assert_array_equal(bounded[finite], unbounded[finite])

                newly_nan = ~finite & np.isfinite(unbounded)
                # Planning measured 18/61 (na0.4_z61) and 12/61 (na0.6_z61)
                # newly-NaN positions; deliberately not pinned to those exact
                # counts so a psfmodels version bump does not break this
                # test -- only that bounding has a real, non-vacuous effect.
                self.assertGreaterEqual(int(newly_nan.sum()), 1)

                self.assertEqual(len(caught), 1)
                text = str(caught[0].message)
                self.assertTrue(text.startswith("measure_beam_width_profile: "))
                parts = text[len("measure_beam_width_profile: ") :].split("; ")
                for part in parts:
                    self.assertFalse(
                        part.startswith("no local minimum before array edge")
                    )
                narrow_prefix = "central lobe too narrow to reach half-max at (um): "
                narrow_part = next(p for p in parts if p.startswith(narrow_prefix))
                warned_positions = ast.literal_eval(
                    narrow_part[len(narrow_prefix) :]
                )
                expected_positions = {
                    round(float(p), 6) for p in positions_um[newly_nan]
                }
                self.assertEqual(set(warned_positions), expected_positions)

                # For every newly-NaN position, at least one side's
                # _bound_to_first_local_minimum call must have actually
                # found a flanking minimum before half-max -- proving the
                # NaN comes from the bounded search, not some other cause.
                psf = seeds.generate_theoretical_psf(
                    detection_na=entry["params"]["illumination_na"],
                    **{
                        k: v
                        for k, v in entry["params"].items()
                        if k != "illumination_na"
                    },
                )
                peak_y, peak_x = np.unravel_index(
                    np.argmax(psf), psf.shape
                )[1:]
                half_max_scale = 0.5
                newly_nan_indices = np.where(newly_nan)[0]
                for z in newly_nan_indices:
                    profile = psf[z, :, peak_x].astype(np.float64)
                    peak_value = profile[peak_y]
                    half_max = peak_value * half_max_scale
                    left_indices = np.arange(peak_y, -1, -1)
                    right_indices = np.arange(peak_y, profile.size)
                    left_bounded, left_found = beam_profile._bound_to_first_local_minimum(
                        profile, left_indices, peak_value
                    )
                    right_bounded, right_found = beam_profile._bound_to_first_local_minimum(
                        profile, right_indices, peak_value
                    )
                    found_and_above_half_max = (
                        left_found and (profile[left_bounded] >= half_max).all()
                    ) or (
                        right_found and (profile[right_bounded] >= half_max).all()
                    )
                    self.assertTrue(
                        found_and_above_half_max,
                        f"combo={key!r} z={z!r}: expected at least one side to "
                        "find a bounding minimum before half-max",
                    )

    def test_synthetic_sub_tolerance_wobble_is_not_a_lobe_boundary(self):
        volume = _synthetic_volume(_Y_WOBBLE_SUB_TOL, _Y_WOBBLE_SUB_TOL, _Y_WOBBLE_SUB_TOL)
        widths_um, messages = _measure_synthetic(volume)

        self.assertEqual(messages, [])
        for width in widths_um:
            self.assertAlmostEqual(float(width), 0.5607692, delta=1e-6)
        self.assertEqual(beam_profile._LOBE_MIN_RTOL, 1e-6)

        # Non-vacuity guard: without this, the test could pass only because
        # float32 rounding erased the deliberate sub-tolerance wobble.
        p = seeds.normalise_psf(volume)[0, :, 7].astype(np.float64)
        self.assertTrue(p[8] > p[9] < p[10])
        self.assertLess(p[10] - p[9], beam_profile._LOBE_MIN_RTOL * p[7])

    def test_synthetic_lobe_tolerance_is_load_bearing(self):
        volume = _synthetic_volume(_Y_WOBBLE_SUB_TOL, _Y_WOBBLE_SUB_TOL, _Y_WOBBLE_SUB_TOL)
        with mock.patch.object(beam_profile, "_LOBE_MIN_RTOL", 0.0):
            widths_um, messages = _measure_synthetic(volume)

        self.assertTrue(np.isnan(widths_um).all())
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0],
            "measure_beam_width_profile: central lobe too narrow to reach "
            "half-max at (um): [0.0, 0.3, 0.6]",
        )

    def test_synthetic_tolerance_is_relative_to_the_peak_not_the_local_value(self):
        volume = _synthetic_volume(_Y_PEAK_RELATIVE, _Y_PEAK_RELATIVE, _Y_PEAK_RELATIVE)
        widths_um, messages = _measure_synthetic(volume)

        self.assertEqual(messages, [])
        for width in widths_um:
            self.assertAlmostEqual(float(width), 0.522, delta=1e-6)

        p = seeds.normalise_psf(volume)[0, :, 7].astype(np.float64)
        self.assertLess(p[10] - p[9], 1e-6 * p[7])
        self.assertGreater(p[10] - p[9], 1e-6 * p[9])

    def test_synthetic_above_tolerance_dip_bounds_the_lobe(self):
        volume = _synthetic_volume(_Y_DIP_ABOVE_TOL, _Y_DIP_ABOVE_TOL, _Y_DIP_ABOVE_TOL)
        widths_um, messages = _measure_synthetic(volume)

        self.assertTrue(np.isnan(widths_um).all())
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0],
            "measure_beam_width_profile: central lobe too narrow to reach "
            "half-max at (um): [0.0, 0.3, 0.6]",
        )

    def test_synthetic_first_flanking_minimum_wins_over_a_deeper_one(self):
        # A deepest-minimum search would bound at index 12 and measure
        # 0.546 um; first-found wins instead, mirroring rayleigh_range.py's
        # D-07 first-crossing-wins precedent.
        volume = _synthetic_volume(_Y_FIRST_MIN_WINS, _Y_FIRST_MIN_WINS, _Y_FIRST_MIN_WINS)
        widths_um, messages = _measure_synthetic(volume)

        self.assertTrue(np.isnan(widths_um).all())
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0],
            "measure_beam_width_profile: central lobe too narrow to reach "
            "half-max at (um): [0.0, 0.3, 0.6]",
        )

    def test_synthetic_minimum_below_half_max_is_kept_inclusive(self):
        volume = _synthetic_volume(_Y_MIN_BELOW_HALF, _Y_MIN_BELOW_HALF, _Y_MIN_BELOW_HALF)
        widths_um, messages = _measure_synthetic(volume)

        self.assertEqual(messages, [])
        for width in widths_um:
            self.assertAlmostEqual(float(width), 0.351, delta=1e-6)

    def test_synthetic_side_reaching_the_edge_without_a_minimum_is_tagged_no_local_minimum(self):
        volume = _synthetic_volume(_Y_EDGE_NO_MIN, _Y_EDGE_NO_MIN, _Y_EDGE_NO_MIN)
        widths_um, messages = _measure_synthetic(volume)

        self.assertTrue(np.isnan(widths_um).all())
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0],
            "measure_beam_width_profile: no local minimum before array edge "
            "at (um): [0.0, 0.3, 0.6]",
        )

    def test_synthetic_mixed_side_failure_prefers_no_local_minimum(self):
        volume = _synthetic_volume(_Y_MIXED_SIDES, _Y_MIXED_SIDES, _Y_MIXED_SIDES)
        widths_um, messages = _measure_synthetic(volume)

        self.assertTrue(np.isnan(widths_um).all())
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0],
            "measure_beam_width_profile: no local minimum before array edge "
            "at (um): [0.0, 0.3, 0.6]",
        )
        self.assertNotIn("central lobe too narrow", messages[0])

    def test_synthetic_zero_peak_slice_is_tagged_no_local_minimum(self):
        volume = _synthetic_volume(_Y_ZERO, _Y_CLEAN, _Y_CLEAN)
        widths_um, messages = _measure_synthetic(volume)

        self.assertTrue(np.isnan(widths_um[0]))
        for width in widths_um[1:]:
            self.assertAlmostEqual(float(width), 0.324, delta=1e-6)
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0],
            "measure_beam_width_profile: no local minimum before array edge "
            "at (um): [0.0]",
        )

    def test_synthetic_peak_on_the_y_boundary_is_tagged_no_local_minimum(self):
        raw = np.zeros((3, 9, 9), dtype=np.float32)
        raw[:, 0, 4] = 1.0
        raw[:, 1, 4] = 0.1
        widths_um, messages = _measure_synthetic(raw)

        self.assertTrue(np.isnan(widths_um).all())
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0],
            "measure_beam_width_profile: no local minimum before array edge "
            "at (um): [0.0, 0.3, 0.6]",
        )

    def test_synthetic_both_reasons_share_one_grouped_warning(self):
        volume = _synthetic_volume(_Y_MIXED_SIDES, _Y_MERGED_RING, _Y_CLEAN)
        widths_um, messages = _measure_synthetic(volume)

        self.assertTrue(np.isnan(widths_um[0]))
        self.assertTrue(np.isnan(widths_um[1]))
        self.assertAlmostEqual(float(widths_um[2]), 0.324, delta=1e-6)
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0],
            "measure_beam_width_profile: no local minimum before array edge at (um): [0.0]; central lobe too narrow to reach half-max at (um): [0.3]",
        )


if __name__ == "__main__":
    unittest.main()
