from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tiresias import beam_profile
from tiresias import seeds

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

    def test_returns_two_parallel_float64_arrays_one_entry_per_z_voxel(self):
        from tiresias.beam_profile import measure_beam_width_profile

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
        from tiresias.beam_profile import measure_beam_width_profile

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
        from tiresias.beam_profile import measure_beam_width_profile

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
        from tiresias.beam_profile import measure_beam_width_profile

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

        allowed_modules = {"__future__", "warnings", "numpy", "seeds"}
        allowed_names = {"annotations", "generate_theoretical_psf"}
        self.assertTrue(modules <= allowed_modules, modules)
        self.assertTrue(names <= allowed_names, names)

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


if __name__ == "__main__":
    unittest.main()
