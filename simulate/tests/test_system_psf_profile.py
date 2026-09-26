from __future__ import annotations

import ast
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.signal import fftconvolve

import simulate
from simulate import system_psf_profile
from tiresias import seeds as tiresias_seeds
from tiresias.seeds import generate_psf_seed, generate_theoretical_psf

# D-11/D-12: locked from 08.1-REGIME-MEASUREMENTS.md (2026-09-26, measured
# against the real sweep-integrated gate implemented by plans 08.1-01/02).
# Do not widen or re-tune -- if a measured deviation exceeds these, STOP and
# report rather than re-deriving a looser tolerance.
TAU_THICK = 0.05
TAU_THIN = 0.15

# Shared tracer params (Task 1).
_TRACER_PARAMS = dict(
    detection_na=1.0,
    illumination_na=0.45,
    wavelength=0.561,
    ni=1.33,
    ns=1.33,
    ni0=1.33,
    dxy=0.108,
    dz=0.108,
    psf_size_z=61,
    psf_size_xy=128,
)


class SystemPsfProfileTracerTests(unittest.TestCase):
    def test_light_sheet_profile_is_measured_through_generate_psf_seed_end_to_end(self):
        positions_um, fwhm_um = simulate.measure_light_sheet_system_fwhm_profile(
            positions_um=(0.0, 5.0), **_TRACER_PARAMS
        )

        self.assertEqual(positions_um.dtype, np.float64)
        self.assertEqual(fwhm_um.dtype, np.float64)
        self.assertEqual(positions_um.shape, (2,))
        self.assertEqual(fwhm_um.shape, (2,))
        np.testing.assert_array_equal(positions_um, np.array([0.0, 5.0]))

        reference_seed = generate_psf_seed(
            psf_mode="light_sheet",
            na=_TRACER_PARAMS["detection_na"],
            detection_na=_TRACER_PARAMS["detection_na"],
            illumination_na=_TRACER_PARAMS["illumination_na"],
            wavelength=_TRACER_PARAMS["wavelength"],
            ni=_TRACER_PARAMS["ni"],
            ns=_TRACER_PARAMS["ns"],
            ni0=_TRACER_PARAMS["ni0"],
            tg=None,
            tg0=None,
            ng=None,
            ng0=None,
            ti0=None,
            oversample_factor=3,
            psf_model="vectorial",
            dxy=_TRACER_PARAMS["dxy"],
            dz=_TRACER_PARAMS["dz"],
            psf_size_z=_TRACER_PARAMS["psf_size_z"],
            psf_size_xy=_TRACER_PARAMS["psf_size_xy"],
            background=0.0,
            polar_deg=90.0,
            azimuthal_deg=0.0,
            emitter_offset=0.0,
        )
        expected_fwhm0 = system_psf_profile._axial_fwhm(reference_seed, _TRACER_PARAMS["dz"])
        self.assertEqual(fwhm_um[0], expected_fwhm0)

        # Off-waist emitter sees a broader sheet (planner prototype:
        # 0.579 um at p=0, 0.672 um at p=5).
        self.assertGreater(fwhm_um[1], fwhm_um[0])

    def test_new_functions_are_exported_from_simulate(self):
        names = (
            "measure_light_sheet_system_fwhm_profile",
            "measure_aslm_system_fwhm_profile",
            "measure_detection_dof",
            "measure_sheet_thickness",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertIn(name, simulate.__all__)
                self.assertTrue(callable(getattr(simulate, name)))


# Task 2 shared params.
_ASLM_FLAT_PARAMS = dict(
    detection_na=1.0,
    illumination_na=0.45,
    wavelength=0.561,
    ni=1.33,
    ns=1.33,
    ni0=1.33,
    dxy=0.108,
    dz=0.108,
    psf_size_z=61,
    psf_size_xy=64,
    slit_width=2.0,
)

_REGIME_GRID_PARAMS = dict(
    detection_na=1.1,
    wavelength=0.561,
    ni=1.33,
    ns=1.33,
    ni0=1.33,
    dxy=0.1,
    dz=0.1,
    psf_size_z=65,
    psf_size_xy=65,
)

_D13_PARAMS = dict(
    detection_na=1.1,
    illumination_na=0.6,
    wavelength=0.561,
    ni=1.33,
    ns=1.33,
    ni0=1.33,
    dxy=0.1,
    dz=0.1,
    psf_size_z=61,
    psf_size_xy=64,
)

# D-13: monotonic slit-width ladder, exactly as specified in the plan.
_SLIT_LADDER: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 6.1, 12.2, 1e6)


def _swept_average_reference_fwhm(
    *,
    detection_na,
    illumination_na,
    wavelength,
    ni,
    ns,
    ni0,
    dxy,
    dz,
    psf_size_z,
    psf_size_xy,
) -> float:
    """The uniform-window (swept-average) sheet, built from seeds primitives directly.

    Not through `generate_psf_seed(psf_mode="aslm", ...)` -- this reference is
    what a very-wide Gaussian slit converges to, computed independently with
    a uniform (ones) kernel instead of `_gaussian_slit_window`.
    """
    detection = tiresias_seeds.generate_theoretical_psf(
        detection_na=detection_na,
        illumination_na=illumination_na,
        wavelength=wavelength,
        ni=ni,
        ns=ns,
        ni0=ni0,
        dxy=dxy,
        dz=dz,
        psf_size_z=psf_size_z,
        psf_size_xy=psf_size_xy,
        background=0.0,
    )
    illumination = tiresias_seeds.generate_theoretical_psf(
        detection_na=illumination_na,
        illumination_na=illumination_na,
        wavelength=wavelength,
        ni=ni,
        ns=ns,
        ni0=ni0,
        dxy=dxy,
        dz=dz,
        psf_size_z=psf_size_z,
        psf_size_xy=psf_size_xy,
        background=0.0,
    )
    n = illumination.shape[0]
    kernel = np.ones((2 * n - 1, 1, 1), dtype=np.float32)
    swept = np.clip(
        fftconvolve(illumination, kernel, mode="same", axes=0), 0.0, None
    ).astype(np.float32)
    rotated = tiresias_seeds.rotate_illumination(
        swept, polar_deg=90.0, azimuthal_deg=0.0, dxy=dxy, dz=dz
    )
    seed = tiresias_seeds.normalise_psf(detection * rotated)
    return system_psf_profile._axial_fwhm(seed, dz)


class SystemPsfProfileBehaviorTests(unittest.TestCase):
    def test_aslm_profile_is_flat_across_fov_positions(self):
        positions_um, fwhm_um = simulate.measure_aslm_system_fwhm_profile(
            positions_um=(-20.0, 0.0, 5.0, 20.0), **_ASLM_FLAT_PARAMS
        )
        np.testing.assert_array_equal(fwhm_um, np.full(fwhm_um.shape, fwhm_um[0]))

        reference_seed = generate_psf_seed(
            psf_mode="aslm",
            na=_ASLM_FLAT_PARAMS["detection_na"],
            detection_na=_ASLM_FLAT_PARAMS["detection_na"],
            illumination_na=_ASLM_FLAT_PARAMS["illumination_na"],
            wavelength=_ASLM_FLAT_PARAMS["wavelength"],
            ni=_ASLM_FLAT_PARAMS["ni"],
            ns=_ASLM_FLAT_PARAMS["ns"],
            ni0=_ASLM_FLAT_PARAMS["ni0"],
            tg=None,
            tg0=None,
            ng=None,
            ng0=None,
            ti0=None,
            oversample_factor=3,
            psf_model="vectorial",
            dxy=_ASLM_FLAT_PARAMS["dxy"],
            dz=_ASLM_FLAT_PARAMS["dz"],
            psf_size_z=_ASLM_FLAT_PARAMS["psf_size_z"],
            psf_size_xy=_ASLM_FLAT_PARAMS["psf_size_xy"],
            background=0.0,
            polar_deg=90.0,
            azimuthal_deg=0.0,
            slit_width=_ASLM_FLAT_PARAMS["slit_width"],
            emitter_offset=0.0,
        )
        expected = system_psf_profile._axial_fwhm(reference_seed, _ASLM_FLAT_PARAMS["dz"])
        self.assertEqual(fwhm_um[0], expected)

    def test_references_are_measured_from_the_simulator(self):
        dof = simulate.measure_detection_dof(**_REGIME_GRID_PARAMS)

        reference_seed = generate_psf_seed(
            psf_mode="single",
            na=_REGIME_GRID_PARAMS["detection_na"],
            detection_na=_REGIME_GRID_PARAMS["detection_na"],
            illumination_na=None,
            wavelength=_REGIME_GRID_PARAMS["wavelength"],
            ni=_REGIME_GRID_PARAMS["ni"],
            ns=_REGIME_GRID_PARAMS["ns"],
            ni0=_REGIME_GRID_PARAMS["ni0"],
            tg=None,
            tg0=None,
            ng=None,
            ng0=None,
            ti0=None,
            oversample_factor=3,
            psf_model="vectorial",
            dxy=_REGIME_GRID_PARAMS["dxy"],
            dz=_REGIME_GRID_PARAMS["dz"],
            psf_size_z=_REGIME_GRID_PARAMS["psf_size_z"],
            psf_size_xy=_REGIME_GRID_PARAMS["psf_size_xy"],
            background=0.0,
        )
        expected_dof = system_psf_profile._axial_fwhm(reference_seed, _REGIME_GRID_PARAMS["dz"])
        self.assertEqual(dof, expected_dof)

        sheet_kwargs = {k: v for k, v in _REGIME_GRID_PARAMS.items() if k != "detection_na"}
        sheet_thick = simulate.measure_sheet_thickness(illumination_na=0.1, **sheet_kwargs)
        sheet_thin = simulate.measure_sheet_thickness(illumination_na=0.6, **sheet_kwargs)

        illum_psf = generate_theoretical_psf(
            detection_na=0.1,
            illumination_na=0.1,
            wavelength=_REGIME_GRID_PARAMS["wavelength"],
            ni=_REGIME_GRID_PARAMS["ni"],
            ns=_REGIME_GRID_PARAMS["ns"],
            ni0=_REGIME_GRID_PARAMS["ni0"],
            dxy=_REGIME_GRID_PARAMS["dxy"],
            dz=_REGIME_GRID_PARAMS["dz"],
            psf_size_z=_REGIME_GRID_PARAMS["psf_size_z"],
            psf_size_xy=_REGIME_GRID_PARAMS["psf_size_xy"],
        )
        peak_z, peak_y, peak_x = np.unravel_index(np.argmax(illum_psf), illum_psf.shape)
        profile = illum_psf[peak_z, peak_y, :].astype(np.float64)
        expected_sheet_thick = system_psf_profile._half_max_width(
            profile, peak_x, _REGIME_GRID_PARAMS["dxy"]
        )
        self.assertEqual(sheet_thick, expected_sheet_thick)

        self.assertGreater(sheet_thick, dof)
        self.assertGreater(dof, sheet_thin)

    def test_aslm_axial_fwhm_is_monotonic_in_slit_width(self):
        fwhms = []
        for slit_width in _SLIT_LADDER:
            _positions, fwhm = simulate.measure_aslm_system_fwhm_profile(
                positions_um=(0.0,), slit_width=slit_width, **_D13_PARAMS
            )
            fwhms.append(float(fwhm[0]))

        for i in range(1, len(fwhms)):
            with self.subTest(index=i, slit_width=_SLIT_LADDER[i]):
                self.assertGreaterEqual(fwhms[i], fwhms[i - 1] - 1e-6)

        _positions, light_sheet_fwhm = simulate.measure_light_sheet_system_fwhm_profile(
            positions_um=(0.0,), **_D13_PARAMS
        )
        self.assertEqual(fwhms[0], light_sheet_fwhm[0])

        reference = _swept_average_reference_fwhm(**_D13_PARAMS)
        relative_deviation = abs(fwhms[-1] - reference) / reference
        self.assertLessEqual(relative_deviation, 1e-4)
        self.assertGreaterEqual(fwhms[-1], 1.005 * fwhms[0])


# Task 3 shared params.
_VALID_LIGHT_SHEET_KWARGS = dict(
    positions_um=(0.0,),
    detection_na=1.0,
    illumination_na=0.45,
    wavelength=0.561,
    ni=1.33,
    ns=1.33,
    ni0=1.33,
    dxy=0.108,
    dz=0.108,
    psf_size_z=61,
    psf_size_xy=64,
)

_VALID_ASLM_KWARGS = dict(_VALID_LIGHT_SHEET_KWARGS, slit_width=2.0)


class SystemPsfProfileHygieneTests(unittest.TestCase):
    def test_system_psf_axial_fwhm_reproduces_both_regimes(self):
        # D-11/D-12: reproduce the user's two regimes with the locked
        # tau_thick/tau_thin tolerances. If the observed deviation exceeds
        # the locked tolerance, this must fail loudly -- do not widen the
        # tolerance to make it pass; report and stop instead.
        grid = dict(
            detection_na=1.1,
            wavelength=0.561,
            ni=1.33,
            ns=1.33,
            ni0=1.33,
            dxy=0.1,
            dz=0.1,
            psf_size_z=65,
            psf_size_xy=65,
        )
        dof = simulate.measure_detection_dof(**grid)
        sheet_kwargs = {k: v for k, v in grid.items() if k != "detection_na"}
        sheet_thick = simulate.measure_sheet_thickness(illumination_na=0.1, **sheet_kwargs)
        sheet_thin = simulate.measure_sheet_thickness(illumination_na=0.6, **sheet_kwargs)

        # Regime-separation preconditions (08.1-REGIME-MEASUREMENTS.md).
        self.assertGreaterEqual(sheet_thick / dof, 2.0)
        self.assertLessEqual(sheet_thin / dof, 0.75)

        for mode_name, profile_fn in (
            ("light_sheet", simulate.measure_light_sheet_system_fwhm_profile),
            ("aslm", simulate.measure_aslm_system_fwhm_profile),
        ):
            extra = {"slit_width": 0.2} if mode_name == "aslm" else {}
            with self.subTest(mode=mode_name, regime="thick"):
                _positions, fwhm = profile_fn(
                    positions_um=(0.0,), illumination_na=0.1, **grid, **extra
                )
                f = float(fwhm[0])
                self.assertLessEqual(abs(f - dof) / dof, TAU_THICK)
                self.assertLess(abs(f - dof), abs(f - sheet_thick))
            with self.subTest(mode=mode_name, regime="thin"):
                _positions, fwhm = profile_fn(
                    positions_um=(0.0,), illumination_na=0.6, **grid, **extra
                )
                f = float(fwhm[0])
                self.assertLessEqual(abs(f - sheet_thin) / sheet_thin, TAU_THIN)
                self.assertLess(abs(f - sheet_thin), abs(f - dof))

    def test_module_imports_are_confined_to_numpy_warnings_and_the_seed_functions(self):
        source = Path(system_psf_profile.__file__).read_text(encoding="utf-8")
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
        allowed_names = {"annotations", "generate_psf_seed", "generate_theoretical_psf"}
        self.assertTrue(modules <= allowed_modules, modules)
        self.assertTrue(names <= allowed_names, names)

    def test_measurement_path_contains_no_closed_form_or_fitted_model(self):
        # Stripping '#' comment lines lets this test module's own explanatory
        # prose name what was rejected without invalidating this gate -- only
        # simulate/system_psf_profile.py is scanned, never this test file.
        source = Path(system_psf_profile.__file__).read_text(encoding="utf-8")
        stripped_lines = [
            line for line in source.splitlines() if not line.strip().startswith("#")
        ]
        stripped_source = "\n".join(stripped_lines)

        forbidden_tokens = (
            "curve_fit",
            "polyfit",
            "optimize",
            "peak_widths",
            "interp1d",
            "gaussian_filter",
            "savgol",
            "sqrt",
            "tot_psf",
            "nan_to_num",
        )
        for token in forbidden_tokens:
            with self.subTest(token=token):
                self.assertNotIn(token, stripped_source)

        self.assertIn("generate_psf_seed", stripped_source)
        self.assertIn("emitter_offset", stripped_source)

    def test_rejects_invalid_parameters_before_generating_any_psf(self):
        with mock.patch.object(system_psf_profile, "generate_psf_seed") as mocked:
            invalid_positions = ((), np.zeros((2, 2)), (0.0, float("nan")))
            for bad_positions in invalid_positions:
                with self.subTest(positions_um=bad_positions):
                    with self.assertRaises(ValueError):
                        simulate.measure_light_sheet_system_fwhm_profile(
                            **dict(_VALID_LIGHT_SHEET_KWARGS, positions_um=bad_positions)
                        )
                    self.assertEqual(mocked.call_count, 0)

            with self.subTest(param="detection_na"):
                with self.assertRaises(ValueError):
                    simulate.measure_light_sheet_system_fwhm_profile(
                        **dict(_VALID_LIGHT_SHEET_KWARGS, detection_na=0)
                    )
                self.assertEqual(mocked.call_count, 0)

            with self.subTest(param="psf_size_z"):
                with self.assertRaises(ValueError):
                    simulate.measure_light_sheet_system_fwhm_profile(
                        **dict(_VALID_LIGHT_SHEET_KWARGS, psf_size_z=0)
                    )
                self.assertEqual(mocked.call_count, 0)

            with self.subTest(param="slit_width"):
                with self.assertRaises(ValueError):
                    simulate.measure_aslm_system_fwhm_profile(
                        **dict(_VALID_ASLM_KWARGS, slit_width=0)
                    )
                self.assertEqual(mocked.call_count, 0)

    def test_unmeasurable_positions_stay_nan_and_are_named_in_one_warning(self):
        with mock.patch.object(
            system_psf_profile,
            "generate_psf_seed",
            side_effect=lambda **kwargs: np.ones((9, 9, 9), dtype=np.float32),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _positions_um, fwhm_um = simulate.measure_light_sheet_system_fwhm_profile(
                    **dict(_VALID_LIGHT_SHEET_KWARGS, positions_um=(0.0, 5.0))
                )

        self.assertTrue(np.isnan(fwhm_um).all())
        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, UserWarning))
        text = str(caught[0].message)
        self.assertIn("0.0", text)
        self.assertIn("5.0", text)


if __name__ == "__main__":
    unittest.main()
