from __future__ import annotations

import inspect
import json
import math
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tiresias import seeds


def _load_legacy_rotation_baseline():
    """Load the ROT-04 regression fixture captured by plan 07-01 Task 1.

    Loaded by path, never by package import -- tests/fixtures/ has no
    __init__.py.
    """
    fixture_path = Path(__file__).parent / "fixtures" / "legacy_rotation_baseline.json"
    with fixture_path.open() as handle:
        return json.load(handle)


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

    def test_single_seed_returns_normalized_detection_psf(self):
        detection = np.arange(3 * 5 * 5, dtype=np.float32).reshape(3, 5, 5)
        expected = detection / detection.sum(dtype=np.float64)

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            return_value=detection,
        ) as generate_theoretical_psf:
            psf = seeds.generate_psf_seed(
                psf_mode="single",
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

        self.assertEqual(psf.dtype, np.float32)
        np.testing.assert_allclose(psf, expected, rtol=1e-6, atol=1e-8)
        self.assertEqual(generate_theoretical_psf.call_count, 1)

    def test_generate_psf_seed_rejects_unsupported_psf_mode(self):
        detection = np.ones((3, 5, 5), dtype=np.float32)

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            return_value=detection,
        ):
            with self.assertRaisesRegex(ValueError, "(?i)nsupported psf_mode"):
                seeds.generate_psf_seed(
                    psf_mode="not_a_real_mode",
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

    def test_light_sheet_seed_at_non_right_angle(self):
        detection = np.ones((5, 5, 5), dtype=np.float32)
        illumination = np.zeros((5, 5, 5), dtype=np.float32)
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
                psf_size_z=5,
                psf_size_xy=5,
                background=0.0,
                light_sheet_angle=45.0,
            )

        self.assertEqual(psf.shape, illumination.shape)
        self.assertEqual(psf.dtype, np.float32)
        self.assertTrue(np.isclose(psf.sum(dtype=np.float64), 1.0))

        right_angle_rotated = seeds._center_crop_or_pad(
            np.rot90(illumination, k=1, axes=(0, 2)), illumination.shape
        )
        right_angle_reference = seeds.normalise_psf(detection * right_angle_rotated)
        self.assertFalse(np.allclose(psf, right_angle_reference))

    def test_aslm_seed_real_numeric_end_to_end(self):
        common_kwargs = dict(
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
            oversample_factor=1,
            psf_model="vectorial",
            dxy=0.108,
            dz=0.3,
            psf_size_z=15,
            psf_size_xy=15,
            background=0.0,
            light_sheet_angle=90.0,
        )

        light_sheet = seeds.generate_psf_seed(psf_mode="light_sheet", **common_kwargs)
        aslm = seeds.generate_psf_seed(
            psf_mode="aslm", slit_width=0.4, **common_kwargs
        )

        self.assertEqual(aslm.shape, (15, 15, 15))
        self.assertEqual(aslm.dtype, np.float32)
        self.assertLess(abs(float(aslm.sum(dtype=np.float64)) - 1.0), 1e-5)
        self.assertFalse(np.allclose(aslm, light_sheet))

        def axis0_variance(psf):
            marginal = psf.sum(axis=(1, 2), dtype=np.float64)
            idx = np.arange(marginal.shape[0], dtype=np.float64)
            centroid = float((marginal * idx).sum() / marginal.sum())
            return float((marginal * (idx - centroid) ** 2).sum() / marginal.sum())

        self.assertLess(axis0_variance(aslm), axis0_variance(light_sheet))

    def test_psfmodels_centers_beam_waist_at_geometric_midpoint(self):
        illumination = seeds.generate_theoretical_psf(
            na=0.2,
            detection_na=0.2,
            illumination_na=0.2,
            wavelength=0.561,
            ni=1.33,
            ns=1.33,
            oversample_factor=1,
            psf_model="vectorial",
            dxy=0.108,
            dz=0.3,
            psf_size_z=15,
            psf_size_xy=15,
            background=0.0,
        )

        def centroid(marginal):
            idx = np.arange(marginal.shape[0], dtype=np.float64)
            return float((marginal * idx).sum() / marginal.sum(dtype=np.float64))

        axis0_marginal = illumination.sum(axis=(1, 2), dtype=np.float64)
        axis2_marginal = illumination.sum(axis=(0, 1), dtype=np.float64)

        self.assertLess(abs(centroid(axis0_marginal) - 7.0), 0.5)
        self.assertLess(abs(centroid(axis2_marginal) - 7.0), 0.5)

    def test_aslm_seed_multiplies_detection_by_gated_illumination(self):
        detection = np.ones((5, 5, 5), dtype=np.float32)
        illumination = np.ones((5, 5, 5), dtype=np.float32)

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            side_effect=[detection, illumination],
        ):
            psf = seeds.generate_psf_seed(
                psf_mode="aslm",
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
                psf_size_z=5,
                psf_size_xy=5,
                background=0.0,
                light_sheet_angle=90.0,
                slit_width=0.216,
            )

        sigma = (0.216 / 0.108) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        idx = np.arange(5, dtype=np.float64)
        window = np.exp(-0.5 * ((idx - 2.0) / sigma) ** 2).astype(np.float32)
        gated = illumination * window.reshape(1, 1, 5)
        expected = seeds.normalise_psf(
            detection * seeds.rotate_illumination_psf(gated, 90.0)
        )

        self.assertEqual(psf.shape, (5, 5, 5))
        np.testing.assert_allclose(psf, expected, rtol=1e-6, atol=1e-8)
        self.assertTrue(np.isclose(psf.sum(dtype=np.float64), 1.0))

    def test_aslm_slit_axis_override_forces_axis(self):
        detection = np.ones((9, 9, 9), dtype=np.float32)
        illumination = np.ones((9, 9, 9), dtype=np.float32)
        captured = {}

        def _record(illumination_arg, angle):
            captured["gated"] = np.array(illumination_arg, copy=True)
            captured["angle"] = angle
            return illumination_arg

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            side_effect=[detection, illumination],
        ), mock.patch.object(seeds, "rotate_illumination_psf", side_effect=_record):
            seeds.generate_psf_seed(
                psf_mode="aslm",
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
                psf_size_z=9,
                psf_size_xy=9,
                background=0.0,
                light_sheet_angle=90.0,
                slit_width=0.216,
                slit_axis=0,
            )

        axis0_profile = captured["gated"].sum(axis=(1, 2))
        axis2_profile = captured["gated"].sum(axis=(0, 1))
        self.assertEqual(int(np.argmax(axis0_profile)), 4)
        self.assertLess(axis0_profile[0], axis0_profile[4])
        self.assertTrue(np.allclose(axis2_profile, axis2_profile[0]))

    def test_aslm_rejects_invalid_slit_axis(self):
        detection = np.ones((9, 9, 9), dtype=np.float32)

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            return_value=detection,
        ) as generate_theoretical_psf:
            with self.assertRaisesRegex(ValueError, "slit_axis must be 0 or 2"):
                seeds.generate_psf_seed(
                    psf_mode="aslm",
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
                    psf_size_z=9,
                    psf_size_xy=9,
                    background=0.0,
                    light_sheet_angle=90.0,
                    slit_width=0.216,
                    slit_axis=1,
                )

        generate_theoretical_psf.assert_not_called()

    def _capture_aslm_gate(self, **overrides):
        """Run generate_psf_seed(psf_mode="aslm", ...), capturing the pre-rotation
        gated illumination array and the angle passed to rotate_illumination_psf."""
        detection = np.ones((9, 9, 9), dtype=np.float32)
        illumination = np.ones((9, 9, 9), dtype=np.float32)
        captured = {}

        def _record(illumination_arg, angle):
            captured["gated"] = np.array(illumination_arg, copy=True)
            captured["angle"] = angle
            return illumination_arg

        kwargs = dict(
            psf_mode="aslm",
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
            psf_size_z=9,
            psf_size_xy=9,
            background=0.0,
            slit_width=0.216,
        )
        kwargs.update(overrides)

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            side_effect=[detection, illumination],
        ), mock.patch.object(seeds, "rotate_illumination_psf", side_effect=_record):
            seeds.generate_psf_seed(**kwargs)

        return captured

    def test_aslm_gate_axis_selection_by_angle(self):
        # Table computed from int(round(angle / 90.0)) % 4 under Python's
        # round-half-to-even; see this plan's interface_context tie-angle table.
        cases = [
            (0.0, 0),
            (45.0, 0),
            (90.0, 2),
            (135.0, 0),
            (180.0, 0),
            (270.0, 2),
        ]
        for angle, expected_axis in cases:
            with self.subTest(angle=angle):
                captured = self._capture_aslm_gate(light_sheet_angle=angle)
                gated = captured["gated"]
                axis0_profile = gated.sum(axis=(1, 2))
                axis1_profile = gated.sum(axis=(0, 2))
                axis2_profile = gated.sum(axis=(0, 1))

                # Axis 1 (Y) is never gated by any code path.
                self.assertTrue(np.allclose(axis1_profile, axis1_profile[0]))

                if expected_axis == 0:
                    narrowed_profile, flat_profile = axis0_profile, axis2_profile
                else:
                    narrowed_profile, flat_profile = axis2_profile, axis0_profile

                self.assertEqual(int(np.argmax(narrowed_profile)), 4)
                self.assertLess(narrowed_profile[0], narrowed_profile[4])
                self.assertTrue(np.allclose(flat_profile, flat_profile[0]))

    def test_aslm_rotation_uses_true_angle_not_snapped_quadrant(self):
        captured = self._capture_aslm_gate(light_sheet_angle=45.0)

        axis0_profile = captured["gated"].sum(axis=(1, 2))
        self.assertEqual(int(np.argmax(axis0_profile)), 4)
        self.assertLess(axis0_profile[0], axis0_profile[4])

        # D-02: rotate_illumination_psf receives the TRUE angle, not the
        # quadrant-snapped gate axis (which resolved to axis 0 here, matching
        # the even-quadrant snap of 45.0, while the rotation angle stays 45.0).
        self.assertEqual(captured["angle"], 45.0)

    def test_aslm_invalid_slit_width_raises(self):
        detection = np.ones((9, 9, 9), dtype=np.float32)

        base_kwargs = dict(
            psf_mode="aslm",
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
            psf_size_z=9,
            psf_size_xy=9,
            background=0.0,
            light_sheet_angle=90.0,
        )

        cases = [
            ("neither form supplied", {}, "Exactly one of slit_width or slit_width_px"),
            (
                "both forms supplied",
                {"slit_width": 0.2, "slit_width_px": 2},
                "Exactly one of slit_width or slit_width_px",
            ),
            ("slit_width zero", {"slit_width": 0.0}, "slit_width must be > 0"),
            ("slit_width negative", {"slit_width": -0.5}, "slit_width must be > 0"),
            ("slit_width_px zero", {"slit_width_px": 0}, "slit_width_px must be > 0"),
            ("slit_width_px negative", {"slit_width_px": -3}, "slit_width_px must be > 0"),
        ]

        for label, extra_kwargs, expected_message in cases:
            with self.subTest(label=label):
                with mock.patch.object(
                    seeds,
                    "generate_theoretical_psf",
                    return_value=detection,
                ) as generate_theoretical_psf:
                    with self.assertRaisesRegex(ValueError, expected_message):
                        seeds.generate_psf_seed(**base_kwargs, **extra_kwargs)

                    generate_theoretical_psf.assert_not_called()

    def test_aslm_slit_width_px_matches_physical_equivalent(self):
        common_kwargs = dict(
            psf_mode="aslm",
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
            psf_size_z=9,
            psf_size_xy=9,
            background=0.0,
            light_sheet_angle=90.0,
        )

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            side_effect=[
                np.ones((9, 9, 9), dtype=np.float32),
                np.ones((9, 9, 9), dtype=np.float32),
            ],
        ):
            psf_px = seeds.generate_psf_seed(slit_width_px=2, **common_kwargs)

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            side_effect=[
                np.ones((9, 9, 9), dtype=np.float32),
                np.ones((9, 9, 9), dtype=np.float32),
            ],
        ):
            psf_physical = seeds.generate_psf_seed(
                slit_width=2 * common_kwargs["dxy"], **common_kwargs
            )

        # Doubling is exact in binary floating point, so slit_width_px=2 at
        # dxy=0.108 must gate the same pixels exactly as slit_width=2*0.108 —
        # ASLM-03's actual claim, checked without a tolerance.
        np.testing.assert_array_equal(psf_px, psf_physical)

    def test_aslm_slit_width_px_converts_via_dxy_even_on_the_z_axis(self):
        # D-09's conversion pin: slit_width_px * dxy is unconditional, even
        # though the gate axis is forced to axis 0 here (Z, spaced by dz, not
        # dxy). This test goes red if the conversion is ever "fixed" to be
        # axis-aware without revisiting the locked decision.
        common_kwargs = dict(
            psf_mode="aslm",
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
            psf_size_z=9,
            psf_size_xy=9,
            background=0.0,
            light_sheet_angle=90.0,
            slit_axis=0,
        )

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            side_effect=[
                np.ones((9, 9, 9), dtype=np.float32),
                np.ones((9, 9, 9), dtype=np.float32),
            ],
        ):
            psf_px = seeds.generate_psf_seed(slit_width_px=2, **common_kwargs)

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            side_effect=[
                np.ones((9, 9, 9), dtype=np.float32),
                np.ones((9, 9, 9), dtype=np.float32),
            ],
        ):
            psf_dxy_equivalent = seeds.generate_psf_seed(
                slit_width=2 * common_kwargs["dxy"], **common_kwargs
            )

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            side_effect=[
                np.ones((9, 9, 9), dtype=np.float32),
                np.ones((9, 9, 9), dtype=np.float32),
            ],
        ):
            psf_dz_equivalent = seeds.generate_psf_seed(
                slit_width=2 * common_kwargs["dz"], **common_kwargs
            )

        # Proves the conversion used dxy...
        np.testing.assert_array_equal(psf_px, psf_dxy_equivalent)
        # ...and NOT dz — this assertion is what turns red if someone "fixes"
        # the conversion to be axis-aware.
        self.assertFalse(np.allclose(psf_px, psf_dz_equivalent))

    def test_aslm_too_narrow_slit_width_raises(self):
        # All energy sits at gate-axis (axis 2, since light_sheet_angle=90.0)
        # index 0, distance 4 from the size-9 axis's centre index 4.
        illumination = np.zeros((9, 9, 9), dtype=np.float32)
        illumination[:, :, 0] = 1.0

        dxy = 0.108
        distance_px = 4.0

        # Positive-but-negligible slit_width, inverted from a target surviving
        # fraction rather than a magic constant: at slit_width below, the
        # Gaussian window value at distance_px equals target_fraction exactly.
        target_fraction = 1e-9  # well under the 1e-7 relative epsilon
        sigma_px = distance_px / math.sqrt(-2.0 * math.log(target_fraction))
        fwhm_px = sigma_px * 2.0 * math.sqrt(2.0 * math.log(2.0))
        negligible_slit_width = fwhm_px * dxy

        base_kwargs = dict(
            psf_mode="aslm",
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
            dxy=dxy,
            dz=0.3,
            psf_size_z=9,
            psf_size_xy=9,
            background=0.0,
            light_sheet_angle=90.0,
        )

        cases = [
            ("underflow to exact zero", 0.001),
            ("positive but negligible", negligible_slit_width),
        ]

        for label, slit_width in cases:
            with self.subTest(label=label):
                detection = np.ones((9, 9, 9), dtype=np.float32)
                with mock.patch.object(
                    seeds,
                    "generate_theoretical_psf",
                    side_effect=[detection, illumination.copy()],
                ):
                    with self.assertRaisesRegex(ValueError, "too narrow"):
                        seeds.generate_psf_seed(slit_width=slit_width, **base_kwargs)

        # Companion assertion (ASLM-06): normalise_psf's silent zero-energy
        # pass-through is exactly the behaviour the guard above must be
        # distinct from — it returns an all-zero array unchanged, no error.
        zero_psf = seeds.normalise_psf(np.zeros((9, 9, 9), dtype=np.float32))
        np.testing.assert_array_equal(zero_psf, np.zeros((9, 9, 9), dtype=np.float32))

    def test_aslm_full_extent_equals_light_sheet(self):
        dxy = 0.108
        psf_size_xy = 5
        full_extent = psf_size_xy * dxy

        base_kwargs = dict(
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
            dxy=dxy,
            dz=0.3,
            psf_size_z=5,
            psf_size_xy=psf_size_xy,
            background=0.0,
            light_sheet_angle=90.0,
        )

        def _fresh_pair():
            return [
                np.ones((5, 5, 5), dtype=np.float32),
                np.ones((5, 5, 5), dtype=np.float32),
            ]

        with mock.patch.object(
            seeds, "generate_theoretical_psf", side_effect=_fresh_pair()
        ):
            light_sheet_reference = seeds.generate_psf_seed(
                psf_mode="light_sheet", **base_kwargs
            )

        cases = [
            ("below full extent", 0.9 * full_extent, False),
            ("at full extent", full_extent, True),
            ("above full extent", 2 * full_extent, True),
        ]

        for label, slit_width, expect_equal in cases:
            with self.subTest(label=label):
                with mock.patch.object(
                    seeds, "generate_theoretical_psf", side_effect=_fresh_pair()
                ):
                    aslm_psf = seeds.generate_psf_seed(
                        psf_mode="aslm", slit_width=slit_width, **base_kwargs
                    )

                if expect_equal:
                    np.testing.assert_array_equal(aslm_psf, light_sheet_reference)
                else:
                    self.assertFalse(np.array_equal(aslm_psf, light_sheet_reference))

    def test_generate_psf_seed_has_no_timing_jitter_parameter(self):
        parameter_names = tuple(inspect.signature(seeds.generate_psf_seed).parameters)

        forbidden_substrings = (
            "jitter",
            "sync",
            "desync",
            "timing",
            "shutter",
            "delay",
        )
        for name in parameter_names:
            lowered = name.lower()
            for substring in forbidden_substrings:
                self.assertNotIn(
                    substring,
                    lowered,
                    msg=(
                        f"parameter {name!r} carries timing-jitter semantics "
                        f"(matched substring {substring!r})"
                    ),
                )

        expected_parameter_names = (
            "psf_mode", "na", "detection_na", "illumination_na", "wavelength",
            "ni", "ns", "ni0", "tg", "tg0", "ng", "ng0", "ti0",
            "oversample_factor", "psf_model", "dxy", "dz", "psf_size_z",
            "psf_size_xy", "background", "light_sheet_angle",
            "slit_width", "slit_axis", "slit_width_px",
        )
        self.assertEqual(parameter_names, expected_parameter_names)

    def test_generate_psf_seed_docstring_states_perfect_sync_assumption(self):
        # G-01-24: the negative test above (no timing-jitter *parameter*) is
        # satisfiable by an empty docstring and says nothing about what ASLM
        # mode actually models. This positive assertion is the deliberate
        # complement: __doc__ must disclose the static, midpoint-centered,
        # assumed-perfectly-synchronized model (ASLM-04), not just omit a
        # timing knob.
        doc = (seeds.generate_psf_seed.__doc__ or "").lower()
        self.assertTrue(doc, msg="generate_psf_seed.__doc__ must not be empty")

        required_substrings = (
            "static",
            "slit",
            "geometric midpoint",
            "perfectly synchronized",
            "assum",
            "jitter",
            "out of scope",
        )
        for substring in required_substrings:
            self.assertIn(
                substring,
                doc,
                msg=(
                    f"generate_psf_seed.__doc__ must disclose {substring!r} as "
                    "part of the assumed-perfect-synchronization ASLM model "
                    "(ASLM-04, G-01-24)"
                ),
            )

        # G-01-24 error-message half: the too-narrow-slit ValueError is the
        # ASLM error a user is most likely to hit while forming a mental
        # model of the mode, so it must carry the same disclosure as the
        # docstring above. Call the private helper directly rather than
        # routing through generate_psf_seed -- no mocking or psfmodels call
        # is needed, and this keeps the assertion about message text, not
        # about PSF numerics.
        illumination = np.zeros((9, 9, 9), dtype=np.float32)
        illumination[:, :, 0] = 1.0
        with self.assertRaises(ValueError) as ctx:
            seeds._apply_aslm_slit_gate(illumination, 2, 0.001, 0.108, 0.3)
        message = str(ctx.exception).lower()
        self.assertIn(
            "perfectly synchronized",
            message,
            msg="too-narrow-slit ValueError must disclose the perfectly-synchronized model (G-01-24)",
        )
        self.assertIn(
            "slit_width",
            message,
            msg="too-narrow-slit ValueError must point the user at slit_width (G-01-24)",
        )

    # Tracer (plan 07-02 Task 1): drives cli.estimate_psf_main twice -- once
    # at an oblique 3D direction, once at the default -- through the real
    # generate_psf_seed, proving the whole stack (CLI flag -> direction
    # vector -> gate-axis resolver -> physical-space rotation -> normalized
    # seed) end to end for a direction the old 1-DOF API could never reach.
    def test_light_sheet_seed_at_oblique_3d_direction_end_to_end(self):
        from tiresias import cli

        def _capture_seed(*, psf_seed, **kwargs):
            del kwargs
            return psf_seed

        common_argv = [
            "--image-path",
            "volume.tif",
            "--output-path",
            "estimated_psf.tif",
            "--detection-na",
            "1.0",
            "--illumination-na",
            "0.2",
            "--wavelength",
            "0.561",
            "--ni",
            "1.33",
            "--ns",
            "1.33",
            "--dxy",
            "0.108",
            "--dz",
            "0.3",
            "--oversample-factor",
            "1",
            "--psf-size-z",
            "15",
            "--psf-size-xy",
            "15",
            "--psf-mode",
            "light_sheet",
        ]

        seeds_by_key = {}
        with mock.patch.object(cli, "imwrite"):
            for extra_args, key in (
                ([], "default"),
                (
                    ["--illumination-polar-deg", "60", "--illumination-azimuthal-deg", "35"],
                    "oblique",
                ),
            ):
                with mock.patch.object(
                    cli, "estimate_psf_from_chunks", side_effect=_capture_seed
                ) as estimate:
                    cli.estimate_psf_main(common_argv + extra_args)
                    seeds_by_key[key] = estimate.call_args.kwargs["psf_seed"]

        default_seed = seeds_by_key["default"]
        oblique_seed = seeds_by_key["oblique"]

        self.assertEqual(oblique_seed.shape, (15, 15, 15))
        self.assertEqual(oblique_seed.dtype, np.float32)
        self.assertTrue(np.isfinite(oblique_seed).all())
        self.assertAlmostEqual(float(oblique_seed.sum(dtype=np.float64)), 1.0, delta=1e-5)
        self.assertGreater(int(np.count_nonzero(oblique_seed > 0)), 100)
        self.assertFalse(np.allclose(oblique_seed, default_seed))

    # Contract with plan 07-02 Task 2: this test exercises the pre-v1.1 API
    # today (rotate_illumination_psf(volume, angle)). 07-02 rewires this test
    # in place, under the same name and against the same fixture, to call the
    # replacement rotation API -- so `pytest -k legacy` selects the same gate
    # before and after the migration. This is a rewiring, not a rewrite or a
    # weakening of the assertion.
    def test_legacy_cardinal_rotation_matches_captured_baseline(self):
        baseline = _load_legacy_rotation_baseline()
        volumes = {
            "cubic": np.arange(1, 126, dtype=np.float32).reshape(5, 5, 5),
            "anisotropic": np.arange(1, 61, dtype=np.float32).reshape(3, 4, 5),
        }
        for volume_key, volume in volumes.items():
            for angle_key, expected in baseline["rotation"][volume_key].items():
                with self.subTest(volume=volume_key, angle=angle_key):
                    legacy_angle = float(angle_key)
                    actual = seeds.rotate_illumination_psf(volume, legacy_angle)
                    np.testing.assert_array_equal(
                        actual, np.array(expected, dtype=np.float32)
                    )

    # Contract with plan 07-02 Task 2: this test exercises the pre-v1.1 API
    # today (_resolve_slit_axis(angle)). 07-02 rewires this test in place,
    # under the same name and against the same fixture, to call the
    # replacement gate-axis API -- so `pytest -k legacy` selects the same gate
    # before and after the migration. This is a rewiring, not a rewrite or a
    # weakening of the assertion.
    def test_legacy_gate_axis_matches_captured_baseline(self):
        baseline = _load_legacy_rotation_baseline()
        # The four tie angles 45/135/225/315 are the falsification target for
        # 07-RESEARCH.md's claim that numpy.argmax's first-occurrence ordering
        # on ties reproduces the legacy round-half-to-even rule with no extra
        # tie-break code -- see 07-RESEARCH.md "Gate-Axis Resolution".
        for angle_key, expected_axis in baseline["gate_axis"].items():
            with self.subTest(angle=angle_key):
                legacy_angle = float(angle_key)
                actual_axis = seeds._resolve_slit_axis(legacy_angle)
                self.assertEqual(actual_axis, expected_axis)


if __name__ == "__main__":
    unittest.main()
