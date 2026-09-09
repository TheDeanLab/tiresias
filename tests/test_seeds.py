from __future__ import annotations

import math
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tiresias import seeds


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


if __name__ == "__main__":
    unittest.main()
