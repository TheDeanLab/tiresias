from __future__ import annotations

import inspect
import json
import math
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.signal import fftconvolve

from tiresias import seeds


def _half_max_width(profile, spacing):
    """Linear-interpolation half-max crossing width, walking outward from the peak.

    Test-local helper (plan 08.1-01, D-14) -- mirrors the linear-interpolation
    half-max crossing convention already used by simulate/beam_profile.py,
    but is not itself production code.
    """
    profile = np.asarray(profile, dtype=np.float64)
    peak_idx = int(np.argmax(profile))
    half = profile[peak_idx] / 2.0

    def _walk(step):
        idx = peak_idx
        while 0 <= idx + step < len(profile):
            prev = idx
            idx += step
            if profile[idx] <= half:
                span = profile[prev] - profile[idx]
                frac = 0.0 if span == 0 else (profile[prev] - half) / span
                return prev + frac * step
        raise ValueError("half-max crossing not found within profile bounds")

    left_idx = _walk(-1)
    right_idx = _walk(1)
    return (right_idx - left_idx) * spacing


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
                polar_deg=90.0,
                azimuthal_deg=0.0,
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
                polar_deg=90.0,
                azimuthal_deg=0.0,
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
                    polar_deg=90.0,
                    azimuthal_deg=0.0,
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
                polar_deg=45.0,
                azimuthal_deg=0.0,
            )

        self.assertEqual(psf.shape, illumination.shape)
        self.assertEqual(psf.dtype, np.float32)
        self.assertTrue(np.isclose(psf.sum(dtype=np.float64), 1.0))

        right_angle_rotated = seeds._center_crop_or_pad(
            np.rot90(illumination, k=1, axes=(0, 2)), illumination.shape
        )
        right_angle_reference = seeds.normalise_psf(detection * right_angle_rotated)
        self.assertFalse(np.allclose(psf, right_angle_reference))

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

    def test_aslm_seed_multiplies_detection_by_sweep_integrated_illumination(self):
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
                polar_deg=90.0,
                azimuthal_deg=0.0,
                slit_width=0.6,
            )

        sigma = (0.6 / 0.3) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        k = np.arange(9, dtype=np.float64)
        taps = np.exp(-0.5 * ((k - 4.0) / sigma) ** 2).astype(np.float32)
        gated = np.clip(
            fftconvolve(illumination, taps.reshape(-1, 1, 1), mode="same", axes=0),
            0.0,
            None,
        ).astype(np.float32)
        expected = seeds.normalise_psf(
            detection
            * seeds.rotate_illumination(
                gated, polar_deg=90.0, azimuthal_deg=0.0, dxy=0.108, dz=0.3
            )
        )

        self.assertEqual(psf.shape, (5, 5, 5))
        np.testing.assert_allclose(psf, expected, rtol=1e-6, atol=1e-8)
        self.assertTrue(np.isclose(psf.sum(dtype=np.float64), 1.0))

    def test_aslm_rejects_out_of_range_slit_axis(self):
        # D-05: slit_axis=1 (Y) is now a VALID gate axis; only out-of-range
        # values are rejected.
        detection = np.ones((9, 9, 9), dtype=np.float32)

        for invalid_slit_axis in (3, -1):
            with self.subTest(slit_axis=invalid_slit_axis):
                with mock.patch.object(
                    seeds,
                    "generate_theoretical_psf",
                    return_value=detection,
                ) as generate_theoretical_psf:
                    with self.assertRaisesRegex(ValueError, "slit_axis must be 0, 1, or 2"):
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
                            polar_deg=90.0,
                            azimuthal_deg=0.0,
                            slit_width=0.216,
                            slit_axis=invalid_slit_axis,
                        )

                generate_theoretical_psf.assert_not_called()

    def _capture_aslm_gate(self, **overrides):
        """Run generate_psf_seed(psf_mode="aslm", ...), capturing the pre-rotation
        gated illumination array and the (polar_deg, azimuthal_deg) passed to
        rotate_illumination."""
        detection = np.ones((9, 9, 9), dtype=np.float32)
        illumination = np.ones((9, 9, 9), dtype=np.float32)
        captured = {}

        def _record(illumination_arg, *, polar_deg, azimuthal_deg, dxy, dz):
            captured["gated"] = np.array(illumination_arg, copy=True)
            captured["polar_deg"] = polar_deg
            captured["azimuthal_deg"] = azimuthal_deg
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
            # D-18: 0.9 is 3 samples at dz=0.3, well above the one-dz-sample
            # floor, so this default always exercises the convolution branch
            # (0.216 now falls below one dz sample and would skip it).
            slit_width=0.9,
        )
        kwargs.update(overrides)

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            side_effect=[detection, illumination],
        ), mock.patch.object(seeds, "rotate_illumination", side_effect=_record):
            seeds.generate_psf_seed(**kwargs)

        return captured

    def test_aslm_convolves_pre_rotation_axis_0_for_every_direction(self):
        # D-02: the convolution runs on pre-rotation axis 0 before rotation,
        # for every direction including oblique/out-of-plane ones -- no
        # camera-axis snapping decides the integration axis.
        directions = [
            (0.0, 0.0),
            (45.0, 0.0),
            (90.0, 0.0),
            (135.0, 0.0),
            (180.0, 0.0),
            (90.0, 180.0),
            (90.0, 90.0),
            (90.0, 45.0),
            (70.0, 40.0),
        ]
        for polar_deg, azimuthal_deg in directions:
            with self.subTest(polar_deg=polar_deg, azimuthal_deg=azimuthal_deg):
                captured = self._capture_aslm_gate(
                    polar_deg=polar_deg, azimuthal_deg=azimuthal_deg
                )
                gated = captured["gated"]
                axis0_profile = gated.sum(axis=(1, 2))
                axis1_profile = gated.sum(axis=(0, 2))
                axis2_profile = gated.sum(axis=(0, 1))

                self.assertEqual(int(np.argmax(axis0_profile)), 4)
                self.assertLess(axis0_profile[0], axis0_profile[4])
                self.assertTrue(np.allclose(axis1_profile, axis1_profile[0]))
                self.assertTrue(np.allclose(axis2_profile, axis2_profile[0]))

    def test_aslm_rotation_receives_true_direction_not_snapped_gate_axis(self):
        captured = self._capture_aslm_gate(polar_deg=45.0, azimuthal_deg=0.0)

        axis0_profile = captured["gated"].sum(axis=(1, 2))
        self.assertEqual(int(np.argmax(axis0_profile)), 4)
        self.assertLess(axis0_profile[0], axis0_profile[4])

        # D-02: the convolution is always axis 0 in the pre-rotation frame --
        # there is no gate-axis snapping anymore. rotate_illumination
        # receives the TRUE direction (polar_deg=45.0, azimuthal_deg=0.0).
        self.assertEqual(captured["polar_deg"], 45.0)
        self.assertEqual(captured["azimuthal_deg"], 0.0)

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
            polar_deg=90.0,
            azimuthal_deg=0.0,
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

    def test_aslm_slit_width_px_converts_via_dz(self):
        # D-17: slit_width_px converts via dz, the sample spacing of the
        # pre-rotation integration axis -- not dxy (Phase 1 D-09, retired).
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
            polar_deg=90.0,
            azimuthal_deg=0.0,
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
            psf_dz_equivalent = seeds.generate_psf_seed(
                slit_width=2 * common_kwargs["dz"], **common_kwargs
            )

        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            side_effect=[
                np.ones((9, 9, 9), dtype=np.float32),
                np.ones((9, 9, 9), dtype=np.float32),
            ],
        ):
            # 2*dxy = 0.216 < dz = 0.3, one dz sample -- this falls under the
            # D-18 waist-limited fallback (no convolution), a completely
            # different result from the real convolution at 2*dz.
            psf_dxy_equivalent = seeds.generate_psf_seed(
                slit_width=2 * common_kwargs["dxy"], **common_kwargs
            )

        # Doubling is exact in binary floating point, so slit_width_px=2 at
        # dz=0.3 must gate identically to slit_width=2*0.3, checked without a
        # tolerance.
        np.testing.assert_array_equal(psf_px, psf_dz_equivalent)
        # ...and NOT dxy — this assertion is what turns red if someone
        # "fixes" the conversion back to Phase 1 D-09's dxy-always rule.
        self.assertFalse(np.allclose(psf_px, psf_dxy_equivalent))

    def test_aslm_sub_sample_slit_falls_back_to_the_waist_limited_seed(self):
        # D-18: below one dz sample, no convolution runs -- the seed is
        # bit-identical to light_sheet (the waist-limited limit).
        dz = 0.3
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
            dxy=0.108,
            dz=dz,
            psf_size_z=9,
            psf_size_xy=9,
            background=0.0,
            polar_deg=90.0,
            azimuthal_deg=0.0,
        )

        def _fresh_arms():
            detection = np.ones((9, 9, 9), dtype=np.float32)
            illumination = (
                np.random.default_rng(0).random((9, 9, 9), dtype=np.float32) + 0.1
            )
            return [detection, illumination.copy()]

        with mock.patch.object(
            seeds, "generate_theoretical_psf", side_effect=_fresh_arms()
        ):
            light_sheet = seeds.generate_psf_seed(psf_mode="light_sheet", **base_kwargs)

        with mock.patch.object(
            seeds, "generate_theoretical_psf", side_effect=_fresh_arms()
        ):
            below_sample = seeds.generate_psf_seed(
                psf_mode="aslm", slit_width=0.999 * dz, **base_kwargs
            )

        with mock.patch.object(
            seeds, "generate_theoretical_psf", side_effect=_fresh_arms()
        ):
            at_sample = seeds.generate_psf_seed(
                psf_mode="aslm", slit_width=dz, **base_kwargs
            )

        np.testing.assert_array_equal(below_sample, light_sheet)
        self.assertFalse(np.array_equal(at_sample, light_sheet))

    def test_aslm_rejects_illumination_with_no_positive_finite_energy(self):
        # D-18 guard: the helper raises directly on zero and NaN illumination...
        with self.assertRaisesRegex(ValueError, "slit_width"):
            seeds._apply_aslm_slit_gate(
                np.zeros((9, 9, 9), dtype=np.float32), 0.9, 0.3
            )

        nan_illumination = np.ones((9, 9, 9), dtype=np.float32)
        nan_illumination[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "slit_width"):
            seeds._apply_aslm_slit_gate(nan_illumination, 0.9, 0.3)

        # ...and generate_psf_seed raises through the same guard when the
        # mocked illumination arm is all zero.
        detection = np.ones((9, 9, 9), dtype=np.float32)
        illumination = np.zeros((9, 9, 9), dtype=np.float32)
        with mock.patch.object(
            seeds,
            "generate_theoretical_psf",
            side_effect=[detection, illumination],
        ):
            with self.assertRaisesRegex(ValueError, "slit_width"):
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
                    polar_deg=90.0,
                    azimuthal_deg=0.0,
                    slit_width=0.9,
                )

    def test_aslm_wide_slit_converges_to_the_swept_average_sheet(self):
        # D-15: a very wide slit no longer takes the old full-extent
        # shortcut back to exact light_sheet -- it follows the physics and
        # converges to the swept-average sheet (uniform window over the
        # whole simulated window).
        common_kwargs = dict(
            na=1.0,
            detection_na=1.0,
            illumination_na=0.4,
            wavelength=0.561,
            ni=1.33,
            ns=1.33,
            ni0=1.33,
            tg=None,
            tg0=None,
            ng=None,
            ng0=None,
            ti0=None,
            oversample_factor=1,
            psf_model="vectorial",
            dxy=0.1,
            dz=0.1,
            psf_size_z=15,
            psf_size_xy=15,
            background=0.0,
            polar_deg=90.0,
            azimuthal_deg=0.0,
        )
        arm_kwargs = {
            key: value
            for key, value in common_kwargs.items()
            if key not in ("detection_na", "illumination_na", "polar_deg", "azimuthal_deg")
        }

        light_sheet = seeds.generate_psf_seed(psf_mode="light_sheet", **common_kwargs)

        detection_arm = seeds.generate_theoretical_psf(
            detection_na=common_kwargs["detection_na"],
            illumination_na=common_kwargs["illumination_na"],
            **arm_kwargs,
        )
        illumination_arm = seeds.generate_theoretical_psf(
            detection_na=common_kwargs["illumination_na"],
            illumination_na=common_kwargs["illumination_na"],
            **arm_kwargs,
        )

        n = common_kwargs["psf_size_z"]
        uniform_window = np.ones(2 * n - 1, dtype=np.float32)
        swept_average_gated = np.clip(
            fftconvolve(
                illumination_arm,
                uniform_window.reshape(-1, 1, 1),
                mode="same",
                axes=0,
            ),
            0.0,
            None,
        ).astype(np.float32)
        swept_average_rotated = seeds.rotate_illumination(
            swept_average_gated,
            polar_deg=common_kwargs["polar_deg"],
            azimuthal_deg=common_kwargs["azimuthal_deg"],
            dxy=common_kwargs["dxy"],
            dz=common_kwargs["dz"],
        )
        reference = seeds.normalise_psf(detection_arm * swept_average_rotated)

        wide_seed = seeds.generate_psf_seed(
            psf_mode="aslm", slit_width=1e6, **common_kwargs
        )
        np.testing.assert_allclose(wide_seed, reference, rtol=1e-4, atol=1e-9)
        self.assertFalse(np.allclose(wide_seed, light_sheet))

        length = common_kwargs["psf_size_z"] * common_kwargs["dz"]
        widths = [0.5 * length, length, 2 * length, 8 * length]
        distances = []
        for slit_width in widths:
            seed = seeds.generate_psf_seed(
                psf_mode="aslm", slit_width=slit_width, **common_kwargs
            )
            distances.append(
                float(
                    np.linalg.norm(
                        (seed.astype(np.float64) - reference.astype(np.float64)).ravel()
                    )
                )
            )

        for earlier, later in zip(distances, distances[1:]):
            self.assertLessEqual(later, earlier + 1e-12)

    def test_light_sheet_and_single_match_independent_recomposition(self):
        common_kwargs = dict(
            na=1.0,
            detection_na=1.0,
            illumination_na=0.4,
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
        )
        arm_kwargs = {
            key: value
            for key, value in common_kwargs.items()
            if key not in ("detection_na", "illumination_na")
        }

        directions = [(90.0, 0.0), (70.0, 40.0)]
        for polar_deg, azimuthal_deg in directions:
            with self.subTest(polar_deg=polar_deg, azimuthal_deg=azimuthal_deg):
                light_sheet = seeds.generate_psf_seed(
                    psf_mode="light_sheet",
                    polar_deg=polar_deg,
                    azimuthal_deg=azimuthal_deg,
                    **common_kwargs,
                )
                single = seeds.generate_psf_seed(
                    psf_mode="single",
                    polar_deg=polar_deg,
                    azimuthal_deg=azimuthal_deg,
                    **common_kwargs,
                )

                detection_arm = seeds.generate_theoretical_psf(
                    detection_na=common_kwargs["detection_na"],
                    illumination_na=common_kwargs["illumination_na"],
                    **arm_kwargs,
                )
                illumination_arm = seeds.generate_theoretical_psf(
                    detection_na=common_kwargs["illumination_na"],
                    illumination_na=common_kwargs["illumination_na"],
                    **arm_kwargs,
                )
                rotated = seeds.rotate_illumination(
                    illumination_arm,
                    polar_deg=polar_deg,
                    azimuthal_deg=azimuthal_deg,
                    dxy=common_kwargs["dxy"],
                    dz=common_kwargs["dz"],
                )
                expected_light_sheet = seeds.normalise_psf(detection_arm * rotated)
                expected_single = seeds.normalise_psf(detection_arm)

                np.testing.assert_array_equal(light_sheet, expected_light_sheet)
                np.testing.assert_array_equal(single, expected_single)

    def test_aslm_uses_the_same_pencil_beam_illumination_as_light_sheet(self):
        # D-04: ASLM uses the identical 3-D pencil-beam illumination arm as
        # light_sheet -- only the sweep-integration convolution is new.
        recorded_calls = []

        def _record(*args, **kwargs):
            recorded_calls.append(kwargs)
            return np.ones((9, 9, 9), dtype=np.float32)

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
            oversample_factor=3,
            psf_model="vectorial",
            dxy=0.108,
            dz=0.3,
            psf_size_z=9,
            psf_size_xy=9,
            background=0.0,
            polar_deg=90.0,
            azimuthal_deg=0.0,
        )

        with mock.patch.object(seeds, "generate_theoretical_psf", side_effect=_record):
            seeds.generate_psf_seed(psf_mode="light_sheet", **common_kwargs)
        light_sheet_illumination_call = recorded_calls[1]
        recorded_calls.clear()

        with mock.patch.object(seeds, "generate_theoretical_psf", side_effect=_record):
            seeds.generate_psf_seed(psf_mode="aslm", slit_width=0.5, **common_kwargs)
        aslm_illumination_call = recorded_calls[1]

        self.assertEqual(light_sheet_illumination_call, aslm_illumination_call)

    def test_aslm_seed_matches_sweep_integrated_recomposition(self):
        common_kwargs = dict(
            na=1.0,
            detection_na=1.0,
            illumination_na=0.4,
            wavelength=0.561,
            ni=1.33,
            ns=1.33,
            ni0=1.33,
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
            slit_width=0.9,
        )
        arm_kwargs = dict(
            na=common_kwargs["na"],
            wavelength=common_kwargs["wavelength"],
            ni=common_kwargs["ni"],
            ns=common_kwargs["ns"],
            ni0=common_kwargs["ni0"],
            tg=common_kwargs["tg"],
            tg0=common_kwargs["tg0"],
            ng=common_kwargs["ng"],
            ng0=common_kwargs["ng0"],
            ti0=common_kwargs["ti0"],
            oversample_factor=common_kwargs["oversample_factor"],
            psf_model=common_kwargs["psf_model"],
            dxy=common_kwargs["dxy"],
            dz=common_kwargs["dz"],
            psf_size_z=common_kwargs["psf_size_z"],
            psf_size_xy=common_kwargs["psf_size_xy"],
            background=common_kwargs["background"],
        )
        light_sheet_kwargs = {
            key: value for key, value in common_kwargs.items() if key != "slit_width"
        }

        directions = [(90.0, 0.0), (70.0, 40.0)]
        for polar_deg, azimuthal_deg in directions:
            with self.subTest(polar_deg=polar_deg, azimuthal_deg=azimuthal_deg):
                aslm = seeds.generate_psf_seed(
                    psf_mode="aslm",
                    polar_deg=polar_deg,
                    azimuthal_deg=azimuthal_deg,
                    **common_kwargs,
                )
                light_sheet = seeds.generate_psf_seed(
                    psf_mode="light_sheet",
                    polar_deg=polar_deg,
                    azimuthal_deg=azimuthal_deg,
                    **light_sheet_kwargs,
                )

                detection_arm = seeds.generate_theoretical_psf(
                    detection_na=common_kwargs["detection_na"],
                    illumination_na=common_kwargs["illumination_na"],
                    **arm_kwargs,
                )
                illumination_arm = seeds.generate_theoretical_psf(
                    detection_na=common_kwargs["illumination_na"],
                    illumination_na=common_kwargs["illumination_na"],
                    **arm_kwargs,
                )

                n = common_kwargs["psf_size_z"]
                sigma = (0.9 / 0.3) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
                k = np.arange(2 * n - 1, dtype=np.float64)
                taps = np.exp(-0.5 * ((k - (n - 1)) / sigma) ** 2).astype(np.float32)

                gated = np.clip(
                    fftconvolve(
                        illumination_arm, taps.reshape(-1, 1, 1), mode="same", axes=0
                    ),
                    0.0,
                    None,
                ).astype(np.float32)

                rotated = seeds.rotate_illumination(
                    gated,
                    polar_deg=polar_deg,
                    azimuthal_deg=azimuthal_deg,
                    dxy=common_kwargs["dxy"],
                    dz=common_kwargs["dz"],
                )
                expected = seeds.normalise_psf(detection_arm * rotated)

                np.testing.assert_allclose(aslm, expected, rtol=1e-5, atol=1e-9)
                self.assertFalse(np.allclose(aslm, light_sheet))

    def test_aslm_removes_out_of_focus_energy_at_every_z_beyond_dof(self):
        common_kwargs = dict(
            na=1.1,
            detection_na=1.1,
            illumination_na=0.6,
            wavelength=0.561,
            ni=1.33,
            ns=1.33,
            ni0=1.33,
            tg=None,
            tg0=None,
            ng=None,
            ng0=None,
            ti0=None,
            oversample_factor=3,
            psf_model="vectorial",
            dxy=0.1,
            dz=0.1,
            psf_size_z=61,
            psf_size_xy=64,
            background=0.0,
            polar_deg=90.0,
            azimuthal_deg=0.0,
        )
        dxy = common_kwargs["dxy"]
        dz = common_kwargs["dz"]

        single = seeds.generate_psf_seed(psf_mode="single", **common_kwargs)
        light_sheet = seeds.generate_psf_seed(psf_mode="light_sheet", **common_kwargs)
        aslm = seeds.generate_psf_seed(psf_mode="aslm", slit_width=2.0, **common_kwargs)

        single_peak = np.unravel_index(np.argmax(single), single.shape)
        z0_single = single_peak[0]
        axial_profile_single = single[:, single_peak[1], single_peak[2]]
        dof = _half_max_width(axial_profile_single, dz)

        lateral_profile_single = single[z0_single, single_peak[1], :]
        r_core = _half_max_width(lateral_profile_single, dxy) / 2.0

        def _core_fraction_out(psf):
            peak = np.unravel_index(np.argmax(psf), psf.shape)
            z0, y0, x0 = peak
            yy, xx = np.meshgrid(
                np.arange(psf.shape[1]), np.arange(psf.shape[2]), indexing="ij"
            )
            core_mask = np.hypot((yy - y0) * dxy, (xx - x0) * dxy) <= r_core
            e_plane = psf.sum(axis=(1, 2), dtype=np.float64)
            e_core = (psf * core_mask[None, :, :]).sum(axis=(1, 2), dtype=np.float64)
            fraction_out = np.ones_like(e_plane)
            positive = e_plane > 0
            fraction_out[positive] = 1.0 - (e_core[positive] / e_plane[positive])
            return z0, e_plane, fraction_out

        z0_light_sheet, e_plane_light_sheet, fraction_out_light_sheet = (
            _core_fraction_out(light_sheet)
        )
        _, e_plane_aslm, fraction_out_aslm = _core_fraction_out(aslm)

        max_e_light_sheet = e_plane_light_sheet.max()
        max_e_aslm = e_plane_aslm.max()

        tested_planes = []
        for z in range(single.shape[0]):
            if abs(z - z0_light_sheet) * dz <= dof + dz:
                continue
            if e_plane_light_sheet[z] < 1e-6 * max_e_light_sheet:
                continue
            if e_plane_aslm[z] < 1e-6 * max_e_aslm:
                continue
            tested_planes.append(z)

        self.assertGreaterEqual(len(tested_planes), 10)
        self.assertTrue(any(z < z0_light_sheet for z in tested_planes))
        self.assertTrue(any(z > z0_light_sheet for z in tested_planes))

        violations = [
            (z, float(fraction_out_aslm[z]), float(fraction_out_light_sheet[z]))
            for z in tested_planes
            if fraction_out_aslm[z] > fraction_out_light_sheet[z] + 1e-6
        ]
        self.assertEqual(
            violations,
            [],
            msg=f"D-14 violated at planes (z, aslm, light_sheet): {violations}",
        )

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
            "psf_size_xy", "background", "polar_deg", "azimuthal_deg",
            "slit_width", "slit_axis", "slit_width_px", "emitter_offset",
        )
        self.assertEqual(parameter_names, expected_parameter_names)

    def test_generate_theoretical_psf_focus_offset_zero_keeps_the_integer_z_request(self):
        raw = np.ones((5, 7, 7), dtype=np.float32)

        with mock.patch.object(seeds.pm, "make_psf", return_value=raw) as make_psf:
            seeds.generate_theoretical_psf(
                detection_na=1.0,
                wavelength=0.561,
                ni=1.33,
                ns=1.33,
                dxy=0.108,
                dz=0.3,
                psf_size_z=5,
                psf_size_xy=7,
                focus_offset=0.0,
            )
        self.assertEqual(make_psf.call_args.kwargs["z"], 5)
        self.assertIn("dz", make_psf.call_args.kwargs)

        with mock.patch.object(seeds.pm, "make_psf", return_value=raw) as make_psf:
            seeds.generate_theoretical_psf(
                detection_na=1.0,
                wavelength=0.561,
                ni=1.33,
                ns=1.33,
                dxy=0.108,
                dz=0.3,
                psf_size_z=5,
                psf_size_xy=7,
                focus_offset=0.5,
            )
        lim = (5 - 1) * 0.3 / 2
        expected_z = np.linspace(-lim + 0.5, lim + 0.5, 5)
        np.testing.assert_array_equal(make_psf.call_args.kwargs["z"], expected_z)
        self.assertNotIn("dz", make_psf.call_args.kwargs)

        with mock.patch.object(seeds.pm, "make_psf", return_value=raw) as make_psf:
            with self.assertRaisesRegex(ValueError, "focus_offset must be finite"):
                seeds.generate_theoretical_psf(
                    detection_na=1.0,
                    wavelength=0.561,
                    ni=1.33,
                    ns=1.33,
                    dxy=0.108,
                    dz=0.3,
                    psf_size_z=5,
                    psf_size_xy=7,
                    focus_offset=float("nan"),
                )
            make_psf.assert_not_called()

    def test_emitter_offset_validation_rejects_before_any_psf_generation(self):
        detection = np.ones((5, 5, 5), dtype=np.float32)
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
            dxy=0.108,
            dz=0.3,
            psf_size_z=5,
            psf_size_xy=5,
            background=0.0,
            polar_deg=90.0,
            azimuthal_deg=0.0,
        )

        cases = [
            (
                "single, nan",
                "single",
                {"emitter_offset": float("nan")},
                "emitter_offset must be finite",
            ),
            (
                "light_sheet, inf",
                "light_sheet",
                {"emitter_offset": float("inf")},
                "emitter_offset must be finite",
            ),
            (
                "aslm, nan",
                "aslm",
                {"emitter_offset": float("nan"), "slit_width": 0.9},
                "emitter_offset must be finite",
            ),
            (
                "single, nonzero",
                "single",
                {"emitter_offset": 1.0},
                "emitter_offset applies only to",
            ),
        ]

        for label, psf_mode, extra_kwargs, expected_message in cases:
            with self.subTest(label=label):
                with mock.patch.object(
                    seeds,
                    "generate_theoretical_psf",
                    return_value=detection,
                ) as generate_theoretical_psf:
                    with self.assertRaisesRegex(ValueError, expected_message):
                        seeds.generate_psf_seed(
                            psf_mode=psf_mode, **base_kwargs, **extra_kwargs
                        )
                    generate_theoretical_psf.assert_not_called()

    def test_emitter_offset_zero_is_bit_identical_to_omitting_it(self):
        # Real psfmodels, parameters as in plan 08.1-01's broadside_aniso SC5
        # config, with slit_width=0.9 added for aslm.
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
            polar_deg=90.0,
            azimuthal_deg=0.0,
        )

        for psf_mode, extra in (
            ("single", {}),
            ("light_sheet", {}),
            ("aslm", {"slit_width": 0.9}),
        ):
            with self.subTest(psf_mode=psf_mode):
                omitted = seeds.generate_psf_seed(
                    psf_mode=psf_mode, **common_kwargs, **extra
                )
                explicit_zero = seeds.generate_psf_seed(
                    psf_mode=psf_mode, emitter_offset=0.0, **common_kwargs, **extra
                )
                np.testing.assert_array_equal(omitted, explicit_zero)

    def test_aslm_seed_is_invariant_to_emitter_offset(self):
        # D-07: the ASLM waist is swept over the whole simulated window with
        # the rolling-shutter slit perfectly synchronized to the emitter's
        # row, so the ASLM seed is bit-identical for every finite
        # emitter_offset.
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
            polar_deg=90.0,
            azimuthal_deg=0.0,
            slit_width=0.9,
        )

        baseline = seeds.generate_psf_seed(
            psf_mode="aslm", emitter_offset=0.0, **common_kwargs
        )
        for emitter_offset in (5.0, -12.0):
            with self.subTest(emitter_offset=emitter_offset):
                offset_seed = seeds.generate_psf_seed(
                    psf_mode="aslm", emitter_offset=emitter_offset, **common_kwargs
                )
                np.testing.assert_array_equal(offset_seed, baseline)

    def test_light_sheet_offset_changes_the_seed(self):
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
            polar_deg=90.0,
            azimuthal_deg=0.0,
        )

        baseline = seeds.generate_psf_seed(
            psf_mode="light_sheet", emitter_offset=0.0, **common_kwargs
        )
        offset_seed = seeds.generate_psf_seed(
            psf_mode="light_sheet", emitter_offset=1.0, **common_kwargs
        )
        self.assertFalse(np.allclose(baseline, offset_seed))

    def test_light_sheet_offset_is_applied_before_rotation_at_an_oblique_direction(self):
        # D-02: the offset is applied in the pre-rotation frame, before
        # rotate_illumination, so it stays correct at an oblique direction.
        arm_kwargs = dict(
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
        )
        detection_na = 1.0
        illumination_na = 0.2
        emitter_offset = 1.0

        actual = seeds.generate_psf_seed(
            psf_mode="light_sheet",
            na=detection_na,
            detection_na=detection_na,
            illumination_na=illumination_na,
            polar_deg=70.0,
            azimuthal_deg=40.0,
            emitter_offset=emitter_offset,
            **arm_kwargs,
        )

        detection_arm = seeds.generate_theoretical_psf(
            detection_na=detection_na,
            illumination_na=illumination_na,
            **arm_kwargs,
        )
        illumination_arm = seeds.generate_theoretical_psf(
            detection_na=illumination_na,
            illumination_na=illumination_na,
            focus_offset=emitter_offset,
            **arm_kwargs,
        )
        rotated = seeds.rotate_illumination(
            illumination_arm,
            polar_deg=70.0,
            azimuthal_deg=40.0,
            dxy=arm_kwargs["dxy"],
            dz=arm_kwargs["dz"],
        )
        expected = seeds.normalise_psf(detection_arm * rotated)

        np.testing.assert_array_equal(actual, expected)

    def test_generate_psf_seed_docstring_states_perfect_sync_assumption(self):
        # G-01-24: the negative test above (no timing-jitter *parameter*) is
        # satisfiable by an empty docstring and says nothing about what ASLM
        # mode actually models. This positive assertion is the deliberate
        # complement: __doc__ must disclose the sweep-integrated,
        # assumed-perfectly-synchronized model (D-01/D-02), not just omit a
        # timing knob.
        doc = (seeds.generate_psf_seed.__doc__ or "").lower()
        self.assertTrue(doc, msg="generate_psf_seed.__doc__ must not be empty")

        required_substrings = (
            "sweep",
            "convol",
            "propagation",
            "slit",
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
                    "part of the sweep-integrated, assumed-perfect-synchronization "
                    "ASLM model (D-01/D-02)"
                ),
            )

        # G-01-24 error-message half: the no-positive-energy ValueError is the
        # ASLM error a user is most likely to hit while forming a mental
        # model of the mode, so it must carry the same disclosure as the
        # docstring above. Call the private helper directly rather than
        # routing through generate_psf_seed -- no mocking or psfmodels call
        # is needed, and this keeps the assertion about message text, not
        # about PSF numerics.
        with self.assertRaises(ValueError) as ctx:
            seeds._apply_aslm_slit_gate(
                np.zeros((9, 9, 9), dtype=np.float32), 0.9, 0.3
            )
        message = str(ctx.exception).lower()
        self.assertIn(
            "perfectly synchronized",
            message,
            msg="ASLM energy-guard ValueError must disclose the perfectly-synchronized model (G-01-24)",
        )
        self.assertIn(
            "slit_width",
            message,
            msg="ASLM energy-guard ValueError must point the user at slit_width (G-01-24)",
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

    # Rewired by plan 07-02 Task 2: exercises the new rotate_illumination(...)
    # replacement API against the exact same unmodified 07-01 fixture, so
    # `pytest -k legacy` selects the same gate before and after the
    # migration. This is a rewiring, not a rewrite or a weakening of the
    # assertion; tests/fixtures/legacy_rotation_baseline.json is untouched.
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
                    # D-03's mapping: the legacy single-angle parameter maps
                    # onto (polar=legacy_angle, azimuthal=0), because
                    # Ry(angle) applied to the pre-rotation +Z axis gives
                    # exactly the Z/X-plane direction the old parameter
                    # described. dxy != dz is deliberate: at a legacy
                    # cardinal direction the fast path must ignore both, so a
                    # green assertion under anisotropic spacing proves
                    # delegation happened rather than the isotropic pipeline.
                    actual = seeds.rotate_illumination(
                        volume,
                        polar_deg=legacy_angle,
                        azimuthal_deg=0.0,
                        dxy=0.108,
                        dz=0.300,
                    )
                    np.testing.assert_array_equal(
                        actual, np.array(expected, dtype=np.float32)
                    )

    # Rewired by plan 07-02 Task 2: exercises the new
    # _resolve_slit_axis(_spherical_direction(...)) replacement API against
    # the exact same unmodified 07-01 fixture, so `pytest -k legacy` selects
    # the same gate before and after the migration. This is a rewiring, not
    # a rewrite or a weakening of the assertion.
    def test_legacy_gate_axis_matches_captured_baseline(self):
        baseline = _load_legacy_rotation_baseline()
        # The four tie angles 45/135/225/315 are the falsification target for
        # 07-RESEARCH.md's claim that numpy.argmax's first-occurrence ordering
        # on ties reproduces the legacy round-half-to-even rule with no extra
        # tie-break code -- see 07-RESEARCH.md "Gate-Axis Resolution".
        for angle_key, expected_axis in baseline["gate_axis"].items():
            with self.subTest(angle=angle_key):
                legacy_angle = float(angle_key)
                actual_axis = seeds._resolve_slit_axis(
                    seeds._spherical_direction(legacy_angle, 0.0)
                )
                self.assertEqual(actual_axis, expected_axis)

    # Plan 07-03 Task 1 (ROT-02): STATE.md records rotation-formula
    # validation against dz != dxy as this phase's principal research risk,
    # requiring an explicit physical-versus-index-space regression test.
    # 07-RESEARCH.md's "Architecture Patterns" Pattern 1 empirically measured
    # 0.0102 physical-pipeline error and 0.641 naive-rotation error for this
    # exact recipe; this test makes that verification permanent.
    def test_rotation_is_physical_not_index_space_under_anisotropic_voxels(self):
        dxy = 0.108
        dz = 0.300
        z = np.arange(61, dtype=np.float64)
        # Intensity varies only along Z as a sigma=1.0um Gaussian, constant
        # across Y/X -- dz (0.300) samples Z 2.78x more coarsely than dxy
        # (0.108) samples the lateral axes. This anisotropy is the whole
        # point of the fixture and must not be changed to equal spacings.
        profile = np.exp(-0.5 * (((z - 30) * dz) / 1.0) ** 2).astype(np.float32)
        volume = np.broadcast_to(
            profile[:, None, None], (61, 61, 61)
        ).astype(np.float32).copy()

        # (polar_deg=90, azimuthal_deg=90) is pure +Y -- deliberately NOT one
        # of the four legacy cardinal directions, so the call exercises the
        # general _rotate_isotropic path rather than the np.rot90 fast path.
        # Asserted explicitly so a future change routing +Y through the fast
        # path cannot make this test silently vacuous.
        direction = seeds._spherical_direction(90.0, 90.0)
        self.assertIsNone(seeds._match_legacy_cardinal(direction))

        rotated = seeds.rotate_illumination(
            volume, polar_deg=90.0, azimuthal_deg=90.0, dxy=dxy, dz=dz
        )
        lineout = rotated[30, :, 30]
        expected = np.exp(-0.5 * (((np.arange(61) - 30) * dxy) / 1.0) ** 2)
        physical_error = float(np.abs(lineout - expected).max())
        # Measured 0.0102 (07-RESEARCH.md) / 0.01016 (this session, this
        # commit -- see 07-03-SUMMARY.md); 0.05 leaves headroom for
        # array-size differences while staying far below the naive error.
        # Do not widen this bound if it fails -- the pipeline is wrong; check
        # `matrix = rotation.T` and grid_mode consistency in
        # _rotate_isotropic first.
        self.assertLessEqual(physical_error, 0.05)

        # Contrast arm: a pure index-space rotation mapping Z onto Y with no
        # spacing correction -- exactly what the pre-v1.1 code did in its own
        # plane. It transplants 61 samples spaced at 0.300um onto an axis
        # spaced at 0.108um, so the beam appears 2.78x too narrow -- the
        # failure mode ROT-02 exists to eliminate. Without this arm the
        # tolerance assertion alone would not distinguish a correct pipeline
        # from a lucky one.
        naive = np.rot90(volume, k=1, axes=(0, 1))
        naive_lineout = naive[30, :, 30]
        naive_error = float(np.abs(naive_lineout - expected).max())
        self.assertGreaterEqual(naive_error, 0.3)

    def test_azimuthal_has_no_effect_at_the_poles(self):
        # 07-RESEARCH.md Pitfall 4: this invariance is a property of matching
        # on the direction vector rather than the raw angle pair, so it needs
        # no special-case branch. This test exists so that adding such a
        # branch -- which would be incorrect -- goes red.
        dxy = 0.108
        dz = 0.300
        z = np.arange(15, dtype=np.float64)
        profile = np.exp(-0.5 * (((z - 7) * dz) / 1.0) ** 2).astype(np.float32)
        volume = np.broadcast_to(
            profile[:, None, None], (15, 15, 15)
        ).astype(np.float32).copy()

        for polar_deg in (0.0, 180.0):
            with self.subTest(polar_deg=polar_deg):
                first = seeds.rotate_illumination(
                    volume, polar_deg=polar_deg, azimuthal_deg=45.0, dxy=dxy, dz=dz
                )
                second = seeds.rotate_illumination(
                    volume, polar_deg=polar_deg, azimuthal_deg=200.0, dxy=dxy, dz=dz
                )
                np.testing.assert_array_equal(first, second)

    def test_non_finite_rotation_angles_reject_before_any_psf_generation(self):
        # normalise_psf already applies nan_to_num, so without this guard a
        # NaN/inf angle would produce a silently all-zero seed flowing into
        # blind-RL estimation looking valid -- the same failure class the
        # existing D-08 slit-energy guard addresses. Covers both rotating
        # modes (light_sheet and aslm).
        detection = np.ones((5, 5, 5), dtype=np.float32)
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
            dxy=0.108,
            dz=0.3,
            psf_size_z=5,
            psf_size_xy=5,
            background=0.0,
        )
        cases = [
            (
                "light_sheet, nan polar_deg",
                "light_sheet",
                {"polar_deg": float("nan"), "azimuthal_deg": 0.0},
                "polar_deg must be finite",
            ),
            (
                "aslm, inf azimuthal_deg",
                "aslm",
                {
                    "polar_deg": 90.0,
                    "azimuthal_deg": float("inf"),
                    "slit_width": 0.2,
                },
                "azimuthal_deg must be finite",
            ),
        ]
        for label, psf_mode, extra_kwargs, expected_message in cases:
            with self.subTest(label=label):
                with mock.patch.object(
                    seeds,
                    "generate_theoretical_psf",
                    return_value=detection,
                ) as generate_theoretical_psf:
                    with self.assertRaisesRegex(ValueError, expected_message):
                        seeds.generate_psf_seed(
                            psf_mode=psf_mode, **base_kwargs, **extra_kwargs
                        )
                    generate_theoretical_psf.assert_not_called()


if __name__ == "__main__":
    unittest.main()
