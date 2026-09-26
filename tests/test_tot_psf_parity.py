"""D-10: cross-check tiresias.seeds' theoretical PSF arms against
psfmodels.tot_psf, the independent reference for an orthogonal
illumination/detection (SPIM-style) system PSF.

generate_theoretical_psf's illumination arm, summed over pre-rotation Y and
transposed, is compared against tot_psf's ex_psf; its detection arm is
compared against tot_psf's em_psf. Both comparisons are unit-sum normalised
so only shape (not absolute scale) is asserted.
"""

from __future__ import annotations

import unittest

import numpy as np
import psfmodels as pm

from tiresias import seeds

N = 65
D = 0.1
ILLUMINATION_NA = 0.4
EX_WVL = 0.488
NI = 1.33
NS = 1.33
NI0 = 1.33
OVERSAMPLE_FACTOR = 3
PSF_MODEL = "vectorial"


def _unit_sum(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    return array / array.sum(dtype=np.float64)


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    denom = np.linalg.norm(expected.ravel())
    return float(np.linalg.norm((actual - expected).ravel()) / denom)


class TotPsfParityTests(unittest.TestCase):
    def test_illumination_arm_matches_tot_psf_ex_psf_at_every_offset(self):
        for x_offset in (0.0, 0.3, 1.0, -2.5):
            with self.subTest(x_offset=x_offset):
                illumination = seeds.generate_theoretical_psf(
                    detection_na=ILLUMINATION_NA,
                    illumination_na=ILLUMINATION_NA,
                    wavelength=EX_WVL,
                    ni=NI,
                    ns=NS,
                    ni0=NI0,
                    oversample_factor=OVERSAMPLE_FACTOR,
                    psf_model=PSF_MODEL,
                    dxy=D,
                    dz=D,
                    psf_size_z=N,
                    psf_size_xy=N,
                    focus_offset=x_offset,
                )
                # RESEARCH.md Pattern 3: sum over pre-rotation Y, transpose
                # to (transverse, propagation) -- the like-for-like shape
                # matching tot_psf's ex_psf.
                ours = _unit_sum(illumination.sum(axis=1).T)

                ex_psf, _, _ = pm.tot_psf(
                    nx=N,
                    nz=N,
                    dxy=D,
                    dz=D,
                    x_offset=x_offset,
                    ex_wvl=EX_WVL,
                    ex_params={"NA": ILLUMINATION_NA, "ni0": NI0, "ni": NI, "ns": NS},
                    em_params={"NA": 1.1, "ni0": NI0, "ni": NI, "ns": NS},
                )
                ref = _unit_sum(ex_psf)

                self.assertLessEqual(_relative_l2(ours, ref), 1e-5)
                self.assertEqual(int(np.argmax(ours)), int(np.argmax(ref)))

    def test_positive_emitter_offset_moves_the_waist_to_a_smaller_axis0_index(self):
        # D-06: a positive emitter_offset moves the waist to a smaller
        # pre-rotation axis-0 index, the psfmodels.tot_psf x_offset
        # convention. Measured (planner prototype, this session): index
        # 32 (centre, x_offset=0.0) -> 29 (x_offset=0.3).
        illumination_at_0 = seeds.generate_theoretical_psf(
            detection_na=ILLUMINATION_NA,
            illumination_na=ILLUMINATION_NA,
            wavelength=EX_WVL,
            ni=NI,
            ns=NS,
            ni0=NI0,
            oversample_factor=OVERSAMPLE_FACTOR,
            psf_model=PSF_MODEL,
            dxy=D,
            dz=D,
            psf_size_z=N,
            psf_size_xy=N,
            focus_offset=0.0,
        )
        illumination_at_offset = seeds.generate_theoretical_psf(
            detection_na=ILLUMINATION_NA,
            illumination_na=ILLUMINATION_NA,
            wavelength=EX_WVL,
            ni=NI,
            ns=NS,
            ni0=NI0,
            oversample_factor=OVERSAMPLE_FACTOR,
            psf_model=PSF_MODEL,
            dxy=D,
            dz=D,
            psf_size_z=N,
            psf_size_xy=N,
            focus_offset=0.3,
        )

        profile_at_0 = illumination_at_0.sum(axis=(1, 2), dtype=np.float64)
        profile_at_offset = illumination_at_offset.sum(axis=(1, 2), dtype=np.float64)

        self.assertEqual(int(np.argmax(profile_at_0)), 32)
        self.assertEqual(int(np.argmax(profile_at_offset)), 29)

    def test_light_sheet_seed_offset_is_composed_from_the_parity_checked_arm(self):
        # D-10: generate_psf_seed(psf_mode="light_sheet", emitter_offset=p)
        # is composed from exactly the parity-checked arm --
        # normalise_psf(D * rotate(generate_theoretical_psf(..., focus_offset=p))).
        detection_na = 1.1
        emitter_offset = 1.0
        arm_kwargs = dict(
            wavelength=EX_WVL,
            ni=NI,
            ns=NS,
            ni0=NI0,
            tg=None,
            tg0=None,
            ng=None,
            ng0=None,
            ti0=None,
            oversample_factor=OVERSAMPLE_FACTOR,
            psf_model=PSF_MODEL,
            dxy=D,
            dz=D,
            psf_size_z=N,
            psf_size_xy=N,
            background=0.0,
        )

        actual = seeds.generate_psf_seed(
            psf_mode="light_sheet",
            na=detection_na,
            detection_na=detection_na,
            illumination_na=ILLUMINATION_NA,
            polar_deg=90.0,
            azimuthal_deg=0.0,
            emitter_offset=emitter_offset,
            **arm_kwargs,
        )

        detection_arm = seeds.generate_theoretical_psf(
            detection_na=detection_na,
            illumination_na=ILLUMINATION_NA,
            **arm_kwargs,
        )
        illumination_arm = seeds.generate_theoretical_psf(
            detection_na=ILLUMINATION_NA,
            illumination_na=ILLUMINATION_NA,
            focus_offset=emitter_offset,
            **arm_kwargs,
        )
        rotated = seeds.rotate_illumination(
            illumination_arm, polar_deg=90.0, azimuthal_deg=0.0, dxy=D, dz=D
        )
        expected = seeds.normalise_psf(detection_arm * rotated)

        np.testing.assert_array_equal(actual, expected)

    def test_detection_arm_matches_tot_psf_em_psf(self):
        for na, em_wvl in ((1.1, 0.525), (1.0, 0.561)):
            with self.subTest(na=na, em_wvl=em_wvl):
                detection = seeds.generate_theoretical_psf(
                    detection_na=na,
                    wavelength=em_wvl,
                    ni=NI,
                    ns=NS,
                    ni0=NI0,
                    oversample_factor=OVERSAMPLE_FACTOR,
                    psf_model=PSF_MODEL,
                    dxy=D,
                    dz=D,
                    psf_size_z=N,
                    psf_size_xy=N,
                )
                ours = _unit_sum(detection)

                _, em_psf, _ = pm.tot_psf(
                    nx=N,
                    nz=N,
                    dxy=D,
                    dz=D,
                    em_wvl=em_wvl,
                    em_params={"NA": na, "ni0": NI0, "ni": NI, "ns": NS},
                )
                ref = _unit_sum(em_psf)

                self.assertLessEqual(_relative_l2(ours, ref), 1e-5)


if __name__ == "__main__":
    unittest.main()
