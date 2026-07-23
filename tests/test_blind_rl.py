from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import numpy as np
from scipy.signal import fftconvolve

from tiresias import blind_rl


class _FakeGpuArray:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def max(self):
        return self.values.max()


def _fake_gpu_modules(events, richardson_lucy):
    class Device:
        def __init__(self, device_id):
            self.device_id = device_id

        def __enter__(self):
            events.append(("device_enter", self.device_id))
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            events.append(("device_exit", self.device_id))

    class PlanCache:
        def clear(self):
            events.append("plan_clear")

    class MemoryPool:
        def free_all_blocks(self):
            events.append("pool_free")

    fake_cp = types.ModuleType("cupy")
    fake_cp.float32 = np.float32
    fake_cp.asarray = lambda values, dtype=None: _FakeGpuArray(values)
    fake_cp.asnumpy = lambda values: values.values.copy()
    fake_cp.cuda = types.SimpleNamespace(
        Device=Device,
        Stream=types.SimpleNamespace(
            null=types.SimpleNamespace(
                synchronize=lambda: events.append("synchronize")
            )
        ),
    )
    fake_cp.fft = types.SimpleNamespace(
        config=types.SimpleNamespace(get_plan_cache=lambda: PlanCache())
    )
    fake_cp.get_default_memory_pool = lambda: MemoryPool()

    fake_cucim = types.ModuleType("cucim")
    fake_skimage = types.ModuleType("cucim.skimage")
    fake_restoration = types.ModuleType("cucim.skimage.restoration")
    fake_restoration.richardson_lucy = richardson_lucy
    return {
        "cupy": fake_cp,
        "cucim": fake_cucim,
        "cucim.skimage": fake_skimage,
        "cucim.skimage.restoration": fake_restoration,
    }


class BlindRlTests(unittest.TestCase):
    def test_convolution_adjoints_match_for_even_psf(self):
        rng = np.random.default_rng(7)
        image = rng.random((5, 6, 7), dtype=np.float32)
        psf = rng.random((2, 3, 4), dtype=np.float32)
        values = rng.random(image.shape, dtype=np.float32)

        forward = blind_rl.convolve_same(image, psf, fftconvolve)
        image_back = blind_rl.image_adjoint(values, psf, np, fftconvolve)
        psf_back = blind_rl.psf_adjoint(values, image, psf.shape, np, fftconvolve)

        expected = np.vdot(forward, values)
        self.assertTrue(np.allclose(expected, np.vdot(image, image_back), rtol=2e-5))
        self.assertTrue(np.allclose(expected, np.vdot(psf, psf_back), rtol=2e-5))

    def test_blind_rl_preserves_psf_constraints(self):
        image = np.zeros((9, 17, 17), dtype=np.float32)
        image[4, 5, 6] = 3.0
        image[4, 11, 12] = 2.0
        true_psf = np.zeros((3, 5, 5), dtype=np.float32)
        true_psf[1, 2, 2] = 0.7
        true_psf[1, 2, 3] = 0.2
        true_psf[2, 2, 2] = 0.1
        observed = fftconvolve(image, true_psf, mode="same")
        seed = np.ones_like(true_psf) / true_psf.size

        estimated, history = blind_rl.estimate_blind_psf_scipy(
            observed, seed, 4, return_history=True
        )

        self.assertEqual(estimated.shape, seed.shape)
        self.assertTrue(np.isfinite(estimated).all())
        self.assertGreaterEqual(float(np.min(estimated)), 0.0)
        self.assertTrue(np.isclose(np.sum(estimated, dtype=np.float64), 1.0, atol=1e-6))
        self.assertEqual(len(history), 4)

    def test_lazy_latent_updates_reduce_fft_work_and_preserve_constraints(self):
        image = np.zeros((9, 17, 17), dtype=np.float32)
        image[4, 5, 6] = 3.0
        image[4, 11, 12] = 2.0
        true_psf = np.zeros((3, 5, 5), dtype=np.float32)
        true_psf[1, 2, 2] = 0.7
        true_psf[1, 2, 3] = 0.2
        true_psf[2, 2, 2] = 0.1
        observed = fftconvolve(image, true_psf, mode="same")
        seed = np.ones_like(true_psf) / true_psf.size
        baseline_calls = 0
        accelerated_calls = 0

        def baseline_fft(*args, **kwargs):
            nonlocal baseline_calls
            baseline_calls += 1
            return fftconvolve(*args, **kwargs)

        def accelerated_fft(*args, **kwargs):
            nonlocal accelerated_calls
            accelerated_calls += 1
            return fftconvolve(*args, **kwargs)

        blind_rl.estimate_blind_psf(
            observed,
            seed,
            4,
            xp=np,
            fftconvolve=baseline_fft,
            latent_update_period=1,
        )
        estimated = blind_rl.estimate_blind_psf(
            observed,
            seed,
            4,
            xp=np,
            fftconvolve=accelerated_fft,
            latent_update_period=2,
        )

        self.assertLess(accelerated_calls, baseline_calls)
        self.assertEqual(estimated.shape, seed.shape)
        self.assertTrue(np.isfinite(estimated).all())
        self.assertGreaterEqual(float(np.min(estimated)), 0.0)
        self.assertTrue(np.isclose(np.sum(estimated, dtype=np.float64), 1.0, atol=1e-6))

    def test_latent_update_period_must_be_positive(self):
        with self.assertRaises(ValueError):
            blind_rl.estimate_blind_psf_scipy(
                np.ones((3, 5, 5), dtype=np.float32),
                np.ones((3, 3, 3), dtype=np.float32),
                2,
                latent_update_period=0,
            )

    def test_damping_neutralizes_small_residuals(self):
        observed = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        model = np.array([1.05, 1.5, 2.95], dtype=np.float32)
        ratio = blind_rl._error_ratio(observed, model, np, 1e-7, 0.1)
        self.assertEqual(ratio[0], 1.0)
        self.assertEqual(ratio[2], 1.0)
        self.assertNotEqual(ratio[1], 1.0)

    def test_cucim_cleanup_releases_fft_plans_and_memory_pool_after_success(self):
        events = []

        def richardson_lucy(image, psf, **kwargs):
            return _FakeGpuArray(np.full((2, 2, 2), 7, dtype=np.float32))

        with (
            mock.patch.dict(sys.modules, _fake_gpu_modules(events, richardson_lucy)),
            mock.patch.object(blind_rl, "_normalise_psf", side_effect=lambda psf, xp, epsilon: psf),
        ):
            restored = blind_rl.deconvolve_with_cucim(
                np.ones((2, 2, 2), dtype=np.float32),
                np.ones((1, 1, 1), dtype=np.float32),
                2,
            )

        self.assertTrue(np.all(restored == 7))
        self.assertIn("plan_clear", events)
        self.assertIn("pool_free", events)
        self.assertLess(events.index("plan_clear"), events.index("pool_free"))

    def test_cucim_cleanup_releases_fft_plans_and_memory_pool_after_error(self):
        events = []

        def richardson_lucy(image, psf, **kwargs):
            raise RuntimeError("restoration failed")

        with (
            mock.patch.dict(sys.modules, _fake_gpu_modules(events, richardson_lucy)),
            mock.patch.object(blind_rl, "_normalise_psf", side_effect=lambda psf, xp, epsilon: psf),
        ):
            with self.assertRaisesRegex(RuntimeError, "restoration failed"):
                blind_rl.deconvolve_with_cucim(
                    np.ones((2, 2, 2), dtype=np.float32),
                    np.ones((1, 1, 1), dtype=np.float32),
                    2,
                )

        self.assertIn("plan_clear", events)
        self.assertIn("pool_free", events)
        self.assertLess(events.index("plan_clear"), events.index("pool_free"))


if __name__ == "__main__":
    unittest.main()
