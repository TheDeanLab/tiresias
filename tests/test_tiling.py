from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tiresias import tiling


class TilingTests(unittest.TestCase):
    def test_representative_tiles_choose_best_candidate_per_spatial_region(self):
        scores = np.array(
            [
                [1, 2, 3, 9],
                [4, 8, 7, 6],
                [3, 2, 8, 1],
                [9, 4, 2, 7],
            ],
            dtype=np.float32,
        )
        volume = scores[np.newaxis, :, :]
        tiles = [
            (row, column, row + 1, column + 1)
            for row in range(4)
            for column in range(4)
        ]

        with mock.patch.object(
            tiling,
            "_snr_weight",
            side_effect=lambda core, weight_cap: float(core[0, 0, 0]),
        ):
            selected = tiling.select_representative_tiles(
                volume, tiles, max_tiles=4, snr_weight_cap=100.0
            )

        self.assertEqual(
            selected,
            [(0, 3, 1, 4), (1, 1, 2, 2), (2, 2, 3, 3), (3, 0, 4, 1)],
        )

    def test_coarse_to_fine_representative_tiles_sample_best_coarse_regions(self):
        scores = np.array(
            [
                [9, 8, 1, 1],
                [7, 6, 1, 1],
                [1, 1, 5, 4],
                [1, 1, 3, 2],
            ],
            dtype=np.float32,
        )
        volume = scores[np.newaxis, :, :]
        tiles = [
            (row, column, row + 1, column + 1)
            for row in range(4)
            for column in range(4)
        ]

        with mock.patch.object(
            tiling,
            "_snr_weight",
            side_effect=lambda core, weight_cap: float(np.max(core)),
        ):
            selected = tiling.select_representative_tiles(
                volume,
                tiles,
                max_tiles=4,
                snr_weight_cap=100.0,
                strategy="coarse_to_fine_snr",
                coarse_region_limit=2,
                coarse_region_rows=2,
                coarse_region_columns=2,
            )

        self.assertEqual(
            selected,
            [(0, 0, 1, 1), (0, 1, 1, 2), (2, 2, 3, 3), (2, 3, 3, 4)],
        )

    def test_representative_tiles_keep_full_grid_without_scoring_when_limit_is_zero(self):
        tiles = [(0, 0, 1, 1), (0, 1, 1, 2)]

        with mock.patch.object(tiling, "_snr_weight") as snr_weight:
            selected = tiling.select_representative_tiles(
                np.ones((1, 1, 2), dtype=np.float32),
                tiles,
                max_tiles=0,
                snr_weight_cap=100.0,
            )

        self.assertEqual(selected, tiles)
        snr_weight.assert_not_called()

    def test_cache_key_separates_representative_and_full_grid_psfs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "image.tif"
            image_path.write_bytes(b"image")
            common = {
                "image_path": image_path,
                "psf_seed": np.ones((3, 3, 3), dtype=np.float32),
                "n_iters": 20,
                "chunk_xy": 160,
                "pad_xy": 32,
                "pad_z": 20,
                "merge_mode": "snr_weighted_mean",
                "snr_weight_cap": 100.0,
                "z_window": (0, 128),
                "blind_peak_normalization": "none",
                "blind_peak_gamma_max": 2.5,
                "blind_latent_update_period": 2,
                "cupy_fft_engine": "cupyx",
            }

            representative = tiling.psf_cache_key(**common, blind_max_tiles=16)
            full_grid = tiling.psf_cache_key(**common, blind_max_tiles=0)

        self.assertNotEqual(representative, full_grid)

    def test_cupy_array_path_skips_chunk_tiff_roundtrip(self):
        volume = np.ones((3, 8, 8), dtype=np.float32)
        psf_seed = np.ones((3, 3, 3), dtype=np.float32)
        psf_seed /= psf_seed.sum()

        with (
            mock.patch.object(
                tiling,
                "_write_chunk",
                side_effect=AssertionError("CuPy path should not write chunk TIFFs"),
            ) as write_chunk,
            mock.patch.object(
                tiling,
                "_run_cupy_tile_array",
                return_value=psf_seed.copy(),
            ) as run_cupy,
        ):
            idx, psf_chunk, weight, error = tiling.estimate_one_tile(
                0,
                1,
                volume,
                (0, 0, 8, 8),
                psf_seed,
                pad_xy=0,
                pad_z=0,
                n_iters=2,
                peak_normalization="none",
                peak_gamma_max=2.5,
                latent_update_period=2,
                cupy_fft_engine="cupyx",
                snr_weight_cap=100.0,
            )

        self.assertEqual(idx, 0)
        self.assertIsNone(error)
        self.assertGreater(weight, 0)
        np.testing.assert_allclose(psf_chunk, psf_seed, rtol=1e-6, atol=1e-8)
        write_chunk.assert_not_called()
        run_cupy.assert_called_once()

    def test_small_edge_tile_is_padded_to_psf_support(self):
        volume = np.ones((3, 128, 144), dtype=np.float32)
        psf_seed = np.ones((3, 128, 128), dtype=np.float32)
        psf_seed /= psf_seed.sum()

        with mock.patch.object(
            tiling,
            "_run_cupy_tile_array",
            return_value=psf_seed.copy(),
        ) as run_cupy:
            idx, psf_chunk, weight, error = tiling.estimate_one_tile(
                0,
                1,
                volume,
                (80, 0, 128, 80),
                psf_seed,
                pad_xy=32,
                pad_z=0,
                n_iters=2,
                peak_normalization="none",
                peak_gamma_max=2.5,
                latent_update_period=2,
                cupy_fft_engine="cupyx",
                snr_weight_cap=100.0,
            )

        self.assertEqual(idx, 0)
        self.assertIsNone(error)
        self.assertGreater(weight, 0)
        self.assertEqual(run_cupy.call_args.args[0].shape, (3, 128, 144))
        np.testing.assert_allclose(psf_chunk, psf_seed, rtol=1e-6, atol=1e-8)

    def test_chunks_are_padded_to_common_shape_before_stacking(self):
        chunks = [
            np.ones((3, 112, 144), dtype=np.float32),
            np.ones((3, 144, 144), dtype=np.float32),
        ]

        padded = tiling.pad_chunks_to_common_shape(chunks)

        self.assertEqual(
            [chunk.shape for chunk in padded],
            [(3, 144, 144), (3, 144, 144)],
        )

    def test_adaptive_scout_tile_selection_keeps_tiles_with_best_peer_agreement(self):
        psf_seed = np.ones((3, 3, 3), dtype=np.float32)
        psf_seed /= psf_seed.sum()
        close_a = psf_seed.copy()
        close_a[1, 1, 1] += 0.1
        close_a = tiling.normalise_psf(close_a)
        close_b = psf_seed.copy()
        close_b[1, 1, 2] += 0.1
        close_b = tiling.normalise_psf(close_b)
        outlier = np.zeros_like(psf_seed)
        outlier[0, 0, 0] = 1.0
        origins = [
            (0, 0, 8, 8),
            (0, 8, 8, 16),
            (8, 0, 16, 8),
        ]

        kept, weights, scout_seed = tiling.select_adaptive_scout_tiles(
            origins,
            [close_a, outlier, close_b],
            [1.0, 100.0, 2.0],
            keep_tiles=2,
            snr_weight_cap=100.0,
        )

        self.assertEqual(kept, [(0, 0, 8, 8), (8, 0, 16, 8)])
        self.assertEqual(weights, [1.0, 2.0])
        self.assertEqual(scout_seed.shape, psf_seed.shape)
        self.assertTrue(np.isclose(float(scout_seed.sum()), 1.0, atol=1e-6))

    def test_scout_engine_filters_tiles_then_runs_cupyx_on_kept_subset(self):
        volume = np.ones((3, 16, 16), dtype=np.float32)
        psf_seed = np.ones((3, 3, 3), dtype=np.float32)
        psf_seed /= psf_seed.sum()
        scout_a = psf_seed.copy()
        scout_b = np.zeros_like(psf_seed)
        scout_b[0, 0, 0] = 1.0
        scout_c = psf_seed.copy()
        scout_c[1, 1, 1] += 0.1
        scout_c = tiling.normalise_psf(scout_c)
        final_a = psf_seed.copy()
        final_b = psf_seed.copy()
        final_b[1, 1, 1] += 0.1
        final_b = tiling.normalise_psf(final_b)
        origins = [
            (0, 0, 8, 8),
            (0, 8, 8, 16),
            (8, 0, 16, 8),
        ]

        with (
            mock.patch.object(
                tiling,
                "_run_blind_tile_batch_pass",
                side_effect=[
                    ([scout_a, scout_b, scout_c], [1.0, 100.0, 2.0]),
                    ([final_a, final_b], [1.0, 2.0]),
                ],
            ) as run_scout,
            mock.patch.object(tiling, "clear_cupy_memory"),
        ):
            estimates, weights = tiling._run_blind_tile_pass(
                volume,
                psf_seed,
                origins,
                pad_xy=0,
                pad_z=0,
                n_iters=10,
                max_workers=1,
                prefetch_chunks=0,
                peak_normalization="none",
                peak_gamma_max=2.5,
                latent_update_period=2,
                cupy_fft_engine="scout",
                snr_weight_cap=100.0,
                cupy_pool_trim_bytes=None,
                adaptive_scout_iters=2,
                adaptive_keep_tiles=2,
            )

        self.assertEqual(len(estimates), 1)
        self.assertEqual(weights, [3.0])
        expected = tiling.merge_weighted_psfs([final_a, final_b], [1.0, 2.0], 100.0)
        np.testing.assert_allclose(estimates[0], expected, rtol=1e-6, atol=1e-8)
        scout_call, final_call = run_scout.call_args_list
        self.assertEqual(scout_call.kwargs["n_iters"], 2)
        self.assertEqual(scout_call.kwargs["single_tile_engine"], "cupyx")
        self.assertEqual(scout_call.kwargs["initial_batch_size"], 1)
        self.assertEqual(final_call.args[2], [(0, 0, 8, 8), (8, 0, 16, 8)])
        expected_scout_seed = tiling.merge_weighted_psfs(
            [scout_a, scout_c], [1.0, 1.0], 100.0
        )
        np.testing.assert_allclose(
            final_call.args[1], expected_scout_seed, rtol=1e-6, atol=1e-8
        )
        self.assertEqual(final_call.kwargs["n_iters"], 8)
        self.assertEqual(final_call.kwargs["single_tile_engine"], "cupyx")
        self.assertEqual(final_call.kwargs["initial_batch_size"], 1)

    def test_cupy_blind_sizing_reduces_chunk_to_fit_vram_budget(self):
        chunk_xy, detail = tiling.resolve_cupy_blind_chunk_xy(
            256,
            (128, 512, 512),
            (61, 128, 128),
            halo_xy=32,
            pad_z=20,
            vram_gb=1.0,
        )

        self.assertLess(chunk_xy, 256)
        self.assertIn("free_vram", detail)


if __name__ == "__main__":
    unittest.main()
