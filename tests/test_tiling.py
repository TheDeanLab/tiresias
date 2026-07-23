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
                snr_weight_cap=100.0,
            )

        self.assertEqual(idx, 0)
        self.assertIsNone(error)
        self.assertGreater(weight, 0)
        np.testing.assert_allclose(psf_chunk, psf_seed, rtol=1e-6, atol=1e-8)
        write_chunk.assert_not_called()
        run_cupy.assert_called_once()

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
