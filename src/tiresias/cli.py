"""Command-line entry points for Tiresias."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from tifffile import imread, imwrite

from .blind_rl import deconvolve_with_cucim
from .seeds import generate_theoretical_psf, load_psf_seed, resolve_dxy
from .tiling import (
    DEFAULT_ADAPTIVE_KEEP_TILES,
    DEFAULT_ADAPTIVE_SCOUT_ITERS,
    DEFAULT_BLIND_CHUNK_XY,
    DEFAULT_BLIND_LATENT_UPDATE_PERIOD,
    DEFAULT_BLIND_MAX_TILES,
    DEFAULT_BLIND_PEAK_GAMMA_MAX,
    DEFAULT_BLIND_PEAK_NORMALIZATION,
    DEFAULT_BLIND_Z_SLICES,
    DEFAULT_COARSE_REGION_COLUMNS,
    DEFAULT_COARSE_REGION_LIMIT,
    DEFAULT_COARSE_REGION_ROWS,
    DEFAULT_SNR_WEIGHT_CAP,
    TILE_SELECTION_STRATEGIES,
    BLIND_TILE_SELECTION_STRATEGY,
    estimate_psf_from_chunks,
)


def _add_optical_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--na", type=float, default=None)
    parser.add_argument("--detection-na", dest="detection_na", type=float, default=None)
    parser.add_argument("--illumination-na", dest="illumination_na", type=float, default=None)
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--ni", type=float, default=None)
    parser.add_argument("--ns", type=float, default=None)
    parser.add_argument("--ni0", type=float, default=None)
    parser.add_argument("--tg", type=float, default=None)
    parser.add_argument("--tg0", type=float, default=None)
    parser.add_argument("--ng", type=float, default=None)
    parser.add_argument("--ng0", type=float, default=None)
    parser.add_argument("--ti0", type=float, default=None)
    parser.add_argument("--oversample-factor", dest="oversample_factor", type=int, default=3)
    parser.add_argument(
        "--psf-model",
        choices=("vectorial", "scalar", "gaussian"),
        default="vectorial",
    )
    parser.add_argument("--camera-pixel-size", dest="camera_pixel_size", type=float, default=None)
    parser.add_argument("--magnification", type=float, default=None)
    parser.add_argument("--dxy", type=float, default=None)
    parser.add_argument("--dz", type=float, default=None)
    parser.add_argument("--psf-size-z", dest="psf_size_z", type=int, default=61)
    parser.add_argument("--psf-size-xy", dest="psf_size_xy", type=int, default=128)
    parser.add_argument("--background", type=float, default=0.0)


def build_estimate_psf_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate a blind PSF from a TIFF volume.")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--psf-seed-path",
        type=Path,
        default=None,
        help="Calibrated TIFF PSF seed; bypasses theoretical seed generation.",
    )
    parser.add_argument("--n-iters", dest="n_iters", type=int, default=10)
    parser.add_argument("--chunk-xy", dest="chunk_xy", type=int, default=DEFAULT_BLIND_CHUNK_XY)
    parser.add_argument("--blind-max-tiles", dest="blind_max_tiles", type=int, default=DEFAULT_BLIND_MAX_TILES)
    parser.add_argument("--pad-xy", dest="pad_xy", type=int, default=32)
    parser.add_argument("--pad-z", dest="pad_z", type=int, default=20)
    parser.add_argument("--prefetch-chunks", dest="prefetch_chunks", type=int, default=0)
    parser.add_argument("--vram-gb", dest="vram_gb", type=float, default=None)
    parser.add_argument("--cache-dir", dest="cache_dir", type=Path, default=None)
    parser.add_argument("--no-psf-cache", dest="use_cache", action="store_false", default=True)
    parser.add_argument(
        "--peak-normalization",
        choices=("none", "gamma", "unit"),
        default=DEFAULT_BLIND_PEAK_NORMALIZATION,
    )
    parser.add_argument("--peak-gamma-max", dest="peak_gamma_max", type=float, default=DEFAULT_BLIND_PEAK_GAMMA_MAX)
    parser.add_argument(
        "--latent-update-period",
        dest="latent_update_period",
        type=int,
        default=DEFAULT_BLIND_LATENT_UPDATE_PERIOD,
    )
    parser.add_argument("--blind-z-slices", dest="blind_z_slices", type=int, default=DEFAULT_BLIND_Z_SLICES)
    parser.add_argument("--snr-weight-cap", dest="snr_weight_cap", type=float, default=DEFAULT_SNR_WEIGHT_CAP)
    parser.add_argument(
        "--cupy-fft-engine",
        dest="cupy_fft_engine",
        choices=("cupyx", "scout"),
        default="scout",
    )
    parser.add_argument(
        "--adaptive-scout-iters",
        dest="adaptive_scout_iters",
        type=int,
        default=DEFAULT_ADAPTIVE_SCOUT_ITERS,
    )
    parser.add_argument(
        "--adaptive-keep-tiles",
        dest="adaptive_keep_tiles",
        type=int,
        default=DEFAULT_ADAPTIVE_KEEP_TILES,
    )
    parser.add_argument(
        "--tile-selection-strategy",
        dest="tile_selection_strategy",
        choices=TILE_SELECTION_STRATEGIES,
        default=BLIND_TILE_SELECTION_STRATEGY,
    )
    parser.add_argument("--coarse-region-rows", dest="coarse_region_rows", type=int, default=DEFAULT_COARSE_REGION_ROWS)
    parser.add_argument("--coarse-region-columns", dest="coarse_region_columns", type=int, default=DEFAULT_COARSE_REGION_COLUMNS)
    parser.add_argument("--coarse-region-limit", dest="coarse_region_limit", type=int, default=DEFAULT_COARSE_REGION_LIMIT)
    _add_optical_arguments(parser)
    return parser


def estimate_psf_main(argv: Sequence[str] | None = None) -> None:
    args = build_estimate_psf_parser().parse_args(argv)
    psf_shape = (args.psf_size_z, args.psf_size_xy, args.psf_size_xy)
    if args.psf_seed_path is not None:
        psf_seed = load_psf_seed(args.psf_seed_path, psf_shape)
    else:
        dxy = resolve_dxy(args.dxy, args.camera_pixel_size, args.magnification)
        psf_seed = generate_theoretical_psf(
            na=args.na,
            detection_na=args.detection_na,
            illumination_na=args.illumination_na,
            wavelength=args.wavelength,
            ni=args.ni,
            ns=args.ns,
            ni0=args.ni0,
            tg=args.tg,
            tg0=args.tg0,
            ng=args.ng,
            ng0=args.ng0,
            ti0=args.ti0,
            oversample_factor=args.oversample_factor,
            psf_model=args.psf_model,
            dxy=dxy,
            dz=args.dz,
            psf_size_z=args.psf_size_z,
            psf_size_xy=args.psf_size_xy,
            background=args.background,
        )
    estimated = estimate_psf_from_chunks(
        image_path=args.image_path,
        psf_seed=psf_seed,
        n_iters=args.n_iters,
        chunk_xy=args.chunk_xy,
        pad_xy=args.pad_xy,
        pad_z=args.pad_z,
        prefetch_chunks=args.prefetch_chunks,
        vram_gb=args.vram_gb,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        snr_weight_cap=args.snr_weight_cap,
        peak_normalization=args.peak_normalization,
        peak_gamma_max=args.peak_gamma_max,
        latent_update_period=args.latent_update_period,
        cupy_fft_engine=args.cupy_fft_engine,
        blind_z_slices=args.blind_z_slices,
        blind_max_tiles=args.blind_max_tiles,
        adaptive_scout_iters=args.adaptive_scout_iters,
        adaptive_keep_tiles=args.adaptive_keep_tiles,
        tile_selection_strategy=args.tile_selection_strategy,
        coarse_region_rows=args.coarse_region_rows,
        coarse_region_columns=args.coarse_region_columns,
        coarse_region_limit=args.coarse_region_limit,
    )
    imwrite(args.output_path, estimated)


def build_deconvolve_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cuCIM Richardson-Lucy restoration on a TIFF volume.")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--psf-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--n-iters", dest="n_iters", type=int, default=20)
    parser.add_argument("--device-id", dest="device_id", type=int, default=0)
    return parser


def deconvolve_main(argv: Sequence[str] | None = None) -> None:
    args = build_deconvolve_parser().parse_args(argv)
    image = imread(args.image_path)
    psf = imread(args.psf_path)
    restored = deconvolve_with_cucim(
        image,
        psf,
        args.n_iters,
        device_id=args.device_id,
    )
    imwrite(args.output_path, restored)


__all__ = [
    "build_estimate_psf_parser",
    "estimate_psf_main",
    "build_deconvolve_parser",
    "deconvolve_main",
]
