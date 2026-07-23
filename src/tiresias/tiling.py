"""TIFF-based tiled CuPy blind PSF estimation."""

from __future__ import annotations

import concurrent.futures as futures
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import next_fast_len
from tifffile import TiffFile, imread, imwrite, memmap as tiff_memmap

from .blind_rl import clear_cupy_memory, estimate_psf_array_cupy, trim_cupy_memory_pool
from .seeds import normalise_psf

DEFAULT_SNR_WEIGHT_CAP = 100.0
DEFAULT_BLIND_CHUNK_XY = 256
DEFAULT_BLIND_MAX_TILES = 16
DEFAULT_BLIND_Z_SLICES = 128
DEFAULT_BLIND_LATENT_UPDATE_PERIOD = 2
DEFAULT_BLIND_PEAK_NORMALIZATION = "none"
DEFAULT_BLIND_PEAK_GAMMA_MAX = 2.5
DEFAULT_CUPY_VRAM_FRACTION = 0.72
DEFAULT_CUPY_FFT_BYTES_PER_VOXEL = 208
BLIND_CHUNK_ALIGNMENT = 32
BLIND_TILE_SELECTION_STRATEGY = "spatial_snr_v1"


def ensure_3d_volume(volume: np.ndarray) -> np.ndarray:
    if volume.ndim == 2:
        return volume[np.newaxis, :, :]
    return volume


def open_tiff_memmap(path: str | Path) -> np.ndarray:
    """Return a read-only array-like TIFF volume without forcing a full RAM load."""
    path = Path(path)
    try:
        return ensure_3d_volume(tiff_memmap(str(path), mode="r"))
    except Exception:
        with TiffFile(str(path)) as tif:
            return ensure_3d_volume(tif.asarray(out="memmap"))


def open_psf_source(path: str | Path) -> np.ndarray:
    """Open a TIFF volume for PSF estimation."""
    path = Path(path)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"Tiresias currently estimates PSFs from TIFF inputs, got {path}")
    return open_tiff_memmap(path)


def detect_vram_bytes() -> int | None:
    """Best-effort free VRAM query using nvidia-smi."""
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    command = ["nvidia-smi"]
    if visible_device and visible_device not in ("NoDevFiles", "-1"):
        command.extend(["--id", visible_device])
    command.extend(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    free_mb = []
    for line in result.stdout.splitlines():
        try:
            free_mb.append(int(line.strip().split()[0]))
        except (IndexError, ValueError):
            continue
    if not free_mb:
        return None
    return min(free_mb) * 1024 * 1024


def _cupy_blind_fft_bytes(
    core_xy: int,
    volume_z: int,
    halo_xy: int,
    pad_z: int,
    psf_shape: tuple[int, int, int],
) -> int:
    image_shape = (
        int(volume_z) + 2 * int(pad_z),
        int(core_xy) + 2 * int(halo_xy),
        int(core_xy) + 2 * int(halo_xy),
    )
    fft_shape = tuple(
        next_fast_len(image_size + int(kernel_size) - 1)
        for image_size, kernel_size in zip(image_shape, psf_shape)
    )
    return int(math.prod(fft_shape) * DEFAULT_CUPY_FFT_BYTES_PER_VOXEL)


def resolve_cupy_blind_chunk_xy(
    requested_xy: int,
    volume_shape: tuple[int, int, int],
    psf_shape: tuple[int, int, int],
    halo_xy: int,
    pad_z: int,
    vram_gb: float | None = None,
) -> tuple[int, str]:
    """Clamp a requested CuPy core size to the available FFT workspace."""
    _, ny, nx = volume_shape
    maximum = min(ny, nx, requested_xy if requested_xy > 0 else min(ny, nx))
    minimum = min(
        maximum,
        max(
            64,
            int(math.ceil(max(int(psf_shape[-2]), int(psf_shape[-1])) / BLIND_CHUNK_ALIGNMENT))
            * BLIND_CHUNK_ALIGNMENT,
        ),
    )
    candidate = max(minimum, (maximum // BLIND_CHUNK_ALIGNMENT) * BLIND_CHUNK_ALIGNMENT)
    vram_bytes = (
        int(vram_gb * (1024 ** 3))
        if vram_gb and vram_gb > 0
        else detect_vram_bytes()
    )
    if not vram_bytes:
        return candidate, "VRAM unavailable; OOM retry enabled"

    budget = int(vram_bytes * DEFAULT_CUPY_VRAM_FRACTION)
    while candidate > minimum:
        estimated = _cupy_blind_fft_bytes(candidate, volume_shape[0], halo_xy, pad_z, psf_shape)
        if estimated <= budget:
            break
        candidate = max(minimum, candidate - BLIND_CHUNK_ALIGNMENT)

    estimated = _cupy_blind_fft_bytes(candidate, volume_shape[0], halo_xy, pad_z, psf_shape)
    detail = (
        f"free_vram={vram_bytes / (1024 ** 3):.1f}GiB, "
        f"budget={budget / (1024 ** 3):.1f}GiB, "
        f"estimated_peak={estimated / (1024 ** 3):.1f}GiB"
    )
    return candidate, detail


def is_cupy_out_of_memory(exc: BaseException) -> bool:
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        name = f"{type(current).__module__}.{type(current).__name__}"
        message = f"{name}: {current}"
        if "cupy.cuda.memory.OutOfMemoryError" in message or (
            type(current).__name__ == "OutOfMemoryError" and "cupy" in name
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def next_smaller_blind_chunk_xy(current: int, minimum: int) -> int:
    if current <= minimum:
        return current
    return max(minimum, ((current - 1) // BLIND_CHUNK_ALIGNMENT) * BLIND_CHUNK_ALIGNMENT)


def select_blind_z_window(
    volume: np.ndarray,
    max_z_slices: int = DEFAULT_BLIND_Z_SLICES,
    sample_planes: int = 64,
) -> tuple[slice, str]:
    nz = volume.shape[0]
    if max_z_slices <= 0 or nz <= max_z_slices:
        return slice(None), f"full_z=0:{nz}"
    sample_count = max(1, min(sample_planes, nz))
    sample_indices = np.unique(np.linspace(0, nz - 1, sample_count, dtype=int))
    scores = []
    for z in sample_indices:
        plane = np.asarray(volume[z], dtype=np.float32)
        scores.append(float(np.percentile(plane, 99.9)))
    if max(scores) <= min(scores):
        center_z = nz // 2
        score_detail = "flat_sample_scores"
    else:
        center_z = int(sample_indices[int(np.argmax(scores))])
        score_detail = "brightest_sample"
    start = max(0, center_z - (max_z_slices // 2))
    stop = min(nz, start + max_z_slices)
    start = max(0, stop - max_z_slices)
    return (
        slice(start, stop),
        f"bright_z_window={start}:{stop}, center={center_z}, "
        f"sampled_planes={len(sample_indices)}, selector={score_detail}",
    )


def adapt_psf_seed_to_volume(psf_seed: np.ndarray, volume_shape: tuple[int, int, int]) -> np.ndarray:
    volume_z = int(volume_shape[0])
    if volume_z <= 0:
        raise ValueError(f"Volume Z size must be positive, got {volume_shape}")
    if psf_seed.shape[0] <= volume_z:
        return psf_seed
    z_start = (psf_seed.shape[0] - volume_z) // 2
    return normalise_psf(psf_seed[z_start : z_start + volume_z, :, :])


def tile_origins(ny: int, nx: int, chunk_xy: int) -> list[tuple[int, int, int, int]]:
    min_tile = max(1, chunk_xy // 2)
    origins = []
    for y0 in range(0, ny, chunk_xy):
        for x0 in range(0, nx, chunk_xy):
            y1 = min(y0 + chunk_xy, ny)
            x1 = min(x0 + chunk_xy, nx)
            if (y1 - y0) >= min_tile and (x1 - x0) >= min_tile:
                origins.append((y0, x0, y1, x1))
    return origins


def extract_tile_with_halo(
    volume: np.ndarray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    halo_xy: int,
) -> np.ndarray:
    _, ny, nx = volume.shape
    read_y0 = max(0, y0 - halo_xy)
    read_y1 = min(ny, y1 + halo_xy)
    read_x0 = max(0, x0 - halo_xy)
    read_x1 = min(nx, x1 + halo_xy)
    chunk = np.asarray(volume[:, read_y0:read_y1, read_x0:read_x1])
    before_y = read_y0 - (y0 - halo_xy)
    after_y = (y1 + halo_xy) - read_y1
    before_x = read_x0 - (x0 - halo_xy)
    after_x = (x1 + halo_xy) - read_x1
    if before_y or after_y or before_x or after_x:
        chunk = np.pad(
            chunk,
            pad_width=((0, 0), (before_y, after_y), (before_x, after_x)),
            mode="reflect",
        )
    return chunk


def _snr_weight(core: np.ndarray, weight_cap: float = DEFAULT_SNR_WEIGHT_CAP) -> float:
    sample = np.asarray(core, dtype=np.float32)
    if sample.size == 0:
        return 0.0
    p50, p90, p99 = np.percentile(sample, [50, 90, 99])
    noise_region = sample[sample <= p90]
    if noise_region.size == 0:
        noise_region = sample
    mad = np.median(np.abs(noise_region - np.median(noise_region)))
    noise = max(1.4826 * float(mad), float(np.std(noise_region)), 1.0)
    snr = max(0.0, float(p99 - p50)) / noise
    weight = max(1e-3, snr * snr)
    if weight_cap > 0:
        weight = min(weight, weight_cap)
    return weight


def select_representative_tiles(
    volume,
    origins: list[tuple[int, int, int, int]],
    max_tiles: int,
    snr_weight_cap: float,
) -> list[tuple[int, int, int, int]]:
    """Select high-SNR tiles across balanced spatial regions."""
    if max_tiles < 0:
        raise ValueError(f"blind_max_tiles cannot be negative, got {max_tiles}")
    if max_tiles == 0 or len(origins) <= max_tiles:
        return list(origins)

    y_positions = sorted({tile[0] for tile in origins})
    x_positions = sorted({tile[1] for tile in origins})
    y_indices = {position: index for index, position in enumerate(y_positions)}
    x_indices = {position: index for index, position in enumerate(x_positions)}
    region_rows = max(
        1,
        min(
            len(y_positions),
            max_tiles,
            int(round(math.sqrt(max_tiles * len(y_positions) / len(x_positions)))),
        ),
    )
    region_columns = max(1, min(len(x_positions), max_tiles // region_rows))
    scored_tiles: list[tuple[float, tuple[int, int, int, int]]] = []
    regions: dict[tuple[int, int], list[tuple[float, tuple[int, int, int, int]]]] = {}
    for tile in origins:
        y0, x0, y1, x1 = tile
        score = _snr_weight(np.asarray(volume[:, y0:y1, x0:x1]), weight_cap=snr_weight_cap)
        scored = (score, tile)
        scored_tiles.append(scored)
        region = (
            min(region_rows - 1, y_indices[y0] * region_rows // len(y_positions)),
            min(region_columns - 1, x_indices[x0] * region_columns // len(x_positions)),
        )
        regions.setdefault(region, []).append(scored)

    selected: list[tuple[float, tuple[int, int, int, int]]] = []
    selected_tiles: set[tuple[int, int, int, int]] = set()
    for region in sorted(regions):
        best = min(regions[region], key=lambda item: (-item[0], item[1]))
        selected.append(best)
        selected_tiles.add(best[1])
    for scored in sorted(scored_tiles, key=lambda item: (-item[0], item[1])):
        if len(selected) >= max_tiles:
            break
        if scored[1] not in selected_tiles:
            selected.append(scored)
            selected_tiles.add(scored[1])
    selected.sort(key=lambda item: item[1])
    return [tile for _, tile in selected]


def psf_cache_key(
    image_path: Path,
    psf_seed: np.ndarray,
    n_iters: int,
    chunk_xy: int,
    pad_xy: int,
    pad_z: int,
    merge_mode: str,
    snr_weight_cap: float,
    z_window: tuple[int | None, int | None],
    blind_peak_normalization: str,
    blind_peak_gamma_max: float,
    blind_latent_update_period: int,
    blind_max_tiles: int,
) -> str:
    stat = image_path.stat()
    payload: dict[str, Any] = {
        "image": str(image_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "seed_shape": psf_seed.shape,
        "seed_sha256": hashlib.sha256(np.ascontiguousarray(psf_seed).view(np.uint8)).hexdigest(),
        "n_iters": n_iters,
        "chunk_xy": chunk_xy,
        "pad_xy": pad_xy,
        "pad_z": pad_z,
        "merge_mode": merge_mode,
        "snr_weight_cap": snr_weight_cap,
        "z_window": z_window,
        "backend": "cupy",
        "blind_peak_normalization": blind_peak_normalization,
        "blind_peak_gamma_max": blind_peak_gamma_max,
        "blind_latent_update_period": blind_latent_update_period,
        "blind_max_tiles": blind_max_tiles,
        "tile_selection_strategy": BLIND_TILE_SELECTION_STRATEGY,
        "version": 1,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _write_chunk(chunk: np.ndarray, path: Path) -> None:
    imwrite(str(path), np.asarray(chunk))


def _run_cupy_tile_array(
    chunk: np.ndarray,
    psf_seed: np.ndarray,
    n_iters: int,
    pad_z: int,
    peak_normalization: str,
    peak_gamma_max: float,
    latent_update_period: int,
    cupy_pool_trim_bytes: int | None,
) -> np.ndarray:
    psf = estimate_psf_array_cupy(
        chunk,
        psf_seed,
        n_iters,
        pad_z,
        peak_normalization=peak_normalization,
        peak_gamma_max=peak_gamma_max,
        clear_plan_cache=False,
        free_memory_pool=False,
        latent_update_period=latent_update_period,
    )
    trim_cupy_memory_pool(cupy_pool_trim_bytes)
    return psf


def estimate_one_tile(
    idx: int,
    total_tiles: int,
    volume: np.ndarray,
    tile: tuple[int, int, int, int],
    psf_seed: np.ndarray,
    *,
    pad_xy: int,
    pad_z: int,
    n_iters: int,
    peak_normalization: str,
    peak_gamma_max: float,
    latent_update_period: int,
    snr_weight_cap: float,
    cupy_pool_trim_bytes: int | None = None,
) -> tuple[int, np.ndarray | None, float, str | None]:
    del total_tiles
    y0, x0, y1, x1 = tile
    core = np.asarray(volume[:, y0:y1, x0:x1])
    weight = _snr_weight(core, weight_cap=snr_weight_cap)
    chunk = extract_tile_with_halo(volume, y0, x0, y1, x1, pad_xy)
    try:
        psf_chunk = _run_cupy_tile_array(
            chunk,
            psf_seed,
            n_iters,
            pad_z,
            peak_normalization,
            peak_gamma_max,
            latent_update_period,
            cupy_pool_trim_bytes,
        )
    except BaseException as exc:
        if is_cupy_out_of_memory(exc):
            raise
        if not isinstance(exc, RuntimeError):
            raise
        return idx, None, weight, str(exc)
    if psf_chunk.shape != psf_seed.shape:
        return idx, None, weight, f"PSF shape {psf_chunk.shape} != seed shape {psf_seed.shape}"
    return idx, normalise_psf(psf_chunk), weight, None


def _run_blind_tile_pass(
    volume: np.ndarray,
    psf_seed: np.ndarray,
    origins: list[tuple[int, int, int, int]],
    *,
    pad_xy: int,
    pad_z: int,
    n_iters: int,
    max_workers: int,
    prefetch_chunks: int,
    peak_normalization: str,
    peak_gamma_max: float,
    latent_update_period: int,
    snr_weight_cap: float,
    cupy_pool_trim_bytes: int | None,
) -> tuple[list[np.ndarray], list[float]]:
    psf_estimates: list[np.ndarray] = []
    psf_weights: list[float] = []
    failure_details: list[str] = []
    failed_chunks = 0
    completed_chunks = 0
    prefetch_limit = prefetch_chunks if prefetch_chunks > 0 else max_workers
    heartbeat_seconds = 60.0
    last_heartbeat = time.perf_counter()
    try:
        with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            next_idx = 0
            pending: set[futures.Future] = set()
            while next_idx < len(origins) or pending:
                while next_idx < len(origins) and len(pending) < prefetch_limit:
                    pending.add(
                        executor.submit(
                            estimate_one_tile,
                            next_idx,
                            len(origins),
                            volume,
                            origins[next_idx],
                            psf_seed,
                            pad_xy=pad_xy,
                            pad_z=pad_z,
                            n_iters=n_iters,
                            peak_normalization=peak_normalization,
                            peak_gamma_max=peak_gamma_max,
                            latent_update_period=latent_update_period,
                            snr_weight_cap=snr_weight_cap,
                            cupy_pool_trim_bytes=cupy_pool_trim_bytes,
                        )
                    )
                    next_idx += 1
                done, pending = futures.wait(
                    pending,
                    timeout=heartbeat_seconds,
                    return_when=futures.FIRST_COMPLETED,
                )
                if not done:
                    now = time.perf_counter()
                    if now - last_heartbeat >= heartbeat_seconds:
                        print(
                            f"Blind PSF heartbeat: submitted={next_idx}/{len(origins)}, "
                            f"completed={completed_chunks}, failed={failed_chunks}, pending={len(pending)}",
                            flush=True,
                        )
                        last_heartbeat = now
                    continue
                for future in done:
                    idx, psf_chunk, weight, error = future.result()
                    completed_chunks += 1
                    if error:
                        failed_chunks += 1
                        failure_details.append(f"chunk {idx}: {error}")
                        if failed_chunks >= 3 and not psf_estimates:
                            raise RuntimeError(
                                "First three chunks failed during PSF estimation; aborting.\n\n"
                                + "\n\n".join(failure_details[:3])
                            )
                        print(f"WARNING: chunk {idx} failed, skipping. {error}", flush=True)
                        continue
                    if psf_chunk is not None:
                        psf_estimates.append(psf_chunk)
                        psf_weights.append(weight)
    finally:
        clear_cupy_memory(clear_plan_cache=True, free_memory_pool=True)
    return psf_estimates, psf_weights


def _ensure_writable_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path, prefix=".write_test_", delete=True):
        pass


def _resolve_cache_root(image_path: Path, cache_dir: str | Path | None) -> Path:
    if cache_dir:
        root = Path(cache_dir)
        _ensure_writable_dir(root)
        return root
    root = image_path.parent / ".psf_cache"
    _ensure_writable_dir(root)
    return root


def estimate_psf_from_chunks(
    image_path: str | Path,
    psf_seed: np.ndarray,
    n_iters: int = 10,
    chunk_xy: int = DEFAULT_BLIND_CHUNK_XY,
    pad_xy: int = 32,
    pad_z: int = 20,
    max_workers: int = 1,
    prefetch_chunks: int = 0,
    vram_gb: float | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    snr_weight_cap: float = DEFAULT_SNR_WEIGHT_CAP,
    peak_normalization: str = DEFAULT_BLIND_PEAK_NORMALIZATION,
    peak_gamma_max: float = DEFAULT_BLIND_PEAK_GAMMA_MAX,
    latent_update_period: int = DEFAULT_BLIND_LATENT_UPDATE_PERIOD,
    blind_z_slices: int = DEFAULT_BLIND_Z_SLICES,
    blind_max_tiles: int = DEFAULT_BLIND_MAX_TILES,
) -> np.ndarray:
    """Estimate and merge tiled CuPy blind-RL PSFs from a TIFF volume."""
    image_path = Path(image_path)
    volume = open_psf_source(image_path)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {volume.shape}")
    z_window, z_detail = select_blind_z_window(volume, blind_z_slices)
    z_start, z_stop = z_window.start, z_window.stop
    volume = volume[z_window]
    nz, ny, nx = volume.shape
    psf_seed = adapt_psf_seed_to_volume(psf_seed, volume.shape)
    pad_z = max(0, int(pad_z))
    if nz == 1:
        pad_z = 0
    latent_update_period = max(1, int(latent_update_period))
    max_workers = max(1, int(max_workers))
    snr_weight_cap = max(0.0, float(snr_weight_cap))
    if blind_max_tiles < 0:
        raise ValueError(f"blind_max_tiles cannot be negative, got {blind_max_tiles}")

    chunk_xy, sizing_detail = resolve_cupy_blind_chunk_xy(
        int(chunk_xy),
        volume.shape,
        psf_seed.shape,
        int(pad_xy),
        pad_z,
        vram_gb=vram_gb,
    )
    min_chunk_xy = min(
        ny,
        nx,
        max(
            64,
            int(math.ceil(max(psf_seed.shape[-2:]) / BLIND_CHUNK_ALIGNMENT))
            * BLIND_CHUNK_ALIGNMENT,
        ),
    )
    vram_bytes = int(vram_gb * (1024 ** 3)) if vram_gb and vram_gb > 0 else detect_vram_bytes()
    cupy_pool_trim_bytes = (
        int(vram_bytes * DEFAULT_CUPY_VRAM_FRACTION) if vram_bytes else None
    )
    cache_root = _resolve_cache_root(image_path, cache_dir) if use_cache else None

    print(
        f"Volume shape: {volume.shape}; {z_detail}; chunk_xy={chunk_xy}; {sizing_detail}",
        flush=True,
    )

    def cache_path_for(resolved_chunk_xy: int) -> Path | None:
        if cache_root is None:
            return None
        key = psf_cache_key(
            image_path=image_path,
            psf_seed=psf_seed,
            n_iters=n_iters,
            chunk_xy=resolved_chunk_xy,
            pad_xy=pad_xy,
            pad_z=pad_z,
            merge_mode="snr_weighted_mean",
            snr_weight_cap=snr_weight_cap,
            z_window=(z_start, z_stop),
            blind_peak_normalization=peak_normalization,
            blind_peak_gamma_max=peak_gamma_max,
            blind_latent_update_period=latent_update_period,
            blind_max_tiles=blind_max_tiles,
        )
        return cache_root / f"estimated_psf_{key}.tif"

    cache_path = cache_path_for(chunk_xy)
    if cache_path is not None and cache_path.exists():
        return normalise_psf(imread(str(cache_path)))

    while True:
        origins = select_representative_tiles(
            volume,
            tile_origins(ny, nx, chunk_xy),
            max_tiles=blind_max_tiles,
            snr_weight_cap=snr_weight_cap,
        )
        started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Starting {len(origins)} CuPy blind PSF tile(s) at {started_at}", flush=True)
        try:
            estimates, weights = _run_blind_tile_pass(
                volume,
                psf_seed,
                origins,
                pad_xy=pad_xy,
                pad_z=pad_z,
                n_iters=n_iters,
                max_workers=1,
                prefetch_chunks=prefetch_chunks,
                peak_normalization=peak_normalization,
                peak_gamma_max=peak_gamma_max,
                latent_update_period=latent_update_period,
                snr_weight_cap=snr_weight_cap,
                cupy_pool_trim_bytes=cupy_pool_trim_bytes,
            )
            break
        except BaseException as exc:
            reduced = next_smaller_blind_chunk_xy(chunk_xy, min_chunk_xy)
            if not is_cupy_out_of_memory(exc) or reduced >= chunk_xy:
                raise
            print(
                f"WARNING: CuPy exhausted VRAM at chunk_xy={chunk_xy}; retrying all tiles with {reduced}.",
                flush=True,
            )
            chunk_xy = reduced
            cache_path = cache_path_for(chunk_xy)
            if cache_path is not None and cache_path.exists():
                return normalise_psf(imread(str(cache_path)))

    if not estimates:
        raise RuntimeError("All chunks failed during PSF estimation.")
    stack = np.stack(estimates, axis=0)
    weight_array = np.asarray(weights, dtype=np.float32)
    max_weight = snr_weight_cap if snr_weight_cap > 0 else None
    weight_array = np.clip(weight_array, 1e-3, max_weight)
    weight_array = weight_array / weight_array.sum()
    merged = normalise_psf(np.tensordot(weight_array, stack, axes=(0, 0)).astype(np.float32))
    if cache_path is not None:
        imwrite(str(cache_path), merged)
    return merged
