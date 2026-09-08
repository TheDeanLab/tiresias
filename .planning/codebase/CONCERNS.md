# Codebase Concerns

**Analysis Date:** 2026-09-08

## Tech Debt

**GPU Memory Management Complexity:**
- Issue: `_estimate_shared_blind_psf_streamed_cupy` in `src/tiresias/blind_rl.py` (lines 782-912) implements streaming GPU computation with manual resource cleanup across multiple nested loops. If exceptions occur mid-iteration during the latent update or PSF refinement loops, GPU memory may not be fully released despite the `finally` block cleanup.
- Files: `src/tiresias/blind_rl.py` (lines 782-912, 1210-1223)
- Impact: Long-running PSF estimation on large volumes could accumulate GPU memory over multiple tile iterations, potentially leading to cascading OOM errors and requiring manual CUDA device reset.
- Fix approach: Extract the inner loop bodies into separate functions with proper context managers; consolidate GPU resource lifecycle to ensure cleanup even during nested exception handling.

**OOM Retry Loop Without Upper Bound Check:**
- Issue: In `src/tiresias/tiling.py` (lines 1059-1107), the `while True` loop retries with progressively smaller chunks on OOM, but `next_smaller_blind_chunk_xy()` could theoretically return a value >= current `chunk_xy`, creating an infinite retry.
- Files: `src/tiresias/tiling.py` (lines 1076-1107)
- Impact: If `next_smaller_blind_chunk_xy()` encounters an edge case, the loop will retry indefinitely, consuming CPU cycles and blocking the estimation pipeline.
- Fix approach: Add an explicit guard: `if reduced >= chunk_xy: raise` before updating `chunk_xy`; add a maximum retry counter as a safety net.

**Cache Key Fragility:**
- Issue: Cache key in `src/tiresias/tiling.py` (lines 1031-1053) includes seed content hash and algorithmic settings, but does not version the algorithm itself. Changes to epsilon calculation, normalization logic, or convolution algorithms won't invalidate old cached PSFs.
- Files: `src/tiresias/tiling.py` (lines 1031-1053)
- Impact: After algorithm updates, users may silently load stale cached PSFs, producing degraded results without warning.
- Fix approach: Add an algorithm version field to the cache key; implement cache versioning with migration strategy.

**Streaming Function Resource Leak on Exception:**
- Issue: In `_estimate_shared_blind_psf_streamed_cupy` (lines 835-910), if an exception occurs during the loop over batch_slices, GPU-allocated `observed_gpu`, `latent_gpu`, `psf_batch` arrays are explicitly deleted in the del statement (lines 862-871), but if the exception occurs before reaching the del, those arrays leak.
- Files: `src/tiresias/blind_rl.py` (lines 835-910)
- Impact: Large PSF batches on high-iteration counts could accumulate leaked GPU memory, degrading performance over the course of a long estimation run.
- Fix approach: Wrap each microbatch loop iteration in a try-finally or context manager to guarantee cleanup.

## Known Bugs

**Epsilon Calculation Hardcoded for Float32:**
- Symptoms: Numerical precision loss when working with very small or very large image intensities; observed in Richardson-Lucy iterations when gradient steps are very small.
- Files: `src/tiresias/blind_rl.py` (lines 472, 608, 690, 822)
- Trigger: Input images with peak intensity <1e-6 or >1e6, or images with extreme dynamic range.
- Workaround: Manually normalize input images to [0, 1] or [0, 65535] before calling Tiresias.

**Subprocess VRAM Detection Fails Silently on Systems Without nvidia-smi:**
- Symptoms: `detect_vram_bytes()` returns None on systems where nvidia-smi is not in PATH, causing chunk sizing to default to maximum without VRAM constraint.
- Files: `src/tiresias/tiling.py` (lines 75-96)
- Trigger: Windows systems with NVIDIA driver but nvidia-smi not in standard locations; WSL2 setups.
- Workaround: Explicitly pass `--vram-gb` flag to CLI to override automatic detection.

**Silent Tile Failure Accumulation:**
- Symptoms: When `--blind-max-tiles` includes many tiles and some fail, errors are printed to stdout but aggregated as warnings. If 3 consecutive tiles fail at the start, the run aborts, but users may not see the stdout in log files if buffering is enabled.
- Files: `src/tiresias/tiling.py` (lines 722-735)
- Trigger: Heterogeneous tile quality; some tiles with insufficient signal or extreme noise.
- Workaround: Run with stdout/stderr unbuffered; check CLI output explicitly before interpreting PSF results.

**Cache Read on Reduced Chunk Size Doesn't Check Compatibility:**
- Symptoms: After OOM and chunk reduction, the retry checks cache for the reduced chunk_xy (line 1106) but doesn't verify that the cached PSF was estimated with compatible parameters (seed, n_iters, etc.).
- Files: `src/tiresias/tiling.py` (lines 1096-1107)
- Trigger: User estimates PSF at chunk_xy=256, then re-runs with smaller --vram-gb causing fallback to 128; if old cache exists, it's loaded even though it was estimated at different chunk size.
- Workaround: Clear cache before re-running with different parameters.

## Security Considerations

**Subprocess Command Injection in nvidia-smi:**
- Risk: `detect_vram_bytes()` constructs nvidia-smi command with `CUDA_VISIBLE_DEVICES` env var without validation.
- Files: `src/tiresias/tiling.py` (lines 75-96)
- Current mitigation: Environment variable is split by comma but not validated; command is executed via `subprocess.run` with `shell=False`, limiting injection surface.
- Recommendations: Validate `CUDA_VISIBLE_DEVICES` against pattern `^\d+([,]\d+)*$`; log constructed command for debugging.

**TIFF Input Validation Limited:**
- Risk: `imread()` calls from tifffile on untrusted TIFF files could trigger parsing vulnerabilities in the library.
- Files: `src/tiresias/tiling.py` (lines 57-73), `src/tiresias/blind_rl.py` (line 1402)
- Current mitigation: File type checked by suffix; no parsing of metadata.
- Recommendations: Document that Tiresias should only process TIFF files from trusted sources; consider adding magic number verification.

**CuPy Kernel JIT Cache Directory:**
- Risk: `_configure_cupy_cache()` creates cache directory in `/tmp` or `$SLURM_TMPDIR` with world-readable permissions by default.
- Files: `src/tiresias/blind_rl.py` (lines 18-32)
- Current mitigation: Cache directory is user-specific (`f"cupy-kernel-cache-{os.getuid()}"`), restricting to owner.
- Recommendations: Verify that directory permissions are 0o700 after creation; document CuPy JIT cache requirements for HPC environments.

## Performance Bottlenecks

**Adaptive Scout Filtering Double-Pass:**
- Problem: `_run_blind_tile_adaptive_cupyx_pass()` runs all tiles through adaptive scout iters, filters based on agreement, then re-runs kept tiles with remaining iters. If many tiles are filtered out, initial scout work is wasted.
- Files: `src/tiresias/tiling.py` (lines 745-809)
- Cause: Scout pass over-allocates computation before filtering; no early stopping based on tile similarity.
- Improvement path: Implement online tile comparison during scout pass; stop scout early if best tiles have converged.

**VRAM Estimation Pessimistic:**
- Problem: `_cupy_blind_fft_bytes()` uses hardcoded 208 bytes per voxel (DEFAULT_CUPY_FFT_BYTES_PER_VOXEL), which is conservative. Real FFT workspace is typically 50-100 bytes; this causes unnecessary chunk size reduction.
- Files: `src/tiresias/tiling.py` (lines 36, 99-115)
- Cause: Multiplier includes allocator overhead, fragmentation, and kernel cache; may not match actual usage on all GPUs.
- Improvement path: Measure actual VRAM usage on a small test chunk and calibrate multiplier dynamically per GPU.

**CPU-GPU Transfer Overhead in Streaming Path:**
- Problem: `_estimate_shared_blind_psf_streamed_cupy()` transfers latent estimate back to CPU after each microbatch (line 859), requiring synchronization. For large batches, this creates many H2D/D2H transfers.
- Files: `src/tiresias/blind_rl.py` (lines 859-861)
- Cause: Streaming design to fit large batches in VRAM; no option to batch GPU updates.
- Improvement path: If VRAM permits, batch multiple latent updates on GPU before H2D transfer; add parameter to control GPU residence time.

**Tile Boundary Padding Inefficiency:**
- Problem: Small edge tiles are padded to PSF support size (src/tiresias/tiling.py, test_small_edge_tile_is_padded_to_psf_support), creating redundant padding overhead near image boundaries.
- Files: `src/tiresias/tiling.py` (implicit in tile padding logic)
- Cause: Uniform padding strategy doesn't account for tile position; corner tiles are padded more than necessary.
- Improvement path: Use asymmetric padding based on tile position relative to image edges.

## Fragile Areas

**FFT Engine Validation Minimal:**
- Files: `src/tiresias/tiling.py` (lines 850)
- Why fragile: String-based FFT engine selection ("cupyx" vs "scout") with minimal validation; typos silently fall through to default.
- Safe modification: Add explicit enum or constants for engine choices; validate at entry point before passing to `_run_blind_tile_batch_pass`.
- Test coverage: Only happy path tested in `test_blind_rl.py`; no coverage of invalid engine names.

**Epsilon Propagation Inconsistency:**
- Files: `src/tiresias/blind_rl.py` (lines 352-360, 422-433, 506-514)
- Why fragile: Epsilon is computed per-image but reused across batch operations; batch epsilon is computed from max of batch but used for all batch elements.
- Safe modification: Compute per-image epsilon in batch functions; ensure batch epsilon respects individual image ranges.
- Test coverage: No test for heterogeneous image ranges in batch.

**Normalized PSF Assumption in Adjoints:**
- Files: `src/tiresias/blind_rl.py` (lines 204-222, 331-349)
- Why fragile: `image_adjoint` and `psf_adjoint` assume kernel is normalized (sum=1), but this is enforced only at end of each RL iteration.
- Safe modification: Add optional normalization parameter; document precondition.
- Test coverage: Tests assume normalized kernels; no edge case for unnormalized kernels.

**Tile Weight Cap Not Enforced in Merge:**
- Files: `src/tiresias/tiling.py` (merge_weighted_psfs function, implicit SNR weighting)
- Why fragile: `snr_weight_cap` is passed to tile selection but not re-enforced during merge; a tile with outlier SNR could dominate if weights are recomputed.
- Safe modification: Clamp weights to cap before merge; document weight cap semantics.
- Test coverage: No test for merge with weight-capped SNR values.

## Scaling Limits

**GPU Memory Per Tile:**
- Current capacity: ~1-4 GB per chunk on consumer GPUs; 10-80 GB on data center GPUs.
- Limit: Chunk size is clamped to fit FFT workspace; very large PSFs (>256x256 XY) or deep volumes (>256 Z) require small chunks, increasing overhead.
- Scaling path: Implement tiled FFT decomposition for larger kernels; use hierarchical FFT (Cooley-Tukey) or GPU memory-mapped approaches.

**Number of Tiles Per Volume:**
- Current capacity: Default `--blind-max-tiles=16` selects 16 representative tiles from full grid.
- Limit: Adaptive scout pass runs all tiles first, then filters; memory pooling assumes single-process worker.
- Scaling path: Implement online tile selection to avoid full grid evaluation; add support for multi-GPU tile parallelism with explicit GPU assignment.

**Batch Size in Streaming PSF Estimation:**
- Current capacity: Default `fft_batch_size=1` in `_estimate_shared_blind_psf_streamed_cupy` streams one tile at a time.
- Limit: For large number of tiles (>32), repeated GPU allocation/deallocation cycles create fragmentation.
- Scaling path: Implement adaptive batch sizing based on available GPU memory; expose `fft_batch_size` as CLI parameter.

## Dependencies at Risk

**psfmodels API Compatibility:**
- Risk: `generate_theoretical_psf()` in `src/tiresias/seeds.py` (lines 37-114) introspects `psfmodels.make_psf` signature using inspect module to handle API changes. If psfmodels introduces breaking changes, this will fail at runtime.
- Files: `src/tiresias/seeds.py` (lines 98-112)
- Impact: Users cannot generate theoretical PSF seeds if psfmodels updates incompatibly; blind PSF estimation requires external seed or fallback.
- Migration plan: Pin `psfmodels>=0.3,<0.4` in `pyproject.toml`; add integration tests to detect psfmodels API changes; maintain fallback seed generation.

**CuPy CUDA Version Lock:**
- Risk: `cupy-cuda11x>=13,<14` requirement locks to CUDA 11.x. NVIDIA is phasing out CUDA 11.x support; 12.x and later require new cupy wheels.
- Files: `pyproject.toml` (line 30)
- Impact: Package will not work on systems with CUDA 12.x installed without manual wheel installation.
- Migration plan: Add support for multiple cupy variants; detect CUDA version at install time; provide instructions for CUDA 12 users.

**tifffile Version Compatibility:**
- Risk: `tifffile>=2024.0` is a moving target with frequent updates; tiff_memmap or imread API may change.
- Files: `src/tiresias/tiling.py` (lines 19, 57-73), `src/tiresias/blind_rl.py` (line 1399)
- Impact: Major tifffile update could break TIFF loading or memmap support.
- Migration plan: Pin to `tifffile>=2024.0,<2025.0` to allow patches while preventing major version jumps; add compatibility layer for memmap variants.

## Missing Critical Features

**Multi-GPU Support:**
- Problem: Entire PSF estimation pipeline is single-GPU. For systems with multiple GPUs, only one can be utilized per CLI invocation.
- Blocks: High-throughput batch processing; scaling estimation across multiple volumes in parallel.

**Progressive Result Saving:**
- Problem: PSF results are saved only after all tiles complete. Crash mid-estimation results in complete data loss.
- Blocks: Checkpointing; resuming interrupted estimation runs; monitoring progress via saved intermediate results.

**Tile-Level Error Logging:**
- Problem: Chunk failures print to stdout; no structured error log or machine-readable error report.
- Blocks: Automated failure analysis; integration with job schedulers (SLURM, etc.) that parse exit codes and error logs.

**Input Validation Report:**
- Problem: Minimal reporting on input data quality (NaN prevalence, dynamic range, background estimation).
- Blocks: Pre-flight diagnostic mode; users cannot assess input feasibility before launching expensive estimation.

## Test Coverage Gaps

**GPU OOM Error Handling:**
- What's not tested: Explicit CuPy OOM exception paths in `_run_blind_tile_batch_pass` (line 723).
- Files: `src/tiresias/tiling.py` (lines 722-735), `src/tiresias/blind_rl.py` (lines 1210-1223)
- Risk: OOM recovery paths are untested; retry logic could have off-by-one errors in chunk size reduction.
- Priority: High

**Heterogeneous Batch Input in Blind RL:**
- What's not tested: `estimate_blind_psf_batch` with images of different peak intensities; epsilon calculation correctness for outlier batches.
- Files: `src/tiresias/blind_rl.py` (lines 557-648)
- Risk: Batch operations assume homogeneous signal ranges; outlier images could corrupt gradient steps.
- Priority: High

**Cache Key Collision:**
- What's not tested: Two different PSF estimation runs with identical cache key but different algorithms.
- Files: `src/tiresias/tiling.py` (lines 1031-1053)
- Risk: Cache incorrectly assumes algorithm version is stable; no coverage of cache invalidation after code changes.
- Priority: Medium

**CLI Integration with Invalid TIFF:**
- What's not tested: `estimate_psf_main` in `src/tiresias/cli.py` with corrupted or multi-channel TIFF inputs.
- Files: `src/tiresias/cli.py` (lines 200+)
- Risk: Error handling is implicit; users may encounter confusing error messages.
- Priority: Medium

**Streaming Cupy Function Exception Handling:**
- What's not tested: Mid-loop exceptions in `_estimate_shared_blind_psf_streamed_cupy` microbatch iterations.
- Files: `src/tiresias/blind_rl.py` (lines 835-910)
- Risk: Resource cleanup on exception is not verified; GPU memory leaks possible.
- Priority: High

**Epsilon Boundary Cases:**
- What's not tested: Images with all-zero regions; peak intensity exactly 1.0; negative values after preprocessing.
- Files: `src/tiresias/blind_rl.py` (lines 466-473, 602-609, 684-691, 816-822)
- Risk: Epsilon calculation may produce unexpected values; division by zero or invalid conversions.
- Priority: Medium

---

*Concerns audit: 2026-09-08*
