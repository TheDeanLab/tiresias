<!-- refreshed: 2026-09-08 -->
# Architecture

**Analysis Date:** 2026-09-08

## System Overview

Tiresias is a GPU-first Python package for blind PSF (point-spread function) estimation and CuPy Richardson-Lucy deconvolution from 3-D TIFF microscopy volumes. The system follows a layered architecture with dual backends (SciPy for reference, CuPy for GPU production) and adaptive memory management for large-scale microscopy datasets.

```text
┌──────────────────────────────────────────────────────────────────┐
│                      CLI Layer                                    │
│  tiresias-estimate-psf / tiresias-deconvolve                      │
│              `src/tiresias/cli.py`                                │
└──────────────┬──────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────────┐
│            High-Level API Layer (Tiling/Chunking)                │
│         `src/tiresias/tiling.py`                                 │
│  • estimate_psf_from_chunks (main entry point)                   │
│  • Tile selection & SNR weighting                                │
│  • VRAM detection & adaptive chunk sizing                        │
│  • Cache management                                              │
└──────────────┬──────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────────┐
│            Seed/PSF Initialization Layer                         │
│         `src/tiresias/seeds.py`                                  │
│  • Theoretical PSF generation (psfmodels)                        │
│  • Calibrated PSF loading & fitting                              │
│  • PSF normalization                                             │
└──────────────┬──────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────────┐
│            Algorithm Layer (Core Computations)                   │
│         `src/tiresias/blind_rl.py`                               │
│  • FFT Convolution Engines (single-tile & batched)               │
│  • Alternating Richardson-Lucy (RL) updates                      │
│  • SciPy backend (reference)                                     │
│  • CuPy backend (GPU production)                                 │
└──────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| CLI Entry Points | Parse arguments, orchestrate PSF estimation or deconvolution workflows | `src/tiresias/cli.py` |
| Tiling & Chunking | Split large volumes into GPU-manageable tiles, merge results, manage cache | `src/tiresias/tiling.py` |
| PSF Seeds | Generate theoretical PSFs via optical models or load calibrated seeds | `src/tiresias/seeds.py` |
| FFT Engines | Implement efficient "same" convolutions and adjoints via FFT | `src/tiresias/blind_rl.py` |
| RL Algorithms | Execute alternating latent/kernel update iterations | `src/tiresias/blind_rl.py` |
| Memory/Device | Detect VRAM, manage CuPy memory pools, handle out-of-memory recovery | `src/tiresias/tiling.py`, `src/tiresias/blind_rl.py` |

## Pattern Overview

**Overall:** Tiled blind Richardson-Lucy deconvolution with automatic GPU memory adaptation

**Key Characteristics:**
- **Tile-based processing**: Large 3-D volumes are decomposed into overlapping XY tiles with Z-axis padding to fit GPU VRAM
- **Adaptive chunk sizing**: Automatic VRAM detection and FFT budget estimation clamp tile size to available GPU memory
- **SNR-weighted selection**: Tiles are scored by SNR; high-quality tiles receive more training emphasis
- **Alternating updates**: PSF and latent image are updated on alternating iterations (configurable period)
- **FFT convolution**: "Same" mode convolutions via FFT with center-roll alignment and pre-padding
- **Dual backend support**: SciPy implementation for numerical reference; CuPy for GPU acceleration
- **Cache-aware**: Deterministic PSF merge caching allows repeated runs without re-estimation

## Layers

**CLI Layer:**
- Purpose: User-facing entry points for PSF estimation and deconvolution
- Location: `src/tiresias/cli.py`
- Contains: Argument parsing, workflow orchestration
- Depends on: Tiling, Seeds, Algorithm layers
- Used by: Command-line users via `tiresias-estimate-psf` and `tiresias-deconvolve` scripts

**Tiling/High-Level API Layer:**
- Purpose: Decompose large volumes into tiles, manage VRAM, coordinate tile-wise estimation, merge results
- Location: `src/tiresias/tiling.py`
- Contains: `estimate_psf_from_chunks`, tile extraction, SNR scoring, VRAM budgeting, cache logic
- Depends on: Seeds, Algorithm layers (for per-tile estimation)
- Used by: CLI, Python API callers

**Seed/PSF Initialization Layer:**
- Purpose: Generate or load PSF seeds for blind estimation
- Location: `src/tiresias/seeds.py`
- Contains: Theoretical PSF generation via `psfmodels`, calibrated PSF loading, normalization
- Depends on: External `psfmodels` library
- Used by: Tiling layer to initialize blind-RL iterations

**Algorithm Layer (Core Computations):**
- Purpose: Execute blind Richardson-Lucy updates and deconvolution
- Location: `src/tiresias/blind_rl.py`
- Contains: FFT convolution engines, alternating RL updates, SciPy/CuPy backends
- Depends on: NumPy/SciPy (reference) or CuPy (GPU)
- Used by: Tiling layer for per-tile PSF estimation

## Data Flow

### Primary Request Path: PSF Estimation from TIFF Volume

1. **Entry (CLI)** (`src/tiresias/cli.py::estimate_psf_main`) — Parse arguments, resolve PSF seed
2. **Seed generation** (`src/tiresias/seeds.py::generate_theoretical_psf` or `load_psf_seed`) — Create or load initial PSF
3. **Volume opening** (`src/tiresias/tiling.py::open_psf_source`) — Load TIFF as memmap (no full RAM load)
4. **Z-window selection** (`src/tiresias/tiling.py::select_blind_z_window`) — Sample planes to find brightest Z region
5. **VRAM budgeting** (`src/tiresias/tiling.py::resolve_cupy_blind_chunk_xy`) — Query GPU memory, compute safe tile size
6. **Tile enumeration** (`src/tiresias/tiling.py::tile_origins`) — Generate XY tile grid
7. **Per-tile estimation** (loop over tiles):
   - Extract tile with halo (`src/tiresias/tiling.py::extract_tile_with_halo`)
   - Estimate PSF via alternating RL (`src/tiresias/blind_rl.py::estimate_blind_psf_cupy`)
   - Store PSF and SNR score
8. **Tile selection & merge** (`src/tiresias/tiling.py`) — Rank tiles by SNR, merge selected PSFs via weighted averaging
9. **Output** (`src/tiresias/cli.py::estimate_psf_main`) — Write merged PSF to TIFF file

**State Management:**
- Large image volume held as memmap; only active tiles loaded to GPU
- PSF estimates accumulated in host memory (not GPU)
- CuPy memory pools cleared between tiles to prevent fragmentation
- Cache stores merged PSF hashes for deterministic repeated runs

### Secondary Flow: CuPy Richardson-Lucy Deconvolution

1. **Entry (CLI)** (`src/tiresias/cli.py::deconvolve_main`) — Load image and PSF TIFFs
2. **GPU transfer** (`src/tiresias/blind_rl.py::deconvolve_with_cupy`) — Transfer to GPU via CuPy
3. **Restoration loop** (N iterations):
   - Forward convolution: `model = convolve(latent, psf)`
   - Error ratio: `ratio = observed / (model + epsilon)`
   - Latent update: `latent *= adjoint_convolve(ratio, psf)`
   - PSF normalization
4. **Output** — Transfer result to host, write TIFF

**Intermediate Flow: FFT Convolution Setup**

Within `estimate_blind_psf` or `estimate_blind_psf_cupy`:
- Initialize `FftConvolutionEngine` (single-tile) or `BatchedFftConvolutionEngine` (batched)
- Pre-compute FFT shapes, crop slices, and embed slices
- Reuse engine across iterations to amortize FFT plan generation

## Key Abstractions

**FftConvolutionEngine:**
- Purpose: Encapsulate fixed-shape "same" convolution and adjoint operations via real FFT
- Examples: `src/tiresias/blind_rl.py` lines 118–223
- Pattern: Stateful class pre-computing geometric constants; reused across iterations
- Methods: `convolve_same`, `image_adjoint`, `psf_adjoint`

**BatchedFftConvolutionEngine:**
- Purpose: Extend FftConvolutionEngine to handle a leading batch dimension (multiple tiles in parallel)
- Examples: `src/tiresias/blind_rl.py` lines 225–349
- Pattern: Mirrors FftConvolutionEngine logic with batch-aware indexing

**Tile Metadata:**
- Purpose: Represent tile origin, size, and quality metrics for selection
- Examples: Implicit in `tile_origins` return, SNR scores in tiling loop
- Pattern: Tuple-based representation; selection via ranking on SNR

## Entry Points

**CLI Entry: PSF Estimation**
- Location: `src/tiresias/cli.py::estimate_psf_main`
- Triggers: Shell command `tiresias-estimate-psf --image-path <TIFF> --output-path <TIFF> [options]`
- Responsibilities: Parse arguments, load/generate PSF seed, call `estimate_psf_from_chunks`, write output

**CLI Entry: Deconvolution**
- Location: `src/tiresias/cli.py::deconvolve_main`
- Triggers: Shell command `tiresias-deconvolve --image-path <TIFF> --psf-path <TIFF> --output-path <TIFF> [options]`
- Responsibilities: Load image and PSF, call `deconvolve_with_cupy`, write output

**Python API Entry: Tiled PSF Estimation**
- Location: `src/tiresias/tiling.py::estimate_psf_from_chunks`
- Triggers: Direct import and call from Python
- Responsibilities: Orchestrate full tiled workflow; main user-facing API

**Python API Entry: Low-level PSF Estimation**
- Locations: 
  - `src/tiresias/blind_rl.py::estimate_blind_psf_cupy` (GPU, batched tiles)
  - `src/tiresias/blind_rl.py::estimate_blind_psf_scipy` (CPU reference)
- Triggers: Direct calls from tiling layer or advanced users
- Responsibilities: Run single-tile blind-RL iterations with specified backend

## Architectural Constraints

- **Threading:** Single-threaded event loop per worker. Tile-wise parallelization via ProcessPoolExecutor (prefetching) or sequential per-tile processing. GPU work always on single CUDA device.
- **Global state:** CuPy JIT cache location configured at module load via `_configure_cupy_cache()` (`src/tiresias/blind_rl.py:18–32`). FFT plan caches are released between tiles.
- **Circular imports:** None detected; imports follow clean layering (CLI → Tiling → Algorithm/Seeds).
- **Memory constraints:** Tile sizes auto-clamped to fit FFT workspace within VRAM budget; smaller tiles trigger automatic retry with reduced chunk_xy.
- **Device affinity:** Single GPU device per process; `device_id` parameter in `deconvolve_with_cupy` selects which GPU.
- **Data format:** 3-D volumes only (Z, Y, X). 2-D images auto-promoted to 3-D with Z=1 in some paths.

## Anti-Patterns

### Direct CuPy Import Dependency

**What happens:** Some modules (e.g., `blind_rl.py`) import and handle CuPy only conditionally/dynamically; if CuPy is not installed, import fails at runtime despite being "optional."

**Why it's wrong:** Advertised as supporting both SciPy (no GPU) and CuPy (GPU) backends, but the current code structure makes CuPy a hard dependency even though SciPy path should work standalone.

**Do this instead:** Defer CuPy import to function call sites and handle ImportError gracefully. Wrap CuPy-dependent functions with try-except or lazy loading (`importlib.util.find_spec`). Example: `src/tiresias/blind_rl.py` should only import CuPy inside functions that need it, not at module level.

### Tile Boundary Artifacts

**What happens:** Extracted tiles use halo padding (neighboring pixels) for convolution adjacency, but the padding mode is "reflect," which can introduce boundary artifacts in low-SNR regions near tile edges.

**Why it's wrong:** Reflect padding replicates edge structure that may not match the true physical blur; can lead to inconsistent PSF estimates at tile boundaries.

**Do this instead:** Consider higher-order boundary conditions (e.g., synthetic extension based on PSF decay rate) or taper tile results at boundaries before merging. Currently hardcoded in `extract_tile_with_halo` (`src/tiresias/tiling.py:254`).

### SNR Weight Capping Without Justification

**What happens:** SNR weights are capped at `DEFAULT_SNR_WEIGHT_CAP=100.0` during tile selection (`src/tiresias/tiling.py:296`), which suppresses contribution from very high-SNR tiles.

**Why it's wrong:** Caps outlier tiles to prevent over-influence, but the cap value is not adaptive; may discard valuable signal in extremely clean regions.

**Do this instead:** Make weight cap a function of dataset statistics (e.g., percentile of observed SNR scores) rather than a fixed constant. Or use rank-based selection instead of weight-based averaging.

## Error Handling

**Strategy:** Graceful degradation with OOM recovery. CuPy out-of-memory errors trigger automatic chunk size reduction and retry.

**Patterns:**
- **OOM Detection** (`src/tiresias/tiling.py::is_cupy_out_of_memory`) — Inspect exception chain for CuPy-specific OutOfMemoryError
- **Chunk Downsizing** (`src/tiresias/tiling.py::next_smaller_blind_chunk_xy`) — Reduce tile XY size in fixed steps (aligned to `BLIND_CHUNK_ALIGNMENT=32`) and retry
- **Input Validation** — Early checks on array shapes, dimensionality, and finite values (e.g., `_normalise_psf` clips NaNs/Infs)
- **Epsilon Guards** — Float operations protect against division-by-zero with epsilon=max(machine eps, signal_peak * 1e-7)

## Cross-Cutting Concerns

**Logging:** Print statements to stdout (no formal logger). Key milestones: volume shape, chunk sizing details, tile processing progress, merge results. Example: `src/tiresias/tiling.py:1023–1026`.

**Validation:** Input shape and dimensionality checks at layer boundaries. PSF seeds validated for positive finite energy. Observed images sanitized (NaN→0, Inf→0, negatives clipped).

**Memory Management:** VRAM detection via `nvidia-smi` query; CuPy memory pool cleared between tiles via `_release_cupy_workspace` (`src/tiresias/blind_rl.py:89–99`). Trimming API exposed (`trim_cupy_memory_pool`, `clear_cupy_memory`) for user control in long-running jobs.

---

*Architecture analysis: 2026-09-08*
