# Coding Conventions

**Analysis Date:** 2026-09-08

## Naming Patterns

**Files:**
- Module files: `snake_case.py` (e.g., `blind_rl.py`, `tiling.py`)
- Test files: `test_<module_name>.py` (e.g., `test_blind_rl.py`, `test_seeds.py`)

**Functions:**
- Public functions: `snake_case` (e.g., `estimate_blind_psf`, `fit_psf_to_shape`, `load_psf_seed`)
- Private functions (module-internal): Leading underscore `_snake_case` (e.g., `_normalise_psf`, `_shape`, `_scalar`)
- Type alias definitions: Uppercase (e.g., `Array = Any`, `Convolve = Callable[..., Array]`)

**Variables:**
- Snake_case for all variables and parameters
- Use explicit names: `observed`, `image_shape`, `latent_update_period` rather than abbreviations
- Loop variables: Single letters acceptable (`z`, `axis`) in context-obvious cases

**Types/Classes:**
- Class names: PascalCase (e.g., `FftConvolutionEngine`, `BatchedFftConvolutionEngine`)
- Enum-like constants: UPPER_CASE with explanatory names
  - Example: `DEFAULT_BLIND_CHUNK_XY`, `DEFAULT_BLIND_MAX_TILES`, `RIGHT_ANGLE_TOLERANCE`
  - Located at module level for configuration values

**Constants:**
- Module-level constants: UPPER_CASE
- Located near top of module after imports
- Examples: `DEFAULT_SNR_WEIGHT_CAP = 100.0`, `BLIND_CHUNK_ALIGNMENT = 32`

## Code Style

**Import Organization:**
1. `from __future__ import annotations` (always first)
2. Standard library imports (sys, os, pathlib, etc.)
3. Third-party imports (numpy, scipy, tifffile, etc.)
4. Local/relative imports (`.blind_rl`, `.seeds`)

Example from `blind_rl.py`:
```python
from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.fft import next_fast_len

Array = Any
Convolve = Callable[..., Array]
```

**Type Hints:**
- Use `from __future__ import annotations` for all files (enables forward references)
- Type hint all function parameters and return types
- Use union syntax: `Array | None` rather than `Optional[Array]`
- Use `tuple[int, ...]` for variable-length tuples
- Generic type aliases at module level: `Array = Any`, `Convolve = Callable[..., Array]`

Example from `blind_rl.py`:
```python
def estimate_blind_psf(
    observed: Array,
    initial_psf: Array,
    n_iters: int,
    *,
    xp: Any,
    fftconvolve: Convolve,
    dampar: float = 0.0,
    return_history: bool = False,
    latent_update_period: int = 1,
    fft_engine: str | FftConvolutionEngine | None = None,
) -> Array | tuple[Array, list[float]]:
```

## Error Handling

**Exception Types:**
- `ValueError`: Invalid parameter values, shape mismatches, invalid state
- `RuntimeError`: Runtime failures (missing CuPy, API mismatches)
- `OSError`: File system or subprocess errors
- Custom exceptions: None currently used; use built-in types

**Error Message Pattern:**
- Clear, descriptive messages with context
- Include actual values and expected values
- Use f-strings for formatting

Examples from `blind_rl.py`:
```python
if not result or any(value <= 0 for value in result):
    raise ValueError(f"Invalid shape: {result}")

if array.ndim != len(target_shape):
    raise ValueError(
        "PSF and image dimensionality must match: "
        f"psf={array.ndim}, image={len(target_shape)}"
    )
```

**Validation Pattern:**
- Validate inputs early, before expensive operations
- Convert types explicitly: `int(value)`, `float(value)`, `np.asarray(value, dtype=...)`
- Use guards with descriptive messages

## Docstrings

**Module Docstrings:**
- One-line description of module purpose
- Located right after imports

Examples:
```python
"""Shared SciPy/CuPy blind Richardson-Lucy PSF estimation."""
"""Theoretical PSF seed generation."""
"""TIFF-based tiled CuPy blind PSF estimation."""
```

**Function Docstrings:**
- One-line summary (imperative mood: "Estimate", "Convert", "Return")
- Multi-line docstrings only for complex functions
- No parameter/return documentation beyond the signature (types are in annotations)

Examples from `blind_rl.py`:
```python
def fit_psf_to_shape(psf: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    """Center-crop a PSF only along axes that exceed the image shape."""
    ...

def psf_to_otf(psf: Array, output_shape: Sequence[int], *, xp: Any = np) -> Array:
    """Convert a PSF to an OTF using post-padding and center-roll alignment."""
    ...
```

## Comments

**When to Comment:**
- Explain "why", not "what" (code should be self-documenting for "what")
- Complex algorithmic choices or mathematical operations
- Non-obvious design decisions
- References to external algorithms or papers

Example from `blind_rl.py`:
```python
# Keep CuPy JIT artifacts off read-only container home mounts.
if os.environ.get("CUPY_CACHE_DIR"):
    return
```

**Avoid:**
- Restating what the code obviously does
- Outdated comments (they drift from code)

## Function Design

**Size Guidelines:**
- Prefer focused functions under 100 lines
- Break large algorithms into named helpers (private functions with `_` prefix)
- Example: `blind_rl.py` has `_normalise_psf`, `_embed_same_adjoint`, `_error_ratio` as helpers

**Parameters:**
- Positional parameters: Required inputs (e.g., `observed`, `initial_psf`)
- Keyword-only parameters (after `*`): Optional configuration and control flow
- Use `xp` parameter to abstract numpy/cupy (pattern: `xp: Any` where `xp` is either `np` or `cp`)

Example:
```python
def estimate_blind_psf(
    observed: Array,              # positional: required
    initial_psf: Array,           # positional: required
    n_iters: int,                 # positional: required
    *,                            # force remaining to keyword-only
    xp: Any,                      # required keyword
    fftconvolve: Convolve,        # required keyword
    dampar: float = 0.0,          # optional keyword with default
    return_history: bool = False, # optional keyword with default
):
```

**Return Values:**
- Single value: Return directly
- Multiple related values: Return as `tuple[Type1, Type2]` with unpacking in docstring
- No return value: Omit return statement or return `None` explicitly

## Module Design

**Public API:**
- Explicit `__all__` list in each module (e.g., in `__init__.py`)
- Document in module docstring what is exported
- Private functions/classes prefixed with underscore

Example from `src/tiresias/__init__.py`:
```python
__all__ = [
    "clear_cupy_memory",
    "deconvolve_with_cupy",
    "estimate_blind_psf",
    # ... etc
]
```

**Barrel Files:**
- Main `__init__.py` re-exports from submodules
- No circular imports

## Language Features

**NumPy/CuPy Abstraction:**
- Pass array module (`xp`) as parameter to functions that need it
- Never import `cupy` at module level (leads to hard failures if CUDA unavailable)
- Import `cupy` inside functions with try/except and descriptive error messages

Example from `blind_rl.py`:
```python
def estimate_blind_psf_cupy(...) -> Array:
    try:
        import cupy as cp
        from cupyx.scipy.signal import fftconvolve
    except ImportError as exc:
        raise RuntimeError(
            "blind_backend='cupy' requires CuPy with cupyx.scipy"
        ) from exc
```

**Context Managers:**
- Use for GPU device context: `with cp.cuda.Device(int(device_id)):`
- Use for temporary directories: `with tempfile.TemporaryDirectory() as tmpdir:`
- Use for resource cleanup: `try/finally` blocks for GPU memory

Example from `blind_rl.py`:
```python
with cp.cuda.Device(int(device_id)):
    try:
        # GPU operations
        result = accelerated_richardson_lucy(...)
    finally:
        # Cleanup GPU memory
        output_gpu = None
        _release_cupy_workspace(cp)
```

---

*Convention analysis: 2026-09-08*
