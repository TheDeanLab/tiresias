# Testing Patterns

**Analysis Date:** 2026-09-08

## Test Framework

**Runner:**
- pytest 8.0+
- Config: `pyproject.toml` with `[tool.pytest.ini_options]`
- Test paths: `testpaths = ["tests"]`

**Assertion Library:**
- unittest.TestCase assertions (`self.assertEqual`, `self.assertTrue`, `self.assertRaises`, etc.)
- NumPy assertions: `np.testing.assert_allclose`, `np.testing.assert_array_equal`

**Run Commands:**
```bash
pytest                          # Run all tests
pytest tests/test_blind_rl.py  # Run specific test file
pytest -v                       # Verbose output
pytest -k test_name            # Run tests matching pattern
```

## Test File Organization

**Location:**
- Mirror of source structure: `src/tiresias/` → `tests/test_`
- Pattern: Co-located by module, not by test type

**Naming:**
- `test_<module_name>.py` (e.g., `test_blind_rl.py`, `test_seeds.py`)
- Test classes: `<ModuleName>Tests` (e.g., `BlindRlTests`, `SeedTests`)
- Test methods: `test_<behavior_description>` (e.g., `test_convolution_adjoints_match_for_even_psf`)

**Structure:**
```
tests/
├── __init__.py           # Empty, marks as package
├── test_blind_rl.py      # Tests for blind_rl module
├── test_seeds.py         # Tests for seeds module
├── test_tiling.py        # Tests for tiling module
├── test_cli.py           # Tests for CLI module
├── test_public_api.py    # API export tests
└── test_benchmark_scripts.py  # Benchmark tests
```

## Test Structure

**Suite Organization:**
```python
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from tiresias import blind_rl


class BlindRlTests(unittest.TestCase):
    def test_descriptive_test_name(self):
        # Arrange: setup test data
        rng = np.random.default_rng(seed_number)
        image = rng.random((shape,), dtype=np.float32)
        
        # Act: perform operation
        result = blind_rl.some_function(image)
        
        # Assert: verify behavior
        self.assertEqual(result.shape, expected_shape)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
```

**Patterns:**
- **Setup**: Create test data inline within test method, use seeded RNG for reproducibility
- **Teardown**: None typically needed (unittest cleans up test case instances)
- **Assertions**: Use specific assertions; avoid bare `self.assertTrue(x == y)` in favor of `self.assertEqual(x, y)`

## Mocking

**Framework:** unittest.mock

**Patterns:**

1. **Mock function return values:**
```python
with mock.patch.object(module, "function_name", return_value=expected) as mock_func:
    result = code_under_test()
    mock_func.assert_called_once()
    self.assertEqual(mock_func.call_args.kwargs["param_name"], value)
```

2. **Mock with side effects (multiple calls or custom behavior):**
```python
with mock.patch.object(module, "function", side_effect=lambda x: x * 2):
    result = code_under_test()
```

3. **Mock module imports (for optional dependencies like CuPy):**
```python
fake_cp = _fake_numpy_cupy_module()  # Helper in test file
with (
    mock.patch.dict(sys.modules, {"cupy": fake_cp}),
    mock.patch.object(blind_rl, "estimate_blind_psf_cupy", return_value=seed),
):
    result = function_using_cupy()
```

4. **Verify call arguments:**
```python
imread.assert_called_once_with(Path("calibrated_psf.tif"))
self.assertEqual(estimate.call_args.kwargs["image_path"], Path("volume.tif"))
estimate.assert_not_called()
```

**What to Mock:**
- External I/O: `imread`, `imwrite`, file operations
- Optional dependencies: CuPy (if not available for CI)
- Expensive operations: GPU computations in unit tests
- System calls: `subprocess.run`, GPU queries

**What NOT to Mock:**
- Core NumPy/SciPy functions
- The functions being tested
- Simple utilities (math functions, type conversions)

## Test Data and Fixtures

**Test Array Creation:**
```python
# Explicit inline creation for simple cases
psf = np.zeros((3, 5, 5), dtype=np.float32)
psf[1, 2, 2] = 0.7
psf[1, 2, 3] = 0.2

# Seeded RNG for reproducibility
rng = np.random.default_rng(seed_number)
image = rng.random((5, 6, 7), dtype=np.float32)

# Derived test data (e.g., convolved images)
observed = fftconvolve(image, true_psf, mode="same").astype(np.float32)
```

**Fixture Pattern:**
- No `pytest.fixture` decorators in use
- Test data created per test method (isolation)
- Helper functions for common data generation (e.g., `_fake_numpy_cupy_module()`)

**Location:**
- Small fixtures inline in test methods
- Reusable helper functions in test file (e.g., `_fake_numpy_cupy_module()` in `test_blind_rl.py`)

## Typical Test Patterns

**Numerical Correctness Tests:**
```python
def test_fft_convolution_engine_matches_fftconvolve(self):
    rng = np.random.default_rng(13)
    image = rng.random((5, 6, 7), dtype=np.float32)
    psf = rng.random((3, 4, 5), dtype=np.float32)
    engine = blind_rl.FftConvolutionEngine(np, image.shape, psf.shape)
    
    np.testing.assert_allclose(
        engine.convolve_same(image, psf),
        blind_rl.convolve_same(image, psf, fftconvolve),
        rtol=2e-5,
        atol=2e-5,
    )
```

**Error Handling Tests:**
```python
def test_load_psf_seed_rejects_zero_energy(self):
    with mock.patch.object(
        seeds,
        "imread",
        return_value=np.zeros((3, 3, 3), dtype=np.float32),
    ):
        with self.assertRaisesRegex(ValueError, "no positive finite energy"):
            seeds.load_psf_seed("empty.tif", (3, 3, 3))
```

**API/Integration Tests:**
```python
def test_estimate_psf_cli_loads_calibrated_seed_without_optical_arguments(self):
    seed = np.ones((5, 7, 7), dtype=np.float32) / 35.0
    
    with (
        mock.patch.object(cli, "load_psf_seed", return_value=seed) as load_seed,
        mock.patch.object(cli, "estimate_psf_from_chunks", return_value=seed) as estimate,
        mock.patch.object(cli, "imwrite"),
    ):
        cli.estimate_psf_main([
            "--image-path", "volume.tif",
            "--output-path", "estimated_psf.tif",
            "--psf-seed-path", "calibrated_psf.tif",
            "--psf-size-z", "5",
            "--psf-size-xy", "7",
        ])
    
    load_seed.assert_called_once_with(Path("calibrated_psf.tif"), (5, 7, 7))
```

**Shape/Constraint Tests:**
```python
def test_batched_blind_rl_matches_single_tile_results(self):
    # ... setup ...
    batched = blind_rl.estimate_blind_psf_batch(observed_batch, seed, 3, xp=np)
    
    self.assertEqual(batched.shape, (2,) + seed.shape)
    np.testing.assert_allclose(
        batched.sum(axis=(1, 2, 3)),
        np.ones(2, dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(batched, expected, rtol=5e-4, atol=5e-4)
```

## Async/GPU Testing

**Pattern:**
- No async tests (synchronous operations)
- GPU tests: Mocked with fake CuPy module in CI
- Helper functions create mock GPU modules: `_fake_numpy_cupy_module()` returns mock with:
  - `asarray`, `asnumpy`, `broadcast_to`
  - `cuda.Device` context manager
  - `fft` functions mapped to SciPy FFT
  - Memory pool stubs

Example from `test_blind_rl.py`:
```python
def _fake_numpy_cupy_module():
    class Device:
        def __init__(self, device_id):
            self.device_id = device_id
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_value, traceback):
            return None
    
    return types.SimpleNamespace(
        asarray=lambda values, dtype=None: np.asarray(values, dtype=dtype),
        asnumpy=lambda values: np.asarray(values).copy(),
        cuda=types.SimpleNamespace(
            Device=Device,
            Stream=types.SimpleNamespace(
                null=types.SimpleNamespace(synchronize=lambda: None)
            )
        ),
        # ... more stubs ...
    )
```

## Coverage

**Current State:**
- No coverage requirements enforced
- No coverage badges/targets in configuration
- Unit tests provide reasonable coverage of core algorithms

**How to Measure (if needed):**
```bash
pytest --cov=tiresias --cov-report=html
```

## Test Types

**Unit Tests:**
- Scope: Individual functions and classes
- Location: All `test_*.py` files
- Approach: Isolated, fast, use mocking for dependencies
- Examples:
  - `test_convolution_adjoints_match_for_even_psf` - Tests convolution math
  - `test_load_psf_seed_rejects_zero_energy` - Tests error handling
  - `test_resolve_dxy_accepts_direct_pixel_size` - Tests parameter validation

**Integration Tests:**
- Scope: Multiple components working together
- Location: `test_blind_rl.py`, `test_tiling.py`
- Approach: Full algorithm runs (not mocked)
- Examples:
  - `test_batched_blind_rl_matches_single_tile_results` - Validates batch mode vs single mode
  - `test_shared_blind_rl_matches_single_tile_for_identical_observations` - Validates shared PSF estimation

**End-to-End / CLI Tests:**
- Scope: Full CLI workflows
- Location: `test_cli.py`
- Approach: Mock file I/O, verify argument parsing and function calls
- Examples:
  - `test_estimate_psf_cli_loads_calibrated_seed_without_optical_arguments`
  - `test_deconvolve_cli_reads_inputs_and_writes_restored_tiff`

**API Tests:**
- Scope: Public module exports
- Location: `test_public_api.py`
- Approach: Verify `__all__` exports and import correctness
- Example: `test_public_api_exports_core_functions`

## Common Assertions

**NumPy Equality:**
```python
np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-5)
np.testing.assert_array_equal(result, expected)
np.testing.assert_array_almost_equal(result, expected, decimal=5)
```

**Shape/Type/Value:**
```python
self.assertEqual(array.shape, (3, 5, 5))
self.assertEqual(array.dtype, np.float32)
self.assertTrue(np.isfinite(array).all())
self.assertGreaterEqual(float(np.min(array)), 0.0)
self.assertTrue(np.isclose(np.sum(array), 1.0, atol=1e-6))
```

**Exception Handling:**
```python
with self.assertRaisesRegex(ValueError, "pattern to match"):
    function_that_raises()
```

**Mock Calls:**
```python
mock_obj.assert_called_once()
mock_obj.assert_called_once_with(arg1, arg2, kwarg=value)
mock_obj.assert_not_called()
self.assertEqual(mock_obj.call_args.kwargs["name"], expected_value)
```

## Test Isolation

**Independence:**
- Each test method creates its own data
- Tests can run in any order
- No shared state between tests
- Use `setUp`/`tearDown` only if needed (not currently used)

**Environment:**
- Tests run in `tests/` directory
- `sys.modules` mocked for optional imports (CuPy)
- Temporary directories used with `tempfile.TemporaryDirectory()`

---

*Testing analysis: 2026-09-08*
