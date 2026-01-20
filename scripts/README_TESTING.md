# Testing and Benchmarking Guide

This guide shows how to test correctness and benchmark performance of the fp8 2D quantized matmul kernels.

## Quick Start

```bash
# 1. Test correctness (all versions must pass before benchmarking)
python scripts/test_kernel_correctness.py

# 2. Benchmark performance (compare all versions)
python scripts/benchmark_kernels.py

# 3. Quick benchmark (fewer iterations, faster)
python scripts/benchmark_kernels.py --quick
```

---

## Correctness Testing

**File:** `scripts/test_kernel_correctness.py`

Tests all three kernel versions against a reference JAX implementation to verify correctness.

### What it does:

1. Creates random test data
2. Computes reference output using pure JAX (no Pallas)
3. Runs each kernel version (v1, v2, v3)
4. Compares outputs with tolerance checking
5. Reports PASS/FAIL for each version

### Test configurations:

The script tests on multiple problem sizes:
- Small: 512×512 (128 blocks)
- Medium: 1024×1024 (256 blocks)
- Large: 2048×2048 (512 blocks)
- Non-aligned: 1536×1536 (512 blocks)
- Asymmetric: 1024×2048×4096

### Running:

```bash
python scripts/test_kernel_correctness.py
```

### Expected output:

```
================================================================================
FP8 2D Quantized Matmul - Correctness Test Suite
================================================================================

================================================================================
Test: 512×512 @ 512×512, quant_block=128
================================================================================

Computing reference (pure JAX)...
  Reference computed: (512, 512)

Testing v1 (Auto Double Buffer)...
------------------------------------------------------------
  Shape: (512, 512)
  Max relative error: 0.023456
  Mean relative error: 0.001234
  Tolerance: 0.05
  ✅ PASS - Within tolerance

Testing v2 (Manual Async DMA)...
------------------------------------------------------------
  ✅ PASS - Within tolerance

Testing v3 (SMEM Scales)...
------------------------------------------------------------
  ✅ PASS - Within tolerance

...

================================================================================
TEST SUMMARY
================================================================================

v1: ✅ PASS
  Passed: 5/5
  Failed: 0/5

v2: ✅ PASS
  Passed: 5/5
  Failed: 0/5

v3: ✅ PASS
  Passed: 5/5
  Failed: 0/5

================================================================================
✅ ALL TESTS PASSED - All versions are correct!

Next step: Run benchmarks to compare performance
  python scripts/benchmark_kernels.py
```

### If tests fail:

If any version fails correctness tests:

```
❌ SOME TESTS FAILED - Fix errors before benchmarking

Debug steps:
  1. Check HLO output: ./scripts/quick_hlo_check.sh
  2. Review failed kernel implementations
  3. Verify quantization is consistent
```

**Do NOT benchmark until all correctness tests pass!**

---

## Performance Benchmarking

**File:** `scripts/benchmark_kernels.py`

Compares performance of all three kernel versions on various problem sizes.

### What it measures:

1. **Execution time** (mean, min, max, std dev)
2. **Throughput** (TFLOPS/s)
3. **Relative performance** (which version is fastest)

### Benchmark configurations:

- Small: 512×512 (128 blocks)
- Medium: 1024×1024 (256 blocks)
- Large: 2048×2048 (512 blocks)
- XLarge: 4096×4096 (512 blocks)
- Asymmetric: 1024×2048×4096
- Wide: 2048×4096×2048

### Running:

```bash
# Full benchmark (100 iterations, ~5-10 minutes)
python scripts/benchmark_kernels.py

# Quick test (10 iterations, ~1 minute)
python scripts/benchmark_kernels.py --quick

# Custom iterations
python scripts/benchmark_kernels.py --iterations 50 --warmup 5
```

### Expected output:

```
================================================================================
FP8 2D Quantized Matmul - Performance Benchmark Suite
================================================================================

Device: TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)
Device kind: TPU v4

================================================================================
Configuration: Large (2048×2048, 512 blocks)
  Batch: 2048, N_in: 2048, N_out: 2048, Quant block: 512
================================================================================

Creating test data...

  Benchmarking v1 (Auto Double Buffer)...
    Input shape: (2048, 2048), Weight shape: (2048, 2048)
    Warming up (3 iterations)... done
    Running 100 iterations... done
    Mean time: 1.234 ms (±0.056 ms)
    Best time: 1.189 ms
    Throughput (mean): 13.94 TFLOPS/s
    Throughput (best): 14.47 TFLOPS/s

  Benchmarking v2 (Manual Async DMA)...
    ...
    Throughput (best): 15.23 TFLOPS/s

  Benchmarking v3 (SMEM Scales)...
    ...
    Throughput (best): 16.12 TFLOPS/s

  ----------------------------------------------------------------------------
  Version                   Mean (ms)       Best (ms)       TFLOPS/s
  ----------------------------------------------------------------------------
     v1                     1.234           1.189           14.47
     v2                     1.128           1.096           15.23
  🏆 v3                     1.067           1.035           16.12
  ----------------------------------------------------------------------------

...

================================================================================
BENCHMARK SUMMARY
================================================================================

Large (2048×2048, 512 blocks):
  Best: v3 - 16.12 TFLOPS/s
    v1: 14.47 TFLOPS/s (89.8% of best)
    v2: 15.23 TFLOPS/s (94.5% of best)
    v3: 16.12 TFLOPS/s (100.0% of best)

...

OVERALL ANALYSIS:
--------------------------------------------------------------------------------

Wins by version:
  v1: 0/6 configurations
  v2: 1/6 configurations
  v3: 5/6 configurations

🏆 Overall winner: v3

================================================================================

RECOMMENDATIONS:
--------------------------------------------------------------------------------
✓ v3 (SMEM Scales) performs best
  → SMEM for scales provides measurable speedup
  → Use v3 as the default implementation

Next steps:
  1. Profile on actual TPU hardware if not already
  2. Check MXU utilization in profiler
  3. Analyze memory bandwidth usage
  4. Consider hybrid approaches if results are mixed
```

### Options:

```bash
# Quick mode (10 iterations, 2 warmup)
python scripts/benchmark_kernels.py --quick

# Custom iterations
python scripts/benchmark_kernels.py --iterations 200 --warmup 10

# Help
python scripts/benchmark_kernels.py --help
```

---

## Understanding Results

### Correctness Testing

**Tolerance:** 5e-2 (5% relative error)

This is appropriate for fp8 quantization because:
- fp8 has limited precision (~2 decimal digits)
- Quantization introduces approximation error
- Different computation orders can accumulate error differently

**What PASS means:**
- Kernel produces numerically correct results
- Safe to use in production
- Differences are within expected fp8 precision

**What FAIL means:**
- Bug in kernel implementation
- Wrong scaling or quantization
- Memory corruption or indexing error
- DO NOT use until fixed!

### Performance Benchmarking

**TFLOPS/s:** Tera (trillion) floating-point operations per second
- Higher is better
- Typical TPU v4: 10-30 TFLOPS/s for fp8 matmul (depends on size)
- Typical TPU v5e: 20-50 TFLOPS/s

**Interpreting results:**

1. **v1 wins** → Compiler's auto-optimization is sufficient
2. **v2 wins** → Manual DMA control helps
3. **v3 wins** → SMEM for scales provides speedup

**Relative performance:**
- <5% difference: Negligible, use simpler version (v1)
- 5-10% difference: Noticeable, consider complexity vs. gain
- >10% difference: Significant, use faster version

---

## Workflow

### Standard workflow:

```bash
# Step 1: Verify HLO (optional but recommended)
./scripts/quick_hlo_check.sh

# Step 2: Test correctness (REQUIRED)
python scripts/test_kernel_correctness.py

# Step 3: Benchmark performance
python scripts/benchmark_kernels.py

# Step 4: Choose best version based on results
```

### If developing/debugging:

```bash
# Check HLO patterns
./scripts/quick_hlo_check.sh

# Verify bug fixes
./scripts/verify_bugfixes.sh

# Quick correctness check
python scripts/test_kernel_correctness.py

# Quick benchmark
python scripts/benchmark_kernels.py --quick
```

---

## Troubleshooting

### Correctness tests fail

**Symptom:** `❌ FAIL - Exceeds tolerance`

**Debug steps:**

1. Check which version failed (v1, v2, or v3)
2. Look at the debug info (output range, error magnitude)
3. Verify HLO patterns: `./scripts/quick_hlo_check.sh`
4. Check for known issues in `BUGFIXES.md`

**Common causes:**
- Wrong memory space (should be HBM for inputs)
- Missing async DMA operations
- Incorrect scale application
- Buffer indexing bug

### Benchmark crashes

**Symptom:** `ERROR: ...` during benchmark

**Solutions:**

1. Try quick mode first: `--quick`
2. Check memory: TPU might be out of VMEM
3. Reduce problem sizes in benchmark configs
4. Check JAX version compatibility

### Low performance

**Symptom:** TFLOPS/s much lower than expected

**Possible causes:**

1. **Not on TPU:** Running on CPU/GPU instead
   - Check: `jax.devices()` should show TPU
   - Fix: Set `JAX_PLATFORMS=tpu`

2. **Cold cache:** First run is slow
   - Warmup iterations address this
   - Use `--warmup 10` for more warmup

3. **Small problem size:** Overhead dominates
   - TPUs excel at large matmuls
   - Try larger configurations (4096×4096)

4. **Wrong block size:** Not aligned with TPU tiles
   - Ensure quant_block_size is 128, 256, or 512
   - Check that dimensions are divisible

---

## Advanced Usage

### Custom test configurations

Edit `test_kernel_correctness.py`:

```python
test_configs = [
    # Add your custom configuration
    (your_batch, your_n_in, your_n_out, your_quant_block, tolerance),
]
```

### Custom benchmark configurations

Edit `benchmark_kernels.py`:

```python
benchmark_configs = [
    # Add your custom configuration
    (batch, n_in, n_out, quant_block, "Description"),
]
```

### Profiling

For detailed profiling:

```python
import jax.profiler

# In your code
jax.profiler.start_trace("/tmp/tensorboard")
# ... run kernel ...
jax.profiler.stop_trace()

# View in TensorBoard
# tensorboard --logdir=/tmp/tensorboard
```

---

## File Summary

| File | Purpose | When to Use |
|------|---------|-------------|
| `test_kernel_correctness.py` | Verify numerical correctness | Before benchmarking, after changes |
| `benchmark_kernels.py` | Compare performance | After correctness tests pass |
| `quick_hlo_check.sh` | Inspect HLO output | When debugging, verifying optimizations |
| `verify_bugfixes.sh` | Check v2/v3 bug fixes | After kernel changes, troubleshooting |
| `analyze_hlo.py` | Deep HLO analysis | When investigating performance issues |
| `inspect_fp8_2d_kernel_compilation.py` | Generate HLO dumps | First step in HLO inspection |

---

## Next Steps After Benchmarking

### If v1 wins:

```python
# Use v1 as default in __init__.py
from tpu_inference.kernels.fp8_quantized_matmul_2d.v1 import fp8_quantized_matmul_2d_kernel
```

### If v2 or v3 wins:

1. Update default export
2. Document the performance gains
3. Consider keeping all versions for flexibility

### Production deployment:

1. Run full benchmark suite (not `--quick`)
2. Test on actual workload sizes
3. Profile to check MXU utilization
4. Monitor memory usage
5. Add integration tests

---

## Questions?

See also:
- `scripts/README_HLO_INSPECTION.md` - HLO analysis guide
- `scripts/HLO_VERIFICATION_QUICK_GUIDE.md` - Bug fix verification
- `tpu_inference/kernels/fp8_quantized_matmul_2d/README.md` - Kernel overview
- `tpu_inference/kernels/fp8_quantized_matmul_2d/VERSION_COMPARISON.md` - Version differences
