#!/usr/bin/env python3
"""Correctness tests for fp8 2D quantized matmul kernels.

Tests all three kernel versions (v1, v2, v3) against a reference implementation
to verify correctness before benchmarking performance.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tpu_inference.kernels.fp8_quantized_matmul_2d.v1.kernel import (
    fp8_quantized_matmul_2d_kernel as kernel_v1,
)
from tpu_inference.kernels.fp8_quantized_matmul_2d.v2.kernel import (
    fp8_quantized_matmul_2d_kernel as kernel_v2,
)
from tpu_inference.kernels.fp8_quantized_matmul_2d.v3.kernel import (
    fp8_quantized_matmul_2d_kernel as kernel_v3,
)
from tpu_inference.kernels.fp8_quantized_matmul_2d.v1.util import (
    quantize_tensor_2d,
    xla_quantized_matmul_2d,
)


def create_test_data(batch_size, n_in, n_out, quant_block_size, seed=42):
    """Create test data for correctness testing."""
    key = jax.random.PRNGKey(seed)

    # Split keys
    k1, k2 = jax.random.split(key)

    # Create random inputs (use smaller values for better fp8 precision)
    x = jax.random.normal(k1, (batch_size, n_in), dtype=jnp.bfloat16) * 0.5
    w = jax.random.normal(k2, (n_out, n_in), dtype=jnp.bfloat16) * 0.5

    # Quantize weights with 2D blocks
    w_q, w_scale = quantize_tensor_2d(
        w, jnp.float8_e4m3fn,
        block_size_m=quant_block_size,
        block_size_n=quant_block_size
    )

    return x, w, w_q, w_scale


def compute_reference(x, w, w_q, w_scale, quant_block_size):
    """Compute reference output using pure JAX implementation."""
    return xla_quantized_matmul_2d(
        x, w_q, w_scale,
        x_quantize=True,
        quant_block_size=quant_block_size,
    )


def relative_error(a, b):
    """Compute relative error between two arrays."""
    diff = jnp.abs(a - b)
    ref_norm = jnp.abs(b)
    # Avoid division by zero
    rel_err = jnp.where(ref_norm > 1e-6, diff / (ref_norm + 1e-8), diff)
    return jnp.max(rel_err), jnp.mean(rel_err)


def test_kernel_correctness(kernel_fn, version_name, x, w, w_q, w_scale,
                            reference, quant_block_size, tolerance=1e-2):
    """Test a kernel version for correctness."""
    print(f"\nTesting {version_name}...")
    print("-" * 60)

    try:
        # Run kernel
        output = kernel_fn(
            x, w_q, w_scale,
            x_q_dtype=jnp.float8_e4m3fn,
            quant_block_size=quant_block_size,
        )

        # Check shape
        if output.shape != reference.shape:
            print(f"  ❌ Shape mismatch: {output.shape} vs {reference.shape}")
            return False

        # Check numerical accuracy
        max_err, mean_err = relative_error(output, reference)

        print(f"  Shape: {output.shape}")
        print(f"  Max relative error: {max_err:.6f}")
        print(f"  Mean relative error: {mean_err:.6f}")
        print(f"  Tolerance: {tolerance}")

        # Check against tolerance
        if max_err <= tolerance:
            print(f"  ✅ PASS - Within tolerance")
            return True
        else:
            print(f"  ❌ FAIL - Exceeds tolerance")

            # Show some statistics for debugging
            print(f"\n  Debug info:")
            print(f"    Output range: [{jnp.min(output):.4f}, {jnp.max(output):.4f}]")
            print(f"    Reference range: [{jnp.min(reference):.4f}, {jnp.max(reference):.4f}]")
            print(f"    Absolute error range: [{jnp.min(jnp.abs(output - reference)):.6f}, "
                  f"{jnp.max(jnp.abs(output - reference)):.6f}]")

            return False

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_test_suite():
    """Run comprehensive correctness test suite."""
    print("=" * 80)
    print("FP8 2D Quantized Matmul - Correctness Test Suite")
    print("=" * 80)

    # Test configurations
    test_configs = [
        # (batch, n_in, n_out, quant_block, tolerance)
        (512, 512, 512, 128, 5e-2),      # Small, 128 blocks
        (1024, 1024, 1024, 256, 5e-2),   # Medium, 256 blocks
        (2048, 2048, 2048, 512, 5e-2),   # Large, 512 blocks (aligned)
        (1536, 1536, 1536, 512, 5e-2),   # Non-aligned size
        (1024, 2048, 4096, 512, 5e-2),   # Asymmetric
    ]

    results = {
        "v1": {"passed": 0, "failed": 0},
        "v2": {"passed": 0, "failed": 0},
        "v3": {"passed": 0, "failed": 0},
    }

    for batch, n_in, n_out, quant_block, tolerance in test_configs:
        print(f"\n{'=' * 80}")
        print(f"Test: {batch}×{n_in} @ {n_in}×{n_out}, quant_block={quant_block}")
        print(f"{'=' * 80}")

        # Create test data
        x, w, w_q, w_scale = create_test_data(batch, n_in, n_out, quant_block)

        # Compute reference
        print("\nComputing reference (pure JAX)...")
        reference = compute_reference(x, w, w_q, w_scale, quant_block)
        print(f"  Reference computed: {reference.shape}")

        # Test all versions
        versions = [
            (kernel_v1, "v1 (Auto Double Buffer)"),
            (kernel_v2, "v2 (Manual Async DMA)"),
            (kernel_v3, "v3 (SMEM Scales)"),
        ]

        for kernel_fn, version_name in versions:
            version_key = version_name.split()[0]  # Extract v1, v2, v3

            passed = test_kernel_correctness(
                kernel_fn, version_name, x, w, w_q, w_scale,
                reference, quant_block, tolerance
            )

            if passed:
                results[version_key]["passed"] += 1
            else:
                results[version_key]["failed"] += 1

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total_tests = len(test_configs)
    all_passed = True

    for version in ["v1", "v2", "v3"]:
        passed = results[version]["passed"]
        failed = results[version]["failed"]

        status = "✅ PASS" if failed == 0 else "❌ FAIL"
        print(f"\n{version}: {status}")
        print(f"  Passed: {passed}/{total_tests}")
        print(f"  Failed: {failed}/{total_tests}")

        if failed > 0:
            all_passed = False

    print("\n" + "=" * 80)

    if all_passed:
        print("✅ ALL TESTS PASSED - All versions are correct!")
        print("\nNext step: Run benchmarks to compare performance")
        print("  python scripts/benchmark_kernels.py")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Fix errors before benchmarking")
        print("\nDebug steps:")
        print("  1. Check HLO output: ./scripts/quick_hlo_check.sh")
        print("  2. Review failed kernel implementations")
        print("  3. Verify quantization is consistent")
        return 1


if __name__ == "__main__":
    import os

    # Ensure we're using the correct backend
    if "JAX_PLATFORMS" not in os.environ:
        print("Note: JAX_PLATFORMS not set. Using default backend.")
        print("For TPU testing, set: export JAX_PLATFORMS=tpu")
        print()

    sys.exit(run_test_suite())
