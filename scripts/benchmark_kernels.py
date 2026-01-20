#!/usr/bin/env python3
"""Performance benchmarking for fp8 2D quantized matmul kernels.

Compares all three kernel versions (v1, v2, v3) on various problem sizes
to measure throughput, MXU utilization, and identify the fastest implementation.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
)


def create_benchmark_data(batch_size, n_in, n_out, quant_block_size, seed=42):
    """Create test data for benchmarking."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    # Create random inputs
    x = jax.random.normal(k1, (batch_size, n_in), dtype=jnp.bfloat16) * 0.5
    w = jax.random.normal(k2, (n_out, n_in), dtype=jnp.bfloat16) * 0.5

    # Quantize weights with 2D blocks
    w_q, w_scale = quantize_tensor_2d(
        w, jnp.float8_e4m3fn,
        block_size_m=quant_block_size,
        block_size_n=quant_block_size
    )

    return x, w_q, w_scale


def compute_flops(batch_size, n_in, n_out):
    """Compute FLOPs for matmul: 2 * batch * n_in * n_out."""
    return 2 * batch_size * n_in * n_out


def warmup_kernel(kernel_fn, x, w_q, w_scale, quant_block_size, num_warmup=3):
    """Warm up kernel to ensure compilation and cache warming."""
    print(f"    Warming up ({num_warmup} iterations)...", end=" ", flush=True)

    for _ in range(num_warmup):
        output = kernel_fn(
            x, w_q, w_scale,
            x_q_dtype=jnp.float8_e4m3fn,
            quant_block_size=quant_block_size,
        )
        output.block_until_ready()

    print("done")


def benchmark_kernel(kernel_fn, version_name, x, w_q, w_scale, quant_block_size,
                     num_iterations=100, num_warmup=3):
    """Benchmark a single kernel version."""
    print(f"\n  Benchmarking {version_name}...")
    print(f"    Input shape: {x.shape}, Weight shape: {w_q.shape}")

    # Warmup
    try:
        warmup_kernel(kernel_fn, x, w_q, w_scale, quant_block_size, num_warmup)
    except Exception as e:
        print(f"    ❌ Warmup failed: {e}")
        return None

    # Benchmark
    print(f"    Running {num_iterations} iterations...", end=" ", flush=True)
    times = []

    try:
        for _ in range(num_iterations):
            start = time.perf_counter()
            output = kernel_fn(
                x, w_q, w_scale,
                x_q_dtype=jnp.float8_e4m3fn,
                quant_block_size=quant_block_size,
            )
            output.block_until_ready()
            end = time.perf_counter()
            times.append(end - start)

        print("done")

        # Compute statistics
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        median_time = np.median(times)

        # Compute throughput
        batch_size, n_in = x.shape
        n_out, _ = w_q.shape
        flops = compute_flops(batch_size, n_in, n_out)
        tflops_mean = (flops / mean_time) / 1e12
        tflops_best = (flops / min_time) / 1e12

        results = {
            "mean_time_ms": mean_time * 1000,
            "std_time_ms": std_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "median_time_ms": median_time * 1000,
            "tflops_mean": tflops_mean,
            "tflops_best": tflops_best,
            "flops": flops,
        }

        print(f"    Mean time: {mean_time*1000:.3f} ms (±{std_time*1000:.3f} ms)")
        print(f"    Best time: {min_time*1000:.3f} ms")
        print(f"    Throughput (mean): {tflops_mean:.2f} TFLOPS/s")
        print(f"    Throughput (best): {tflops_best:.2f} TFLOPS/s")

        return results

    except Exception as e:
        print(f"\n    ❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_benchmark_suite(num_iterations=100, num_warmup=3):
    """Run comprehensive benchmark suite."""
    print("=" * 80)
    print("FP8 2D Quantized Matmul - Performance Benchmark Suite")
    print("=" * 80)
    print()

    # Get device info
    devices = jax.devices()
    print(f"Device: {devices[0]}")
    print(f"Device kind: {devices[0].device_kind}")
    print()

    # Benchmark configurations
    # (batch, n_in, n_out, quant_block, description)
    benchmark_configs = [
        (512, 512, 512, 128, "Small (512×512, 128 blocks)"),
        (1024, 1024, 1024, 256, "Medium (1024×1024, 256 blocks)"),
        (2048, 2048, 2048, 512, "Large (2048×2048, 512 blocks)"),
        (4096, 4096, 4096, 512, "XLarge (4096×4096, 512 blocks)"),
        (1024, 2048, 4096, 512, "Asymmetric (1024×2048×4096)"),
        (2048, 4096, 2048, 512, "Wide (2048×4096×2048)"),
    ]

    # Store all results
    all_results = {}

    for batch, n_in, n_out, quant_block, description in benchmark_configs:
        print("=" * 80)
        print(f"Configuration: {description}")
        print(f"  Batch: {batch}, N_in: {n_in}, N_out: {n_out}, Quant block: {quant_block}")
        print("=" * 80)

        # Create test data
        print("\nCreating test data...")
        x, w_q, w_scale = create_benchmark_data(batch, n_in, n_out, quant_block)

        # Benchmark all versions
        versions = [
            (kernel_v1, "v1 (Auto Double Buffer)"),
            (kernel_v2, "v2 (Manual Async DMA)"),
            (kernel_v3, "v3 (SMEM Scales)"),
        ]

        config_results = {}

        for kernel_fn, version_name in versions:
            version_key = version_name.split()[0]  # Extract v1, v2, v3

            results = benchmark_kernel(
                kernel_fn, version_name, x, w_q, w_scale, quant_block,
                num_iterations, num_warmup
            )

            if results is not None:
                config_results[version_key] = results

        all_results[description] = config_results

        # Print comparison for this config
        if len(config_results) > 0:
            print("\n  " + "-" * 76)
            print(f"  {'Version':<25} {'Mean (ms)':<15} {'Best (ms)':<15} {'TFLOPS/s':<15}")
            print("  " + "-" * 76)

            best_tflops = max(r["tflops_best"] for r in config_results.values())

            for version in ["v1", "v2", "v3"]:
                if version in config_results:
                    r = config_results[version]
                    marker = "🏆" if r["tflops_best"] == best_tflops else "  "
                    print(f"  {marker} {version:<23} {r['mean_time_ms']:<15.3f} "
                          f"{r['min_time_ms']:<15.3f} {r['tflops_best']:<15.2f}")

            print("  " + "-" * 76)

    # Print final summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    for config_name, results in all_results.items():
        if len(results) == 0:
            continue

        print(f"\n{config_name}:")

        best_tflops = max(r["tflops_best"] for r in results.values())
        best_version = [v for v, r in results.items() if r["tflops_best"] == best_tflops][0]

        print(f"  Best: {best_version} - {best_tflops:.2f} TFLOPS/s")

        # Show relative performance
        for version in ["v1", "v2", "v3"]:
            if version in results:
                relative = (results[version]["tflops_best"] / best_tflops) * 100
                print(f"    {version}: {results[version]['tflops_best']:.2f} TFLOPS/s ({relative:.1f}% of best)")

    print("\n" + "=" * 80)

    # Overall winner
    print("\nOVERALL ANALYSIS:")
    print("-" * 80)

    version_wins = {"v1": 0, "v2": 0, "v3": 0}

    for config_name, results in all_results.items():
        if len(results) == 0:
            continue
        best_tflops = max(r["tflops_best"] for r in results.values())
        best_version = [v for v, r in results.items() if r["tflops_best"] == best_tflops][0]
        version_wins[best_version] += 1

    print("\nWins by version:")
    for version in ["v1", "v2", "v3"]:
        wins = version_wins[version]
        total = len(all_results)
        print(f"  {version}: {wins}/{total} configurations")

    winner = max(version_wins.items(), key=lambda x: x[1])[0]
    print(f"\n🏆 Overall winner: {winner}")

    print("\n" + "=" * 80)
    print("\nRECOMMENDATIONS:")
    print("-" * 80)

    if winner == "v1":
        print("✓ v1 (Auto Double Buffer) performs best")
        print("  → Use v1 as the default implementation")
        print("  → Compiler's automatic optimizations are sufficient")
    elif winner == "v2":
        print("✓ v2 (Manual Async DMA) performs best")
        print("  → Manual DMA control provides better performance")
        print("  → Use v2 as the default implementation")
    elif winner == "v3":
        print("✓ v3 (SMEM Scales) performs best")
        print("  → SMEM for scales provides measurable speedup")
        print("  → Use v3 as the default implementation")

    print("\nNext steps:")
    print("  1. Profile on actual TPU hardware if not already")
    print("  2. Check MXU utilization in profiler")
    print("  3. Analyze memory bandwidth usage")
    print("  4. Consider hybrid approaches if results are mixed")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark fp8 2D quantized matmul kernels")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of benchmark iterations (default: 100)")
    parser.add_argument("--warmup", type=int, default=3,
                       help="Number of warmup iterations (default: 3)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with fewer iterations (10 iters, 2 warmup)")

    args = parser.parse_args()

    if args.quick:
        print("Running in quick mode (fewer iterations)")
        num_iterations = 10
        num_warmup = 2
    else:
        num_iterations = args.iterations
        num_warmup = args.warmup

    print(f"Benchmark parameters: {num_iterations} iterations, {num_warmup} warmup")
    print()

    run_benchmark_suite(num_iterations=num_iterations, num_warmup=num_warmup)
