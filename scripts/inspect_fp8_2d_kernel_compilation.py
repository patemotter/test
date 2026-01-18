#!/usr/bin/env python3
"""Inspect HLO/LLO compilation output for fp8 2D quantized matmul kernels.

This script compiles all three kernel versions and dumps their HLO/LLO for analysis.
"""

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

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
from tpu_inference.kernels.fp8_quantized_matmul_2d.v1.util import quantize_tensor_2d


def create_test_inputs(batch_size=2048, n_in=2048, n_out=4096, quant_block_size=512):
    """Create test inputs for kernel compilation."""
    print(f"Creating test inputs: {batch_size}×{n_in} @ {n_in}×{n_out}")
    print(f"Quantization block size: {quant_block_size}×{quant_block_size}")

    # Create random inputs
    x = jnp.ones((batch_size, n_in), dtype=jnp.bfloat16)
    w = jnp.ones((n_out, n_in), dtype=jnp.bfloat16)

    # Quantize weights
    w_q, w_scale = quantize_tensor_2d(
        w, jnp.float8_e4m3fn,
        block_size_m=quant_block_size,
        block_size_n=quant_block_size
    )

    print(f"Input shapes:")
    print(f"  x: {x.shape} ({x.dtype})")
    print(f"  w_q: {w_q.shape} ({w_q.dtype})")
    print(f"  w_scale: {w_scale.shape} ({w_scale.dtype})")
    print()

    return x, w_q, w_scale


def inspect_kernel(kernel_fn, version_name, x, w_q, w_scale, quant_block_size):
    """Compile kernel and dump HLO/LLO for inspection."""
    print(f"{'='*80}")
    print(f"Compiling {version_name}")
    print(f"{'='*80}")

    try:
        # Lower to StableHLO
        lowered = jax.jit(
            lambda x, w_q, w_scale: kernel_fn(
                x, w_q, w_scale,
                x_q_dtype=jnp.float8_e4m3fn,
                quant_block_size=quant_block_size,
            )
        ).lower(x, w_q, w_scale)

        # Get HLO text
        hlo_text = lowered.as_text()

        # Save to file
        output_dir = Path("compilation_output")
        output_dir.mkdir(exist_ok=True)

        hlo_file = output_dir / f"{version_name}_hlo.txt"
        with open(hlo_file, "w") as f:
            f.write(hlo_text)
        print(f"✓ HLO saved to: {hlo_file}")

        # Try to get compiled output (this may fail if not on TPU)
        try:
            compiled = lowered.compile()

            # Get cost analysis if available
            if hasattr(compiled, 'cost_analysis'):
                cost = compiled.cost_analysis()
                print(f"\nCost Analysis:")
                for key, value in sorted(cost[0].items()):
                    print(f"  {key}: {value}")

            # Try to get LLO/assembly (TPU-specific)
            if hasattr(compiled, 'as_text'):
                llo_text = compiled.as_text()
                llo_file = output_dir / f"{version_name}_llo.txt"
                with open(llo_file, "w") as f:
                    f.write(llo_text)
                print(f"✓ LLO saved to: {llo_file}")
        except Exception as e:
            print(f"⚠ Could not get compiled output (not on TPU?): {e}")

        # Analyze HLO for key patterns
        print(f"\nHLO Analysis for {version_name}:")
        analyze_hlo(hlo_text, version_name)

    except Exception as e:
        print(f"✗ Error compiling {version_name}: {e}")
        import traceback
        traceback.print_exc()

    print()


def analyze_hlo(hlo_text, version_name):
    """Analyze HLO text for key optimization patterns."""

    # Key patterns to look for
    patterns = {
        "fp8 matmuls": ["dot", "f8e4m3fn"],
        "async copies": ["async-start", "async-done", "async-copy-start", "async-copy-done"],
        "collective-permute": ["collective-permute"],  # For prefetching
        "while loops": ["while"],  # Grid iteration
        "custom-call": ["custom-call"],  # Pallas kernel calls
        "memory spaces": ["memory_space", "VMEM", "SMEM", "HBM"],
    }

    findings = {}

    for pattern_name, keywords in patterns.items():
        count = 0
        matches = []
        for keyword in keywords:
            keyword_count = hlo_text.count(keyword)
            count += keyword_count
            if keyword_count > 0:
                matches.append(f"{keyword}({keyword_count})")

        if count > 0:
            findings[pattern_name] = f"{count} occurrences: {', '.join(matches)}"

    # Print findings
    for pattern_name, result in findings.items():
        print(f"  ✓ {pattern_name}: {result}")

    # Version-specific checks
    if version_name == "v1":
        # V1 should have automatic prefetching via collective-permute or loops
        if "while loops" in findings:
            print(f"  → V1: Grid iteration detected (automatic double buffering)")
        else:
            print(f"  ⚠ V1: No explicit grid iteration found (check if single iteration)")

    elif version_name == "v2":
        # V2 should have explicit async copies
        if "async copies" in findings:
            print(f"  → V2: Explicit async DMA detected ✓")
        else:
            print(f"  ⚠ V2: No async copies found! Manual DMA may not be working")

    elif version_name == "v3":
        # V3 should have both async copies AND SMEM
        has_async = "async copies" in findings
        has_smem = any("SMEM" in findings.get(k, "") for k in findings)

        if has_async and has_smem:
            print(f"  → V3: Both async DMA and SMEM detected ✓✓")
        elif has_async:
            print(f"  → V3: Async DMA detected, but SMEM not found")
        elif has_smem:
            print(f"  → V3: SMEM detected, but async DMA not found")
        else:
            print(f"  ⚠ V3: Neither async DMA nor SMEM found!")

    # Check for fp8 native matmul usage
    if "fp8 matmuls" in findings:
        print(f"  → Native fp8 MXU usage detected ✓")
    else:
        print(f"  ⚠ No fp8 operations found - may be using fp32 fallback")


def main():
    print("="*80)
    print("FP8 2D Quantized Matmul Kernel Compilation Inspector")
    print("="*80)
    print()

    # Configuration
    batch_size = 2048
    n_in = 2048
    n_out = 4096
    quant_block_size = 512  # Aligned blocks for fair comparison

    # Create test inputs
    x, w_q, w_scale = create_test_inputs(batch_size, n_in, n_out, quant_block_size)

    # Inspect all three versions
    versions = [
        (kernel_v1, "v1_auto_double_buffer"),
        (kernel_v2, "v2_manual_async_dma"),
        (kernel_v3, "v3_smem_scales"),
    ]

    for kernel_fn, version_name in versions:
        inspect_kernel(kernel_fn, version_name, x, w_q, w_scale, quant_block_size)

    print("="*80)
    print("Compilation inspection complete!")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Review HLO files in compilation_output/")
    print("2. Look for:")
    print("   - Async copy operations (v2, v3)")
    print("   - SMEM memory spaces (v3)")
    print("   - Native fp8 dot operations (all versions)")
    print("   - Prefetch patterns (v1)")
    print("3. Compare instruction counts and memory usage")
    print("4. Run on actual TPU to get LLO and performance data")


if __name__ == "__main__":
    # Set JAX to CPU backend for compilation inspection
    # (can be changed to TPU if available)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    main()
