#!/usr/bin/env python3
"""Detailed HLO analyzer for fp8 2D quantized matmul kernels.

This script performs deep analysis of HLO dumps to identify:
- Double buffering patterns
- Async DMA operations
- Memory hierarchy usage (VMEM, SMEM, HBM)
- Native fp8 operations
- Prefetch scheduling
"""

import re
import sys
from pathlib import Path
from collections import defaultdict


def analyze_hlo_file(hlo_file: Path):
    """Deep analysis of HLO file."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {hlo_file.name}")
    print(f"{'='*80}\n")

    with open(hlo_file, 'r') as f:
        hlo_text = f.read()

    # Split into lines for detailed analysis
    lines = hlo_text.split('\n')

    # Analysis sections
    print("📊 OVERALL STATISTICS")
    print("-" * 80)
    print_overall_stats(hlo_text, lines)

    print("\n🔍 MEMORY HIERARCHY")
    print("-" * 80)
    analyze_memory_hierarchy(hlo_text, lines)

    print("\n⚡ COMPUTATION PATTERNS")
    print("-" * 80)
    analyze_computation(hlo_text, lines)

    print("\n🔄 ASYNC OPERATIONS")
    print("-" * 80)
    analyze_async_operations(hlo_text, lines)

    print("\n📦 DOUBLE BUFFERING DETECTION")
    print("-" * 80)
    detect_double_buffering(hlo_text, lines)

    print("\n🎯 OPTIMIZATION CHECKLIST")
    print("-" * 80)
    optimization_checklist(hlo_text, lines)


def print_overall_stats(hlo_text, lines):
    """Print overall statistics."""
    total_lines = len(lines)
    total_ops = hlo_text.count('\n  %')  # Rough count of operations

    print(f"Total lines: {total_lines:,}")
    print(f"Estimated ops: {total_ops:,}")

    # Count major operation types
    op_counts = defaultdict(int)
    for line in lines:
        if '=' in line and '%' in line:
            # Extract operation type
            match = re.search(r'=\s+\w+\[', line)
            if match:
                op = match.group(0).split('[')[0].strip('= ')
                op_counts[op] += 1

    if op_counts:
        print(f"\nTop operations:")
        for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {op}: {count}")


def analyze_memory_hierarchy(hlo_text, lines):
    """Analyze memory space usage."""

    # Look for memory space annotations
    memory_spaces = {
        'HBM': hlo_text.count('memory_space=0') + hlo_text.count('S(0)'),
        'VMEM': hlo_text.count('memory_space=1') + hlo_text.count('S(1)'),
        'SMEM': hlo_text.count('memory_space=2') + hlo_text.count('S(2)'),
    }

    print("Memory space usage:")
    for space, count in memory_spaces.items():
        if count > 0:
            print(f"  ✓ {space}: {count} references")
            if space == 'SMEM':
                print(f"    → SMEM detected! (v3 optimization)")
        else:
            print(f"  ✗ {space}: Not used")

    # Look for memory allocation patterns
    vmem_allocs = []
    smem_allocs = []

    for line in lines:
        if 'allocate' in line.lower() or 'scratch' in line.lower():
            # Try to extract size
            size_match = re.search(r'\[(\d+(?:,\d+)*)\]', line)
            if size_match:
                dims = size_match.group(1)
                if 'S(1)' in line or 'VMEM' in line:
                    vmem_allocs.append(dims)
                elif 'S(2)' in line or 'SMEM' in line:
                    smem_allocs.append(dims)

    if vmem_allocs:
        print(f"\nVMEM allocations detected: {len(vmem_allocs)}")
        for alloc in vmem_allocs[:5]:  # Show first 5
            print(f"  - [{alloc}]")

    if smem_allocs:
        print(f"\nSMEM allocations detected: {len(smem_allocs)}")
        print(f"  → V3 optimization confirmed!")
        for alloc in smem_allocs[:5]:
            print(f"  - [{alloc}]")


def analyze_computation(hlo_text, lines):
    """Analyze computation patterns."""

    # Look for dot operations (matmuls)
    dot_ops = []
    for i, line in enumerate(lines):
        if 'dot(' in line or 'dot-general' in line:
            # Check for fp8
            is_fp8 = 'f8e4m3' in line or 'f8e5m2' in line
            is_fp32 = 'f32' in line
            is_bf16 = 'bf16' in line

            dtype = 'fp8' if is_fp8 else ('fp32' if is_fp32 else ('bf16' if is_bf16 else 'unknown'))
            dot_ops.append((i, dtype, line.strip()))

    print(f"Matrix multiplication operations: {len(dot_ops)}")

    if dot_ops:
        # Categorize by dtype
        dtype_counts = defaultdict(int)
        for _, dtype, _ in dot_ops:
            dtype_counts[dtype] += 1

        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} operations")
            if dtype == 'fp8':
                print(f"    → Native fp8 MXU usage ✓")
            elif dtype == 'fp32':
                print(f"    ⚠ Using fp32 (should be fp8 for performance)")

        # Show a sample
        print(f"\nSample matmul operations:")
        for i, dtype, line in dot_ops[:3]:
            print(f"  Line {i}: [{dtype}] {line[:100]}...")

    # Look for scale operations
    scale_ops = 0
    for line in lines:
        if 'multiply' in line and ('scale' in line.lower() or 'f32' in line):
            scale_ops += 1

    if scale_ops > 0:
        print(f"\nScale/multiply operations: {scale_ops}")
        print(f"  → Likely post-matmul scaling")


def analyze_async_operations(hlo_text, lines):
    """Analyze async copy and DMA operations."""

    # Look for async patterns
    async_patterns = {
        'async-copy-start': 'Async copy initiation',
        'async-copy-done': 'Async copy completion',
        'async-start': 'Async operation start',
        'async-done': 'Async operation completion',
        'copy-start': 'DMA copy start',
        'copy-done': 'DMA copy completion',
    }

    found_async = False
    for pattern, description in async_patterns.items():
        count = hlo_text.count(pattern)
        if count > 0:
            print(f"✓ {description}: {count} occurrences")
            found_async = True

    if found_async:
        print(f"\n→ Async DMA detected! (v2/v3 optimizations)")

        # Try to identify async pairs
        starts = hlo_text.count('async-copy-start') + hlo_text.count('async-start')
        dones = hlo_text.count('async-copy-done') + hlo_text.count('async-done')

        if starts > 0 and dones > 0:
            print(f"  Async pairs: {starts} starts, {dones} completions")
            if abs(starts - dones) <= 1:  # Allow for off-by-one
                print(f"  → Balanced async operations ✓")
            else:
                print(f"  ⚠ Imbalanced async operations (may indicate issue)")
    else:
        print("✗ No async operations found")
        print("  → Using synchronous copies (v1 automatic double buffering)")


def detect_double_buffering(hlo_text, lines):
    """Detect double buffering patterns."""

    # Look for buffer dimension of 2 (x2 pattern)
    double_buffer_patterns = []

    for i, line in enumerate(lines):
        # Look for shapes with leading dimension of 2
        if 'f32[2,' in line or 'bf16[2,' in line or 'f8e4m3fn[2,' in line:
            double_buffer_patterns.append((i, line.strip()))

    if double_buffer_patterns:
        print(f"✓ Double buffering detected: {len(double_buffer_patterns)} instances")
        print(f"  → Explicit double buffering (v2/v3)")

        # Show samples
        print(f"\nSample double-buffered arrays:")
        for i, line in double_buffer_patterns[:5]:
            print(f"  Line {i}: {line[:80]}...")
    else:
        print("✗ No explicit double buffering found")
        print("  → May be using automatic compiler double buffering (v1)")

    # Look for collective-permute (compiler prefetching)
    collective_count = hlo_text.count('collective-permute')
    if collective_count > 0:
        print(f"\n✓ Collective-permute operations: {collective_count}")
        print(f"  → Compiler-managed prefetching (v1 automatic)")


def optimization_checklist(hlo_text, lines):
    """Generate optimization checklist."""

    checks = []

    # 1. Native fp8 operations
    has_fp8 = 'f8e4m3' in hlo_text or 'f8e5m2' in hlo_text
    checks.append(("Native fp8 matmuls", has_fp8, "Using hardware MXU"))

    # 2. Async operations
    has_async = any(p in hlo_text for p in ['async-copy', 'async-start', 'async-done'])
    checks.append(("Async DMA", has_async, "Manual prefetching (v2/v3)"))

    # 3. Double buffering
    has_double_buf = 'f32[2,' in hlo_text or 'bf16[2,' in hlo_text or 'f8e4m3fn[2,' in hlo_text
    checks.append(("Explicit double buffering", has_double_buf, "2× VMEM arrays (v2/v3)"))

    # 4. SMEM usage
    has_smem = 'S(2)' in hlo_text or 'memory_space=2' in hlo_text or 'SMEM' in hlo_text
    checks.append(("SMEM for scales", has_smem, "Faster memory (v3)"))

    # 5. Prefetching
    has_prefetch = 'collective-permute' in hlo_text or has_async
    checks.append(("Prefetching", has_prefetch, "Overlap compute/memory"))

    # Print checklist
    for check_name, passed, description in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}: {description}")
        if not passed and check_name == "Native fp8 matmuls":
            print(f"  ⚠ WARNING: Not using fp8 - performance will be poor!")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_hlo.py <hlo_file1> [hlo_file2] ...")
        print("\nExample:")
        print("  python analyze_hlo.py compilation_output/*.txt")
        sys.exit(1)

    hlo_files = [Path(f) for f in sys.argv[1:]]

    print("="*80)
    print("HLO Deep Analysis Tool")
    print("="*80)

    for hlo_file in hlo_files:
        if not hlo_file.exists():
            print(f"✗ File not found: {hlo_file}")
            continue

        analyze_hlo_file(hlo_file)

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
