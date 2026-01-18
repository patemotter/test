# HLO/LLO Inspection Guide for FP8 2D Quantized Matmul Kernels

This guide explains how to inspect and analyze the compilation output of the three kernel versions to verify optimizations are working correctly.

## Quick Start

### Step 1: Generate HLO Dumps

```bash
# On CPU (for initial inspection)
python scripts/inspect_fp8_2d_kernel_compilation.py

# This will create compilation_output/ with HLO dumps
```

### Step 2: Analyze HLO

```bash
# Analyze all versions
python scripts/analyze_hlo.py compilation_output/*_hlo.txt

# Or analyze specific version
python scripts/analyze_hlo.py compilation_output/v2_manual_async_dma_hlo.txt
```

### Step 3: Compare Results

Look at the optimization checklist output for each version and compare.

---

## What to Look For in HLO

### 1. Native FP8 Operations (ALL VERSIONS)

**Critical:** All versions MUST use native fp8 matmuls for good performance.

**Look for:**
```hlo
%dot = f8e4m3fn[...] dot(
  f8e4m3fn[...] %lhs,
  f8e4m3fn[...] %rhs
)
```

**Warning signs:**
```hlo
%dot = f32[...] dot(f32[...] %lhs, f32[...] %rhs)  # BAD - using fp32!
```

**What it means:**
- ✅ `f8e4m3fn` in dot operations → Using hardware fp8 MXU
- ❌ `f32` or `bf16` in dot operations → NOT using fp8 (performance loss)

---

### 2. Double Buffering (V1 - Automatic)

**V1 should show compiler-managed prefetching.**

**Look for:**
```hlo
%collective-permute = ...  # Compiler prefetch mechanism
```

Or look for implicit double buffering in the while loop structure:
```hlo
%while.body {
  %param = (...) parameter(0)
  %async-copy-start = ...  # Implicit prefetch by compiler
  ...
  %async-copy-done = ...
}
```

**What it means:**
- ✅ `collective-permute` → Compiler is auto-prefetching
- ✅ `async-copy-*` in while loop → Compiler inserted async ops
- ❌ No async operations → Double buffering may not be happening

---

### 3. Explicit Async DMA (V2)

**V2 should show manual async copy operations.**

**Look for:**
```hlo
%async-copy-start.1 = (f8e4m3fn[...], ...) async-copy-start(...)
...
%async-copy-done.1 = f8e4m3fn[...] async-copy-done(%async-copy-start.1)
```

**Count pairs:**
- Should see equal numbers of `async-copy-start` and `async-copy-done`
- Multiple pairs indicate double buffering of different arrays

**Look for buffer ID pattern (0 and 1):**
```hlo
dynamic-slice(..., constant(0), ...)  # Buffer 0
dynamic-slice(..., constant(1), ...)  # Buffer 1
```

**What it means:**
- ✅ Many `async-copy-start/done` pairs → Explicit async DMA working
- ✅ Buffer indexing (0/1) → Double buffering confirmed
- ❌ No async ops → Manual DMA not working (code issue)

---

### 4. SMEM Usage (V3)

**V3 should show SMEM memory space for scales.**

**Look for:**
```hlo
%scale = f32[2,1,1]{2,1,0:T(2)} ...  # T(2) = SMEM
# Or
memory_space=2  # SMEM memory space ID
```

**Memory space IDs:**
- `S(0)` or `memory_space=0` = HBM (slow)
- `S(1)` or `memory_space=1` = VMEM (fast)
- `S(2)` or `memory_space=2` = SMEM (fastest for small data)

**What it means:**
- ✅ `S(2)` or `memory_space=2` for scales → SMEM optimization working
- ❌ Scales in `S(1)` (VMEM) → SMEM not being used (check code)
- ⚠️ Data blocks in `S(2)` → Likely won't fit (SMEM too small)

---

### 5. Double Buffering Pattern (V2/V3)

**Explicit double buffering shows leading dimension of 2.**

**Look for:**
```hlo
%x_x2_vmem = f8e4m3fn[2,512,512]{...}  # 2 = double buffer
%w_q_x2_vmem = f8e4m3fn[2,512,512]{...}
%scales_x2_smem = f32[2,1,1]{...}
```

**What it means:**
- ✅ Shape `[2,...]` → Explicit 2-buffer allocation
- ✅ Multiple arrays with `[2,...]` → All inputs double buffered
- Each buffer used alternately (ping-pong pattern)

---

## Detailed Analysis Workflow

### For V1 (Auto Double Buffering):

1. **Check for native fp8 matmuls:**
   ```bash
   grep -c "f8e4m3fn.*dot" compilation_output/v1_auto_double_buffer_hlo.txt
   ```
   Should be > 0

2. **Check for automatic prefetching:**
   ```bash
   grep -c "collective-permute\|async-copy" compilation_output/v1_auto_double_buffer_hlo.txt
   ```
   Should be > 0 (compiler inserted)

3. **Look at while loop structure:**
   ```bash
   grep -A 20 "while.body" compilation_output/v1_auto_double_buffer_hlo.txt
   ```
   Should show grid iteration with implicit prefetching

**Expected:** Compiler-inserted async operations, no explicit buffer ID indexing

---

### For V2 (Manual Async DMA):

1. **Count async operations:**
   ```bash
   grep -c "async-copy-start" compilation_output/v2_manual_async_dma_hlo.txt
   grep -c "async-copy-done" compilation_output/v2_manual_async_dma_hlo.txt
   ```
   Numbers should be equal and > 0

2. **Check for explicit buffer indexing:**
   ```bash
   grep "dynamic-slice.*constant([01])" compilation_output/v2_manual_async_dma_hlo.txt
   ```
   Should see both buffer 0 and buffer 1

3. **Verify double-buffered arrays:**
   ```bash
   grep "\[2," compilation_output/v2_manual_async_dma_hlo.txt | head -10
   ```
   Should see arrays with leading dimension of 2

**Expected:** Many async ops, explicit buffer indexing, `[2,...]` shaped arrays

---

### For V3 (SMEM Scales):

1. **Check SMEM usage:**
   ```bash
   grep -c "S(2)\|memory_space=2" compilation_output/v3_smem_scales_hlo.txt
   ```
   Should be > 0

2. **Verify scales are in SMEM:**
   ```bash
   grep "scale.*S(2)\|scale.*memory_space=2" compilation_output/v3_smem_scales_hlo.txt
   ```
   Should show scale arrays in SMEM

3. **Check data blocks in VMEM:**
   ```bash
   grep "x_x2.*S(1)\|w_q_x2.*S(1)" compilation_output/v3_smem_scales_hlo.txt
   ```
   Large arrays should be in VMEM (S(1))

**Expected:** Small scales in SMEM (S(2)), large data in VMEM (S(1)), async ops present

---

## Performance Indicators

### Good Signs ✅

- **Native fp8 matmuls**: `f8e4m3fn` in dot operations
- **Async operations**: Multiple `async-copy-start/done` pairs
- **Memory hierarchy**: HBM → VMEM → SMEM usage for appropriate data sizes
- **Double buffering**: Leading dimension of 2 or collective-permute
- **Balanced operations**: Equal starts and completions

### Warning Signs ⚠️

- **fp32 matmuls**: Using `f32` instead of `f8e4m3fn` (major perf loss)
- **No async ops**: Missing prefetching (memory bound)
- **Imbalanced async**: Unequal starts/completions (possible deadlock)
- **Wrong memory spaces**: Scales in VMEM instead of SMEM (v3)
- **Single buffering**: No `[2,...]` shapes and no async ops (v2/v3 broken)

### Red Flags 🚨

- **No fp8 anywhere**: Kernel completely broken, using fp32
- **OOM errors**: Allocated more memory than available
- **No computation**: Missing actual matmul operations

---

## Comparing Versions

Create a comparison table:

| Metric | V1 | V2 | V3 | Winner |
|--------|----|----|----|----|
| FP8 matmuls | ✓ | ✓ | ✓ | Tie |
| Async DMA | Auto | Manual | Manual | ? |
| SMEM scales | ✗ | ✗ | ✓ | V3 |
| Code complexity | Low | High | Highest | V1 |
| Buffer control | Auto | Manual | Manual | V2/V3 |

**Decision factors:**
- If all show native fp8 matmuls → Good baseline
- If V1 has async ops → Compiler did the work for us
- If V2 has more/better async → Manual control working
- If V3 shows SMEM → Extra optimization confirmed

---

## Getting LLO (TPU-specific)

On actual TPU hardware:

```python
# Set environment to get detailed output
import os
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/xla_dump'

# Compile and run kernel
output = kernel(x, w_q, w_scale)

# Check /tmp/xla_dump/ for:
# - *.hlo - High-level optimizer output
# - *.ll - LLVM IR
# - *.ptx / *.asm - Assembly code (TPU-specific)
```

**What to look for in LLO:**
- TPU-specific instructions (MXU operations)
- DMA transfer instructions
- Memory barrier instructions
- SMEM load/store instructions

---

## Troubleshooting

### "No fp8 operations found"

**Cause:** Kernel is using fp32 fallback

**Fix:**
1. Check `x_q_dtype=jnp.float8_e4m3fn` is passed
2. Verify weights are quantized to fp8
3. Check TPU backend supports fp8

### "No async operations (V2/V3)"

**Cause:** Manual async DMA not working

**Fix:**
1. Verify `pltpu.make_async_copy()` calls in kernel
2. Check semaphore setup is correct
3. Ensure memory spaces are set to `ANY` for HBM

### "No SMEM usage (V3)"

**Cause:** Scales not being placed in SMEM

**Fix:**
1. Check `pltpu.SMEM(...)` in scratch_shapes
2. Verify scales are small enough for SMEM (~32KB limit)
3. Ensure data is accessed from SMEM in kernel body

---

## Useful Commands

```bash
# Quick grep patterns
alias check-fp8="grep -c 'f8e4m3fn.*dot'"
alias check-async="grep -c 'async-copy-start'"
alias check-smem="grep -c 'S(2)'"
alias check-double-buf="grep -c '\[2,'"

# Use them
check-fp8 compilation_output/v1_*.txt
check-async compilation_output/v2_*.txt
check-smem compilation_output/v3_*.txt
check-double-buf compilation_output/*_hlo.txt
```

---

## Next Steps After HLO Analysis

1. ✅ **Verify correctness**: All versions produce same output
2. ✅ **Confirm optimizations**: HLO shows expected patterns
3. ⏭️ **Run on TPU**: Get actual performance numbers
4. ⏭️ **Profile**: Use TPU profiler to see where time is spent
5. ⏭️ **Tune**: Adjust block sizes based on profiling data
6. ⏭️ **Benchmark**: Compare throughput of all three versions

Remember: HLO analysis is about **correctness**, not performance. The real test is benchmarking on actual hardware!
