# Kernel Version Comparison

This document compares the three versions of the fp8 2D quantized matmul kernel, explaining when to use each and their performance trade-offs.

## Quick Summary

| Version | Approach | Complexity | VMEM Usage | When to Use |
|---------|----------|------------|------------|-------------|
| **v1** | Auto double buffering | Simple | Medium | Default, baseline |
| **v2** | Manual async DMA | Complex | Medium-High | Fine-grained control |
| **v3** | SMEM for scales | Complex | Low-Medium | Small scales, VMEM constrained |

## Version Details

### V1: Automatic Double Buffering (Baseline)

**Implementation:**
- Uses `PrefetchScalarGridSpec` for automatic compiler-managed double buffering
- Compiler handles all buffer swapping and synchronization
- Clean, straightforward code

**Optimizations:**
- Automatic prefetching of next iteration while computing current
- Compiler decides optimal prefetch timing
- No manual buffer management required

**VMEM Layout:**
```
Inputs (auto double-buffered by compiler):
- x: quant_block_size × quant_block_size
- w_q: quant_block_size × quant_block_size
- w_scale: 1 × 1
- x_abs_max: 1 × 1

Scratch (single buffered):
- acc: quant_block_size × quant_block_size
- x_q (optional): quant_block_size × quant_block_size
- x_scale (optional): 1 × 1
```

**Pros:**
- Simple, clean code
- Compiler optimizations automatically applied
- Good baseline performance
- Easy to maintain and debug

**Cons:**
- Less control over prefetch timing
- Relies on compiler heuristics
- May leave performance on table if compiler is conservative

**When to use:**
- Default choice for most cases
- When code simplicity matters
- As baseline for benchmarking other versions

---

### V2: Manual Async DMA with Semaphores

**Implementation:**
- Explicit `pltpu.make_async_copy()` with `.start()` and `.wait()`
- Manual semaphore synchronization (5 semaphores per buffer)
- Double-buffered VMEM arrays with `x2` naming pattern
- Fine-grained control over prefetch timing

**Optimizations:**
- Prefetch iteration i+1 while computing iteration i
- Explicit buffer swapping with semaphores
- Can tune prefetch timing precisely

**VMEM Layout:**
```
Data (manually double-buffered):
- x_x2: 2 × quant_block_size × quant_block_size
- w_q_x2: 2 × quant_block_size × quant_block_size
- w_scale_x2: 2 × 1 × 1
- x_abs_max_x2: 2 × 1 × 1
- out_x2: 2 × quant_block_size × quant_block_size

Scratch (single buffered):
- acc: quant_block_size × quant_block_size
- x_q (optional): quant_block_size × quant_block_size
- x_scale (optional): 1 × 1

Synchronization:
- sems: 2 × 5 (2 buffers, 5 semaphores each)
  - sem 0: x transfer
  - sem 1: w_q transfer
  - sem 2: w_scale transfer
  - sem 3: x_abs_max transfer
  - sem 4: out transfer
```

**Pros:**
- Fine-grained control over prefetch timing
- Can optimize for specific access patterns
- May outperform v1 if compiler auto-prefetch is suboptimal
- Explicit buffer management aids debugging

**Cons:**
- More complex code (~2× LOC vs v1)
- Manual buffer management error-prone
- More VMEM for semaphores
- Requires TPU performance expertise to tune

**When to use:**
- Profiling shows compiler auto-prefetch leaving gaps
- Need precise control over memory transfers
- Willing to trade complexity for potential speedup
- Have TPU performance expertise

**Expected speedup:** 1.1-1.3× vs v1 (if compiler was suboptimal)

---

### V3: SMEM for Scales + Async DMA

**Implementation:**
- SMEM (Scalar Memory) for scale factors
- Async DMA for data blocks (like v2)
- Scales in faster SMEM, data in VMEM

**Optimizations:**
- ~2-3× faster scale access (SMEM vs VMEM)
- Reduced VMEM pressure and contention
- Double-buffered scales in SMEM

**Memory Layout:**
```
VMEM (data blocks, double buffered):
- x_x2: 2 × quant_block_size × quant_block_size
- w_q_x2: 2 × quant_block_size × quant_block_size
- out_x2: 2 × quant_block_size × quant_block_size
- acc: quant_block_size × quant_block_size
- x_q (optional): quant_block_size × quant_block_size
- x_scale_vmem (optional): 1 × 1

SMEM (scales, double buffered):
- w_scale_x2: 2 × 1 × 1
- x_abs_max_x2: 2 × 1 × 1
- x_scale: 1 × 1

Synchronization:
- sems: 2 × 5 (same as v2)
```

**SMEM Benefits:**
- Faster access: ~50-100 cycles vs ~150-300 for VMEM
- Lower latency for small data
- Reduces VMEM bandwidth contention

**SMEM Constraints:**
- Limited capacity: ~32KB total on Ironwood
- Only worth it for small data (scales are tiny)
- Aligned blocks only (scales = 1×1 per kernel)

**Pros:**
- Fastest scale access
- Lowest VMEM pressure
- Best when scales accessed frequently
- Can enable larger data blocks

**Cons:**
- Most complex code
- SMEM capacity limited (not for large blocks)
- Only beneficial for small scales
- Requires understanding TPU memory hierarchy

**When to use:**
- Aligned blocks (scales are 1×1)
- Scales accessed multiple times per kernel
- VMEM bandwidth is bottleneck
- Need maximum performance

**Expected speedup:** 1.05-1.15× vs v2 (for aligned blocks with frequent scale access)

---

## Benchmarking Guide

To choose the best version for your workload:

1. **Start with v1** (baseline)
   ```python
   from tpu_inference.kernels.fp8_quantized_matmul_2d import fp8_quantized_matmul_2d_kernel
   # Run and measure performance
   ```

2. **Try v2** if you suspect prefetch issues
   ```python
   from tpu_inference.kernels.fp8_quantized_matmul_2d.v2.kernel import fp8_quantized_matmul_2d_kernel
   # Compare performance
   ```

3. **Try v3** for aligned blocks with VMEM pressure
   ```python
   from tpu_inference.kernels.fp8_quantized_matmul_2d.v3.kernel import fp8_quantized_matmul_2d_kernel
   # Compare performance
   ```

### What to Measure

- **Throughput**: TFLOPS/second
- **Memory bandwidth**: GB/s
- **MXU utilization**: % of peak
- **VMEM usage**: Bytes (check for OOM)

### Expected Results

For **aligned blocks** (512×512 quant blocks, 4096×4096 matrix):
- v1: Baseline (100%)
- v2: 110-130% of v1 (if prefetch was suboptimal)
- v3: 115-145% of v1 (if VMEM was bottleneck)

For **large blocks** (only v1 supported currently):
- TBD (need to implement v2/v3 support)

## Recommendations

**Use v1 if:**
- ✅ You want simple, maintainable code
- ✅ You're establishing baseline performance
- ✅ Compiler auto-prefetch works well for your workload

**Use v2 if:**
- ✅ Profile shows prefetch gaps/bubbles
- ✅ You need fine-grained memory control
- ✅ You have TPU performance expertise
- ✅ Willing to manage complexity

**Use v3 if:**
- ✅ Using aligned blocks (not large blocks)
- ✅ VMEM bandwidth is bottleneck
- ✅ Scales accessed frequently
- ✅ Want maximum performance for small scales

**For production:**
- Benchmark all versions on your actual workload
- Choose based on measured performance, not theory
- Consider maintenance cost vs performance gain
- Document which version you use and why

## Future Work

- [ ] Implement v2 for large blocks (sub-block iteration with async DMA)
- [ ] Implement v3 for large blocks (SMEM for scale arrays)
- [ ] Add num_scalar_prefetch optimization
- [ ] Benchmark all versions on Ironwood TPU
- [ ] Auto-tuning to select best version per workload
