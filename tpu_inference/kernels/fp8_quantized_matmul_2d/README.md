# 2D FP8 Quantized Matmul Kernel for Ironwood TPU

This kernel implements high-performance matrix multiplication with 2D block-wise quantization for Ironwood TPU (64MB VMEM, 256×256 MXU with native fp8 support).

## Key Features

- **2D Block-wise Quantization**: Each block of size (quant_block_size, quant_block_size) has its own scale factor
- **Native FP8 Compute**: Uses TPU's hardware fp8×fp8 → fp32 accumulation for maximum performance
- **Flexible Block Sizes**: Supports 128×128, 256×256, and 512×512 quantization blocks
- **Two Execution Modes**: Aligned blocks (simple) and large blocks (amortized overhead)

## Architecture Overview

### 2D Quantization

Unlike 1D (per-channel) quantization, 2D quantization uses a grid of scale factors:

```
Weight matrix (n_out × n_in):
┌─────────┬─────────┬─────────┐
│ Block   │ Block   │ Block   │  Each block: quant_block_size × quant_block_size
│ (0,0)   │ (0,1)   │ (0,2)   │  Each block has its own scale factor
│ scale_00│ scale_01│ scale_02│
├─────────┼─────────┼─────────┤
│ Block   │ Block   │ Block   │
│ (1,0)   │ (1,1)   │ (1,2)   │
│ scale_10│ scale_11│ scale_12│
└─────────┴─────────┴─────────┘

Scales shape: (n_out // quant_block_size, n_in // quant_block_size)
```

### Two Execution Modes

#### 1. Aligned Blocks Mode (Default)

**Simple and correct approach:**
- Kernel blocks = quantization blocks
- Each kernel processes exactly ONE quantization block
- One scale per kernel (scalar multiplication)
- Uses native fp8×fp8 matmul

**Example with 512×512 quant blocks:**
```
Matrix: 4096×4096
Quant blocks: 8×8 grid of 512×512 blocks
Kernel blocks: 512×512 (aligned with quant blocks)
Grid size: 8×8 = 64 kernels
Each kernel: fp8×fp8 matmul + scalar scale
```

**Advantages:**
- Simple implementation, easy to verify
- Native fp8 hardware acceleration
- Minimal scaling overhead
- Good starting point for tuning

#### 2. Large Blocks Mode (Optimized)

**Amortize kernel overhead:**
- Kernel blocks > quantization blocks
- Each kernel processes MULTIPLE quantization blocks
- Sub-block iteration with native fp8×fp8 matmuls
- More complex but better for large matrices

**Example with 512×512 quant blocks, 2048×2048 kernel blocks:**
```
Matrix: 4096×4096
Quant blocks: 8×8 grid of 512×512 blocks
Kernel blocks: 2048×2048 (4×4 quant blocks per kernel)
Grid size: 2×2 = 4 kernels
Each kernel: 4×4×4 = 64 sub-matmuls with fp8×fp8
```

**Advantages:**
- Fewer kernel invocations
- Amortizes kernel launch overhead
- Better VMEM utilization
- Potentially higher throughput

**Trade-offs:**
- More complex implementation
- More code in kernel body
- Requires careful sub-block indexing

## Performance Optimization

### Automatic Double Buffering (CRITICAL)

**This kernel uses double buffering to maximize TPU performance by overlapping compute with memory transfers.**

How it works:
- **PrefetchScalarGridSpec**: Enables automatic prefetching of next iteration's data
- **VMEM Allocation**: Extra VMEM is allocated for double buffering (see `get_vmem_limit` in util.py)
- **Grid Iteration**: When iterating over the `i` dimension (in_block), the compiler:
  1. Loads next iteration's data into second buffer while computing current iteration
  2. Swaps buffers automatically (ping-pong pattern)
  3. Hides memory transfer latency behind computation

Benefits:
- **Near-zero memory latency**: Compute and transfer overlap perfectly
- **Maximum MXU utilization**: TPU cores stay busy, not waiting for data
- **Automatic**: Compiler handles all synchronization, no manual buffer management

Example: For grid=(8, 8, 4) with 512×512 blocks:
- While computing iteration i=1, prefetch data for i=2
- While computing iteration i=2, prefetch data for i=3
- Achieves ~2× speedup vs non-double-buffered version

### Critical Design Decisions

#### ✅ Native FP8 Matmul (What We Do)
```python
# Use native fp8×fp8 matmul with fp32 accumulation
acc = jax.lax.dot_general(x_q, w_q, ..., preferred_element_type=jnp.float32)
acc = acc * w_scale * x_scale
```

Benefits:
- Leverages hardware fp8 MXU
- Maximum throughput on Ironwood
- Minimal scaling overhead

#### ❌ Dequantize-First (What We DON'T Do)
```python
# WRONG: Dequantize before matmul
x_dequant = x_q * x_scale  # fp32
w_dequant = w_q * w_scale  # fp32
acc = jax.lax.dot_general(x_dequant, w_dequant, ...)  # fp32×fp32 - SLOW!
```

Problems:
- Does fp32×fp32 matmul instead of fp8×fp8
- Wastes native fp8 hardware support
- Much slower, defeats the purpose

### TPU Constraints

- **VMEM Limit**: 64MB on Ironwood
- **MXU Size**: 256×256 with native fp8 support
- **Alignment**: Last two dimensions must be divisible by (8×128)
- **Block sizes**: Must be divisible by 128 (and by quant_block_size)

## Usage

### Basic Usage (Aligned Blocks)

```python
from tpu_inference.kernels.fp8_quantized_matmul_2d import (
    fp8_quantized_matmul_2d_kernel,
    quantize_tensor_2d,
)

# Quantize weights with 2D blocks
w_q, w_scale = quantize_tensor_2d(
    w,
    jnp.float8_e4m3fn,
    block_size_m=512,
    block_size_n=512,
)

# Run kernel (will use aligned blocks by default)
output = fp8_quantized_matmul_2d_kernel(
    x,  # [batch_size, n_in]
    w_q,  # [n_out, n_in] in fp8
    w_scale,  # [n_out // 512, n_in // 512]
    x_q_dtype=jnp.float8_e4m3fn,
    quant_block_size=512,
)
```

### Advanced Usage (Large Blocks)

```python
from tpu_inference.kernels.fp8_quantized_matmul_2d.tuned_block_sizes import TunedValue

# Manually specify larger kernel blocks for amortization
tuned_value = TunedValue(
    batch_block_size=2048,  # Process 4×4 grid of 512×512 quant blocks
    out_block_size=2048,
    in_block_size=2048,
    quant_block_size=512,
)

output = fp8_quantized_matmul_2d_kernel(
    x, w_q, w_scale,
    x_q_dtype=jnp.float8_e4m3fn,
    quant_block_size=512,
    tuned_value=tuned_value,  # Use large blocks mode
)
```

## Tuning Guidelines

1. **Start with aligned blocks**: Simple, correct, good baseline performance
2. **Benchmark**: Measure throughput on target workload
3. **Try larger blocks**: If overhead is significant, increase kernel block sizes
4. **Monitor VMEM**: Ensure kernel fits within 64MB VMEM limit
5. **Add to database**: Update `TUNED_BLOCK_SIZES` with best configurations

## File Structure

- `kernel.py` - Main kernel implementations (aligned and large blocks modes)
- `util.py` - Quantization utilities and helper functions
- `tuned_block_sizes.py` - Block size configurations database
- `__init__.py` - Public API exports
- `README.md` - This documentation

## Performance Expectations

For 4096×4096 matrices with 512×512 quantization blocks:

**Aligned Blocks Mode:**
- Grid: 8×8 = 64 kernels
- Each kernel: 512×512 fp8×fp8 matmul
- Simple, predictable performance

**Large Blocks Mode (2048×2048):**
- Grid: 2×2 = 4 kernels
- Each kernel: 64 sub-matmuls (4×4×4)
- Potentially 2-4× faster due to amortization (needs benchmarking)

## Testing

See `tests/kernels/fp8_quantized_matmul_2d_kernel_test.py` for comprehensive tests covering:
- Various matrix sizes
- Different quantization block sizes (128, 256, 512)
- Both quantize activation modes
- Correctness vs reference implementation

## References

- Original 1D quantized matmul: `tpu_inference/kernels/quantized_matmul/`
- Fused MoE sub-block iteration: `tpu_inference/kernels/fused_moe/v1/kernel.py`
- Pallas documentation: https://jax.readthedocs.io/en/latest/pallas/index.html
