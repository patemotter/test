# SPDX-License-Identifier: Apache-2.0
"""Tuned block sizes for 2D fp8 quantized matmul kernel on Ironwood TPU."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TunedValue:
    """Tuned block size configuration for the kernel.

    Attributes:
        batch_block_size: Block size for the batch dimension
        out_block_size: Block size for the output features dimension
        in_block_size: Block size for the input features dimension
        quant_block_size: Block size for 2D quantization (128, 256, or 512)
    """
    batch_block_size: int
    out_block_size: int
    in_block_size: int
    quant_block_size: int


# Ironwood TPU configuration
# - 64MB VMEM
# - 256x256 MXU with native fp8 support
# - Blocks must be divisible by (8x128) for last two dimensions
IRONWOOD_VMEM_LIMIT = 64 * 1024 * 1024  # 64 MB


def get_device_vmem_limit():
    """Return the VMEM limit for Ironwood TPU."""
    return IRONWOOD_VMEM_LIMIT


# Tuned block sizes database
# Key: (n_batch, n_out, n_in, x_q_dtype, w_q_dtype, quant_block_size)
# Value: TunedValue(batch_block_size, out_block_size, in_block_size, quant_block_size)
#
# These are starting configurations optimized for:
# - Ironwood TPU (64MB VMEM, 256x256 MXU)
# - fp8_e4m3fn quantization
# - 2D block quantization with blocks of 128x128, 256x256, or 512x512
# - Divisibility by (8x128) for last two dimensions

# TUNED BLOCK SIZES DATABASE
#
# TWO MODES SUPPORTED:
# 1. ALIGNED BLOCKS (kernel_block = quant_block):
#    - Simpler implementation, one quant block per kernel
#    - Uses native fp8×fp8 matmuls
#    - Good for smaller matrices or initial testing
#
# 2. LARGE BLOCKS (kernel_block > quant_block):
#    - Multiple quant blocks per kernel with sub-block iteration
#    - Amortizes kernel launch overhead
#    - Better for large matrices on TPU
#
# Format: (batch, out, in, x_dtype, w_dtype, quant_block) -> TunedValue(batch_block, out_block, in_block, quant_block)

TUNED_BLOCK_SIZES = {
    # ========== ALIGNED BLOCKS MODE (Default for simplicity) ==========
    # All aligned with 128x128 blocks
    (1024, 1024, 1024, "float8_e4m3fn", "float8_e4m3fn", 128): TunedValue(128, 128, 128, 128),
    (2048, 2048, 2048, "float8_e4m3fn", "float8_e4m3fn", 128): TunedValue(128, 128, 128, 128),
    (4096, 4096, 4096, "float8_e4m3fn", "float8_e4m3fn", 128): TunedValue(128, 128, 128, 128),

    # All aligned with 256x256 blocks
    (1024, 1024, 1024, "float8_e4m3fn", "float8_e4m3fn", 256): TunedValue(256, 256, 256, 256),
    (2048, 2048, 2048, "float8_e4m3fn", "float8_e4m3fn", 256): TunedValue(256, 256, 256, 256),
    (4096, 4096, 4096, "float8_e4m3fn", "float8_e4m3fn", 256): TunedValue(256, 256, 256, 256),

    # All aligned with 512x512 blocks
    (1024, 1024, 1024, "float8_e4m3fn", "float8_e4m3fn", 512): TunedValue(512, 512, 512, 512),
    (2048, 2048, 2048, "float8_e4m3fn", "float8_e4m3fn", 512): TunedValue(512, 512, 512, 512),
    (4096, 4096, 4096, "float8_e4m3fn", "float8_e4m3fn", 512): TunedValue(512, 512, 512, 512),
    (8192, 8192, 8192, "float8_e4m3fn", "float8_e4m3fn", 512): TunedValue(512, 512, 512, 512),

    # Mixed precision: bfloat16 activation with fp8 weights (aligned blocks)
    (1024, 1024, 1024, "bfloat16", "float8_e4m3fn", 128): TunedValue(128, 128, 128, 128),
    (2048, 2048, 2048, "bfloat16", "float8_e4m3fn", 128): TunedValue(128, 128, 128, 128),
    (4096, 4096, 4096, "bfloat16", "float8_e4m3fn", 128): TunedValue(128, 128, 128, 128),

    (1024, 1024, 1024, "bfloat16", "float8_e4m3fn", 256): TunedValue(256, 256, 256, 256),
    (2048, 2048, 2048, "bfloat16", "float8_e4m3fn", 256): TunedValue(256, 256, 256, 256),
    (4096, 4096, 4096, "bfloat16", "float8_e4m3fn", 256): TunedValue(256, 256, 256, 256),

    (1024, 1024, 1024, "bfloat16", "float8_e4m3fn", 512): TunedValue(512, 512, 512, 512),
    (2048, 2048, 2048, "bfloat16", "float8_e4m3fn", 512): TunedValue(512, 512, 512, 512),
    (4096, 4096, 4096, "bfloat16", "float8_e4m3fn", 512): TunedValue(512, 512, 512, 512),
}

# Example configurations for LARGE BLOCKS mode (to be added as tuning progresses):
# These amortize kernel overhead by processing multiple quant blocks per kernel
TUNED_BLOCK_SIZES_LARGE = {
    # Example: 512x512 quant blocks with 2048x2048 kernel blocks
    # Each kernel processes a 4×4 grid of quantization blocks
    # (4096, 4096, 4096, "float8_e4m3fn", "float8_e4m3fn", 512): TunedValue(2048, 2048, 2048, 512),
    # (8192, 8192, 8192, "float8_e4m3fn", "float8_e4m3fn", 512): TunedValue(4096, 4096, 4096, 512),

    # Example: 256x256 quant blocks with 1024x1024 kernel blocks
    # (2048, 2048, 2048, "float8_e4m3fn", "float8_e4m3fn", 256): TunedValue(1024, 1024, 1024, 256),
    # (4096, 4096, 4096, "float8_e4m3fn", "float8_e4m3fn", 256): TunedValue(2048, 2048, 2048, 256),
}


def get_tuned_block_sizes(
    n_batch: int,
    n_out: int,
    n_in: int,
    x_q_dtype: str,
    w_q_dtype: str,
    quant_block_size: int,
) -> TunedValue:
    """Get tuned block sizes for the given configuration.

    Args:
        n_batch: Batch size
        n_out: Output feature dimension
        n_in: Input feature dimension
        x_q_dtype: Activation quantization dtype name
        w_q_dtype: Weight quantization dtype name
        quant_block_size: 2D quantization block size (128, 256, or 512)

    Returns:
        TunedValue with optimized block sizes
    """
    key = (n_batch, n_out, n_in, x_q_dtype, w_q_dtype, quant_block_size)

    # Try exact match first
    if key in TUNED_BLOCK_SIZES:
        return TUNED_BLOCK_SIZES[key]

    # Fall back to simple default: align kernel blocks with quantization blocks
    # This is the simplest and most straightforward approach:
    # - Each kernel invocation processes exactly one quantization block
    # - No sub-block iteration needed
    # - Simple scale application (one scale per kernel)
    # - Easy to verify correctness
    # - Can optimize later if benchmarks show benefit from larger blocks

    # For aligned blocks, kernel block size = quant block size
    # This already satisfies all TPU constraints since quant_block_size is validated to be
    # a multiple of 128, and we ensure batch dimension is multiple of 8
    batch_block_size = quant_block_size
    out_block_size = quant_block_size
    in_block_size = quant_block_size

    # Verify all constraints are met
    assert batch_block_size % quant_block_size == 0
    assert out_block_size % quant_block_size == 0
    assert in_block_size % quant_block_size == 0
    assert batch_block_size % 8 == 0  # quant_block_size is always >= 128, so divisible by 8
    assert out_block_size % 128 == 0
    assert in_block_size % 128 == 0

    return TunedValue(
        batch_block_size=batch_block_size,
        out_block_size=out_block_size,
        in_block_size=in_block_size,
        quant_block_size=quant_block_size,
    )
