# SPDX-License-Identifier: Apache-2.0
"""2D fp8 quantized matmul kernel for Ironwood TPU."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.fp8_quantized_matmul_2d.v1 import util
from tpu_inference.kernels.fp8_quantized_matmul_2d.v1.tuned_block_sizes import (
    TunedValue,
    get_device_vmem_limit,
    get_tuned_block_sizes,
)
from tpu_inference.kernels.fp8_quantized_matmul_2d.v1.util import (
    get_kernel_name,
    next_multiple,
)

quantize_tensor_2d = util.quantize_tensor_2d
cdiv = pl.cdiv


def matmul_kernel_2d_large_blocks(
    x_ref: jax.Array,  # (batch_block_size, in_block_size)
    w_q_ref: jax.Array,  # (out_block_size, in_block_size)
    w_scale_ref: jax.Array,  # (n_w_quant_blocks_m, n_w_quant_blocks_n)
    x_abs_max_ref: jax.Array,  # (n_x_quant_blocks_m, n_x_quant_blocks_n)
    out_ref: jax.Array,  # (batch_block_size, out_block_size)
    acc_scratch: jax.Array,  # (batch_block_size, out_block_size)
    x_q_scratch: jax.Array,  # (batch_block_size, in_block_size)
    x_scale_scratch: jax.Array,  # (n_x_quant_blocks_m, n_x_quant_blocks_n)
    *,
    x_q_dtype: jnp.dtype,
    quant_block_size: int,
    batch_block_size: int,
    out_block_size: int,
    in_block_size: int,
    save_acc: bool,
    save_x_q: bool,
):
    """Pallas kernel for 2D quantized matmul with LARGE blocks (kernel blocks > quant blocks).

    OPTIMIZED FOR AMORTIZING KERNEL OVERHEAD:
    - Kernel blocks are MULTIPLES of quantization blocks
    - Each kernel processes MULTIPLE quantization blocks via sub-block iteration
    - Uses native fp8×fp8 matmuls for each sub-block
    - Accumulates partial results with proper 2D scaling
    - More complex but amortizes kernel launch overhead

    For example, with 512×512 quant blocks and 2048×2048 kernel blocks:
    - Each kernel processes a 4×4 grid of quantization blocks
    - 16 sub-matmuls per kernel, each using native fp8 hardware
    - Scales applied per sub-block during accumulation
    """
    # Compile-time assertions for TPU constraints
    assert batch_block_size % quant_block_size == 0
    assert out_block_size % quant_block_size == 0
    assert in_block_size % quant_block_size == 0
    assert quant_block_size % 128 == 0
    assert batch_block_size % 8 == 0
    assert out_block_size % 128 == 0
    assert in_block_size % 128 == 0

    out_idx, in_idx = pl.program_id(1), pl.program_id(2)
    n_in = pl.num_programs(2)
    x_ref_dtype = x_ref.dtype

    quantize_activation = x_q_dtype != x_ref_dtype

    # Initialize conditional logic
    if save_x_q:
        assert quantize_activation
        assert x_q_scratch is not None
        assert x_scale_scratch is not None
        quant = out_idx == 0
    else:
        assert x_q_scratch is None
        assert x_scale_scratch is None
        quant = quantize_activation

    if save_acc:
        assert acc_scratch is not None
        is_first_step = in_idx == 0
        is_last_step = in_idx == (n_in - 1)
    else:
        assert acc_scratch is None
        is_first_step = True
        is_last_step = True

    acc_dtype = jnp.float32
    if quantize_activation and jnp.issubdtype(w_q_ref.dtype, jnp.integer):
        acc_dtype = jnp.int32

    # Compute number of quantization sub-blocks within kernel block
    n_batch_quant_blocks = batch_block_size // quant_block_size
    n_out_quant_blocks = out_block_size // quant_block_size
    n_in_quant_blocks = in_block_size // quant_block_size

    # Start of actual computation logic
    # PERFORMANCE KEY: Iterate sub-blocks with native fp8×fp8 matmuls
    # Similar approach to fused_moe's sub-channel quantization

    if quantize_activation:
        # Quantize or reuse - use jax.lax.cond for potentially-traced quant
        def quantize_branch():
            x_q, x_scale = util.quantize_array_2d(
                x_ref[...],
                x_abs_max_ref[...],
                x_q_dtype,
                quant_block_size,
                quant_block_size,
            )
            if save_x_q:
                x_q_scratch[...] = x_q
                x_scale_scratch[...] = x_scale
            return x_q, x_scale

        def reuse_branch():
            x_q = x_q_scratch[...]
            # Use jax.lax.cond for conditional scale loading
            x_scale = jax.lax.cond(
                is_last_step,
                lambda: x_scale_scratch[...],
                lambda: jnp.zeros((n_batch_quant_blocks, n_in_quant_blocks), dtype=jnp.float32),
            )
            return x_q, x_scale

        x_q_tmp, x_scale_tmp = jax.lax.cond(quant, quantize_branch, reuse_branch)

    # Initialize accumulator - use jax.lax.select
    acc = jax.lax.select(
        is_first_step,
        jnp.zeros((batch_block_size, out_block_size), dtype=jnp.float32),
        acc_scratch[...].astype(jnp.float32)
    )

    # Iterate over quantization sub-blocks in the K dimension
    for k_block in range(n_in_quant_blocks):
        k_start = k_block * quant_block_size
        k_end = k_start + quant_block_size

        # Extract sub-blocks for this K iteration
        if quantize_activation:
            x_sub = pl.ds(x_q_tmp, k_start, quant_block_size, axis=1)
        else:
            x_sub = pl.ds(x_ref[...], k_start, quant_block_size, axis=1)

        w_sub = pl.ds(w_q_ref[...], k_start, quant_block_size, axis=1)

        # Iterate over output blocks
        for i_block in range(n_out_quant_blocks):
            i_start = i_block * quant_block_size
            i_end = i_start + quant_block_size

            w_block = pl.ds(w_sub, i_start, quant_block_size, axis=0)

            # Iterate over batch blocks
            for j_block in range(n_batch_quant_blocks):
                j_start = j_block * quant_block_size
                j_end = j_start + quant_block_size

                x_block = pl.ds(x_sub, j_start, quant_block_size, axis=0)

                # Native fp8×fp8 matmul for this sub-block
                sub_result = jax.lax.dot_general(
                    x_block,
                    w_block,
                    (((1,), (1,)), ((), ())),
                    preferred_element_type=acc_dtype,
                )

                # Apply 2D scales for this sub-block
                sub_result = sub_result.astype(jnp.float32)
                w_scale_scalar = w_scale_ref[i_block, k_block]
                sub_result *= w_scale_scalar

                if quantize_activation:
                    x_scale_scalar = x_scale_tmp[j_block, k_block]
                    sub_result *= x_scale_scalar

                # Accumulate into output position
                acc = acc.at[j_start:j_end, i_start:i_end].add(sub_result)

    # Store result - use jax.lax.cond for is_last_step
    def output_branch():
        out_ref[...] = acc.astype(x_ref_dtype)

    def save_branch():
        acc_scratch[...] = acc.astype(acc_dtype)

    jax.lax.cond(is_last_step, output_branch, save_branch)


def matmul_kernel_2d(
    x_ref: jax.Array,  # (quant_block_size, quant_block_size) - single block
    w_q_ref: jax.Array,  # (quant_block_size, quant_block_size) - single block
    w_scale_ref: jax.Array,  # scalar (1, 1) - one scale per block
    x_abs_max_ref: jax.Array,  # scalar (1, 1) - one abs_max per block
    out_ref: jax.Array,  # (quant_block_size, quant_block_size)
    acc_scratch: jax.Array,  # (quant_block_size, quant_block_size)
    x_q_scratch: jax.Array,  # (quant_block_size, quant_block_size)
    x_scale_scratch: jax.Array,  # scalar (1, 1)
    *,
    x_q_dtype: jnp.dtype,
    quant_block_size: int,
    save_acc: bool,
    save_x_q: bool,
):
    """Pallas kernel for 2D quantized matmul with aligned blocks.

    SIMPLIFIED APPROACH FOR MAXIMUM PERFORMANCE:
    - Kernel blocks are ALIGNED with quantization blocks
    - Each kernel processes exactly ONE quantization block
    - Simple: fp8×fp8 matmul + scale (no sub-block iteration)
    - Leverages native fp8 MXU on Ironwood TPU for maximum speed
    - One scale per kernel = straightforward, easy to verify

    The computation is: out = (x_q @ w_q.T) * x_scale * w_scale

    This is simpler and faster than dequantizing first because:
    - Uses native fp8×fp8 → fp32 accumulation in hardware
    - Single matmul operation per kernel
    - Minimal overhead from scaling (just scalar multiplication at end)
    """
    # Compile-time assertions for TPU constraints
    assert quant_block_size % 128 == 0  # Must be multiple of 128 for TPU
    assert quant_block_size % 8 == 0  # (8x128) divisibility

    out_idx, in_idx = pl.program_id(1), pl.program_id(2)
    n_in = pl.num_programs(2)
    x_ref_dtype = x_ref.dtype

    quantize_activation = x_q_dtype != x_ref_dtype

    # Initialize conditional logic
    if save_x_q:
        assert quantize_activation
        assert x_q_scratch is not None
        assert x_scale_scratch is not None
        quant = out_idx == 0
    else:
        assert x_q_scratch is None
        assert x_scale_scratch is None
        quant = quantize_activation

    if save_acc:
        assert acc_scratch is not None
        is_first_step = in_idx == 0
        is_last_step = in_idx == (n_in - 1)
    else:
        assert acc_scratch is None
        is_first_step = True
        is_last_step = True

    acc_dtype = jnp.float32
    if quantize_activation and jnp.issubdtype(w_q_ref.dtype, jnp.integer):
        acc_dtype = jnp.int32

    # Start of actual computation logic
    # PERFORMANCE KEY: Use native fp8×fp8 matmul, NOT dequantize-first!
    # The TPU MXU does fp8×fp8 → fp32 accumulation in hardware.
    # This is MUCH faster than fp32×fp32 matmul.

    if quantize_activation:
        # Quantize or reuse - use jax.lax.cond for potentially-traced quant
        def quantize_branch():
            x_q, x_scale = util.quantize_array_2d(
                x_ref[...],
                x_abs_max_ref[...],
                x_q_dtype,
                quant_block_size,
                quant_block_size,
            )
            if save_x_q:
                x_q_scratch[...] = x_q
                x_scale_scratch[...] = x_scale
            return x_q, x_scale

        def reuse_branch():
            x_q = x_q_scratch[...]
            # Use jax.lax.cond for conditional scale loading
            x_scale = jax.lax.cond(
                is_last_step,
                lambda: x_scale_scratch[...],
                lambda: jnp.zeros((1, 1), dtype=jnp.float32),  # dummy value
            )
            return x_q, x_scale

        x_q_tmp, x_scale_tmp = jax.lax.cond(quant, quantize_branch, reuse_branch)

        # Native fp8×fp8 matmul with fp32 accumulation
        acc = jax.lax.dot_general(
            x_q_tmp,
            w_q_ref[...],
            (((1,), (1,)), ((), ())),
            preferred_element_type=acc_dtype,
        )
    else:
        # bf16 activation × fp8 weight
        acc = jax.lax.dot_general(
            x_ref[...],
            w_q_ref[...],
            (((1,), (1,)), ((), ())),
            preferred_element_type=acc_dtype,
        )
        x_scale_tmp = None  # Not used when not quantizing

    # Accumulate across in_block dimension - use jax.lax.select
    acc = jax.lax.select(is_first_step, acc, acc + acc_scratch[...])

    # Scale and output - use jax.lax.cond for is_last_step
    def output_branch():
        # Apply both scales: result = matmul * w_scale * x_scale
        # w_scale_ref and x_scale are scalars (1,1), will broadcast
        acc_final = acc.astype(jnp.float32)
        acc_final *= w_scale_ref[0, 0]  # Extract scalar from (1,1) array
        if quantize_activation:
            acc_final *= x_scale_tmp[0, 0]  # Extract scalar from (1,1) array
        out_ref[...] = acc_final.astype(x_ref_dtype)

    def save_branch():
        acc_scratch[...] = acc

    jax.lax.cond(is_last_step, output_branch, save_branch)


@functools.partial(
    jax.jit,
    static_argnames=[
        "x_q_dtype",
        "quant_block_size",
        "tuned_value",
    ],
)
def fp8_quantized_matmul_2d_kernel(
    x: jax.Array,  # [bs, n_in]
    w_q: jax.Array,  # [n_out, n_in]
    w_scale: jax.Array,  # [n_out // quant_block_size, n_in // quant_block_size]
    w_zp: jax.Array | None = None,
    x_q_dtype: jnp.dtype | None = None,
    quant_block_size: int = 128,
    *,
    tuned_value: TunedValue | None = None,
) -> jax.Array:
    """2D fp8 quantized matmul kernel for Ironwood TPU.

    This kernel implements matmul with 2D block-wise quantization, where both
    weights and activations are quantized in blocks of size
    (quant_block_size, quant_block_size).

    PERFORMANCE-CRITICAL OPTIMIZATIONS:
    - **Native fp8×fp8 matmul**: Leverages hardware fp8 MXU for maximum speed
    - **DOUBLE BUFFERING**: Automatic prefetching overlaps compute with memory transfers
    - **Kernel blocks = quantization blocks**: Each kernel processes exactly one block (aligned mode)
    - **Simple scale application**: One scale per kernel, just scalar multiply at end
    - **No sub-block iteration**: Simpler code, easier to verify, less overhead (aligned mode)
    - **Static block sizes**: All dimensions are compile-time constants for TPU compiler
    - **Proper memory layout**: Ensures (8x128) divisibility for TPU constraints
    - **VMEM efficient**: Each kernel uses ~1MB for 512×512 blocks, well within 64MB limit

    This is simpler and faster than dequantizing first because:
    - Uses native fp8×fp8 → fp32 accumulation in hardware (vs fp32×fp32)
    - Minimal scaling overhead (scalar multiplication vs array broadcasting)
    - More kernel invocations but each is highly optimized
    - Double buffering hides memory latency behind computation

    Args:
        x: Input unquantized or pre-quantized array [batch_size, n_in]
        w_q: Weight quantized array [n_out, n_in] in fp8 format
        w_scale: Weight 2D quantization scales [n_out // quant_block_size, n_in // quant_block_size]
        w_zp: Weight zero point (not supported, must be None)
        x_q_dtype: Quantization dtype for activations. If None or same as x.dtype, no quantization.
        quant_block_size: Size of 2D quantization blocks (128, 256, or 512)
        tuned_value: Kernel tuned values for optimal performance

    Returns:
        Matmul result [batch_size, n_out]
    """

    if w_zp is not None:
        raise NotImplementedError("zero_point is not supported.")

    if quant_block_size not in [128, 256, 512]:
        raise ValueError(
            f"quant_block_size must be 128, 256, or 512, got {quant_block_size}"
        )

    if x_q_dtype is None:
        x_q_dtype = x.dtype
    quantize_activation = x_q_dtype != x.dtype

    orig_n_batch, orig_n_in = x.shape
    orig_n_out, _ = w_q.shape

    # Compute 2D abs max for activation blocks if quantizing
    if quantize_activation:
        # Ensure dimensions are divisible by quant_block_size for computing abs_max
        padded_n_batch_for_quant = next_multiple(orig_n_batch, quant_block_size)
        padded_n_in_for_quant = next_multiple(orig_n_in, quant_block_size)

        x_for_abs_max = x
        if orig_n_batch < padded_n_batch_for_quant or orig_n_in < padded_n_in_for_quant:
            x_for_abs_max = jnp.pad(
                x,
                (
                    (0, padded_n_batch_for_quant - orig_n_batch),
                    (0, padded_n_in_for_quant - orig_n_in),
                ),
            )

        # Compute abs max per block
        n_quant_blocks_m = padded_n_batch_for_quant // quant_block_size
        n_quant_blocks_n = padded_n_in_for_quant // quant_block_size

        x_reshaped = x_for_abs_max.reshape(
            n_quant_blocks_m,
            quant_block_size,
            n_quant_blocks_n,
            quant_block_size,
        )
        x_abs_max = jnp.max(jnp.abs(x_reshaped), axis=(1, 3))  # [n_quant_blocks_m, n_quant_blocks_n]
    else:
        # Create dummy abs_max (won't be used)
        x_abs_max = jnp.zeros((1, 1), dtype=jnp.float32)

    if tuned_value is None:
        tuned_value = get_tuned_block_sizes(
            n_batch=orig_n_batch,
            n_out=orig_n_out,
            n_in=orig_n_in,
            x_q_dtype=jnp.dtype(x_q_dtype).name,
            w_q_dtype=jnp.dtype(w_q.dtype).name,
            quant_block_size=quant_block_size,
        )

    batch_block_size = tuned_value.batch_block_size
    out_block_size = tuned_value.out_block_size
    in_block_size = tuned_value.in_block_size

    # Determine if we're using aligned blocks (simple) or large blocks (amortized overhead)
    use_aligned_blocks = (
        batch_block_size == quant_block_size and
        out_block_size == quant_block_size and
        in_block_size == quant_block_size
    )

    # Both approaches are supported:
    # - Aligned blocks: Simple, one quant block per kernel, native fp8 matmul
    # - Large blocks: More complex, multiple quant blocks per kernel, amortizes overhead

    # Pad inputs to be multiple of block size
    padded_n_batch = next_multiple(orig_n_batch, batch_block_size)
    if orig_n_batch < padded_n_batch:
        x = jnp.pad(x, ((0, padded_n_batch - orig_n_batch), (0, 0)))
        if quantize_activation:
            x_abs_max = jnp.pad(
                x_abs_max,
                (
                    (0, (padded_n_batch // quant_block_size) - x_abs_max.shape[0]),
                    (0, 0),
                ),
            )

    padded_n_out = next_multiple(orig_n_out, out_block_size)
    if orig_n_out < padded_n_out:
        w_q = jnp.pad(w_q, ((0, padded_n_out - orig_n_out), (0, 0)))
        w_scale = jnp.pad(
            w_scale,
            (
                (0, (padded_n_out // quant_block_size) - w_scale.shape[0]),
                (0, 0),
            ),
        )

    padded_n_in = next_multiple(orig_n_in, in_block_size)
    if orig_n_in < padded_n_in:
        x = jnp.pad(x, ((0, 0), (0, padded_n_in - orig_n_in)))
        w_q = jnp.pad(w_q, ((0, 0), (0, padded_n_in - orig_n_in)))
        if quantize_activation:
            x_abs_max = jnp.pad(
                x_abs_max,
                (
                    (0, 0),
                    (0, (padded_n_in // quant_block_size) - x_abs_max.shape[1]),
                ),
            )
        w_scale = jnp.pad(
            w_scale,
            (
                (0, 0),
                (0, (padded_n_in // quant_block_size) - w_scale.shape[1]),
            ),
        )

    # Ensure scales are float32
    if w_scale.dtype != jnp.float32:
        w_scale = w_scale.astype(jnp.float32)

    n_batch = padded_n_batch // batch_block_size
    n_out = padded_n_out // out_block_size
    n_in = padded_n_in // in_block_size

    save_acc = n_in > 1
    # Cache quantized input for best performance when single input block per batch
    save_x_q = quantize_activation and n_in == 1 and n_out > 1

    acc_dtype = jnp.float32
    if quantize_activation and jnp.issubdtype(w_q.dtype, jnp.integer):
        acc_dtype = jnp.int32

    vmem_limit_bytes = util.get_vmem_limit(
        n_batch=n_batch,
        n_out=n_out,
        n_in=n_in,
        batch_block_size=batch_block_size,
        out_block_size=out_block_size,
        in_block_size=in_block_size,
        quant_block_size=quant_block_size,
        x_dtype=x.dtype,
        x_q_dtype=x_q_dtype,
        w_q_dtype=w_q.dtype,
        scale_dtype=jnp.float32,
        out_dtype=x.dtype,
        acc_dtype=acc_dtype,
        save_acc=save_acc,
        save_x_q=save_x_q,
        upper_limit_bytes=get_device_vmem_limit(),
    )

    # Choose kernel based on block size configuration
    if use_aligned_blocks:
        # ALIGNED BLOCKS: Simple, one quant block per kernel
        # Each kernel processes exactly one quantization block
        # Scales are 1×1 per kernel (scalar)
        assert batch_block_size == quant_block_size
        assert out_block_size == quant_block_size
        assert in_block_size == quant_block_size

        n_w_blocks_m = 1
        n_w_blocks_n = 1
        n_x_blocks_m = 1
        n_x_blocks_n = 1

        # CRITICAL: PrefetchScalarGridSpec enables AUTOMATIC DOUBLE BUFFERING
        # This overlaps compute with memory transfers for maximum TPU performance.
        # The compiler automatically prefetches the next iteration's data while
        # computing the current iteration.
        kernel = pl.pallas_call(
            functools.partial(
                matmul_kernel_2d,
                x_q_dtype=x_q_dtype,
                quant_block_size=quant_block_size,
                save_acc=save_acc,
                save_x_q=save_x_q,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(
                        (quant_block_size, quant_block_size),
                        lambda b, o, i: (b * quant_block_size, i * quant_block_size),
                    ),  # x
                    pl.BlockSpec(
                        (quant_block_size, quant_block_size),
                        lambda b, o, i: (o * quant_block_size, i * quant_block_size),
                    ),  # w_q
                    pl.BlockSpec(
                        (1, 1),  # Single scale per kernel block
                        lambda b, o, i: (o, i),
                    ),  # w_scale
                    pl.BlockSpec(
                        (1, 1),  # Single abs_max per kernel block
                        lambda b, o, i: (b, i),
                    ),  # x_abs_max
                ],
                out_specs=pl.BlockSpec(
                    (quant_block_size, quant_block_size),
                    lambda b, o, i: (b * quant_block_size, o * quant_block_size),
                ),
                scratch_shapes=[
                    (
                        pltpu.VMEM((quant_block_size, quant_block_size), acc_dtype)
                        if save_acc
                        else None
                    ),  # acc_scratch
                    (
                        pltpu.VMEM((quant_block_size, quant_block_size), x_q_dtype)
                        if save_x_q
                        else None
                    ),  # x_q_scratch
                    (
                        pltpu.VMEM((1, 1), jnp.float32)
                        if save_x_q
                        else None
                    ),  # x_scale_scratch (scalar)
                ],
                grid=(n_batch, n_out, n_in),
            ),
            out_shape=jax.ShapeDtypeStruct((padded_n_batch, padded_n_out), x.dtype),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary", "arbitrary"),
                vmem_limit_bytes=vmem_limit_bytes,
            ),
        )
    else:
        # LARGE BLOCKS: Multiple quant blocks per kernel to amortize overhead
        # Each kernel processes a grid of quantization blocks
        # Scales are 2D arrays per kernel block
        n_w_blocks_m = out_block_size // quant_block_size
        n_w_blocks_n = in_block_size // quant_block_size
        n_x_blocks_m = batch_block_size // quant_block_size
        n_x_blocks_n = in_block_size // quant_block_size

        # CRITICAL: PrefetchScalarGridSpec enables AUTOMATIC DOUBLE BUFFERING
        # This overlaps compute with memory transfers for maximum TPU performance.
        # The compiler automatically prefetches the next iteration's data while
        # computing the current iteration.
        kernel = pl.pallas_call(
            functools.partial(
                matmul_kernel_2d_large_blocks,
                x_q_dtype=x_q_dtype,
                quant_block_size=quant_block_size,
                batch_block_size=batch_block_size,
                out_block_size=out_block_size,
                in_block_size=in_block_size,
                save_acc=save_acc,
                save_x_q=save_x_q,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(
                        (batch_block_size, in_block_size),
                        lambda b, o, i: (b, i),
                    ),  # x
                    pl.BlockSpec(
                        (out_block_size, in_block_size),
                        lambda b, o, i: (o, i),
                    ),  # w_q
                    pl.BlockSpec(
                        (n_w_blocks_m, n_w_blocks_n),
                        lambda b, o, i: (o * n_w_blocks_m, i * n_w_blocks_n),
                    ),  # w_scale (2D)
                    pl.BlockSpec(
                        (n_x_blocks_m, n_x_blocks_n),
                        lambda b, o, i: (b * n_x_blocks_m, i * n_x_blocks_n),
                    ),  # x_abs_max (2D)
                ],
                out_specs=pl.BlockSpec(
                    (batch_block_size, out_block_size),
                    lambda b, o, i: (b, o),
                ),
                scratch_shapes=[
                    (
                        pltpu.VMEM((batch_block_size, out_block_size), acc_dtype)
                        if save_acc
                        else None
                    ),  # acc_scratch
                    (
                        pltpu.VMEM((batch_block_size, in_block_size), x_q_dtype)
                        if save_x_q
                        else None
                    ),  # x_q_scratch
                    (
                        pltpu.VMEM((n_x_blocks_m, n_x_blocks_n), jnp.float32)
                        if save_x_q
                        else None
                    ),  # x_scale_scratch (2D)
                ],
                grid=(n_batch, n_out, n_in),
            ),
            out_shape=jax.ShapeDtypeStruct((padded_n_batch, padded_n_out), x.dtype),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary", "arbitrary"),
                vmem_limit_bytes=vmem_limit_bytes,
            ),
        )

    util.validate_inputs(
        x=x,
        w_q=w_q,
        w_scale=w_scale,
        x_abs_max=x_abs_max,
        x_q_dtype=x_q_dtype,
        batch_block_size=batch_block_size,
        out_block_size=out_block_size,
        in_block_size=in_block_size,
        quant_block_size=quant_block_size,
    )

    # The named_scope is used for autotune
    kernel_name = get_kernel_name(tuned_value)
    with jax.named_scope(kernel_name):
        out = kernel(x, w_q, w_scale, x_abs_max)

    return out[:orig_n_batch, :orig_n_out]
