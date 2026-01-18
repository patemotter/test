# SPDX-License-Identifier: Apache-2.0
"""2D fp8 quantized matmul kernel for Ironwood TPU."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.fp8_quantized_matmul_2d import util
from tpu_inference.kernels.fp8_quantized_matmul_2d.tuned_block_sizes import (
    TunedValue,
    get_device_vmem_limit,
    get_tuned_block_sizes,
)
from tpu_inference.kernels.fp8_quantized_matmul_2d.util import (
    get_kernel_name,
    next_multiple,
    unfold_args,
)

quantize_tensor_2d = util.quantize_tensor_2d
cdiv = pl.cdiv


def matmul_kernel_2d(
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
    """Pallas kernel for 2D quantized matmul.

    This kernel performs matmul with 2D block-wise quantization for both
    weights and activations. Each block of size (quant_block_size, quant_block_size)
    has its own scale factor.

    The computation is: out = (x @ w_q.T) with proper 2D scaling applied.

    Key implementation details:
    - All block sizes are compile-time constants for TPU optimization
    - Uses sub-block iteration pattern similar to fused_moe for 2D scaling
    - Scales are applied after matmul to keep data in quantized format for MXU
    """
    # Compile-time assertions for TPU constraints
    assert batch_block_size % quant_block_size == 0
    assert out_block_size % quant_block_size == 0
    assert in_block_size % quant_block_size == 0
    assert quant_block_size % 128 == 0  # Must be multiple of 128 for TPU
    assert batch_block_size % 8 == 0  # (8x128) divisibility
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

    # Compute number of quantization blocks (compile-time constants)
    n_batch_quant_blocks = batch_block_size // quant_block_size
    n_out_quant_blocks = out_block_size // quant_block_size
    n_in_quant_blocks = in_block_size // quant_block_size

    # Start of actual computation logic
    def matmul_body(quant: bool, is_first_step: bool, is_last_step: bool):
        # For 2D block quantization, we need to break down the matmul into sub-blocks
        # and apply different scales to each sub-block contribution.
        # This follows the pattern from fused_moe kernel (lines 813-890)

        if quantize_activation:
            if quant:
                # Quantize activation with 2D blocks
                x_q_tmp, x_scale_tmp = util.quantize_array_2d(
                    x_ref[...],
                    x_abs_max_ref[...],
                    x_q_dtype,
                    quant_block_size,
                    quant_block_size,
                )

                if save_x_q:
                    x_q_scratch[...] = x_q_tmp
                    x_scale_scratch[...] = x_scale_tmp
            else:
                assert save_x_q
                x_q_tmp = x_q_scratch[...]
                if is_last_step:
                    x_scale_tmp = x_scale_scratch[...]

            x_input = x_q_tmp
        else:
            x_input = x_ref[...]
            x_scale_tmp = None

        # Initialize or load accumulator
        if is_first_step:
            acc = jnp.zeros((batch_block_size, out_block_size), dtype=jnp.float32)
        else:
            acc = acc_scratch[...].astype(jnp.float32)

        # Perform blocked matmul with per-block scaling
        # Iterate over quantization blocks in all three dimensions
        for batch_qblock_id in range(n_batch_quant_blocks):
            for out_qblock_id in range(n_out_quant_blocks):
                # Initialize accumulator for this output block
                partial_acc = jnp.zeros((quant_block_size, quant_block_size), dtype=jnp.float32)

                # Accumulate contributions from all input quantization blocks
                for in_qblock_id in range(n_in_quant_blocks):
                    # Extract quantization blocks
                    x_block = x_input[
                        pl.ds(batch_qblock_id * quant_block_size, quant_block_size),
                        pl.ds(in_qblock_id * quant_block_size, quant_block_size),
                    ]
                    w_block = w_q_ref[
                        pl.ds(out_qblock_id * quant_block_size, quant_block_size),
                        pl.ds(in_qblock_id * quant_block_size, quant_block_size),
                    ]

                    # Compute sub-block matmul
                    sub_result = jnp.dot(x_block, w_block.T, preferred_element_type=acc_dtype).astype(jnp.float32)

                    # Apply 2D block-wise scales
                    w_scale_val = w_scale_ref[out_qblock_id, in_qblock_id]
                    if quantize_activation:
                        x_scale_val = x_scale_tmp[batch_qblock_id, in_qblock_id]
                        combined_scale = x_scale_val * w_scale_val
                    else:
                        combined_scale = w_scale_val

                    sub_result *= combined_scale
                    partial_acc += sub_result

                # Write back the accumulated result for this output block
                acc = acc.at[
                    pl.ds(batch_qblock_id * quant_block_size, quant_block_size),
                    pl.ds(out_qblock_id * quant_block_size, quant_block_size),
                ].set(partial_acc)

        if is_last_step:
            out_ref[...] = acc.astype(x_ref_dtype)
        else:
            assert save_acc
            acc_scratch[...] = acc

    unfold_args((quant, is_first_step, is_last_step), (), matmul_body)


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

    Performance optimizations:
    - All block sizes are compile-time constants for TPU optimization
    - Sub-block iteration pattern (similar to fused_moe) for correct 2D scaling
    - Scales applied after each sub-matmul before accumulation
    - Ensures (8x128) divisibility for Ironwood TPU MXU constraints
    - VMEM management for 64MB limit with proper double buffering

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

    Note:
        For 2D quantization to work correctly, the matmul is broken down into sub-blocks.
        Each sub-block contribution is scaled individually before accumulation, following
        the pattern: result += scale[i,k] * scale[j,k] * (x_block[i,k] @ w_block[j,k].T)
        This ensures each quantization block's scale is properly applied.
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

    # Verify block sizes are divisible by quant_block_size
    assert batch_block_size % quant_block_size == 0
    assert out_block_size % quant_block_size == 0
    assert in_block_size % quant_block_size == 0

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

    # Define BlockSpec for 2D scales
    n_w_blocks_m = out_block_size // quant_block_size
    n_w_blocks_n = in_block_size // quant_block_size
    n_x_blocks_m = batch_block_size // quant_block_size
    n_x_blocks_n = in_block_size // quant_block_size

    kernel = pl.pallas_call(
        functools.partial(
            matmul_kernel_2d,
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
