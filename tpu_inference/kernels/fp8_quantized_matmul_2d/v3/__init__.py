# SPDX-License-Identifier: Apache-2.0
"""V3: SMEM for scales + async DMA.

This version uses SMEM (Scalar Memory) for scale factors combined with async DMA
for data blocks. SMEM is faster than VMEM for small frequently-accessed data.

Key features:
- SMEM for scale storage (~2-3× faster access than VMEM)
- Async DMA for data block transfers
- Double-buffered data in VMEM, scales in SMEM
- Reduced VMEM pressure (scales moved to SMEM)

Trade-offs:
- SMEM capacity limited (~32KB total)
- Only beneficial for small scales (aligned blocks)
- More complex memory hierarchy
- Best when scales are accessed frequently

When to use:
- Aligned blocks (scales are small: 1×1)
- Scales accessed multiple times per kernel
- VMEM is constrained
- Profile shows VMEM bandwidth bottleneck

Performance expectations:
- Faster scale access (~2-3× vs VMEM)
- Lower VMEM contention
- Best for small scale tensors
"""

from tpu_inference.kernels.fp8_quantized_matmul_2d.v3.kernel import (
    fp8_quantized_matmul_2d_kernel,
)
from tpu_inference.kernels.fp8_quantized_matmul_2d.v3.tuned_block_sizes import (
    TunedValue,
    get_device_vmem_limit,
    get_tuned_block_sizes,
)
from tpu_inference.kernels.fp8_quantized_matmul_2d.v3.util import (
    quantize_tensor_2d,
    xla_quantized_matmul_2d,
)

__all__ = [
    "fp8_quantized_matmul_2d_kernel",
    "quantize_tensor_2d",
    "xla_quantized_matmul_2d",
    "TunedValue",
    "get_tuned_block_sizes",
    "get_device_vmem_limit",
]
