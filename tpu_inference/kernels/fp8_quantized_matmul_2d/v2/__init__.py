# SPDX-License-Identifier: Apache-2.0
"""V2: Manual async DMA with explicit semaphores.

This version uses explicit async DMA operations with manual semaphore synchronization,
similar to the fused_moe kernel. This gives fine-grained control over prefetch timing.

Key features:
- Explicit pltpu.make_async_copy() with .start() and .wait()
- Manual semaphore-based synchronization (5 sems per buffer)
- Double-buffered VMEM arrays (x2 pattern)
- Fine-grained control: prefetch next iteration while computing current

Trade-offs:
- More complex code (explicit buffer management)
- More VMEM for semaphores
- Better performance if compiler auto-prefetch is suboptimal
- Currently aligned blocks only (large blocks TBD)

When to use:
- Benchmark shows compiler auto-prefetch leaving gaps
- Need precise control over prefetch timing
- Willing to trade code complexity for potential speedup
"""

from tpu_inference.kernels.fp8_quantized_matmul_2d.v2.kernel import (
    fp8_quantized_matmul_2d_kernel,
)
from tpu_inference.kernels.fp8_quantized_matmul_2d.v2.tuned_block_sizes import (
    TunedValue,
    get_device_vmem_limit,
    get_tuned_block_sizes,
)
from tpu_inference.kernels.fp8_quantized_matmul_2d.v2.util import (
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
