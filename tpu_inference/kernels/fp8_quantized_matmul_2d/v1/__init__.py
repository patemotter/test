# SPDX-License-Identifier: Apache-2.0
"""V1: Automatic double buffering with PrefetchScalarGridSpec.

This is the baseline implementation using automatic compiler-managed double buffering.

Key features:
- PrefetchScalarGridSpec for automatic prefetching
- Compiler manages all buffer swapping
- Simple, clean code
- Good baseline performance

Trade-offs:
- Less control over prefetch timing
- Relies on compiler heuristics
"""

from tpu_inference.kernels.fp8_quantized_matmul_2d.v1.kernel import (
    fp8_quantized_matmul_2d_kernel,
)
from tpu_inference.kernels.fp8_quantized_matmul_2d.v1.tuned_block_sizes import (
    TunedValue,
    get_device_vmem_limit,
    get_tuned_block_sizes,
)
from tpu_inference.kernels.fp8_quantized_matmul_2d.v1.util import (
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
