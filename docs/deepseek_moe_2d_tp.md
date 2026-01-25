# DeepSeek MoE and 2D Tensor Parallelism on TPU

This document describes the Mixture-of-Experts (MoE) implementation for DeepSeek models (V3/R1) on TPU, including the 2D Tensor Parallelism feature and its performance implications.

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Sharding Strategies](#sharding-strategies)
- [MoE Implementation](#moe-implementation)
- [Environment Configuration](#environment-configuration)
- [Performance Analysis](#performance-analysis)
- [Usage Examples](#usage-examples)
- [Key Files](#key-files)

---

## Overview

DeepSeek-V3 and DeepSeek-R1 share the same model architecture (`DeepseekV3ForCausalLM`) and use identical inference code paths. Both models feature:

- **671B total parameters**
- **61 transformer layers** (3 dense + 58 MoE)
- **256 experts per MoE layer**
- **8 experts selected per token**
- **Multi-head Latent Attention (MLA)**

The key difference is training: DeepSeek-R1 includes additional reinforcement learning for enhanced reasoning capabilities.

---

## Model Architecture

### Layer Structure

```
DeepSeek-V3/R1 (61 layers)
├── Layers 0-2: Dense FFN (no MoE)
└── Layers 3-60: MoE layers (256 experts each)
    ├── DeepSeekV3Router (grouped expert selection)
    ├── Shared Experts (dense FFN, always active)
    └── Routed Experts (256 experts, 8 selected per token)
```

### MoE Layer Configuration

```python
# From tpu_inference/models/jax/deepseek_v3.py
num_local_experts = 256        # Total experts per layer
num_experts_per_token = 8      # Top-k experts selected
n_groups = 8                   # Expert groups for routing
topk_groups = 4                # Top groups to consider
hidden_size = 7168             # Model dimension (D)
moe_intermediate_size = 2048   # Expert FFN dimension (F)
```

---

## Sharding Strategies

### Two Approaches

The codebase supports two fundamentally different sharding strategies for MoE layers:

#### 1. Expert Parallelism (EP) - Default

Shards the **expert dimension** across devices. Each device holds a subset of experts.

```
Weight Shape: (E=256, D=7168, F=2048)
Sharding:     (MLP_TENSOR, None, None)  →  Shard on E axis

Device Layout (TP=8):
  Device 0: experts[0:32]   × D[full] × F[full]
  Device 1: experts[32:64]  × D[full] × F[full]
  ...
  Device 7: experts[224:256] × D[full] × F[full]

Communication: All-to-All to route tokens to correct expert device
```

#### 2. 2D Tensor Parallelism (2D TP) - Experimental

Shards the **weight dimensions (D, F)** across devices. All devices hold all experts.

```
Weight Shape: (E=256, D=7168, F=2048)
Sharding:     (None, MODEL_1, MODEL_2)  →  Shard on D and F axes

Device Layout (model=4, expert=2 on 4D mesh):
  Device 0: experts[ALL] × D[0:1792]    × F[0:1024]
  Device 1: experts[ALL] × D[1792:3584] × F[0:1024]
  ...

Communication: All-Reduce after matmuls (no token movement)
```

### Sharding Classes

Located in `tpu_inference/layers/common/sharding.py`:

| Class | Mesh Shape | Use Case |
|-------|------------|----------|
| `ShardingAxisName2D` | `('data', 'model')` | Default 2D mesh, EP sharding |
| `ShardingAxisNameBase` | `('data', 'attn_dp', 'expert', 'model')` | 4D mesh, supports 2D TP |

Key attributes for 2D TP in `ShardingAxisNameBase`:
```python
MODEL_1 = 'model'   # Primary tensor parallelism axis
MODEL_2 = 'expert'  # Secondary axis for weight sharding
```

---

## MoE Implementation

### Functional API (Post PR #1287)

The MoE implementation uses a functional API where forward functions take `moe_instance` as the first parameter:

```python
# tpu_inference/layers/jax/moe/dense_moe.py
def dense_moe_fwd(moe_instance, x_TD, weights):
    ...

def dense_moe_fwd_preapply_router_weights(moe_instance, x_TD, weights_TE):
    ...

# tpu_inference/layers/jax/moe/sparse_moe.py
def sparse_moe_distributed_fwd(moe_instance, x_TD, router_weights_TX, ...):
    ...
```

### MoE Backends

| Backend | Kernel | Mesh Support | Status |
|---------|--------|--------------|--------|
| `DENSE_MAT` | Dense einsum | Any | Default, unoptimized |
| `MEGABLX_GMM` | Megablox GMM | Any | Sparse, generic |
| `RAGGED_DOT` | JAX ragged_dot | Any | Sparse, generic |
| `FUSED_MOE` | Custom Pallas | **2D only** | Optimized, EP only |
| `VLLM_MOE` | vLLM kernel | 2D | Alternative |

### shard_map Configuration

For sparse backends (MEGABLX_GMM, RAGGED_DOT):

```python
in_specs = (
    PartitionSpec(),                    # Replicated MoE instance
    PartitionSpec(*activation_ffw_td),  # Sharded input x_TD
    PartitionSpec(),                    # Replicated router_weights_TX
    PartitionSpec(),                    # Replicated selected_experts_TX
    PartitionSpec(*edf_sharding),       # Sharded gating kernel
    PartitionSpec(*edf_sharding),       # Sharded up-projection kernel
    PartitionSpec(*efd_sharding),       # Sharded down-projection kernel
)
```

---

## Environment Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_2D_TP` | `False` | Enable 2D tensor parallelism sharding |
| `NEW_MODEL_DESIGN` | `False` | Use 4D mesh instead of 2D |
| `USE_MOE_EP_KERNEL` | `False` | Use fused expert-parallel MoE kernel |
| `USE_VLLM_MOE_KERNEL` | `False` | Use vLLM MoE kernel |
| `USE_MEGABLOCKS` | `False` | Use Megablox GMM for sparse MoE |
| `USE_RAGGED_DOT` | `False` | Use ragged dot for sparse MoE |

### Sharding Selection Logic

```python
# From tpu_inference/layers/common/sharding.py
_use_2d_tp_sharding = envs.USE_2D_TP
_use_base_sharding = envs.NEW_MODEL_DESIGN

if _use_2d_tp_sharding or _use_base_sharding:
    ShardingAxisName = ShardingAxisNameBase  # 4D mesh, 2D TP capable
else:
    ShardingAxisName = ShardingAxisName2D    # 2D mesh, EP only
```

---

## Performance Analysis

### When to Use Each Strategy

| Factor | Expert Parallelism (EP) | 2D Tensor Parallelism |
|--------|------------------------|----------------------|
| **TP Size** | ≤ 8 (optimal) | > 8 (better scaling) |
| **Batch Size** | Large (compute-bound) | Small (memory-bound) |
| **Sequence Length** | Short (< 4K) | Long (> 8K, KV cache pressure) |
| **Expert Selection** | Balanced | Imbalanced (avoids hotspots) |
| **Kernel Support** | fused_ep_moe (optimized) | Generic only |

### Communication Patterns

**Expert Parallelism:**
```
Token → Ring All-to-All Scatter → Expert Compute → Ring All-to-All Gather → Output
       (O(N) rounds)                              (O(N) rounds)
```

**2D Tensor Parallelism:**
```
Token → All-Gather (metadata) → Expert Compute → All-Reduce (D axis) → Output
                               (local, all experts)  All-Reduce (F axis)
```

### Current Limitation

The optimized `fused_ep_moe` kernel **requires a 2D mesh**:

```python
# From tpu_inference/kernels/fused_moe/v1/kernel.py
if len(mesh.shape) != 2:
    raise NotImplementedError("Only 2D mesh is supported.")
```

This means 2D TP currently runs with generic (slower) kernels. The infrastructure is in place for when optimized 4D mesh kernels are developed.

### Load Balancing

**EP Problem:** If expert selection is skewed, some devices are idle:
```
Token selects expert 5   → Must go to Device 0
Token selects expert 200 → Must go to Device 6
Uneven distribution = wasted compute
```

**2D TP Solution:** All devices have all experts:
```
Token selects expert 5   → Computed locally (partial weights)
Token selects expert 200 → Computed locally (partial weights)
Always balanced compute, then reduce
```

---

## Usage Examples

### Standard DeepSeek Inference (Expert Parallelism)

```bash
# Default: Uses 2D mesh with expert parallelism
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size=8 \
    --max-model-len=4096
```

### With Optimized Fused Kernel

```bash
# Enable fused MoE kernel (requires 2D mesh, EP sharding)
USE_MOE_EP_KERNEL=1 vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size=8
```

### Experimental 2D TP (4D Mesh)

```bash
# Enable 2D tensor parallelism (uses generic kernels)
USE_2D_TP=1 NEW_MODEL_DESIGN=1 USE_RAGGED_DOT=1 \
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size=8
```

### With Attention Data Parallelism

```bash
# For long sequences with KV cache pressure
NEW_MODEL_DESIGN=1 vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size=8 \
    --additional-config='{"sharding":{"sharding_strategy":{"enable_dp_attention":true}}}'
```

---

## Key Files

| File | Description |
|------|-------------|
| `tpu_inference/models/jax/deepseek_v3.py` | DeepSeek V3/R1 model definition |
| `tpu_inference/layers/jax/moe/moe.py` | MoE layer implementation |
| `tpu_inference/layers/jax/moe/sparse_moe.py` | Sparse MoE forward (functional) |
| `tpu_inference/layers/jax/moe/dense_moe.py` | Dense MoE forward (functional) |
| `tpu_inference/layers/jax/moe/utils.py` | Backend selection, helper functions |
| `tpu_inference/layers/jax/moe/deepseek_v3_moe.py` | DeepSeek-specific router |
| `tpu_inference/layers/common/sharding.py` | Sharding axis definitions |
| `tpu_inference/kernels/fused_moe/v1/kernel.py` | Optimized fused MoE kernel |
| `tpu_inference/runner/tpu_runner.py` | TPU runner, mesh creation |
| `tpu_inference/envs.py` | Environment variable definitions |

---

## Testing

### Unit Tests

```bash
# Test MoE layer functionality
python -m pytest tests/layers/jax/moe/test_deepseek_moe.py -v

# Test sharding configuration
python -m pytest tests/layers/jax/test_sharding.py -v
```

### Test Classes

- `TestDeepSeekV3Router` - Router expert selection
- `TestMoE` - MoE forward pass correctness (Dense, Sparse backends)
- `TestMoE2DTP` - 2D TP sharding verification
- `TestShardingAxisName2DTP` - Sharding axis configuration

---

## History

This implementation evolved through:

1. **Original**: Expert parallelism with class-based MoE engines
2. **PR #1287**: Refactored to functional API (`dense_moe_fwd`, `sparse_moe_distributed_fwd`)
3. **bz/ds-moe3 branch**: Added 2D TP support (`MODEL_1`, `MODEL_2` axes, `USE_2D_TP` flag)
4. **Merge**: Combined functional API with 2D TP sharding infrastructure

The 2D TP infrastructure is forward-looking, designed for when optimized kernels support 4D meshes.
