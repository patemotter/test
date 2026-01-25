# TPU Inference Technical Architecture

This document provides a comprehensive technical walkthrough of the tpu-inference codebase architecture, explaining how all components interact to serve large language models on TPUs.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Core Concepts](#core-concepts)
- [Layers Module](#layers-module)
- [Models Module](#models-module)
- [MoE Implementation Deep Dive](#moe-implementation-deep-dive)
- [Runner & Serving Infrastructure](#runner--serving-infrastructure)
- [Sharding & Distributed Execution](#sharding--distributed-execution)
- [Kernel Implementations](#kernel-implementations)
- [Environment Configuration](#environment-configuration)
- [Request Flow Walkthrough](#request-flow-walkthrough)

---

## Overview

The tpu-inference project provides a unified inference backend for vLLM on TPUs, supporting both native JAX implementations and PyTorch models via Torchax interoperability. The architecture is designed around several key principles:

1. **Multi-Backend Support**: Native JAX models (high performance) and vLLM PyTorch models (compatibility)
2. **Disaggregated Serving**: Separate prefill and decode stages for optimal throughput/latency
3. **Flexible Parallelism**: Tensor, Expert, Data, Pipeline, and Sequence parallelism
4. **Modular Kernel Selection**: Multiple MoE backends optimized for different scenarios

---

## Directory Structure

```
tpu-inference/
├── tpu_inference/              # Main package
│   ├── __init__.py            # Platform initialization (Pathways proxy, TPU info)
│   ├── envs.py                # Environment variable management (~200+ settings)
│   ├── utils.py               # Common utilities (dtype, sharding, HBM)
│   ├── tpu_info.py            # TPU device information
│   │
│   ├── core/                  # Engine/executor core logic
│   │   ├── core_tpu.py        # Async TPU model runner output, execute state
│   │   └── disagg_executor.py # Multi-worker disaggregation orchestrator
│   │
│   ├── runner/                # Model runner and batch management
│   │   ├── tpu_runner.py      # Main TPU executor (~1200 lines)
│   │   ├── input_batch.py     # Batch state management
│   │   ├── kv_cache_manager.py# KV cache allocation
│   │   ├── block_table.py     # Block allocation tracking
│   │   └── persistent_batch_manager.py  # Request persistence
│   │
│   ├── layers/                # Layer implementations
│   │   ├── common/            # Shared abstractions
│   │   ├── jax/               # Native JAX layers (Flax NNX)
│   │   └── vllm/              # PyTorch wrappers via Torchax
│   │
│   ├── models/                # Model architectures
│   │   ├── common/            # Model loader registry
│   │   ├── jax/               # Native JAX models
│   │   └── vllm/              # vLLM model wrapper
│   │
│   ├── kernels/               # Custom kernels
│   │   ├── fused_moe/         # Fused MoE kernels
│   │   └── ...                # Other kernels (attention, quant)
│   │
│   ├── distributed/           # Distributed state management
│   │   ├── jax_parallel_state.py  # Pipeline parallel coordination
│   │   └── tpu_connector.py   # Multi-host TPU connection
│   │
│   └── executors/             # vLLM executor integration
│
├── tests/                     # Test suite
├── scripts/                   # vLLM integration scripts
├── examples/                  # Usage examples
└── docs/                      # Documentation
```

---

## Core Concepts

### Execution Model

The system follows an async execution model where:

1. **Request arrives** → Scheduler batches requests
2. **Input preparation** → Token IDs, attention metadata assembled
3. **Model execution** → Returns `AsyncTPUModelRunnerOutput`
4. **Sampling** → Samples next tokens from logits
5. **Output** → Results returned to client

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   vLLM      │────>│  TPU Runner  │────>│    Model    │
│  Scheduler  │     │ (tpu_runner) │     │  (JAX/PT)   │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           v
                    ┌──────────────┐
                    │ Async Output │
                    │ (device arr) │
                    └──────────────┘
                           │
                           v
                    ┌──────────────┐
                    │   Sampler    │
                    └──────────────┘
```

### Weight Loading Pipeline

```
HuggingFace Checkpoint
        │
        v
Torch CPU Loading (via safetensors/pickle)
        │
        v
Dtype Conversion (torch → jax.numpy dtypes)
        │
        v
Parameter Mapping (_loaded_to_standardized_keys dict)
        │
        v
Transposition (if needed by model architecture)
        │
        v
JAX Array Creation (jax.device_put)
        │
        v
Sharding Distribution (shard_put → NamedSharding)
```

---

## Layers Module

The layers module (`tpu_inference/layers/`) provides layer implementations across three subdirectories.

### Common Layers (`layers/common/`)

Shared abstractions used by all backends:

| File | Purpose |
|------|---------|
| `attention_interface.py` | Unified `attention()` function signature |
| `attention_metadata.py` | `AttentionMetadata` dataclass for execution context |
| `sharding.py` | `ShardingAxisName`, `ShardingStrategy`, `ShardingConfigManager` |
| `quantization/` | Quantization configuration framework |
| `process_weights/` | Weight post-processing utilities |
| `fused_moe_gmm.py` | Grouped matrix multiply utilities |

**Key Classes in `sharding.py`:**

```python
class ShardingAxisNameBase:
    """4D mesh axis naming for advanced parallelism"""
    SEQUENCE = ('data', 'attn_dp')
    ATTN_HEAD = ('model', 'expert')
    MLP_TENSOR = ('attn_dp', 'model', 'expert')
    MODEL_1 = 'model'    # Primary TP axis
    MODEL_2 = 'expert'   # Secondary TP axis (for 2D TP)

class ShardingAxisName2D:
    """Simplified 2D mesh (default)"""
    ATTN_HEAD = 'model'
    MLP_TENSOR = 'model'

@dataclass
class ShardingStrategy:
    tensor_parallelism: int = 1
    expert_parallelism: int = 1
    sequence_parallelism: int = 1
    data_parallelism: int = 1
    attention_data_parallelism: int = 1
```

### JAX Backend (`layers/jax/`)

Native JAX implementations using Flax NNX:

**Core Files:**

| File | Classes/Functions |
|------|------------------|
| `base.py` | `JaxModule` (base class), `Config`, `create_param()` |
| `layers.py` | `FlaxUtils`, `RMSNorm`, `RuntimeParams` |
| `linear.py` | Linear layer implementations |
| `rope_interface.py` | `apply_rope()` - Rotary Position Embedding |

**Attention (`layers/jax/attention/`):**

| File | Description |
|------|-------------|
| `attention.py` | Standard `Attention` with RoPE, KV caching |
| `deepseek_v3_attention.py` | `MLA` - Multi-head Latent Attention |
| `gpt_oss_attention.py` | GPT-OSS specific attention |
| `llama4_attention.py` | Llama4 attention variant |

**MoE (`layers/jax/moe/`):**

| File | Classes/Functions |
|------|------------------|
| `moe.py` | `Router`, `CombineExperts`, `MoE` main layer |
| `dense_moe.py` | `dense_moe_fwd()`, `dense_moe_fwd_preapply_router_weights()` |
| `sparse_moe.py` | `sparse_moe_distributed_fwd()` |
| `utils.py` | `MoEBackend` enum, `select_moe_backend()`, permutation functions |
| `deepseek_v3_moe.py` | `DeepSeekV3Router` |
| `gpt_oss_moe.py` | `GptOssRouter`, custom activation |

**Sampling (`layers/jax/sample/`):**

| File | Purpose |
|------|---------|
| `sampling.py` | `sample()` - temperature/top-k/top-p |
| `rejection_sampler.py` | Speculative decoding rejection |
| `sampling_metadata.py` | TPU sampling configuration |

### vLLM Backend (`layers/vllm/`)

PyTorch layers running via Torchax interop:

| File | Purpose |
|------|---------|
| `attention.py` | Attention wrapper |
| `linear.py` | Linear layer wrapper |
| `fused_moe.py` | `fused_moe_func()` - vLLM MoE interface |
| `quantization/` | AWQ, Compressed Tensors, FP8, INT8 |
| `process_weights/` | `shard_model_to_tpu()` post-load sharding |

---

## Models Module

The models module (`tpu_inference/models/`) contains model architectures.

### Model Loader (`models/common/model_loader.py`)

Central registry for supported architectures:

```python
_MODEL_REGISTRY = {
    "LlamaForCausalLM": ("llama3", "Llama3ForCausalLM"),
    "Llama4ForCausalLM": ("llama4", "Llama4ForCausalLM"),
    "DeepseekV3ForCausalLM": ("deepseek_v3", "DeepSeekV3ForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2_5_VLForConditionalGeneration": ("qwen2_5_vl", "Qwen2_5VLForConditionalGeneration"),
    # ... more models
}
```

**Key Functions:**

- `_get_model_architecture()` - Registry lookup
- `_get_nnx_model()` - JAX model initialization
- Model format detection (JAX native vs vLLM fallback)

### JAX Native Models (`models/jax/`)

| File | Architecture | Key Features |
|------|-------------|--------------|
| `llama3.py` | LlamaForCausalLM | Standard transformer decoder |
| `llama4.py` | Llama4ForCausalLM | Enhanced Llama architecture |
| `llama_eagle3.py` | EagleLlama3ForCausalLM | Speculative decoding proposer |
| `llama_guard_4.py` | LlamaGuard4ForCausalLM | Safety model |
| `qwen2.py`, `qwen3.py` | Qwen models | Qwen family |
| `qwen2_5_vl.py` | Qwen2.5-VL | Multi-modal vision-language |
| `deepseek_v3.py` | DeepSeekV3ForCausalLM | MLA + MoE (see below) |
| `gpt_oss.py` | GptOssForCausalLM | MoE with custom activation |

### DeepSeek V3 Model (`models/jax/deepseek_v3.py`)

The most complex model implementation (~50KB), featuring:

**Architecture:**
- 61 layers (3 dense + 58 MoE)
- Multi-head Latent Attention (MLA)
- 256 experts per MoE layer, 8 selected per token
- Shared experts + routed experts

**Key Components:**

```python
class MLA(JaxModule):
    """Multi-head Latent Attention
    - Projects Q/K/V through latent space
    - Reduces KV cache size significantly
    """

class SharedExpertsTransformerBlock(JaxModule):
    """Transformer block with shared + routed experts
    - Shared experts: always active dense FFN
    - Routed experts: top-k selection from 256 experts
    """

class DeepSeekV3Router(Router):
    """Custom routing with e_score correction bias
    - Grouped expert selection (n_groups=8, topk_groups=4)
    - Per-expert bias correction for load balancing
    """

class DeepSeekV3WeightLoader(BaseWeightLoader):
    """Weight loading with custom transposition maps
    - Parameter alignment for efficient sharding
    - Multi-modal configuration support
    """
```

### vLLM Model Wrapper (`models/vllm/vllm_model_wrapper.py`)

Wraps PyTorch vLLM models for JAX execution:

```python
class VllmModelWrapper:
    """Adapts vLLM PyTorch models to JAX execution

    Key methods:
    - __call__(): Forward pass through wrapped model
    - load_weights(): CPU staging + distributed loading
    """

class _VllmRunner(torch.nn.Module):
    """torch.nn.Module adapter for XLA execution"""
```

### Weight Utilities (`models/jax/utils/weight_utils.py`)

Framework for parameter loading:

```python
class BaseWeightLoader:
    """Abstract weight loader with mapping support

    Key attributes:
    - _loaded_to_standardized_keys: parameter name mapping
    - _params_to_transpose: params needing transposition
    """

def transfer_state_with_mappings(state, loaded, mappings, ...):
    """Apply weight mappings to populate model state"""

def shard_put(param, mesh, sharding):
    """Distribute parameter across mesh with sharding"""
```

---

## MoE Implementation Deep Dive

The Mixture-of-Experts implementation is highly modular with multiple execution strategies.

### Architecture Overview

```
Input Tokens (T, D)
        │
        v
┌───────────────────┐
│      Router       │  → Selects top-k experts per token
│  (DeepSeekV3Router)│  → Returns weights (T, X) and indices (T, X)
└───────────────────┘
        │
        v
┌───────────────────┐
│   Backend Select  │  → FUSED_MOE, VLLM_MOE, MEGABLOX_GMM,
│   (select_moe_backend)│     RAGGED_DOT, or DENSE_MAT
└───────────────────┘
        │
        v
┌───────────────────┐
│   Expert Forward  │  → Gating, Up-projection, Activation,
│   (dense/sparse)  │     Down-projection
└───────────────────┘
        │
        v
┌───────────────────┐
│  Combine Experts  │  → Weighted sum: "TED,TE -> TD"
└───────────────────┘
        │
        v
Output (T, D)
```

### MoE Backends (`layers/jax/moe/utils.py`)

```python
class MoEBackend(enum.Enum):
    FUSED_MOE = "fused_moe"       # Custom TPU kernel (2D mesh only)
    VLLM_MOE = "vllm_moe"         # vLLM's fused kernel
    DENSE_MAT = "dense_mat"       # Dense einsum (baseline)
    MEGABLOX_GMM = "megablox_gmm" # JAX Pallas Megablocks
    RAGGED_DOT = "ragged_dot"     # QWIX with quantization

def select_moe_backend():
    if USE_MOE_EP_KERNEL: return MoEBackend.FUSED_MOE
    elif USE_VLLM_MOE_KERNEL: return MoEBackend.VLLM_MOE
    elif USE_MEGABLOCKS: return MoEBackend.MEGABLOX_GMM
    elif USE_RAGGED_DOT: return MoEBackend.RAGGED_DOT
    else: return MoEBackend.DENSE_MAT
```

### Dense MoE Forward (`layers/jax/moe/dense_moe.py`)

Simple but memory-intensive approach:

```python
def dense_moe_fwd(moe_instance, x_TD, weights):
    """Dense matrix-based MoE forward

    Steps:
    1. Router: (T,D) @ (D,E) -> (T,E) logits
    2. Gating: (T,E,D) @ (E,D,F) -> (T,E,F)
    3. Up-proj: (T,E,D) @ (E,D,F) -> (T,E,F)
    4. Activation: element-wise
    5. Down-proj: (T,E,F) @ (E,F,D) -> (T,E,D)
    6. Combine: (T,E,D) * (T,E) -> (T,D)
    """
```

### Sparse MoE Forward (`layers/jax/moe/sparse_moe.py`)

Distributed sparse execution with token routing:

```python
def sparse_moe_distributed_fwd(
    moe_instance,
    x_TD: jax.Array,
    router_weights_TX: jax.Array,
    selected_experts_TX: jax.Array,
    kernel_gating: jax.Array,
    kernel_up_proj: jax.Array,
    kernel_down_proj: jax.Array,
):
    """Distributed sparse MoE with 4 stages

    Stage 1: Global Permute
    - Sort tokens by assigned expert (global_permute_fn)
    - Compute group sizes (tokens per expert)

    Stage 2: All-to-All Exchange (if EP > 1)
    - Gather group sizes across devices
    - ragged_all_to_all exchanges tokens

    Stage 3: Local Permute
    - Route tokens to local expert instances
    - Handle batch-sharded vs replicated scenarios

    Stage 4: Expert Computation (gmm_fn)
    - Grouped matrix multiply for gate/up/down
    - Activation between up and down

    Cleanup: Unpermute
    - Restore original token order
    - Apply weighted combination
    """
```

### shard_map Configuration

For sparse backends, the MoE layer uses JAX's `shard_map` for distributed execution:

```python
in_specs = (
    PartitionSpec(),                    # Replicated MoE instance
    PartitionSpec(*activation_ffw_td),  # Sharded input x_TD
    PartitionSpec(),                    # Replicated router_weights_TX
    PartitionSpec(),                    # Replicated selected_experts_TX
    PartitionSpec(*edf_sharding),       # Sharded gating kernel (E,D,F)
    PartitionSpec(*edf_sharding),       # Sharded up-projection (E,D,F)
    PartitionSpec(*efd_sharding),       # Sharded down-projection (E,F,D)
)

out_specs = PartitionSpec(*activation_ffw_td)

mapped_fwd = shard_map(
    sparse_moe_distributed_fwd,
    mesh=mesh,
    in_specs=in_specs,
    out_specs=out_specs,
    check_rep=False,
)
```

---

## Runner & Serving Infrastructure

### TPU Model Runner (`runner/tpu_runner.py`)

The main execution interface implementing vLLM's Executor API:

```python
class TPUModelRunner:
    """Main TPU inference executor

    Key methods:
    - __init__(): Load model, create mesh, initialize caches
    - execute_model(): Forward pass → AsyncTPUModelRunnerOutput
    - sample_tokens(): Sample from logits
    - _prepare_inputs(): Build token arrays, attention metadata
    """
```

**Initialization Flow:**

```
TPUModelRunner.__init__()
    │
    ├─> _init_model()
    │   ├─> make_optimized_mesh()  # Create device mesh
    │   ├─> _get_nnx_model()       # Load model architecture
    │   └─> load_weights()         # Distribute parameters
    │
    ├─> KVCacheManager()           # Allocate KV caches
    │
    ├─> PersistentBatchManager()   # Request state tracking
    │
    └─> CompilationManager()       # Graph compilation
```

**Execution Flow:**

```
execute_model(scheduler_output)
    │
    ├─> _prepare_inputs()
    │   ├─> Build token_ids array
    │   ├─> Build positions array
    │   └─> Build attention metadata
    │
    ├─> model(input_ids, positions, attn_metadata)
    │   └─> Returns hidden_states, logits
    │
    └─> Return AsyncTPUModelRunnerOutput
        └─> .get_output() triggers D2H transfer
```

### Async Output (`core/core_tpu.py`)

```python
class AsyncTPUModelRunnerOutput:
    """Async wrapper for device arrays

    - Holds JAX device arrays (on TPU)
    - get_output() triggers device-to-host transfer
    - Enables CPU-GPU pipeline overlap
    """

class ExecuteModelState:
    """Ephemeral state between execute_model() and sample_tokens()

    Fields:
    - scheduler_output: vLLM scheduler result
    - attn_metadata: attention execution context
    - input_ids: token array
    - hidden_states: model output
    - logits: pre-sampling scores
    """
```

### Disaggregation Orchestrator (`core/disagg_executor.py`)

Multi-worker architecture for separating prefill and decode:

```
Driver Process
    │
    ├─> Prefill Engines (workers 0-N)
    │   └─> PrefillThread → TransferThread
    │       - High throughput token processing
    │       - KV cache generation
    │
    ├─> Decode Engines (workers N+1-M)
    │   └─> DecodeThread
    │       - Low latency auto-regressive
    │       - Continuous batching
    │
    └─> Output Queue
        - Collects completed sequences
```

```python
class _DisaggOrchestrator:
    """Manages prefill/decode separation

    Key methods:
    - _prefill_thread(): Run prefill stage
    - _transfer_thread(): Move KV cache between stages
    - _decode_thread(): Run decode stage
    - add_request(): Route new requests

    Backlog management:
    - HBM-aware decode queue sizing
    - Per-worker memory monitoring
    """
```

### Batch Management (`runner/input_batch.py`)

```python
class InputBatch:
    """Manages batched requests

    Fields:
    - token_ids_cpu: (max_reqs, max_model_len) array
    - num_tokens, num_prompt_tokens: counts
    - block_table: MultiGroupBlockTable for KV cache
    - sampling_params: temperature, top_p, top_k
    """

class CachedRequestState:
    """Per-request cached state

    Fields:
    - prompt_token_ids: input tokens
    - output_token_ids: generated tokens
    - sampling_params: generation config
    - generator: stateful RNG
    """
```

### KV Cache Management (`runner/kv_cache_manager.py`)

```python
class KVCacheManager:
    """KV cache allocation and configuration

    Methods:
    - create_kv_caches(): Allocate cache arrays
    - get_attention_page_size_bytes(): Size calculation

    Attention types:
    - Full attention
    - Sliding window
    - MLA (Multi-head Latent Attention)
    """
```

---

## Sharding & Distributed Execution

### Mesh Configuration

The system supports both 2D and 4D mesh configurations:

**2D Mesh (Default):**
```python
MESH_AXIS_NAMES_2D = ("data", "model")

# Typical shape: (1, 8) for 8 TPU devices
mesh = Mesh(devices, ("data", "model"))
```

**4D Mesh (Advanced):**
```python
MESH_AXIS_NAMES = ("data", "attn_dp", "expert", "model")

# Typical shape: (1, 1, 2, 4) for 8 TPU devices
mesh = Mesh(devices, ("data", "attn_dp", "expert", "model"))
```

### Sharding Strategy

```python
@dataclass
class ShardingStrategy:
    tensor_parallelism: int = 1    # Weight sharding
    expert_parallelism: int = 1    # MoE expert distribution
    sequence_parallelism: int = 1  # Activation sharding
    data_parallelism: int = 1      # Batch splitting
    attention_data_parallelism: int = 1  # KV head distribution
```

### Pipeline Parallelism (`distributed/jax_parallel_state.py`)

```python
class GroupCoordinator:
    """Pipeline parallel coordination

    Methods:
    - send_tensor_dict(): Transfer to next PP stage
    - recv_tensor_dict(): Receive from previous PP stage

    Properties:
    - rank_in_group: Position in PP group
    - is_first_rank, is_last_rank: Boundary detection
    """
```

---

## Kernel Implementations

### Fused MoE Kernel (`kernels/fused_moe/v1/kernel.py`)

TPU-specific optimized kernel:

```python
def fused_ep_moe(
    hidden_states,    # Input activations
    w1, w2, w3,       # Expert weights
    topk_weights,     # Router weights
    topk_ids,         # Selected expert indices
    activation,       # "silu", "gelu", "swigluoai"
    mesh,             # Device mesh
    ...
):
    """Fused expert-parallel MoE kernel

    Requirements:
    - 2D mesh only (len(mesh.shape) == 2)
    - Supports quantization (scales for weights)
    - Custom Pallas kernel for TPU

    Optimizations:
    - Fused gate/up/down projections
    - Tuned block sizes per TPU type
    - Sub-character quantization support
    """
```

### Grouped Matrix Multiply (`layers/common/fused_moe_gmm.py`)

```python
def gmm_fn(
    activations,   # (tokens, hidden)
    weights,       # (experts, hidden, intermediate) or similar
    group_sizes,   # tokens per expert
    ...
):
    """Grouped matrix multiply for MoE

    Dispatches to:
    - JAX einsum (dense)
    - Pallas GMM (sparse, optimized)
    - QWIX ragged dot (quantized)
    """
```

---

## Environment Configuration

### Key Environment Variables

**Model Implementation:**
```bash
MODEL_IMPL_TYPE=auto|vllm|flax_nnx|jetpack  # Backend selection
```

**Sharding:**
```bash
USE_2D_TP=true/false        # 2D tensor parallelism
NEW_MODEL_DESIGN=true/false # 4D mesh architecture
```

**MoE Backend:**
```bash
USE_MOE_EP_KERNEL=true      # Fused MoE kernel
USE_VLLM_MOE_KERNEL=true    # vLLM MoE kernel
USE_MEGABLOCKS=true         # Megablocks backend
USE_RAGGED_DOT=true         # QWIX ragged dot
```

**Disaggregation:**
```bash
PREFILL_SLICES=0,1,2        # Prefill worker indices
DECODE_SLICES=3,4,5         # Decode worker indices
NUM_SLICES=6                # Total slices
```

**Compilation:**
```bash
SKIP_JAX_PRECOMPILE=false              # Skip warmup compilation
VLLM_XLA_CHECK_RECOMPILATION=true      # Track recompilations
VLLM_XLA_CACHE_PATH=/path/to/cache     # XLA cache directory
```

---

## Request Flow Walkthrough

### End-to-End Request Processing

```
1. Request Arrives (vLLM API)
   └─> vllm serve receives HTTP request

2. Scheduling
   └─> Scheduler batches multiple requests
   └─> Determines prefill vs decode

3. Input Preparation (TPUModelRunner._prepare_inputs)
   ├─> token_ids: padded token array
   ├─> positions: position indices
   └─> attn_metadata: mask, KV cache pointers

4. Model Execution (TPUModelRunner.execute_model)
   ├─> Embedding lookup
   ├─> For each layer:
   │   ├─> Attention (MLA for DeepSeek)
   │   │   └─> KV cache read/write
   │   └─> FFN/MoE
   │       ├─> Router selects experts
   │       ├─> Backend executes experts
   │       └─> Combine expert outputs
   └─> Final layer norm + LM head

5. Async Output (AsyncTPUModelRunnerOutput)
   └─> Device arrays remain on TPU
   └─> Execution returns immediately

6. Sampling (TPUModelRunner.sample_tokens)
   ├─> get_output() transfers logits to host
   ├─> Apply temperature, top-k, top-p
   └─> Sample next token

7. Output Delivery
   └─> Tokens returned to vLLM
   └─> Streaming to client
```

### MoE Layer Execution Detail

```
MoE Layer Forward (for DeepSeek V3)
│
├─> Shared Experts (always active)
│   └─> Dense FFN: x → gate → up → activation → down
│
└─> Routed Experts (256 experts, 8 selected)
    │
    ├─> Router (DeepSeekV3Router)
    │   ├─> Linear projection: (T,D) @ (D,E) → (T,E)
    │   ├─> Grouped selection: top-k within groups
    │   └─> e_score correction bias
    │
    ├─> Backend Selection
    │   └─> Based on env vars → FUSED_MOE, MEGABLOX_GMM, etc.
    │
    └─> Expert Execution (sparse_moe_distributed_fwd)
        ├─> Global permute: sort tokens by expert
        ├─> All-to-all: exchange tokens between devices
        ├─> Local permute: route to local experts
        ├─> GMM: grouped matrix multiply
        │   ├─> Gate: (t,D) @ (E,D,F) → (t,F)
        │   ├─> Up: (t,D) @ (E,D,F) → (t,F)
        │   ├─> Activation: SiLU(gate) * up
        │   └─> Down: (t,F) @ (E,F,D) → (t,D)
        └─> Unpermute: restore order + weighted combine
```

---

## Tests Structure

```
tests/
├── core/                     # Core engine tests
│   ├── test_core_tpu.py
│   └── test_disagg_executor.py
│
├── distributed/              # Distributed tests
│   └── test_distributed_utils.py
│
├── e2e/                      # End-to-end tests
│   ├── test_data_parallel.py
│   ├── test_pipeline_parallel.py
│   ├── test_speculative_decoding.py
│   └── test_multi_modal_inference.py
│
├── kernels/                  # Kernel tests
│   ├── fused_moe_v1_test.py
│   └── gmm_test.py
│
└── layers/jax/moe/           # MoE layer tests
    ├── test_deepseek_moe.py
    └── test_sharding.py
```

---

## Key File Relationships Summary

```
Request Entry Point
└─> runner/tpu_runner.py:TPUModelRunner

Model Loading
├─> models/common/model_loader.py:_get_nnx_model()
└─> models/jax/*.py:Model classes

Layer Execution
├─> layers/jax/attention/*.py:Attention
├─> layers/jax/moe/moe.py:MoE
│   └─> kernels/fused_moe/v1/kernel.py:fused_ep_moe()
└─> layers/jax/layers.py:RMSNorm

State Management
├─> runner/persistent_batch_manager.py
├─> runner/kv_cache_manager.py
└─> runner/input_batch.py

Distributed
├─> layers/common/sharding.py:ShardingConfigManager
└─> distributed/jax_parallel_state.py:GroupCoordinator

Serving
├─> core/core_tpu.py:AsyncTPUModelRunnerOutput
└─> core/disagg_executor.py:_DisaggOrchestrator
```

---

## Further Reading

- [DeepSeek MoE and 2D Tensor Parallelism](deepseek_moe_2d_tp.md) - Detailed MoE sharding strategies
- vLLM documentation for API integration
- JAX documentation for distributed execution patterns
