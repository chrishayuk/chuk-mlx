# Models Module

A composable, async-native, Pydantic-native model framework for building and training language models on MLX. **Designed for extensibility** - mix transformers, state-space models, and recurrent architectures in hybrid configurations.

## Overview

This module provides:
- **Composable architecture** - Components, blocks, backbones, heads, and models
- **Multiple model families** - Transformers (Llama, Mistral, Gemma), SSMs (Mamba), Recurrent (LSTM, GRU, MinGRU)
- **Hybrid architectures** - Mix attention and SSM layers in a single model
- **LoRA adapters** - Efficient fine-tuning with low-rank adaptation
- **Backend abstraction** - Works on MLX (Apple Silicon) with PyTorch fallback
- **Async model loading** - Non-blocking weight loading from local files or HuggingFace Hub
- **Preset configurations** - Ready-to-use configs for popular models

### Architecture Hierarchy

```
Components -> Blocks -> Backbones -> Heads -> Models
                              \       /
                               Families
```

| Layer | Purpose | Examples |
|-------|---------|----------|
| **Components** | Building blocks | Attention, FFN, Normalization, Embeddings, SSM, Recurrent |
| **Blocks** | Layer-level units | TransformerBlock, MambaBlock, RecurrentBlock, HybridBlock |
| **Backbones** | Stack of blocks | TransformerBackbone, MambaBackbone, HybridBackbone |
| **Heads** | Task-specific outputs | LMHead, ClassifierHead, RegressionHead |
| **Models** | Complete end-to-end | CausalLM, SequenceClassifier, TokenClassifier |
| **Families** | Architecture-specific | LlamaForCausalLM, MambaForCausalLM |

## Design Principles

- **Pydantic-native**: All configs use Pydantic BaseModel with frozen=True for validation
- **Async-native**: All I/O operations (loading, saving) are async
- **No magic strings**: Enums for type safety (AttentionType, FFNType, NormType, etc.)
- **No dictionary goop**: Structured ModelOutput, BlockOutput, BackboneOutput types
- **Backend-agnostic**: Abstractions work across MLX, PyTorch, JAX
- **Composable**: Mix and match components to create custom architectures

## Quick Start

```python
from chuk_lazarus import (
    LlamaConfig,
    LlamaForCausalLM,
    LoRAConfig,
    apply_lora,
    load_model,
)

# Create a model from config
config = LlamaConfig.tiny()  # 42M parameters for testing
model = LlamaForCausalLM(config)

# Or load from HuggingFace Hub
model = await load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Forward pass
import mlx.core as mx
tokens = mx.array([[1, 2, 3, 4, 5]])
output = model(tokens)
print(f"Logits shape: {output.logits.shape}")

# Apply LoRA for efficient fine-tuning
lora_config = LoRAConfig(rank=8, alpha=16.0)
lora_layers = apply_lora(model, lora_config)
print(f"LoRA parameters: {sum(l.lora_A.size + l.lora_B.size for l in lora_layers.values()):,}")
```

## Core Modules

### `core/` - Foundation

Configuration, enums, backend abstraction, and model registry.

#### Enums

```python
from chuk_lazarus.models_v2 import (
    AttentionType,    # MHA, GQA, MQA, SLIDING_WINDOW
    FFNType,          # MLP, SWIGLU, GEGLU, MOE
    NormType,         # RMSNORM, LAYERNORM, GEMMA_NORM
    ActivationType,   # RELU, GELU, SILU, SWISH, TANH
    BlockType,        # TRANSFORMER, MAMBA, RECURRENT, HYBRID
    BackboneType,     # TRANSFORMER, MAMBA, RECURRENT, HYBRID
    HeadType,         # LM, CLASSIFIER, REGRESSION
    PositionEmbeddingType,  # ROPE, ALIBI, LEARNED, SINUSOIDAL
    BackendType,      # MLX, TORCH, JAX
    ModelMode,        # TRAIN, EVAL, GENERATE
)
```

#### Configuration

```python
from chuk_lazarus.models_v2 import (
    ModelConfig,
    AttentionConfig,
    FFNConfig,
    NormConfig,
    SSMConfig,
)

# Create a full model config
config = ModelConfig(
    vocab_size=32000,
    hidden_size=2048,
    num_layers=24,
    num_attention_heads=16,
    num_kv_heads=4,  # GQA with 4 KV heads
    intermediate_size=5632,
    max_position_embeddings=4096,
    rope_theta=10000.0,
    attention=AttentionConfig(type=AttentionType.GQA),
    ffn=FFNConfig(type=FFNType.SWIGLU),
    norm=NormConfig(type=NormType.RMSNORM, eps=1e-5),
)
```

#### Backend Abstraction

```python
from chuk_lazarus.models_v2 import (
    get_backend,
    set_backend,
    BackendType,
)

# Get current backend (auto-detects MLX on macOS)
backend = get_backend()
print(f"Using: {backend.name}")  # BackendType.MLX

# Create tensors through backend
zeros = backend.zeros((2, 3))
ones = backend.ones((2, 3))
randn = backend.randn((2, 3))

# Set backend explicitly
set_backend(BackendType.MLX)
set_backend("torch")  # Also accepts strings
```

#### Model Registry

```python
from chuk_lazarus.models_v2 import (
    register_model,
    get_factory,
    list_models,
)

# List registered models
models = list_models()
# ['llama', 'mamba', 'mistral', 'gemma', ...]

# Get factory function
factory = get_factory("llama")
model = factory(config)

# Register custom model
@register_model("my_model")
def create_my_model(config):
    return MyCustomModel(config)
```

## Components

Reusable building blocks for model architectures.

### Embeddings

```python
from chuk_lazarus.models_v2 import (
    TokenEmbedding,
    RoPE,
    ALiBi,
    LearnedPositionEmbedding,
    SinusoidalPositionEmbedding,
)

# Token embeddings
embed = TokenEmbedding(vocab_size=32000, hidden_size=2048)
embedded = embed(token_ids)

# Or from pretrained weights
embed = TokenEmbedding.from_pretrained(weight_matrix)

# Rotary Position Embeddings (RoPE)
rope = RoPE(dim=64, max_seq_len=4096, theta=10000.0)
cos, sin = rope(seq_len=512)

# ALiBi (Attention with Linear Biases)
alibi = ALiBi(num_heads=16)
bias = alibi(seq_len=512)
```

### Attention

```python
from chuk_lazarus.models_v2 import (
    MultiHeadAttention,
    GroupedQueryAttention,
    SlidingWindowAttention,
)

# Multi-Head Attention
mha = MultiHeadAttention(
    hidden_size=2048,
    num_heads=16,
    head_dim=128,
)
output = mha(x, mask=mask)

# Grouped Query Attention (memory efficient)
gqa = GroupedQueryAttention(
    hidden_size=2048,
    num_heads=16,
    num_kv_heads=4,  # 4x fewer KV heads
    head_dim=128,
)
output = gqa(x, mask=mask)

# Sliding Window Attention (for long sequences)
swa = SlidingWindowAttention(
    hidden_size=2048,
    num_heads=16,
    window_size=4096,
)
output = swa(x, mask=mask)
```

### Feed-Forward Networks

```python
from chuk_lazarus.models_v2 import MLP, SwiGLU, GEGLU, MoE

# Standard MLP
mlp = MLP(
    hidden_size=2048,
    intermediate_size=5632,
    activation=ActivationType.GELU,
)
output = mlp(x)

# SwiGLU (Llama-style)
swiglu = SwiGLU(hidden_size=2048, intermediate_size=5632)
output = swiglu(x)

# GEGLU
geglu = GEGLU(hidden_size=2048, intermediate_size=5632)
output = geglu(x)

# Mixture of Experts
moe = MoE(
    hidden_size=2048,
    intermediate_size=5632,
    num_experts=8,
    num_experts_per_tok=2,
)
output, router_logits = moe(x)
```

### Normalization

```python
from chuk_lazarus.models_v2 import RMSNorm, LayerNorm, GemmaNorm

# RMSNorm (Llama-style)
norm = RMSNorm(dim=2048, eps=1e-5)
output = norm(x)

# LayerNorm
norm = LayerNorm(dim=2048, eps=1e-5)
output = norm(x)

# GemmaNorm (RMSNorm + 1)
norm = GemmaNorm(dim=2048, eps=1e-6)
output = norm(x)
```

### State Space Models (Mamba)

```python
from chuk_lazarus.models_v2 import Mamba, MambaBlock, SelectiveSSM

# Full Mamba layer
mamba = Mamba(
    d_model=2048,
    d_state=16,
    d_conv=4,
    expand=2,
)
output = mamba(x)

# Mamba block (with normalization)
block = MambaBlock(
    d_model=2048,
    d_state=16,
    d_conv=4,
)
output = block(x)

# Low-level selective SSM
ssm = SelectiveSSM(
    d_model=2048,
    d_state=16,
)
output = ssm(x, delta, A, B, C)
```

### Recurrent Cells

```python
from chuk_lazarus.models_v2 import LSTM, GRU, MinGRU

# LSTM
lstm = LSTM(input_size=2048, hidden_size=2048)
output, (h_n, c_n) = lstm(x, initial_state)

# GRU
gru = GRU(input_size=2048, hidden_size=2048)
output, h_n = gru(x, initial_state)

# MinGRU (minimal gated recurrent unit)
mingru = MinGRU(input_size=2048, hidden_size=2048)
output, h_n = mingru(x, initial_state)
```

## Blocks

Layer-level units that combine components.

```python
from chuk_lazarus.models_v2 import (
    TransformerBlock,
    MambaBlockWrapper,
    RecurrentBlockWrapper,
    HybridBlock,
    BlockOutput,
)

# Transformer block (attention + FFN)
block = TransformerBlock(
    hidden_size=2048,
    num_heads=16,
    num_kv_heads=4,
    intermediate_size=5632,
    norm_type=NormType.RMSNORM,
    ffn_type=FFNType.SWIGLU,
)
output: BlockOutput = block(x, mask=mask, position_ids=position_ids)
print(f"Hidden: {output.hidden_states.shape}")

# Mamba block
mamba_block = MambaBlockWrapper(
    d_model=2048,
    d_state=16,
    d_conv=4,
)
output = mamba_block(x)

# Hybrid block (attention + SSM)
hybrid = HybridBlock(
    hidden_size=2048,
    num_heads=16,
    d_state=16,
    use_attention=True,
    use_ssm=True,
)
output = hybrid(x, mask=mask)
```

## Backbones

Stacks of blocks with embeddings.

```python
from chuk_lazarus.models_v2 import (
    TransformerBackbone,
    MambaBackbone,
    RecurrentBackbone,
    HybridBackbone,
    BackboneOutput,
)

# Transformer backbone
backbone = TransformerBackbone(config)
output: BackboneOutput = backbone(token_ids, mask=mask)
print(f"Last hidden: {output.last_hidden_state.shape}")
print(f"All hidden: {len(output.hidden_states)}")

# Mamba backbone
backbone = MambaBackbone(config)
output = backbone(token_ids)

# Hybrid backbone (interleaved attention + SSM)
backbone = HybridBackbone(
    config,
    attention_layers=[0, 2, 4, 6],  # Attention at these layers
    ssm_layers=[1, 3, 5, 7],        # SSM at these layers
)
output = backbone(token_ids)
```

## Heads

Task-specific output layers.

```python
from chuk_lazarus.models_v2 import (
    LMHead,
    ClassifierHead,
    RegressionHead,
    HeadOutput,
)

# Language modeling head (ties with embeddings)
lm_head = LMHead(hidden_size=2048, vocab_size=32000)
logits = lm_head(hidden_states)

# Classification head
classifier = ClassifierHead(
    hidden_size=2048,
    num_labels=5,
    pooling="mean",  # or "first", "last", "cls"
)
output: HeadOutput = classifier(hidden_states)
print(f"Logits: {output.logits.shape}")

# Regression head
regressor = RegressionHead(
    hidden_size=2048,
    num_outputs=1,
)
output = regressor(hidden_states)
```

## Models

Complete end-to-end model architectures.

```python
from chuk_lazarus.models_v2 import (
    Model,
    ModelOutput,
    CausalLM,
    SequenceClassifier,
    TokenClassifier,
)

# Causal Language Model
model = CausalLM(config)
output: ModelOutput = model(token_ids, labels=labels)
print(f"Logits: {output.logits.shape}")
print(f"Loss: {output.loss}")

# Sequence Classifier
classifier = SequenceClassifier(config, num_labels=5)
output = classifier(token_ids)
print(f"Predictions: {output.logits.argmax(-1)}")

# Token Classifier (NER, POS tagging)
token_clf = TokenClassifier(config, num_labels=9)
output = token_clf(token_ids)
print(f"Per-token logits: {output.logits.shape}")
```

### Standalone Classifiers

Simple classifiers for use without a full backbone:

```python
from chuk_lazarus.models_v2.models.classifiers import (
    LinearClassifier,
    MLPClassifier,
    create_classifier,
)

# LinearClassifier - single linear layer
linear_clf = LinearClassifier(
    input_dim=768,
    num_classes=5,
    bias=True,
)
logits = linear_clf(hidden_states)  # (batch, 5)

# MLPClassifier - MLP with hidden layers
mlp_clf = MLPClassifier(
    input_dim=768,
    hidden_dim=256,
    num_classes=5,
    num_layers=2,
    dropout=0.1,
    activation="gelu",
)
logits = mlp_clf(hidden_states)  # (batch, 5)

# Factory function for easy creation
clf = create_classifier(
    classifier_type="mlp",  # or "linear"
    input_dim=768,
    num_classes=10,
    hidden_dim=512,
    num_layers=3,
)
```

| Classifier | Parameters | Use Case |
|------------|------------|----------|
| `LinearClassifier` | input_dim × num_classes | Simple classification, probing |
| `MLPClassifier` | Multiple layers | Complex classification tasks |
| `SequenceClassifier` | Backbone + head | Full sequence classification |
| `TokenClassifier` | Backbone + per-token head | NER, POS tagging |

## Families

Architecture-specific implementations with preset configurations.

### Llama Family

```python
from chuk_lazarus.models_v2 import LlamaConfig, LlamaForCausalLM

# Preset configurations
config = LlamaConfig.tiny()        # Testing
config = LlamaConfig.llama2_7b()   # Llama 2 7B
config = LlamaConfig.llama2_13b()  # Llama 2 13B
config = LlamaConfig.llama2_70b()  # Llama 2 70B
config = LlamaConfig.llama3_8b()   # Llama 3 8B
config = LlamaConfig.llama3_70b()  # Llama 3 70B
config = LlamaConfig.mistral_7b()  # Mistral 7B
config = LlamaConfig.code_llama_7b() # Code Llama 7B

# Create model
model = LlamaForCausalLM(config)

# Forward pass with labels for training
output = model(token_ids, labels=labels)
print(f"Loss: {output.loss}")

# Generate text
generated = model.generate(
    input_ids=prompt_ids,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,              # Top-k sampling
    repetition_penalty=1.1, # Reduce repetition
    stop_tokens=[2],       # EOS token
)
```

### Mamba Family

```python
from chuk_lazarus.models_v2 import MambaConfig, MambaForCausalLM

# Preset configurations
config = MambaConfig.tiny()       # Testing
config = MambaConfig.mamba_130m() # 130M params
config = MambaConfig.mamba_370m() # 370M params
config = MambaConfig.mamba_790m() # 790M params
config = MambaConfig.mamba_1_4b() # 1.4B params
config = MambaConfig.mamba_2_8b() # 2.8B params

# Create model
model = MambaForCausalLM(config)

# Mamba is efficient for long sequences (O(n) complexity)
output = model(long_sequence_ids)

# Generate text
generated = model.generate(
    input_ids=prompt_ids,
    max_new_tokens=100,
    temperature=0.7,
)
```

### Gemma Family

```python
from chuk_lazarus.models_v2.families.gemma import GemmaConfig, GemmaForCausalLM

# Preset configurations
config = GemmaConfig.tiny()              # Testing
config = GemmaConfig.gemma3_270m()       # 270M params (FunctionGemma base)
config = GemmaConfig.functiongemma_270m() # Same as 270M, tuned for function calling
config = GemmaConfig.gemma3_1b()         # 1B params
config = GemmaConfig.gemma3_4b()         # 4B params
config = GemmaConfig.gemma3_12b()        # 12B params
config = GemmaConfig.gemma3_27b()        # 27B params

# Create model
model = GemmaForCausalLM(config)

# Forward pass
output = model(token_ids)

# Generate text
generated = model.generate(
    input_ids=prompt_ids,
    max_new_tokens=100,
    temperature=0.7,
)
```

#### Gemma Architecture Features

Gemma 3 has several unique architectural features:

- **Alternating sliding window / global attention**: Every Nth layer uses global attention (pattern configurable)
- **Query/Key pre-normalization**: Q and K projections have separate RMSNorm layers
- **4 normalization layers per block**: Pre-attn, post-attn, pre-ffn, post-ffn norms
- **Gated GELU activation**: Uses `gelu(gate) * up` pattern in FFN
- **Embedding scaling**: Hidden states scaled by √hidden_size
- **GemmaNorm**: RMSNorm with `(1 + weight)` scaling

```python
# Check which layers use sliding vs global attention
config = GemmaConfig.gemma3_270m()

for i in range(config.num_hidden_layers):
    if config.is_sliding_layer(i):
        print(f"Layer {i}: sliding window ({config.sliding_window} tokens)")
    else:
        print(f"Layer {i}: global attention")
```

## Model Loading

Async-native loading from local files or HuggingFace Hub.

```python
from chuk_lazarus.models_v2 import (
    load_model,
    load_model_async,
    create_model,
    create_from_preset,
)

# Async loading from HuggingFace Hub
model = await load_model_async("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Sync wrapper
model = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Create from config
model = create_model("llama", {
    "vocab_size": 32000,
    "hidden_size": 2048,
    "num_layers": 24,
})

# Create from preset
model = create_from_preset("llama3-8b")
model = create_from_preset("mistral-7b")
model = create_from_preset("mamba-1.4b")

# Load with specific dtype
model = await load_model_async(
    "meta-llama/Llama-2-7b-hf",
    dtype="bfloat16",
)

# Load local weights
model = await load_model_async(
    "./checkpoints/my_model",
    weights_path="./checkpoints/my_model/weights.safetensors",
)
```

## LoRA Adapters

Efficient fine-tuning with Low-Rank Adaptation.

```python
from chuk_lazarus.models_v2 import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    merge_lora_weights,
    count_lora_parameters,
)

# Configure LoRA
lora_config = LoRAConfig(
    rank=8,              # Low-rank dimension
    alpha=16.0,          # Scaling factor
    dropout=0.05,        # Dropout rate
    target_modules=[     # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# Apply LoRA to model
lora_layers = apply_lora(model, lora_config)
print(f"Added LoRA to {len(lora_layers)} layers")

# Count parameters
total_lora = count_lora_parameters(lora_layers)
print(f"LoRA parameters: {total_lora:,}")
print(f"Base parameters: {sum(p.size for p in model.parameters()):,}")
print(f"Trainable ratio: {total_lora / sum(p.size for p in model.parameters()):.2%}")

# Train with LoRA (only LoRA params are trainable)
# ... training loop ...

# Merge LoRA weights for inference (optional)
merge_lora_weights(model, lora_layers)
# Now model has merged weights, no LoRA overhead
```

## Training Utilities

```python
from chuk_lazarus.models_v2 import compute_lm_loss

# Compute cross-entropy loss with label smoothing
loss = compute_lm_loss(
    logits=output.logits,
    labels=labels,
    label_smoothing=0.1,
    ignore_index=-100,
)

# With loss mask for packed sequences
loss = compute_lm_loss(
    logits=output.logits,
    labels=labels,
    loss_mask=loss_mask,  # 1 for compute loss, 0 for ignore
)
```

## Architecture

```
models_v2/
├── __init__.py              # Public API (re-exports)
│
├── core/                    # Foundation
│   ├── backend.py           # Backend abstraction (MLX, Torch)
│   ├── config.py            # ModelConfig, AttentionConfig, etc.
│   ├── enums.py             # Type enums (AttentionType, FFNType, etc.)
│   ├── protocols.py         # ModelProtocol, BlockProtocol
│   └── registry.py          # Model registration and factory
│
├── components/              # Building blocks
│   ├── attention/           # MHA, GQA, Sliding Window
│   ├── embeddings/          # Token, RoPE, ALiBi, Learned, Sinusoidal
│   ├── ffn/                 # MLP, SwiGLU, GEGLU, MoE
│   ├── normalization/       # RMSNorm, LayerNorm, GemmaNorm
│   ├── ssm/                 # Mamba, SelectiveSSM
│   └── recurrent/           # LSTM, GRU, MinGRU
│
├── blocks/                  # Layer-level units
│   ├── base.py              # Block, BlockOutput
│   ├── transformer.py       # TransformerBlock
│   ├── mamba.py             # MambaBlockWrapper
│   ├── recurrent.py         # RecurrentBlockWrapper
│   └── hybrid.py            # HybridBlock
│
├── backbones/               # Stacks of blocks
│   ├── base.py              # Backbone, BackboneOutput
│   ├── transformer.py       # TransformerBackbone
│   ├── mamba.py             # MambaBackbone
│   ├── recurrent.py         # RecurrentBackbone
│   └── hybrid.py            # HybridBackbone
│
├── heads/                   # Task-specific outputs
│   ├── base.py              # Head, HeadOutput
│   ├── lm_head.py           # LMHead
│   ├── classifier.py        # ClassifierHead
│   └── regression.py        # RegressionHead
│
├── models/                  # Complete end-to-end
│   ├── base.py              # Model, ModelOutput
│   ├── causal_lm.py         # CausalLM
│   └── classifiers/         # Classification models
│       ├── linear.py        # LinearClassifier
│       ├── mlp.py           # MLPClassifier
│       ├── sequence.py      # SequenceClassifier
│       ├── token.py         # TokenClassifier
│       └── factory.py       # create_classifier()
│
├── families/                # Architecture-specific
│   ├── llama/               # LlamaConfig, LlamaForCausalLM
│   ├── mamba/               # MambaConfig, MambaForCausalLM
│   └── gemma/               # GemmaConfig, GemmaForCausalLM
│
├── adapters/                # Parameter-efficient fine-tuning
│   └── lora.py              # LoRAConfig, LoRALinear, apply_lora
│
├── losses/                  # Loss functions (pure math)
│   └── loss.py              # compute_lm_loss
│
└── loader.py                # load_model, load_model_async
```

## Extending the Framework

### Custom Component

```python
from chuk_lazarus.models_v2.components.attention.base import BaseAttention
import mlx.core as mx
import mlx.nn as nn

class LinearAttention(BaseAttention):
    """O(n) attention using kernel feature maps."""

    def __init__(self, hidden_size: int, num_heads: int, **kwargs):
        super().__init__(hidden_size, num_heads, **kwargs)
        self.feature_map = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x: mx.array, mask=None, **kwargs) -> mx.array:
        # Linear attention computation
        q = self.feature_map(self.q_proj(x))
        k = self.feature_map(self.k_proj(x))
        v = self.v_proj(x)

        # Kernel trick: softmax(QK^T) ≈ φ(Q)φ(K)^T
        kv = mx.einsum("bnd,bnv->bdv", k, v)
        qkv = mx.einsum("bnd,bdv->bnv", q, kv)

        return self.o_proj(qkv)
```

### Custom Block

```python
from chuk_lazarus.models_v2.blocks.base import Block, BlockOutput

class MyCustomBlock(Block):
    """Custom block with attention + FFN + residual."""

    def __init__(self, config):
        super().__init__()
        self.attn = LinearAttention(config.hidden_size, config.num_heads)
        self.ffn = SwiGLU(config.hidden_size, config.intermediate_size)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)

    def __call__(self, x, mask=None, **kwargs) -> BlockOutput:
        # Pre-norm residual
        h = x + self.attn(self.norm1(x), mask=mask)
        h = h + self.ffn(self.norm2(h))
        return BlockOutput(hidden_states=h)
```

### Register Custom Model

```python
from chuk_lazarus.models_v2 import register_model

@register_model("my_model")
def create_my_model(config):
    return MyCustomModel(config)

# Now usable via registry
model = create_model("my_model", config)
```

## Performance Tips

1. **Use GQA for memory efficiency**: Fewer KV heads reduce memory without much quality loss
2. **Use LoRA for fine-tuning**: 99%+ parameter reduction while maintaining quality
3. **Use async loading**: `load_model_async` doesn't block during I/O
4. **Use appropriate dtypes**: bfloat16 for training, float16 for inference
5. **Merge LoRA for inference**: Call `merge_lora_weights` to eliminate adapter overhead

## Testing

```bash
pytest tests/models_v2/ -v --cov=src/chuk_lazarus/models_v2
```

Coverage target: 90%+ per file
