# Models v2

A composable, Pydantic-native model framework for building and training language models on MLX.

Models v2 treats **model architecture as a composable system**, not a monolith.

## Architecture

```
Components -> Blocks -> Backbones -> Heads -> Models
                              \       /
                               Families
```

## Quick Start

```python
from chuk_lazarus.models_v2 import (
    LlamaConfig,
    LlamaForCausalLM,
    MambaConfig,
    MambaForCausalLM,
)
import mlx.core as mx

# Llama model
config = LlamaConfig.tiny()  # or llama2_7b(), llama3_8b(), mistral_7b()
model = LlamaForCausalLM(config)

# Forward pass
output = model(mx.array([[1, 2, 3, 4, 5]]))
print(f"Logits: {output.logits.shape}")

# Generate text
generated = model.generate(
    input_ids=mx.array([[1, 2, 3]]),
    max_new_tokens=50,
    temperature=0.7,
)

# Mamba model (state-space, O(n) complexity)
mamba_config = MambaConfig.mamba_130m()  # or mamba_370m(), mamba_1_4b()
mamba_model = MambaForCausalLM(mamba_config)
```

## Inference with Pretrained Models

Run inference with HuggingFace models:

```bash
# Using the Llama family inference example
uv run python examples/models/llama/03_llama_family_inference.py --model tinyllama
uv run python examples/models/llama/03_llama_family_inference.py --model smollm2-360m
uv run python examples/models/llama/03_llama_family_inference.py --model smollm2-1.7b

# List all available presets
uv run python examples/models/llama/03_llama_family_inference.py --list-models
```

Available model presets: `tinyllama`, `smollm2-135m`, `smollm2-360m`, `smollm2-1.7b`, `llama3.2-1b`, `llama3.2-3b`, `mistral-7b`

See [docs/inference.md](../../../docs/inference.md) for detailed inference documentation.

## Model Presets

### Llama Family
- `LlamaConfig.tiny()` - Testing
- `LlamaConfig.llama2_7b()`, `llama2_13b()`, `llama2_70b()`
- `LlamaConfig.llama3_8b()`, `llama3_70b()`
- `LlamaConfig.mistral_7b()`
- `LlamaConfig.code_llama_7b()`

### Mamba Family
- `MambaConfig.tiny()` - Testing
- `MambaConfig.mamba_130m()`, `mamba_370m()`, `mamba_790m()`
- `MambaConfig.mamba_1_4b()`, `mamba_2_8b()`

## Submodules

| Module | Purpose |
|--------|---------|
| `core/` | Backend abstraction, configs, enums, registry |
| `components/` | Attention, FFN, embeddings, SSM, recurrent |
| `blocks/` | TransformerBlock, MambaBlock, HybridBlock |
| `backbones/` | TransformerBackbone, MambaBackbone, HybridBackbone |
| `heads/` | LMHead, ClassifierHead, RegressionHead |
| `models/` | CausalLM, SequenceClassifier, TokenClassifier |
| `families/` | LlamaForCausalLM, MambaForCausalLM |
| `adapters/` | LoRA adapters for efficient fine-tuning |
| `losses/` | Loss functions (pure math) |
| `loader.py` | Async model loading |

## Design Principles

- **Pydantic-native**: Configs use BaseModel with frozen=True for validation and serialization
- **Async-native**: Model loading, checkpoint I/O, and external resources are async-safe
- **No magic strings**: Enums for type safety (AttentionType, FFNType, NormType, etc.)
- **No dictionary goop**: Structured output types (ModelOutput, BlockOutput, BackboneOutput)
- **Backend-agnostic by design**: Core abstractions are backend-neutral; MLX is the reference implementation

## Documentation

See [docs/models.md](../../../docs/models.md) for comprehensive documentation.

## Testing

```bash
pytest tests/models_v2/ -v --cov=src/chuk_lazarus/models_v2
```
