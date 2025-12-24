# Models v2

A composable, async-native, Pydantic-native model framework for building and training language models on MLX.

## Architecture

```
Components -> Blocks -> Backbones -> Heads -> Models
                              \       /
                               Families
```

## Quick Start

```python
from chuk_lazarus import (
    LlamaConfig,
    LlamaForCausalLM,
    LoRAConfig,
    apply_lora,
)

# Create model
config = LlamaConfig.tiny()
model = LlamaForCausalLM(config)

# Apply LoRA
lora_layers = apply_lora(model, LoRAConfig(rank=8))

# Forward pass
import mlx.core as mx
output = model(mx.array([[1, 2, 3, 4, 5]]))
```

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
| `training/` | Loss functions |
| `loader.py` | Async model loading |

## Design Principles

- **Pydantic-native**: Configs use BaseModel with frozen=True
- **Async-native**: All I/O is async
- **No magic strings**: Enums for type safety
- **No dictionary goop**: Structured output types
- **Backend-agnostic**: Works on MLX, PyTorch, JAX

## Documentation

See [docs/models.md](../../../docs/models.md) for comprehensive documentation.

## Testing

```bash
pytest tests/models_v2/ -v --cov=src/chuk_lazarus/models_v2
```
