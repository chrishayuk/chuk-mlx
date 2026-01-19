# GPT-OSS-Lite

Memory-efficient variants of OpenAI's GPT-OSS model with reduced expert counts per layer.

## Overview

GPT-OSS is a 20B parameter Mixture of Experts (MoE) model. The original model has 32 experts per layer with top-k=4 routing. This project creates "lite" versions with fewer experts to reduce memory requirements.

## Model Configurations

| Configuration | Experts/Layer | Total Experts | Memory | Quality |
|--------------|---------------|---------------|--------|---------|
| Original | 32 | 768 | ~16GB | Full |
| **gpt-oss-lite-16exp** | 16 | 384 | ~8GB | Degraded* |
| gpt-oss-lite-8exp | 8 | 192 | ~5.7GB | Degraded |
| gpt-oss-lite-6exp | 6 | 144 | ~5.1GB | Degraded |
| gpt-oss-lite-minimal | 4 | 96 | ~4.5GB | Degraded |

**\*Quality Note**: All lite models show degraded quality for certain tasks, particularly math. The 16-expert model predicts "3" for "2+2=" instead of "4". This is due to expert pruning removing essential experts for math computation.

## Key Findings

### 1. Expert Pruning Degrades Quality

The expert selection algorithm preserves the most frequently activated experts across diverse prompts, but this approach does not preserve all capabilities:

```
Prompt: "2 + 2 ="
Original (32 experts): "4" ✓
Lite (16 experts): "3" ✗
```

### 2. Hidden State Explosion is Normal

Both original and lite models show hidden states exploding at layer 6 (std ~7 → ~340). This is normal for the GPT-OSS architecture - the final RMSNorm normalizes the output.

### 3. Architecture Details

- **Attention**: Uses GQA with 64 query heads and 8 key/value heads
- **Attention Biases**: Original model has attention biases; lite safetensors missing them
- **Attention Sinks**: Used for improved long-context attention
- **Activation**: Custom GPT-OSS SwiGLU: `(x_glu * sigmoid(1.702 * x_glu)) * (x_linear + 1)`
- **Quantization**: MXFP4 format (4-bit with 32-element groups)
- **RoPE**: YaRN scaling for extended context

## Usage

### Loading the Model

```python
from lite_loader_optimized import load_gpt_oss_lite_optimized
from transformers import AutoTokenizer

model, args = load_gpt_oss_lite_optimized("./gpt-oss-lite-16exp")
tokenizer = AutoTokenizer.from_pretrained("./gpt-oss-lite-16exp")
```

### Generation

```python
from mlx_lm import generate

result = generate(model, tokenizer, prompt="The capital of France is", max_tokens=20)
print(result)
```

### Benchmarking

```bash
python benchmark_simple.py
```

## File Structure

```
gpt-oss-lite-v2/
├── README.md                    # This file
├── lite_loader_optimized.py     # Main loader (uses mlx_lm model + patched experts)
├── build_minimal_lite.py        # Script to build lite models from original
├── benchmark_simple.py          # Memory/storage benchmark
├── weights.npz                  # 16-expert weights (includes attention biases)
├── config.json                  # Base configuration
├── tokenizer.json               # Tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
└── gpt-oss-lite-16exp/          # 16-expert model (recommended)
    ├── config.json
    ├── model.safetensors        # Note: Missing attention biases
    ├── tokenizer.json
    └── ...
```

## Known Issues

### 1. Safetensors Missing Attention Biases

The `model.safetensors` files in the model directories are missing:
- Attention projection biases (`q_proj.bias`, `k_proj.bias`, `v_proj.bias`, `o_proj.bias`)
- Attention sinks (`self_attn.sinks`)

This is a bug in `build_minimal_lite.py`. The `weights.npz` file has the complete weights.

**Workaround**: Use `lite_loader_optimized.py` which loads from `weights.npz`.

### 2. Quality Degradation

Expert pruning degrades model quality, especially for:
- Math operations
- Precise factual recall
- Complex reasoning

This is fundamental to the approach - removing experts removes capabilities.

## Recommendations

1. **For production use**: Use the original 32-expert model if quality is important
2. **For memory-constrained environments**: The 16-expert model provides ~50% memory reduction with some quality loss
3. **For research**: The architecture and approach are correct; improve expert selection to preserve capabilities

## Technical Details

### MXFP4 Quantization

Weights are stored in MXFP4 format:
- `weight`: uint32 (packed 4-bit values, 8 per uint32)
- `scales`: uint8 (one scale per 32 elements)
- `bias`: bfloat16 (output bias)

### Custom SwiGLU Activation

```python
def gpt_oss_swiglu(x_linear, x_glu, alpha=1.702, limit=7.0):
    x_glu = clip(x_glu, max=limit)
    x_linear = clip(x_linear, min=-limit, max=limit)
    out_glu = x_glu * sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)
```

Key differences from standard SwiGLU:
- Alpha scaling (1.702) on sigmoid input
- Asymmetric clamping
- +1 bias on linear path

### Layer Types

GPT-OSS alternates between:
- `sliding_attention`: Window size 128 (odd layers)
- `full_attention`: Full context (even layers)

## License

See original GPT-OSS license from OpenAI.
