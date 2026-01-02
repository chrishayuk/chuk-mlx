# Getting Started with Lazarus

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/chuk-mlx.git
cd chuk-mlx

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### Load a Model and Generate Text

```python
from chuk_lazarus.inference import UnifiedPipeline

# One-liner model loading - auto-detects family
pipeline = UnifiedPipeline.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Generate text
result = pipeline.chat("Once upon a time")
print(result.text)
print(result.stats.summary)  # "25 tokens in 0.42s (59.5 tok/s)"
```

### Fine-tune with LoRA

```python
from chuk_lazarus.training import SFTTrainer, SFTConfig
from chuk_lazarus.data import SFTDataset

# Create dataset
dataset = SFTDataset("./data/train.jsonl", tokenizer)

# Train with CLI (recommended)
# lazarus train sft --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data train.jsonl --use-lora
```

## Using the CLI

Lazarus includes a CLI for common operations:

```bash
# Inference
lazarus infer --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Hello!"

# Training
lazarus train sft --model model-name --data train.jsonl --use-lora
lazarus train dpo --model ./checkpoints/sft/final --data preferences.jsonl

# Data generation
lazarus generate --type math --output ./data/lazarus

# Tokenizer utilities
lazarus tokenizer encode -t model-name --text "Hello world"
lazarus tokenizer decode -t model-name --ids "1,2,3"
lazarus tokenizer vocab -t model-name --search "hello"
lazarus tokenizer compare -t1 model1 -t2 model2 --text "Test"
```

See [CLI Reference](cli.md) for full documentation.

## Supported Models

- LLaMA / LLaMA 2 / LLaMA 3
- Mistral
- Gemma
- Granite
- StarCoder2
- TinyLlama

## Next Steps

- See `examples/` for more detailed examples
- Check [Training Guide](training.md) for training details
- Check [CLI Reference](cli.md) for command-line usage
- Read [API Reference](api-reference.md) for API documentation
