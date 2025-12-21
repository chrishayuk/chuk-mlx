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

### Load a Model

```python
from chuk_lazarus.models import load_model

model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

### Generate Text

```python
from chuk_lazarus.models import load_model, generate_response

model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

response = generate_response(
    model=model,
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_tokens=100,
    temperature=0.7,
)
print(response)
```

### Fine-tune with LoRA

```python
from chuk_lazarus.models import load_model
from chuk_lazarus.training import SFTTrainer, SFTConfig
from chuk_lazarus.data import SFTDataset

# Load model with LoRA
model, tokenizer = load_model(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    use_lora=True,
)

# Create dataset
dataset = SFTDataset("./data/train.jsonl", tokenizer)

# Train
config = SFTConfig(num_epochs=3, learning_rate=1e-5)
trainer = SFTTrainer(model, tokenizer, config)
trainer.train(dataset)
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
