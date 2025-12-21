# Lazarus

MLX-based LLM training framework for Apple Silicon.

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

### Python API

```python
from chuk_lazarus.models import load_model, generate_response

# Load a model
model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Generate text
response = generate_response(
    model=model,
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_tokens=100,
    temperature=0.7,
)
print(response)
```

### CLI

Lazarus provides a unified CLI for training, inference, and tokenizer utilities.

```bash
# Run inference
lazarus infer --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Hello, world!"

# Train with SFT
lazarus train sft --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data train.jsonl --use-lora

# Train with DPO
lazarus train dpo --model ./checkpoints/sft/final --data preferences.jsonl

# Generate synthetic training data
lazarus generate --type math --output ./data/lazarus

# Tokenizer utilities
lazarus tokenizer encode -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text "Hello world"
lazarus tokenizer vocab -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --search "hello"
```

## Features

- **Training**: SFT, DPO, GRPO, PPO trainers with LoRA support
- **Models**: LLaMA, Mistral, Gemma, Granite, StarCoder2, TinyLlama
- **Data**: Dataset classes, synthetic data generators
- **CLI**: Unified command-line interface for all operations
- **Tokenizer Tools**: Encode, decode, vocabulary inspection, comparison

## Documentation

- [Getting Started](docs/getting-started.md)
- [Training Guide](docs/training.md)
- [CLI Reference](docs/cli.md)
- [API Reference](docs/api-reference.md)

## Examples

See the `examples/` directory for detailed examples organized by area:

- `examples/tokenizer/` - Tokenization examples
- `examples/inference/` - Text generation
- `examples/training/` - SFT and DPO training
- `examples/data/` - Data handling and generation
- `examples/models/` - Model loading and configuration

## Project Structure

```
src/chuk_lazarus/
├── cli/            # Command-line interface
├── data/           # Datasets, generators, tokenizers
├── models/         # Model architectures and loading
├── training/       # Trainers (SFT, DPO, GRPO, PPO)
├── inference/      # Text generation
└── utils/          # Utilities
```

## License

MIT
