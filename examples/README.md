# Lazarus Examples

Examples organized by functionality area.

## Directory Structure

```
examples/
├── tokenizer/          # Tokenization examples
│   ├── basic_tokenization.py
│   ├── custom_tokenizer.py
│   ├── preprocessing_demo.py
│   └── regression_tests.py
├── inference/          # Text generation examples
│   ├── basic_inference.py
│   └── chat_inference.py
├── training/           # Model training examples
│   ├── sft_training.py
│   └── dpo_training.py
├── data/               # Data handling examples
│   ├── generate_math_data.py
│   └── create_sft_dataset.py
└── models/             # Model loading examples
    ├── load_with_lora.py
    └── model_config.py
```

## Quick Start

### Load and Generate

```python
from chuk_lazarus.models import load_model, generate_response

model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
response = generate_response(model, tokenizer, "Hello, ", max_tokens=50)
```

### Fine-tune with SFT

```python
from chuk_lazarus.models import load_model
from chuk_lazarus.training import SFTTrainer, SFTConfig
from chuk_lazarus.data import SFTDataset

model, tokenizer = load_model("model-name", use_lora=True)
dataset = SFTDataset("./data/train.jsonl", tokenizer)
trainer = SFTTrainer(model, tokenizer, SFTConfig())
trainer.train(dataset)
```

### Generate Training Data

```python
from chuk_lazarus.data.generators import generate_lazarus_dataset

paths = generate_lazarus_dataset(
    output_dir="./data",
    sft_samples=1000,
    dpo_samples=500,
)
```

## Tokenizer Examples

### Preprocessing Demo

Demonstrates the preprocessing module features for reducing token waste:

```bash
uv run python examples/tokenizer/preprocessing_demo.py
```

Features demonstrated:
- **Numeric normalization** - Detect and encode numbers (integers, floats, scientific, hex, percentages, fractions)
- **Structure token injection** - Replace UUIDs, URLs, emails, IPs with atomic tokens
- **Hook pipeline** - Composable pre/post tokenization transforms
- **Tokenizer profiles** - Switch between training and inference modes
- **Byte fallback** - Ensure any byte sequence tokenizes without UNK

### Regression Tests

Run tokenizer regression tests:

```bash
uv run python examples/tokenizer/regression_tests.py
```

## Running Examples

```bash
# From project root
uv run python examples/inference/basic_inference.py
```
