# Lazarus Examples

Examples organized by functionality area.

## Directory Structure

```
examples/
├── tokenizer/          # Tokenization examples
│   ├── basic_tokenization.py
│   └── custom_tokenizer.py
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

## Running Examples

```bash
# From project root
PYTHONPATH=src python examples/inference/basic_inference.py
```
