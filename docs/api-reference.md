# API Reference

## Models

### load_model

```python
from chuk_lazarus.models import load_model

model, tokenizer = load_model(
    model_path: str,
    use_lora: bool = False,
    lora_config: LoRAConfig = None,
    adapter_path: str = None,
)
```

Load a model from HuggingFace or local path.

### load_tokenizer

```python
from chuk_lazarus.models import load_tokenizer

tokenizer = load_tokenizer(model_path: str)
```

### generate_response

```python
from chuk_lazarus.models import generate_response

response = generate_response(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
)
```

## Training

### BaseTrainer

All trainers inherit from `BaseTrainer`:

```python
class BaseTrainer:
    def train(dataset, num_epochs, eval_dataset=None, callback=None)
    def evaluate(dataset) -> Dict[str, float]
    def save_checkpoint(name: str)
    def load_checkpoint(path: str)
```

### SFTTrainer

```python
from chuk_lazarus.training import SFTTrainer, SFTConfig

config = SFTConfig(
    num_epochs=3,
    batch_size=4,
    learning_rate=1e-5,
)

trainer = SFTTrainer(model, tokenizer, config)
trainer.train(dataset)
```

### DPOTrainer

```python
from chuk_lazarus.training import DPOTrainer, DPOTrainerConfig

trainer = DPOTrainer(
    policy_model,
    reference_model,
    tokenizer,
    config,
)
trainer.train(dataset)
```

## Data

### SFTDataset

```python
from chuk_lazarus.data import SFTDataset

dataset = SFTDataset(
    path: str,           # JSONL file path
    tokenizer,
    max_length: int = 512,
)

len(dataset)  # Number of samples
dataset[0]    # Get sample by index

for batch in dataset.iter_batches(batch_size=4):
    # batch["input_ids"], batch["labels"], batch["loss_mask"]
```

### PreferenceDataset

```python
from chuk_lazarus.data import PreferenceDataset

dataset = PreferenceDataset(
    path: str,  # JSONL with prompt/chosen/rejected
    tokenizer,
    max_length: int = 512,
)
```

### MathProblemGenerator

```python
from chuk_lazarus.data.generators import MathProblemGenerator

gen = MathProblemGenerator(seed=42)
samples = gen.generate_batch(100, difficulty_range=(1, 3))

# Export
gen.save_sft_dataset(samples, "train_sft.jsonl")
gen.save_dpo_dataset(samples, "train_dpo.jsonl")
```

## Configuration Classes

### ModelConfig

```python
@dataclass
class ModelConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int = 2048
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
```

### LoRAConfig

```python
@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = ("q_proj", "v_proj")
```

### SFTConfig

```python
@dataclass
class SFTConfig:
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    log_interval: int = 10
    checkpoint_interval: int = 500
    checkpoint_dir: str = "./checkpoints/sft"
```
