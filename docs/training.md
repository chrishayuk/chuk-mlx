# Training Guide

Lazarus supports multiple training paradigms:

- **SFT** - Supervised Fine-Tuning
- **DPO** - Direct Preference Optimization
- **GRPO** - Group Relative Policy Optimization
- **PPO** - Proximal Policy Optimization

## SFT Training

Supervised fine-tuning on instruction-following data.

### Data Format

JSONL with `prompt` and `response` fields:

```json
{"prompt": "What is 2+2?", "response": "2+2 equals 4."}
{"prompt": "Explain gravity.", "response": "Gravity is a force..."}
```

### Training

```python
from chuk_lazarus.models import load_model
from chuk_lazarus.training import SFTTrainer, SFTConfig
from chuk_lazarus.data import SFTDataset

model, tokenizer = load_model("model-name", use_lora=True)
dataset = SFTDataset("./data/train.jsonl", tokenizer)

config = SFTConfig(
    num_epochs=3,
    batch_size=4,
    learning_rate=1e-5,
    warmup_steps=100,
)

trainer = SFTTrainer(model, tokenizer, config)
trainer.train(dataset)
```

## DPO Training

Train using preference pairs (chosen vs rejected).

### Data Format

```json
{"prompt": "Question?", "chosen": "Good answer", "rejected": "Bad answer"}
```

### Training

```python
from chuk_lazarus.training import DPOTrainer, DPOTrainerConfig
from chuk_lazarus.data import PreferenceDataset

policy_model, tokenizer = load_model("model-name")
reference_model, _ = load_model("model-name")  # Frozen reference

dataset = PreferenceDataset("./data/preferences.jsonl", tokenizer)

config = DPOTrainerConfig(
    dpo=DPOConfig(beta=0.1),
    learning_rate=1e-6,
)

trainer = DPOTrainer(policy_model, reference_model, tokenizer, config)
trainer.train(dataset)
```

## Configuration Options

### SFTConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_epochs | 3 | Number of training epochs |
| batch_size | 4 | Samples per batch |
| learning_rate | 1e-5 | Learning rate |
| warmup_steps | 100 | LR warmup steps |
| max_grad_norm | 1.0 | Gradient clipping |
| checkpoint_interval | 500 | Steps between saves |

### DPOTrainerConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| beta | 0.1 | KL penalty coefficient |
| learning_rate | 1e-6 | Learning rate (lower for DPO) |
| target_reward_margin | 2.0 | Early stop threshold |

## Checkpoints

Checkpoints are saved automatically:

```
checkpoints/
├── step_500.safetensors
├── step_1000.safetensors
├── best.safetensors
└── final.safetensors
```

Load a checkpoint:

```python
trainer.load_checkpoint("./checkpoints/best.safetensors")
```
