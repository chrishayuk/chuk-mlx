# Comprehensive Refactoring Plan: Unifying chuk-mlx Library Structure

## Executive Summary

This document provides a detailed plan for refactoring the chuk-mlx codebase from its current fragmented structure into a unified, clean architecture. The refactoring addresses duplication between `core/` and `rl/` modules, consolidates training infrastructure, and creates a consistent API for model loading, data handling, and training.

---

## 1. Current State Analysis

### 1.1 Directory Structure Overview

```
Current:
chuk-mlx/
├── core/                  # Legacy base infrastructure
│   ├── batch/             # NPZ batch creation and bucketing
│   ├── dataset/           # Batch dataset loading (NPZ-based)
│   ├── models/            # Model architectures and config
│   ├── tokenizers/        # Custom tokenizer
│   └── utils/             # Model/tokenizer/optimizer loaders
├── training/              # Legacy trainer (NPZ-based)
├── rl/                    # RL trainers (parallel structure)
│   ├── data/              # Preference datasets, rollout buffers
│   ├── env/               # Environment and orchestrator
│   ├── experts/           # RNN experts (GRU, LSTM)
│   ├── losses/            # DPO, GRPO, PPO losses
│   ├── models/            # HF model adapter with LoRA
│   ├── trainers/          # DPO, GRPO, PPO, SFT trainers
│   ├── lazarus/           # New training approach
│   └── utils/             # RL-specific utilities
├── tools/                 # CLI utilities (batch, models, tokenizer, train)
├── parameter_server/      # Distributed training infrastructure
├── training_config/       # YAML configurations
├── sft/                   # Empty/minimal SFT placeholder
├── train.py               # Main training entry point
└── infer.py               # Inference entry point
```

### 1.2 Key Issues Identified

| Issue | Location | Description |
|-------|----------|-------------|
| **Model Loading Duplication** | `core/utils/model_loader.py` vs `rl/models/hf_model_adapter.py` | Two different model loading approaches - one basic, one with LoRA support |
| **Trainer Duplication** | `training/trainer.py` vs `rl/trainers/*` | Base trainer uses NPZ batches; RL trainers use dataclass configs with modern patterns |
| **Data Handling Split** | `core/batch/`, `core/dataset/` vs `rl/data/` | Core uses NPZ files with pre-tokenization; RL uses in-memory datasets with on-the-fly tokenization |
| **Configuration Split** | `training_config/*.yaml` vs dataclasses in RL | No unified configuration approach |
| **Duplicate SFT** | `sft/` (empty) vs `rl/trainers/sft_trainer.py` | Placeholder module exists but real implementation is in RL |
| **Loss Functions Split** | `core/models/chuk_loss_function.py` vs `rl/losses/*` | Different loss function patterns |

---

## 2. Proposed New Structure

```
chuk/
├── __init__.py
├── models/                    # All model loading and architectures
│   ├── __init__.py
│   ├── config.py              # Unified ModelConfig (from core/models/model_config.py)
│   ├── loader.py              # Unified model loading (merge both approaches)
│   ├── lora.py                # LoRA implementation (from rl/models/hf_model_adapter.py)
│   ├── weights.py             # Weight loading utilities (from core/models/load_weights.py)
│   └── architectures/         # Model implementations
│       ├── __init__.py
│       ├── base.py            # TransformerBaseModel
│       ├── attention.py       # AttentionBase
│       ├── llama.py           # LlamaModel
│       ├── mistral.py         # MistralModel
│       ├── gemma.py           # GemmaModel
│       ├── granite.py         # GraniteModel
│       ├── starcoder2.py      # StarCoder2 components
│       ├── mlp/               # MLP variants
│       │   ├── mlx/
│       │   └── torch/
│       └── experimental/      # Lazyfox, math, mock models
│
├── training/                  # All training infrastructure
│   ├── __init__.py
│   ├── config.py              # Unified training configurations (dataclasses)
│   ├── base_trainer.py        # Abstract base trainer with common patterns
│   ├── trainer.py             # Standard supervised trainer
│   ├── sft_trainer.py         # SFT trainer (from rl/trainers/)
│   ├── dpo_trainer.py         # DPO trainer (from rl/trainers/)
│   ├── grpo_trainer.py        # GRPO trainer (from rl/trainers/)
│   ├── ppo_trainer.py         # PPO trainer (from rl/trainers/)
│   ├── losses/                # All loss functions
│   │   ├── __init__.py
│   │   ├── cross_entropy.py   # Standard CE loss (from core)
│   │   ├── dpo_loss.py        # DPO loss
│   │   ├── grpo_loss.py       # GRPO loss
│   │   └── ppo_loss.py        # PPO loss
│   ├── schedulers.py          # Learning rate schedulers
│   └── utils/                 # Training utilities
│       ├── advantage.py
│       ├── kl_divergence.py
│       ├── log_probs.py
│       └── gradient_utils.py
│
├── data/                      # All data handling
│   ├── __init__.py
│   ├── config.py              # Dataset configurations
│   ├── base_dataset.py        # Abstract base dataset
│   ├── batch_dataset.py       # Pre-tokenized batch dataset (from core)
│   ├── sft_dataset.py         # SFT dataset (from rl/trainers/sft_trainer.py)
│   ├── preference_dataset.py  # DPO preference pairs (from rl/data/)
│   ├── rollout_buffer.py      # RL rollout buffer (from rl/data/)
│   ├── episode.py             # Episode/trajectory storage (from rl/data/)
│   ├── generators/            # Data generators
│   │   ├── math_generator.py
│   │   └── batch_generator.py
│   ├── preprocessing/         # Batch creation utilities
│   │   ├── batch_base.py
│   │   ├── bucketing.py
│   │   ├── padding.py
│   │   └── tokenization.py
│   └── tokenizers/            # Tokenizer utilities
│       ├── loader.py
│       ├── custom_tokenizer.py
│       └── vocab_utils.py
│
├── inference/                 # Generation and sampling
│   ├── __init__.py
│   ├── generator.py           # Text generation (from core/models/inference_utility.py)
│   ├── sampling.py            # Sampling strategies
│   └── cache.py               # KV cache management
│
├── experts/                   # RNN experts for hybrid architecture
│   ├── __init__.py
│   ├── config.py              # ExpertConfig
│   ├── base.py                # RNNExpertBase
│   ├── gru_expert.py
│   ├── lstm_expert.py
│   └── registry.py
│
├── env/                       # Environments and orchestration
│   ├── __init__.py
│   ├── config.py
│   ├── orchestrator.py
│   ├── hybrid_env.py
│   ├── observation.py
│   ├── reward.py
│   └── mcp_client.py          # MCP tool client
│
├── distributed/               # Distributed training
│   ├── __init__.py
│   ├── parameter_server.py
│   └── client.py
│
├── cli/                       # Unified CLI entry points
│   ├── __init__.py
│   ├── train.py               # Main training CLI
│   ├── infer.py               # Inference CLI
│   ├── batch.py               # Batch utilities CLI
│   ├── model.py               # Model utilities CLI
│   └── tokenizer.py           # Tokenizer utilities CLI
│
└── utils/                     # Shared utilities
    ├── __init__.py
    ├── huggingface.py         # HF hub utilities
    ├── memory.py              # Memory utilities
    ├── model_adapter.py       # MLX/Torch adapter
    └── config_loader.py       # YAML/dataclass config loading
```

---

## 3. File Migration Map

### 3.1 Files to Move (with consolidation notes)

#### Models Module

| Source | Destination | Notes |
|--------|-------------|-------|
| `core/models/model_config.py` | `chuk/models/config.py` | Keep as-is, add LoRA config options |
| `core/utils/model_loader.py` | `chuk/models/loader.py` | Merge with HF adapter approach |
| `rl/models/hf_model_adapter.py` | `chuk/models/loader.py` + `chuk/models/lora.py` | Split LoRA into separate module |
| `core/models/load_weights.py` | `chuk/models/weights.py` | Keep as-is |
| `core/models/mlx_adapter.py` | `chuk/utils/model_adapter.py` | Merge with existing adapter |
| `core/models/architectures/*` | `chuk/models/architectures/*` | Reorganize, flatten hierarchy |
| `core/models/architectures/attention_base.py` | `chuk/models/architectures/attention.py` | Rename |
| `core/models/architectures/transformer_base_model.py` | `chuk/models/architectures/base.py` | Rename |
| `core/models/architectures/model.py` | `chuk/models/architectures/base.py` | Merge with transformer base |

#### Training Module

| Source | Destination | Notes |
|--------|-------------|-------|
| `training/trainer.py` | `chuk/training/trainer.py` | Refactor to inherit from base |
| `training/batch_processor.py` | `chuk/training/utils/batch_processor.py` | Move to utils |
| `training/epoch_processor.py` | `chuk/training/utils/epoch_processor.py` | Move to utils |
| `training/training_scheduler.py` | `chuk/training/schedulers.py` | Rename |
| `rl/trainers/sft_trainer.py` | `chuk/training/sft_trainer.py` | Keep SFTDataset in data module |
| `rl/trainers/dpo_trainer.py` | `chuk/training/dpo_trainer.py` | Direct move |
| `rl/trainers/grpo_trainer.py` | `chuk/training/grpo_trainer.py` | Direct move |
| `rl/trainers/ppo_trainer.py` | `chuk/training/ppo_trainer.py` | Direct move |
| `core/models/chuk_loss_function.py` | `chuk/training/losses/cross_entropy.py` | Rename |
| `rl/losses/dpo_loss.py` | `chuk/training/losses/dpo_loss.py` | Direct move |
| `rl/losses/grpo_loss.py` | `chuk/training/losses/grpo_loss.py` | Direct move |
| `rl/losses/ppo_loss.py` | `chuk/training/losses/ppo_loss.py` | Direct move |
| `rl/utils/advantage.py` | `chuk/training/utils/advantage.py` | Direct move |
| `rl/utils/kl_divergence.py` | `chuk/training/utils/kl_divergence.py` | Direct move |
| `rl/utils/log_probs.py` | `chuk/training/utils/log_probs.py` | Direct move |

#### Data Module

| Source | Destination | Notes |
|--------|-------------|-------|
| `core/batch/batch_base.py` | `chuk/data/preprocessing/batch_base.py` | Direct move |
| `core/batch/bucketing.py` | `chuk/data/preprocessing/bucketing.py` | Direct move |
| `core/batch/padding_utils.py` | `chuk/data/preprocessing/padding.py` | Rename |
| `core/batch/tokenization_utils.py` | `chuk/data/preprocessing/tokenization.py` | Rename |
| `core/batch/*.py` (others) | `chuk/data/preprocessing/` | Move remaining |
| `core/dataset/batch_dataset_base.py` | `chuk/data/batch_dataset.py` | Rename |
| `core/dataset/train_batch_dataset.py` | `chuk/data/batch_dataset.py` | Merge |
| `rl/data/preference_dataset.py` | `chuk/data/preference_dataset.py` | Direct move |
| `rl/data/rollout_buffer.py` | `chuk/data/rollout_buffer.py` | Direct move |
| `rl/data/episode.py` | `chuk/data/episode.py` | Direct move |
| `rl/data/generators/*` | `chuk/data/generators/` | Direct move |
| `core/tokenizers/*` | `chuk/data/tokenizers/` | Direct move |
| `core/utils/tokenizer_loader.py` | `chuk/data/tokenizers/loader.py` | Move |
| (extract from `rl/trainers/sft_trainer.py`) | `chuk/data/sft_dataset.py` | Extract SFTDataset class |

#### Inference Module

| Source | Destination | Notes |
|--------|-------------|-------|
| `core/models/inference_utility.py` | `chuk/inference/generator.py` | Rename and expand |

#### Experts Module

| Source | Destination | Notes |
|--------|-------------|-------|
| `rl/experts/rnn_expert_base.py` | `chuk/experts/base.py` | Rename |
| `rl/experts/gru_expert.py` | `chuk/experts/gru_expert.py` | Direct move |
| `rl/experts/lstm_expert.py` | `chuk/experts/lstm_expert.py` | Direct move |
| `rl/experts/registry.py` | `chuk/experts/registry.py` | Direct move |

#### Environment Module

| Source | Destination | Notes |
|--------|-------------|-------|
| `rl/env/orchestrator.py` | `chuk/env/orchestrator.py` | Extract MCPToolClient |
| `rl/env/hybrid_env.py` | `chuk/env/hybrid_env.py` | Direct move |
| `rl/env/observation.py` | `chuk/env/observation.py` | Direct move |
| `rl/env/reward.py` | `chuk/env/reward.py` | Direct move |

#### Distributed Module

| Source | Destination | Notes |
|--------|-------------|-------|
| `parameter_server/server.py` | `chuk/distributed/parameter_server.py` | Direct move |
| `parameter_server/client.py` | `chuk/distributed/client.py` | Direct move |
| `parameter_server/llm_*.py` | `chuk/distributed/` | Move LLM-specific clients |

#### CLI Module

| Source | Destination | Notes |
|--------|-------------|-------|
| `train.py` | `chuk/cli/train.py` | Refactor for new structure |
| `infer.py` | `chuk/cli/infer.py` | Refactor |
| `tools/batch/*` | `chuk/cli/batch.py` | Consolidate |
| `tools/models/*` | `chuk/cli/model.py` | Consolidate |
| `tools/tokenizer/*` | `chuk/cli/tokenizer.py` | Consolidate |

#### Utils Module

| Source | Destination | Notes |
|--------|-------------|-------|
| `core/utils/huggingface_utils.py` | `chuk/utils/huggingface.py` | Rename |
| `core/utils/memory_utils.py` | `chuk/utils/memory.py` | Rename |
| `core/utils/model_adapter.py` | `chuk/utils/model_adapter.py` | Direct move |
| `core/utils/training_config_loader.py` | `chuk/utils/config_loader.py` | Expand to support dataclasses |

### 3.2 Files to Delete

| File | Reason |
|------|--------|
| `sft/` (entire directory) | Empty placeholder, real implementation in rl/trainers |
| `core/utils/optimizer_adapter.py` | Likely unused, optimizer_loader.py is the active one |
| `core/models/architectures/lazyfox/xxx_lazyfox_mlp.py` | Prefixed with xxx, appears deprecated |
| `rl/__init__.py` (current) | Will be recreated in new structure |

### 3.3 Files to Keep at Root Level

| File | Action |
|------|--------|
| `pyproject.toml` | Update package structure |
| `README.md` | Update documentation |
| `requirements.txt` | Keep/update |
| `.gitignore` | Keep |
| `training_config/` | Keep for backward compatibility, add deprecation notice |
| `sample_data/` | Keep |
| `docs/` | Keep |
| `tutorial/` | Keep but update imports |

---

## 4. Code Consolidation Details

### 4.1 Model Loader Consolidation

**Current State:**
- `core/utils/model_loader.py`: Simple loader that checks local then HuggingFace
- `rl/models/hf_model_adapter.py`: Full-featured loader with LoRA, quantization, generation

**Consolidated Approach:**

```python
# chuk/models/loader.py

@dataclass
class LoadConfig:
    """Configuration for model loading."""
    model_name: str
    use_lora: bool = False
    lora_config: Optional[LoRAConfig] = None
    use_4bit: bool = False
    load_weights: bool = True
    local_only: bool = False

def load_model(config: Union[str, LoadConfig]) -> Tuple[nn.Module, Any]:
    """
    Unified model loading that supports:
    - Local custom models (core/models/architectures/)
    - HuggingFace models via mlx-lm
    - LoRA adapters
    - Weight quantization
    """
    if isinstance(config, str):
        config = LoadConfig(model_name=config)

    # Try local first
    model = _try_load_local(config.model_name)

    if model is None:
        # Fall back to HuggingFace
        model, tokenizer = _load_from_hf(config.model_name)

    # Apply LoRA if requested
    if config.use_lora:
        model = apply_lora(model, config.lora_config)

    return model, tokenizer
```

### 4.2 Base Trainer Pattern

**Create Abstract Base:**

```python
# chuk/training/base_trainer.py

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class BaseTrainerConfig:
    """Base configuration for all trainers."""
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    log_interval: int = 10
    checkpoint_interval: int = 500
    checkpoint_dir: str = "./checkpoints"
    max_steps: Optional[int] = None

class BaseTrainer(ABC):
    """Abstract base trainer with common functionality."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: BaseTrainerConfig,
        optimizer: Optional[optim.Optimizer] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = optimizer or self._create_optimizer()
        self.global_step = 0
        self.metrics_history = []

    @abstractmethod
    def train_step(self, batch: Dict) -> Tuple[float, Dict]:
        """Perform single training step. Override in subclasses."""
        pass

    def train(self, dataset, eval_dataset=None, callback=None):
        """Common training loop."""
        # ... common training logic
        for epoch in range(self.config.num_epochs):
            for batch in dataset.iter_batches(...):
                loss, metrics = self.train_step(batch)
                # ... logging, checkpointing, etc.

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        # ... common checkpoint logic

    def _clip_gradients(self, grads, max_norm: float):
        """Clip gradients by global norm."""
        # ... common gradient clipping
```

### 4.3 Dataset Unification

**Create Base Dataset Interface:**

```python
# chuk/data/base_dataset.py

from abc import ABC, abstractmethod
from typing import Iterator, Dict

class BaseDataset(ABC):
    """Abstract base dataset interface."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict:
        pass

    @abstractmethod
    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        **kwargs
    ) -> Iterator[Dict]:
        """Iterate over batches."""
        pass
```

### 4.4 Configuration Unification

**Unified Config System:**

```python
# chuk/utils/config_loader.py

from dataclasses import dataclass, fields, asdict
from typing import TypeVar, Type
import yaml

T = TypeVar('T')

def load_config(path: str, config_class: Type[T]) -> T:
    """
    Load configuration from YAML file into dataclass.
    Supports both new dataclass configs and legacy YAML format.
    """
    with open(path) as f:
        raw_config = yaml.safe_load(f)

    # Map legacy YAML keys to dataclass fields
    mapped_config = _map_legacy_config(raw_config, config_class)

    return config_class(**mapped_config)

def _map_legacy_config(raw: dict, cls: Type) -> dict:
    """Map legacy YAML structure to flat dataclass."""
    result = {}
    field_names = {f.name for f in fields(cls)}

    # Flatten nested sections
    for section, values in raw.items():
        if isinstance(values, dict):
            for k, v in values.items():
                if k in field_names:
                    result[k] = v
        elif section in field_names:
            result[section] = values

    return result
```

---

## 5. Breaking Changes and Deprecations

### 5.1 Import Path Changes

| Old Import | New Import | Deprecation Strategy |
|------------|------------|---------------------|
| `from core.utils.model_loader import load_model` | `from chuk.models import load_model` | Create shim module in old location |
| `from training.trainer import Trainer` | `from chuk.training import Trainer` | Create shim module |
| `from rl.trainers.dpo_trainer import DPOTrainer` | `from chuk.training import DPOTrainer` | Create shim module |
| `from rl.data.preference_dataset import PreferenceDataset` | `from chuk.data import PreferenceDataset` | Create shim module |
| `from core.models.architectures.llama.llama_model import LlamaModel` | `from chuk.models.architectures import LlamaModel` | Create shim module |

### 5.2 Configuration Changes

**Legacy YAML configs will still work** but should emit deprecation warnings:

```yaml
# Old format (training_config/finetune/*.yaml)
model:
  name: "mistralai/Mistral-7B-Instruct-v0.2"
optimizer:
  name: AdamW
  initial_lr: 1e-5
training:
  num_epochs: 1
```

**New format uses dataclasses:**

```python
# New format
from chuk.training import TrainerConfig

config = TrainerConfig(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    optimizer="adamw",
    learning_rate=1e-5,
    num_epochs=1
)
```

### 5.3 Backward Compatibility Shims

Create shim modules in old locations:

```python
# core/utils/model_loader.py (deprecated)
import warnings
warnings.warn(
    "core.utils.model_loader is deprecated. Use chuk.models.load_model instead.",
    DeprecationWarning,
    stacklevel=2
)
from chuk.models.loader import load_model

__all__ = ['load_model']
```

---

## 6. Migration Order and Dependencies

### Phase 1: Foundation (No Breaking Changes)

1. **Create new package structure** - Create `chuk/` directory with `__init__.py` files
2. **Move utilities first** - `chuk/utils/` (no dependencies on other new modules)
3. **Move data tokenizers** - `chuk/data/tokenizers/` (standalone)
4. **Move model configs** - `chuk/models/config.py`

### Phase 2: Core Modules

5. **Move model architectures** - `chuk/models/architectures/` (depends on config)
6. **Create unified model loader** - `chuk/models/loader.py` (merge both approaches)
7. **Move LoRA implementation** - `chuk/models/lora.py`
8. **Move data preprocessing** - `chuk/data/preprocessing/`
9. **Move datasets** - `chuk/data/` (batch_dataset, preference_dataset, etc.)

### Phase 3: Training Infrastructure

10. **Create base trainer** - `chuk/training/base_trainer.py`
11. **Move loss functions** - `chuk/training/losses/`
12. **Move training utilities** - `chuk/training/utils/`
13. **Move trainers** - Refactor each to inherit from base trainer
14. **Move schedulers** - `chuk/training/schedulers.py`

### Phase 4: Specialized Modules

15. **Move inference** - `chuk/inference/`
16. **Move experts** - `chuk/experts/`
17. **Move environment** - `chuk/env/`
18. **Move distributed** - `chuk/distributed/`

### Phase 5: Integration and CLI

19. **Create new CLI** - `chuk/cli/`
20. **Create backward compatibility shims**
21. **Update pyproject.toml**
22. **Update documentation and README**

### Phase 6: Cleanup

23. **Add deprecation warnings to old imports**
24. **Update tests**
25. **Remove empty/deprecated files**

---

## 7. Testing Strategy

### 7.1 Tests to Update

- All imports in existing tests need updating
- `core/batch/tests/` -> `tests/data/preprocessing/`
- `core/dataset/tests/` -> `tests/data/`
- `core/models/tests/` -> `tests/models/`
- `training/tests/` -> `tests/training/`

### 7.2 New Integration Tests

- Test unified model loader with all model types
- Test backward compatibility shims
- Test config loading from both YAML and dataclasses
- Test trainer inheritance hierarchy

---

## 8. Potential Challenges

| Challenge | Mitigation |
|-----------|------------|
| **Circular imports** | Careful dependency ordering; use TYPE_CHECKING for type hints |
| **Config migration** | Support both old YAML and new dataclass configs during transition |
| **Breaking external users** | 6-month deprecation period with shim modules |
| **Test coverage gaps** | Add integration tests before major refactoring |
| **HuggingFace compatibility** | Ensure mlx-lm integration works with new loader |
| **LoRA complexity** | Keep LoRA in separate module to avoid coupling |

---

## 9. Estimated Effort

| Phase | Files | Estimated Effort |
|-------|-------|------------------|
| Phase 1: Foundation | ~15 | Low |
| Phase 2: Core Modules | ~40 | Medium-High |
| Phase 3: Training | ~25 | Medium |
| Phase 4: Specialized | ~20 | Medium |
| Phase 5: Integration | ~10 | Medium |
| Phase 6: Cleanup | ~30 | Low |
| **Total** | **~140 files** | |

---

## 10. Quick Start Commands (Post-Refactoring)

After refactoring, the unified API will look like:

```python
# Load any model with optional LoRA
from chuk.models import load_model, LoRAConfig

model, tokenizer = load_model(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    lora=LoRAConfig(rank=8, alpha=16)
)

# Train with any trainer
from chuk.training import SFTTrainer, SFTConfig
from chuk.data import SFTDataset

dataset = SFTDataset("./data/train.jsonl", tokenizer)
trainer = SFTTrainer(model, tokenizer, SFTConfig(epochs=3))
trainer.train(dataset)

# Generate data
from chuk.data.generators import MathProblemGenerator

gen = MathProblemGenerator()
samples = gen.generate_batch(1000)
gen.save_sft_dataset(samples, "./data/math_sft.jsonl")
```

---

## 11. CLI Examples (Post-Refactoring)

```bash
# Training
python -m chuk.cli.train sft --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data ./data
python -m chuk.cli.train dpo --model ... --data ...
python -m chuk.cli.train pipeline --model ... --stages sft,dpo,eval

# Inference
python -m chuk.cli.infer --model ... --prompt "Hello"

# Data generation
python -m chuk.cli.data generate-math --output ./data --samples 10000

# Model utilities
python -m chuk.cli.model download mistralai/Mistral-7B-Instruct-v0.2
python -m chuk.cli.model info ./checkpoints/model
```

---

## Appendix: Key Files Reference

### Critical Files for Implementation

1. `/Users/christopherhay/chris-source/chuk-mlx/core/utils/model_loader.py` - Core model loading logic
2. `/Users/christopherhay/chris-source/chuk-mlx/rl/models/hf_model_adapter.py` - LoRA and HF loading
3. `/Users/christopherhay/chris-source/chuk-mlx/training/trainer.py` - Base trainer pattern
4. `/Users/christopherhay/chris-source/chuk-mlx/rl/trainers/sft_trainer.py` - Modern trainer template
5. `/Users/christopherhay/chris-source/chuk-mlx/core/dataset/batch_dataset_base.py` - Dataset pattern
