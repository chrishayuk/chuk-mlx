# chuk-lazarus: Current Status and Roadmap

## Executive Summary

The codebase has been successfully consolidated from a fragmented structure (`core/`, `rl/`, `training/`, `tools/`, etc.) into a unified `src/chuk_lazarus/` package. Version 0.2.0 represents a major restructuring with clean module boundaries and a consistent import API.

---

## 1. Current State (v0.2.0)

### Package Structure

```
src/chuk_lazarus/              # 134 Python files, 53 directories
├── __init__.py                # Package root (v0.2.0)
├── cli/                       # 2 files - Command line interface
├── data/                      # 25 files - Datasets, preprocessing, tokenizers
│   ├── preprocessing/         # Batching, padding, bucketing
│   ├── tokenizers/            # Custom tokenizer, vocab utils
│   └── generators/            # Math problem generator
├── models/                    # 63 files - Architectures, loading, LoRA
│   ├── architectures/         # Llama, Mistral, Gemma, Starcoder2, Granite, Lazyfox
│   ├── mlp/                   # MLX + Torch MLP variants (SwiGLU, GELU-GLU, ReLU)
│   └── transformers/          # Transformer blocks (MLX + Torch)
├── training/                  # 20 files - Trainers, losses, schedulers
│   ├── losses/                # DPO, GRPO, PPO, SFT losses
│   ├── trainers/              # SFT, DPO, GRPO, PPO trainers
│   └── utils/                 # Log probs, advantage, KL divergence
├── inference/                 # 2 files - Text generation
├── experts/                   # 5 files - GRU/LSTM RNN experts
├── env/                       # 3 files - Hybrid environment, orchestrator
├── distributed/               # 1 file - Placeholder (empty)
└── utils/                     # 12 files - HuggingFace, memory, adapters
```

### Module Import Status

| Module | Status | Notes |
|--------|--------|-------|
| `chuk_lazarus` | ✓ Working | v0.2.0 |
| `chuk_lazarus.inference` | ✓ Working | generate_sequence, generate_response |
| `chuk_lazarus.experts` | ✓ Working | GRUExpert, LSTMExpert, ExpertRegistry |
| `chuk_lazarus.env` | ✓ Working | Orchestrator, HybridEnv |
| `chuk_lazarus.cli` | ✓ Working | CLI entry point |
| `chuk_lazarus.models` | ⚠ Requires pydantic | ModelConfig uses pydantic |
| `chuk_lazarus.data` | ⚠ Requires torch | Some preprocessing utils |
| `chuk_lazarus.training` | ⚠ Requires torch | Trainer adapters |
| `chuk_lazarus.utils` | ⚠ Requires torch | Model/optimizer adapters |

### Dependencies

**pyproject.toml dependencies:**
```
mlx>=0.12.0
mlx-lm>=0.12.0
transformers>=4.40.1
huggingface-hub>=0.23.0,<0.24.0
pyyaml>=6.0
numpy>=1.24.0
tqdm>=4.65.0
```

**Missing from pyproject.toml (required by some modules):**
- `pydantic` - Used by `models/model_config.py`
- `torch` - Used by Torch MLP/transformer variants and some adapters

---

## 2. Issues Identified

### 2.1 Duplicate Files

| File 1 | File 2 | Action |
|--------|--------|--------|
| `models/inference_utility.py` | `inference/generator.py` | Remove `models/inference_utility.py` |
| `utils/huggingface.py` | `utils/huggingface_utils.py` | Consolidate into one |
| `utils/memory.py` | `utils/memory_utils.py` | Consolidate into one |

### 2.2 Deprecated Files

| File | Reason |
|------|--------|
| `models/architectures/lazyfox/xxx_lazyfox_mlp.py` | xxx_ prefix = deprecated |

### 2.3 Tests in src/ (Should be in tests/)

```
src/chuk_lazarus/models/transformers/transformer_block/tests/
src/chuk_lazarus/models/mlp/torch/tests/
src/chuk_lazarus/models/mlp/mlx/tests/
```

### 2.4 Torch Dependencies in MLX Project

10 files import torch:
- `utils/model_adapter.py`
- `utils/optimizer_adapter.py`
- `models/mlp/torch/*.py` (4 files)
- `models/transformers/transformer_block/torch_transformer_block.py`
- `models/architectures/normalization_layer_factory.py`

**Decision needed:** Keep Torch support for compatibility, or make MLX-only?

### 2.5 Empty/Placeholder Modules

- `distributed/__init__.py` - Placeholder only, no actual implementation

### 2.6 Missing from Original Plan

From `docs/REFACTORING_PLAN.md`:

| Planned | Status |
|---------|--------|
| Abstract `BaseTrainer` class | NOT IMPLEMENTED |
| Unified `BaseDataset` interface | NOT IMPLEMENTED |
| Sampling strategies (`inference/sampling.py`) | NOT IMPLEMENTED |
| KV cache management (`inference/cache.py`) | NOT IMPLEMENTED |
| Observation/reward utilities in `env/` | NOT IMPLEMENTED |
| Parameter server in `distributed/` | NOT IMPLEMENTED |
| Shim modules for backward compatibility | NOT NEEDED (clean break) |

---

## 3. What's Working Well

### 3.1 Training Infrastructure
- ✓ All loss functions: DPO, GRPO, PPO, SFT
- ✓ All trainers: SFTTrainer, DPOTrainer, GRPOTrainer, PPOTrainer
- ✓ Batch/epoch processing (classic trainer)
- ✓ Learning rate schedulers (warmup, cosine, linear, exponential)
- ✓ RL utilities (log probs, KL divergence, GAE, advantage normalization)

### 3.2 Data Pipeline
- ✓ SFT, Preference, and Rollout datasets
- ✓ Batch datasets for pre-tokenized NPZ data
- ✓ Preprocessing (batching, padding, bucketing)
- ✓ Custom tokenizer with vocab utils
- ✓ Math problem generator

### 3.3 Model Support
- ✓ Architecture support: Llama, Mistral, Gemma, Starcoder2, Granite, Lazyfox
- ✓ LoRA adapter support
- ✓ Weight loading (safetensors, checkpoints)
- ✓ HuggingFace integration
- ✓ MLP variants (SwiGLU, GELU-GLU, ReLU) for both MLX and Torch

### 3.4 Hybrid Architecture
- ✓ RNN experts (GRU, LSTM) with factory functions
- ✓ Expert registry for managing multiple experts
- ✓ Orchestrator for LLM + Expert + Tool coordination
- ✓ Gym-like environment wrapper

### 3.5 CLI
- ✓ Typer-based CLI with `lazarus` command
- ✓ Training and inference commands

---

## 4. Roadmap

### Phase 1: Cleanup (Immediate)

1. **Fix duplicate files**
   - Remove `models/inference_utility.py` (keep `inference/generator.py`)
   - Consolidate `utils/huggingface*.py` into one file
   - Consolidate `utils/memory*.py` into one file

2. **Remove deprecated files**
   - Delete `models/architectures/lazyfox/xxx_lazyfox_mlp.py`

3. **Fix dependencies**
   - Add `pydantic` to pyproject.toml dependencies
   - Decide on torch support (optional dependency or remove)

4. **Move tests out of src/**
   - Create `tests/models/transformers/` and `tests/models/mlp/`
   - Move test files from src

### Phase 2: Core Improvements (Short-term)

1. **Abstract base classes**
   - Create `BaseTrainer` in `training/base_trainer.py`
   - Create `BaseDataset` in `data/base_dataset.py`
   - Refactor existing trainers/datasets to inherit

2. **Inference improvements**
   - Add `inference/sampling.py` with strategies (temperature, top-k, top-p, nucleus)
   - Add `inference/cache.py` for KV cache management

3. **Update imports in models/__init__.py**
   - Remove inference_utility exports (use inference module)
   - Clean up duplicate exports

### Phase 3: Distributed Training (Medium-term)

1. **Implement parameter server**
   - `distributed/parameter_server.py`
   - `distributed/client.py`
   - `distributed/gradient_sync.py`

2. **Multi-GPU support**
   - Gradient accumulation
   - Model parallelism utilities

### Phase 4: Enhanced RL (Long-term)

1. **Environment improvements**
   - `env/observation.py` - Observation processing
   - `env/reward.py` - Reward shaping utilities
   - `env/mcp_client.py` - MCP tool integration

2. **Advanced training**
   - Curriculum learning support
   - Multi-task training
   - Reward model training

### Phase 5: Production Readiness

1. **Testing**
   - Unit tests for all modules
   - Integration tests
   - Benchmark suite

2. **Documentation**
   - API documentation
   - Usage examples
   - Architecture guide

3. **Optimization**
   - Memory profiling and optimization
   - Training speed benchmarks
   - Quantization support (4-bit, 8-bit)

---

## 5. Quick Reference

### Import Examples

```python
# Models
from chuk_lazarus.models import load_model, LoRAConfig, ModelConfig
from chuk_lazarus.models.architectures import LlamaModel, MistralModel

# Training
from chuk_lazarus.training import SFTTrainer, DPOTrainer, GRPOTrainer, PPOTrainer
from chuk_lazarus.training import dpo_loss, grpo_loss, ppo_loss, sft_loss
from chuk_lazarus.training import Trainer, BatchProcessor, EpochProcessor
from chuk_lazarus.training.schedulers import schedule_learning_rate

# Data
from chuk_lazarus.data import SFTDataset, PreferenceDataset, RolloutBuffer
from chuk_lazarus.data import BatchDatasetBase, TrainBatchDataset
from chuk_lazarus.data.preprocessing import pad_sequences, add_to_buckets
from chuk_lazarus.data.tokenizers import CustomTokenizer

# Inference
from chuk_lazarus.inference import generate_sequence, generate_response

# Experts
from chuk_lazarus.experts import GRUExpert, LSTMExpert, ExpertRegistry

# Environment
from chuk_lazarus.env import Orchestrator, HybridEnv
```

### CLI Usage

```bash
# Install
pip install -e .

# Run CLI
lazarus --help
lazarus train --help
lazarus infer --help
```

---

## 6. Metrics

| Metric | Value |
|--------|-------|
| Total Python files | 134 |
| Total directories | 53 |
| Package version | 0.2.0 |
| Supported architectures | 6 (Llama, Mistral, Gemma, Starcoder2, Granite, Lazyfox) |
| Trainer types | 5 (Trainer, SFT, DPO, GRPO, PPO) |
| Loss functions | 4 (SFT, DPO, GRPO, PPO) |
| RNN expert types | 2 (GRU, LSTM) |

---

*Last updated: 2025-12-20*
