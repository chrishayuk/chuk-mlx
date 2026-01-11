# Lazarus CLI Migration Roadmap

This document captures the audit findings and migration plan to consolidate reimplemented code and properly use Lazarus infrastructure.

## Executive Summary

**Audit Date:** 2026-01-07

**Findings:**
- 3 CRITICAL issues (broken imports, incompatible formats)
- 6 HIGH priority reimplementation issues
- 4 MEDIUM priority inconsistencies
- Several areas already following best practices

**Key Pattern:** Code diverged into multiple approaches for:
1. Model loading (scattered `mlx_lm.load()` calls instead of centralized `HFLoader`)
2. LoRA application (standard vs custom reimplementation)
3. Adapter loading (falls back to `mlx_lm` instead of native Lazarus support)
4. Checkpoint formats (NPZ vs safetensors)
5. Hidden state extraction (manual vs accessor-based)

## Centralized Infrastructure - Current State

### Current Model Loading (FRAGMENTED)

```
UnifiedPipeline.from_pretrained()     <- Inference (has own loading logic)
    └── duplicates HFLoader logic

_load_model_sync()                    <- Introspection (in analyzer/)
    └── HFLoader (partial)
    └── [GAP] Falls back to mlx_lm.load() for adapters

cli/commands/train/sft.py             <- Training
    └── mlx_lm.load() (bypasses Lazarus)

cli/commands/train/grpo.py            <- Training
    └── custom _load_model_with_tokenizer() (reimplements)
```

### PROBLEMS:
1. Model loading logic duplicated across 4+ locations
2. No single entry point for "load a model with optional adapter"
3. Adapter loading falls back to external library (mlx_lm)
4. Training and inference use different loading paths

---

## Target Architecture

### Centralized Model Loader (NEW)

```
src/chuk_lazarus/models_v2/loader.py   <- NEW: Single entry point
    │
    ├── load_model(model_id, adapter_path=None, dtype=BFLOAT16)
    │       -> (model, tokenizer, config)
    │
    ├── load_model_with_lora(model_id, lora_config, adapter_path=None)
    │       -> (model, tokenizer, config, lora_layers)
    │
    └── Implementation:
            ├── HFLoader.download()
            ├── detect_model_family()
            ├── family_info.config_class.from_hf_config()
            ├── family_info.model_class()
            ├── HFLoader.apply_weights_to_model()
            ├── HFLoader.load_tokenizer()
            └── [if adapter_path] apply_lora() + load_adapter_weights()
```

### All Consumers Use Central Loader

```
# Inference
UnifiedPipeline.from_pretrained()
    └── load_model()

# Introspection
ModelAnalyzer.from_pretrained()
    └── load_model()

# Training
train_sft()
    └── load_model_with_lora()

train_grpo()
    └── load_model_with_lora()

train_dpo()
    └── load_model()

train_dual_reward()
    └── load_model_with_lora()
```

### Benefits:
1. Single place to fix bugs / add features
2. Consistent behavior across inference, training, introspection
3. Native adapter loading (no mlx_lm fallback)
4. Config always returned (not None for adapter case)

---

## Issue Categories

### CRITICAL (P0) - Must Fix

| Issue | Location | Problem | Fix |
|-------|----------|---------|-----|
| Broken import | `cli/commands/train/dpo.py:26` | `from ....models import load_model` - package doesn't exist | Use `_load_model_sync()` or HFLoader |
| mlx_lm fallback for adapters | `introspection/analyzer/loader.py:44-50` | `_load_model_sync` falls back to `mlx_lm.load()` for adapters | Implement native adapter loading |
| NPZ checkpoint format | `training/trainers/dpo_trainer.py:197` | Saves `.npz` (deprecated), incompatible | Use `mx.save_safetensors()` |
| Custom LoRA in DualRewardTrainer | `training/trainers/dual_reward_trainer.py:160-224` | Reimplements LoRA from scratch | Use `apply_lora()` from models_v2 |
| mlx_lm.load in SFT | `cli/commands/train/sft.py:37` | Bypasses Lazarus infrastructure | Use HFLoader + apply_lora |
| mlx_lm.load in generation | `cli/commands/introspect/generation.py:41` | Bypasses Lazarus infrastructure | Use _load_model_sync |

### HIGH (P1) - Significant Duplication

| Issue | Location | Problem | Fix |
|-------|----------|---------|-----|
| Custom model loader in GRPO | `cli/commands/train/grpo.py:63-101` | 40-line function duplicating `mlx_lm.load()` | Use `mlx_lm.load()` |
| Manual hidden state extraction | `cli/commands/introspect/probing.py` | Reimplements layer iteration in multiple functions | Use `AsyncModelAccessor.forward_through_layers()` |
| Verbose logit projection | `cli/commands/introspect/classifier.py:296-312` | 15+ lines duplicating `accessor.apply_norm_and_head()` | Use accessor method |
| Manual model loading in classifier | `cli/commands/introspect/classifier.py` | Uses HFLoader directly in `introspect_classifier()` | Use `_load_model_sync()` |

### MEDIUM (P2) - Inconsistencies

| Issue | Location | Problem | Fix |
|-------|----------|---------|-----|
| No LoRA target config in GRPO | `cli/commands/train/grpo.py:95-99` | Uses default targets, no CLI option | Add `--lora-targets` option |
| Inconsistent checkpoint metadata | Various trainers | Some save config, some don't | Standardize to mlx-lm format |
| Duplicated prompt constants | Shell scripts | Prompts defined in multiple files | Extract to shared config |
| Manual mask creation | Various introspect commands | Could use `accessor.create_causal_mask()` | Use accessor method |

---

## Infrastructure Available (Use These)

### Model Loading - CANONICAL APPROACH
```python
# RECOMMENDED: For inference pipelines
from chuk_lazarus.inference import UnifiedPipeline
pipeline = UnifiedPipeline.from_pretrained(model_id)

# RECOMMENDED: For introspection/analysis (sync)
from chuk_lazarus.introspection.analyzer.loader import _load_model_sync
model, tokenizer, config = _load_model_sync(model_id)

# RECOMMENDED: For async analyzers
async with ModelAnalyzer.from_pretrained(model_id) as analyzer:
    ...

# CORE: Direct HFLoader (low-level)
from chuk_lazarus.inference.loader import HFLoader, DType
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info

result = HFLoader.download(model_id)
family_type = detect_model_family(config_data)
family_info = get_family_info(family_type)
config = family_info.config_class.from_hf_config(config_data)
model = family_info.model_class(config)
HFLoader.apply_weights_to_model(model, result.model_path, config, dtype=DType.BFLOAT16)
tokenizer = HFLoader.load_tokenizer(result.model_path)
```

### WRONG - Do NOT use mlx_lm.load() directly
```python
# WRONG - bypasses Lazarus infrastructure
from mlx_lm import load as mlx_load
model, tokenizer = mlx_load(model_id)  # Don't do this
```

### LoRA Application
```python
# CORRECT
from chuk_lazarus.models_v2 import LoRAConfig, apply_lora
lora_config = LoRAConfig(rank=16, target_modules=["v_proj", "o_proj"])
lora_layers = apply_lora(model, lora_config)

# WRONG (DualRewardTrainer does this)
lora_layers[key] = {
    "A": mx.random.normal((in_dim, rank)) * 0.01,
    "B": mx.zeros((rank, out_dim)),
    ...
}
```

### Hidden State Extraction
```python
# CORRECT
accessor = ModelAccessor(model=model, config=config)
captured = await accessor.forward_through_layers(input_ids, layers=[0, 4, 8, 12])

# WRONG (probing.py does this)
for idx, lyr in enumerate(accessor.layers):
    out = lyr(h, mask=mask)
    h = out.hidden_states if hasattr(out, "hidden_states") else out[0]
```

### Logit Projection
```python
# CORRECT
logits = accessor.apply_norm_and_head(hidden_states)

# WRONG (classifier.py does this)
if norm is not None:
    h_normed = norm(h_last)
if use_lm_head:
    head_output = lm_head(h_normed)
    logits = head_output.logits if hasattr(head_output, "logits") else head_output
else:
    logits = h_normed @ embed_weight.T
```

### Checkpoint Saving
```python
# CORRECT - mlx-lm compatible
mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), lora_weights)
with open(adapter_dir / "adapter_config.json", "w") as f:
    json.dump(adapter_config, f, indent=2)

# WRONG - deprecated format
np.savez(path / f"{name}_lora.npz", **lora_weights)
```

---

## Migration Plan

### Phase 1: Create Centralized Model Loader

**Goal:** Create a single entry point for model loading used by all of Lazarus.

#### 1.1 Create `src/chuk_lazarus/models_v2/loader.py`

```python
"""
Centralized model loading for Lazarus.

This is THE place to load models - used by inference, training, and introspection.
"""

from pathlib import Path
from typing import Any

import mlx.core as mx

from ..inference.loader import DType, HFLoader
from .families.registry import detect_model_family, get_family_info
from .adapters.lora import LoRAConfig, LoRALinear, apply_lora


def load_model(
    model_id: str,
    adapter_path: str | Path | None = None,
    dtype: DType = DType.BFLOAT16,
) -> tuple[Any, Any, Any]:
    """
    Load a model with optional adapter weights.

    Args:
        model_id: HuggingFace model ID or local path
        adapter_path: Optional path to LoRA adapter directory
        dtype: Data type for weights

    Returns:
        (model, tokenizer, config) tuple
    """
    # Download and detect family
    result = HFLoader.download(model_id)
    model_path = result.model_path

    with open(model_path / "config.json") as f:
        config_data = json.load(f)

    family_type = detect_model_family(config_data)
    if family_type is None:
        raise ValueError(f"Unsupported model: {config_data.get('model_type')}")

    family_info = get_family_info(family_type)
    config = family_info.config_class.from_hf_config(config_data)
    model = family_info.model_class(config)

    # Load base weights
    HFLoader.apply_weights_to_model(model, model_path, config, dtype=dtype)
    tokenizer = HFLoader.load_tokenizer(model_path)

    # Load adapter if provided
    if adapter_path is not None:
        adapter_path = Path(adapter_path)
        lora_config = _load_adapter_config(adapter_path)
        lora_layers = apply_lora(model, lora_config)
        _load_adapter_weights(lora_layers, adapter_path)

    return model, tokenizer, config


def load_model_with_lora(
    model_id: str,
    lora_config: LoRAConfig,
    adapter_path: str | Path | None = None,
    dtype: DType = DType.BFLOAT16,
) -> tuple[Any, Any, Any, dict[str, LoRALinear]]:
    """
    Load a model and apply LoRA adapters.

    Args:
        model_id: HuggingFace model ID or local path
        lora_config: LoRA configuration
        adapter_path: Optional path to pre-trained adapter weights
        dtype: Data type for weights

    Returns:
        (model, tokenizer, config, lora_layers) tuple
    """
    model, tokenizer, config = load_model(model_id, adapter_path=None, dtype=dtype)
    lora_layers = apply_lora(model, lora_config)

    if adapter_path is not None:
        _load_adapter_weights(lora_layers, Path(adapter_path))

    return model, tokenizer, config, lora_layers


def _load_adapter_config(adapter_path: Path) -> LoRAConfig:
    """Load LoRA config from adapter directory."""
    config_path = adapter_path / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            data = json.load(f)
        lora_params = data.get("lora_parameters", data)
        return LoRAConfig(
            rank=lora_params.get("rank", 8),
            target_modules=lora_params.get("target_modules", ["q_proj", "v_proj"]),
        )
    return LoRAConfig()


def _load_adapter_weights(lora_layers: dict[str, LoRALinear], adapter_path: Path):
    """Load adapter weights into LoRA layers."""
    weights_path = adapter_path / "adapters.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"No adapter weights at {weights_path}")

    weights = mx.load(str(weights_path))

    for name, lora_layer in lora_layers.items():
        a_key = f"model.{name}.lora_a"
        b_key = f"model.{name}.lora_b"
        if a_key in weights and b_key in weights:
            lora_layer.lora_A = weights[a_key]
            lora_layer.lora_B = weights[b_key]
```

#### 1.2 Export from `models_v2/__init__.py`

```python
from .loader import load_model, load_model_with_lora
```

### Phase 2: Update All Consumers

#### 2.1 Update `_load_model_sync` (introspection)
```python
# BEFORE: Has own implementation + mlx_lm fallback
# AFTER: Delegates to central loader

from ...models_v2.loader import load_model

def _load_model_sync(model_id: str, adapter_path: str | None = None):
    return load_model(model_id, adapter_path=adapter_path)
```

#### 2.2 Update `UnifiedPipeline` (inference)
```python
# BEFORE: Has own loading logic
# AFTER: Delegates to central loader

from ..models_v2.loader import load_model

@classmethod
def from_pretrained(cls, model_id: str, ...):
    model, tokenizer, config = load_model(model_id, dtype=dtype)
    return cls(model, tokenizer, config, ...)
```

#### 2.3 Update `train_sft` (training)
```python
# BEFORE: Uses mlx_lm.load()
# AFTER: Uses central loader

from ....models_v2.loader import load_model, load_model_with_lora

if config.use_lora:
    model, tokenizer, model_config, lora_layers = load_model_with_lora(
        config.model,
        LoRAConfig(rank=config.lora_rank, target_modules=lora_targets),
    )
else:
    model, tokenizer, model_config = load_model(config.model)
```

#### 2.4 Update `train_grpo` (training)
```python
# BEFORE: Custom _load_model_with_tokenizer()
# AFTER: Uses central loader

from ....models_v2.loader import load_model_with_lora
# Delete _load_model_with_tokenizer function
```

#### 2.5 Update `train_dpo` (training)
```python
# BEFORE: Broken import from ....models
# AFTER: Uses central loader

from ....models_v2.loader import load_model
model, tokenizer, config = load_model(model_id)
```

#### 2.6 Update `introspect_generate` (generation)
```python
# BEFORE: mlx_lm.load()
# AFTER: Uses central loader

from ....models_v2.loader import load_model
model, tokenizer, config = load_model(args.model)
```

### Phase 3: Fix Checkpoint Format

#### 3.1 Update DPO Trainer
```python
# BEFORE: NPZ format
mx.save(str(path), weights)

# AFTER: Safetensors
mx.save_safetensors(str(path), weights)
```

#### 3.2 Refactor DualRewardTrainer
- Remove custom `_setup_lora()`, `_apply_lora()`, `_get_lora_params()`, `_set_lora_params()`
- Use `apply_lora()` from models_v2
- Save checkpoints in safetensors format with `adapter_config.json`

### Phase 4: CLI Command Consolidation

#### 4.1 Create DualReward CLI Command
```bash
lazarus train dual-reward \
  --model MODEL \
  --data DATA \
  --classifier-layer 12 \
  --classifier-weight 0.4 \
  --lora-targets v_proj,o_proj
```

#### 4.2 Add Missing CLI Options
- Add `--lora-targets` to GRPO trainer (uses defaults currently)
- Add `--adapter` to all training commands for resume/fine-tuning

### Phase 5: Consolidate Introspection Commands

#### 5.1 Refactor probing.py
- Use `load_model()` via `_load_model_sync()` (which now delegates to central loader)
- Use `AsyncModelAccessor.forward_through_layers()` for hidden state extraction

#### 5.2 Refactor classifier.py
- `introspect_classifier()`: Already uses `_load_model_sync()` (now delegates to central loader)
- `introspect_logit_lens()`: Use `accessor.apply_norm_and_head()` for projection

#### 5.3 Audit Remaining Commands
- clustering.py: Minor cleanup
- embedding.py: Already good
- analyze.py: Already good (uses ModelAnalyzer which will delegate to central loader)

### Phase 6: Experiment Cleanup

#### 6.1 Update train_phase1.py
- Use refactored DualRewardTrainer with central loader
- Or replace with `lazarus train dual-reward` CLI

#### 6.2 Update generate_data.py
- Either use `lazarus generate --type math` with format adjustment
- Or document why custom generation is needed

#### 6.3 Extract Shared Config
Create `experiments/cli_classifier_emergence/config/`:
- `models.json` - model ID mappings
- `prompts.json` - test prompt definitions
- Update shell scripts to source from config

---

## Files to Modify

### Phase 1: New Files (Central Loader)
- [ ] `src/chuk_lazarus/models_v2/loader.py` (NEW - central model loading)
- [ ] `src/chuk_lazarus/models_v2/__init__.py` (export load_model, load_model_with_lora)

### Phase 2: Update Consumers to Use Central Loader
- [ ] `src/chuk_lazarus/introspection/analyzer/loader.py` (delegate to central loader)
- [ ] `src/chuk_lazarus/inference/unified.py` (delegate to central loader)
- [ ] `src/chuk_lazarus/cli/commands/train/sft.py` (replace mlx_lm.load)
- [ ] `src/chuk_lazarus/cli/commands/train/grpo.py` (delete custom loader function)
- [ ] `src/chuk_lazarus/cli/commands/train/dpo.py` (fix broken import)
- [ ] `src/chuk_lazarus/cli/commands/introspect/generation.py` (replace mlx_lm.load)

### Phase 3: Fix Checkpoint Formats
- [ ] `src/chuk_lazarus/training/trainers/dpo_trainer.py` (NPZ -> safetensors)
- [ ] `src/chuk_lazarus/training/trainers/dual_reward_trainer.py` (use apply_lora, safetensors)

### Phase 4: CLI Consolidation
- [ ] `src/chuk_lazarus/cli/commands/train/dual_reward.py` (NEW - CLI command)
- [ ] `src/chuk_lazarus/cli/main.py` (add GRPO --lora-targets, register dual-reward)

### Phase 5: Introspection Cleanup
- [ ] `src/chuk_lazarus/cli/commands/introspect/probing.py` (use ModelAccessor)
- [ ] `src/chuk_lazarus/cli/commands/introspect/classifier.py` (use accessor methods)

### Phase 6: Experiment Cleanup
- [ ] `experiments/cli_classifier_emergence/train_phase1.py` (use central loader)
- [ ] `experiments/cli_classifier_emergence/config/` (NEW - shared config)

---

## Already Good (Will Benefit from Central Loader)

These files use abstractions that will automatically benefit when their underlying loaders switch to central loader:

### Introspect Commands (use high-level abstractions)
- `analyze.py` - Uses `ModelAnalyzer.from_pretrained()` -> will delegate to central loader
- `arithmetic.py` - Uses `ModelAnalyzer.from_pretrained()` -> will delegate to central loader
- `ablation.py` - Uses `AblationStudy.from_pretrained()` -> will delegate to central loader
- `patching.py` - Uses `AblationStudy.from_pretrained()` + `CommutativityAnalyzer`
- `neurons.py` - Uses `AblationStudy.from_pretrained()` + `ModelHooks`
- `layer.py` - Uses `LayerAnalyzer.from_pretrained()`

### Experiment Files (properly designed)
- `arithmetic_rewards.py` - Properly designed reward module for GRPO (no model loading)
- `lazarus_cli_experiments.sh` - Properly uses CLI commands (doesn't load models directly)

---

## Validation Checklist

After migration, verify:

### Central Loader Works
- [ ] `from chuk_lazarus.models_v2 import load_model` works
- [ ] `load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")` returns (model, tokenizer, config)
- [ ] `load_model(..., adapter_path="path/to/adapter")` loads adapters natively
- [ ] `load_model_with_lora(...)` returns lora_layers for training
- [ ] No `mlx_lm.load()` calls remain in codebase (search: `from mlx_lm import load`)

### Training Commands
- [ ] `lazarus train sft` uses central loader
- [ ] `lazarus train dpo` runs without import error
- [ ] `lazarus train grpo` uses central loader (no custom function)
- [ ] `lazarus train dual-reward` command exists (new)
- [ ] All trainers save checkpoints as `.safetensors` + `adapter_config.json`

### Inference & Introspection
- [ ] `UnifiedPipeline.from_pretrained()` uses central loader
- [ ] `ModelAnalyzer.from_pretrained()` uses central loader
- [ ] `lazarus introspect logit-lens --adapter` loads adapters natively (no mlx_lm)

### Adapter Compatibility
- [ ] Adapters trained with `lazarus train sft` can be loaded by `load_model()`
- [ ] Adapters trained with `lazarus train grpo` can be loaded by `load_model()`
- [ ] Adapters can be loaded for inference AND introspection

### Experiment
- [ ] `train_phase1.py` uses central loader or CLI
- [ ] Shell scripts use shared config files
