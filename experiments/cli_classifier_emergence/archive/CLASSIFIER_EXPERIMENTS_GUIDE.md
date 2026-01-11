# Classifier Emergence Experiments Guide

## Overview

This guide documents experiments to detect and create **vocabulary-aligned operation classifiers** in language models. We demonstrate that:

1. **Base models have hidden-space classifiers** (detectable via linear probe)
2. **Base models do NOT have vocab-aligned classifiers** (0% via logit lens)
3. **V/O training creates vocab-aligned classifiers** (36-81% via logit lens)
4. **Frozen classifier + routing training enables circuit routing**

---

## Quick Start

```bash
# Run the complete experiment
./experiments/cli_classifier_emergence/run_experiment.sh all

# Or run individual phases:
./experiments/cli_classifier_emergence/run_experiment.sh generate   # Create training data
./experiments/cli_classifier_emergence/run_experiment.sh baseline   # Measure base model
./experiments/cli_classifier_emergence/run_experiment.sh phase1     # Dual-reward training
./experiments/cli_classifier_emergence/run_experiment.sh phase2     # Routing training
./experiments/cli_classifier_emergence/run_experiment.sh verify     # Check results
```

---

## CLI Commands

### 1. Linear Probe Classification

Detect hidden-space classifiers at each layer:

```bash
lazarus introspect classifier -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --classes "multiply:7 * 8 = |12 * 5 = |3 * 9 = |6 * 7 = " \
  --classes "add:23 + 45 = |17 + 38 = |11 + 22 = |5 + 9 = " \
  --classes "subtract:50 - 23 = |89 - 34 = |77 - 11 = |40 - 15 = " \
  --classes "divide:48 / 6 = |81 / 9 = |36 / 4 = |24 / 3 = " \
  --test "11 * 12 = |6 * 9 = |13 + 14 = |25 + 17 = " \
  --output results/classifier.json
```

**Expected Result**: 100% accuracy at all layers (hidden-space classifiers exist)

### 2. Logit Lens Analysis

Check for vocabulary-aligned classifiers:

```bash
lazarus introspect logit-lens -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompts "7 * 8 = |12 * 5 = |23 + 45 = |17 + 38 = " \
  --targets "multiply" --targets "add" --targets "subtract" --targets "divide" \
  --output results/logit_lens.json
```

**Expected Result (base model)**: 0% for all target tokens
**Expected Result (after training)**: 36-81% for correct operation tokens

### 3. SFT Training with Layer Freezing

Train with frozen classifier layer:

```bash
lazarus train sft \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data data/arithmetic_sft.jsonl \
  --freeze-layers 0-12 \
  --use-lora \
  --lora-targets v_proj,o_proj \
  --lora-rank 16 \
  --output checkpoints/routing \
  --max-steps 300
```

---

## YAML Configuration

For reproducible experiments, use YAML configs:

### Phase 1: Dual-Reward Training

```yaml
# configs/dual_reward_phase1.yaml
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
output: checkpoints/phase1_classifier

epochs: 1
max_steps: 500
batch_size: 1
learning_rate: 0.001

use_lora: true
lora_rank: 16
lora_targets: v_proj,o_proj

data: data/arithmetic_sft.jsonl

# Intermediate loss configuration
intermediate_loss:
  enabled: true
  layer: 12
  weight: 0.4
  targets:
    multiply: "multiply"
    add: "add"
    subtract: "subtract"
    divide: "divide"
```

### Phase 2: Routing Training

```yaml
# configs/routing_phase2.yaml
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
adapter: checkpoints/phase1_classifier
output: checkpoints/phase2_routing

epochs: 1
max_steps: 300
learning_rate: 0.0005

freeze_layers: 0-12
use_lora: true
lora_rank: 16
lora_targets: v_proj,o_proj

data: data/arithmetic_sft.jsonl
mask_prompt: true
```

Run with:
```bash
lazarus train sft --config configs/dual_reward_phase1.yaml
lazarus train sft --config configs/routing_phase2.yaml
```

---

## Key Findings

### Two Types of Classifiers

| Type | Detection Method | Base Model | After V/O Training |
|------|-----------------|------------|-------------------|
| Hidden-space | Linear probe | 100% | 100% |
| Vocab-aligned | Logit lens | 0% | 36-81% |

### V/O Projections are the Mechanism

| Target | Success Rate |
|--------|-------------|
| V/O only | 100% (8/8) |
| Q/K only | 0% (0/8) |

**Q/K controls attention routing (where to look)**
**V/O controls value composition (what to extract)**

### Training Requirements

| Method | Vocab-Aligned Classifiers |
|--------|--------------------------|
| No training | 0% |
| Standard LoRA | ~1% |
| SFT on answers | 0% |
| RL with verifiable rewards | 0% |
| Dual-reward V/O | 36-81% |

---

## File Structure

```
experiments/cli_classifier_emergence/
├── run_experiment.sh              # Main experiment runner
├── generate_data.py               # Training data generator
├── arithmetic_rewards.py          # Reward function for GRPO
├── CLASSIFIER_EXPERIMENTS_GUIDE.md  # This file
├── EXPERIMENT_WRITEUP.md          # Detailed results
├── lazarus_cli_experiments.sh     # Original CLI experiments
├── configs/
│   ├── dual_reward_phase1.yaml    # Phase 1 config
│   └── routing_phase2.yaml        # Phase 2 config
├── data/
│   └── arithmetic_sft.jsonl       # Generated training data
├── checkpoints/
│   ├── phase1_classifier/         # Trained classifier
│   └── phase2_routing/            # Trained routing layers
└── results/
    ├── baseline_classifier.json   # Linear probe results
    └── baseline_logit_lens.json   # Logit lens results
```

---

## Model Compatibility

| Model | Layers | Classifier Layer (55%) |
|-------|--------|----------------------|
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 22 | L12 |
| meta-llama/Llama-3.2-1B | 16 | L9 |
| ibm-granite/granite-3.1-2b-base | 40 | L22 |

---

## Next Steps

1. **Scale testing**: Verify on 7B+ models
2. **Generalization**: Test beyond arithmetic (sentiment, code classification)
3. **Virtual experts**: Use classifier signal for runtime routing
4. **Latency benchmarks**: Compare classifier-based vs neural routing
