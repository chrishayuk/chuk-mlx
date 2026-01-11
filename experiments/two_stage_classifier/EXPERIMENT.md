# Two-Stage Classifier Training Experiment

## Research Question

**Can we add explicit classifiers to a model WITHOUT destroying its computation ability?**

Previous experiments showed a fundamental tension:
- **SFT**: Good computation, weak classifiers
- **Dual-reward (70/30)**: Strong classifiers, broken computation

## Approach: Two-Stage Training

```
Stage 1: SFT (500 steps)
  → Build computation circuits
  → Fuse adapter into base weights

Stage 2: Light dual-reward on fused model
  → Train new LoRA on top of fused weights
  → Low LR to preserve computation
  → Balanced or low classifier weight
```

## Results (January 10, 2026)

### TinyLlama-1.1B Results

#### Experiment 1: 20/80 balance (200 steps)

| Stage | Symbolic | Semantic | Classifier |
|-------|----------|----------|------------|
| Baseline | 75% | 50% | 0.1% |
| Stage 1 (SFT) | 75% | 75% | 0.1% |
| Stage 2 (20/80) | **25%** | 25% | 0.1% |

**Result**: Computation DESTROYED. No classifiers emerged.

#### Experiment 2: 50/50 balance (500 steps, very low LR)

| Stage | Symbolic | Semantic | Classifier |
|-------|----------|----------|------------|
| Baseline | 75% | 50% | 0.1% |
| Stage 1 (SFT) | 75% | 75% | 0.1% |
| Stage 2 (50/50) | **75%** | **75%** | **0.1%** |

**Result**: Computation PRESERVED! But no classifiers emerged.

## Key Findings

### 1. Computation Can Be Preserved

With proper hyperparameters (very low LR, small LoRA rank), we can train on top of fused weights without destroying the computation circuits.

**What worked:**
- `learning_rate: 0.00005` (very low)
- `lora_rank: 8` (small)
- Fusing stage 1 adapter before stage 2

### 2. Classifiers Don't Emerge from Dual-Reward on TinyLlama

Despite 500 steps with 50% classifier weight, classifier tokens never appeared at intermediate layers. The classifier loss stayed high (10-13) throughout training.

**Possible reasons:**
- TinyLlama's small size (22 layers) may not have enough capacity
- The classifier layer (L12) may not be the right place for this model
- The classifier tokens need different training dynamics

### 3. Model Architecture Matters

Previous experiments on Llama-3.2-1B showed classifier emergence with dual-reward (60.1% at L8). TinyLlama shows different behavior despite similar size.

## Architecture Details

### Two-Stage Pipeline

```python
# Stage 1: SFT to build computation
mlx_lm lora --model TinyLlama --data train.jsonl

# Fuse stage 1 into base weights
mlx_lm fuse --model TinyLlama --adapter-path stage1/adapters

# Stage 2: Train new LoRA on fused model
DualRewardTrainer(fused_model, classifier_weight=0.5)
```

### Config Used

```yaml
stage1:
  method: sft
  max_steps: 500
  lora:
    rank: 16
    alpha: 32.0

stage2:
  method: dual_reward
  max_steps: 500
  learning_rate: 0.00005  # Very low
  classifier_weight: 0.5
  lora:
    rank: 8  # Small
    alpha: 16.0
```

## Conclusions

1. **Two-stage training can preserve computation** - The key is fusing stage 1 before training stage 2

2. **Classifier emergence depends on model architecture** - What works for Llama-3.2 doesn't work for TinyLlama

3. **The classifier loss objective may not be right** - 500 steps at 50% weight didn't create any classifiers

## Next Steps

1. **Try different classifier layers** - Maybe L12 is wrong for TinyLlama
2. **Try different classifier tokens** - Maybe "multiply"/"add" aren't natural for this model
3. **Try larger models** - Classifier emergence may require more capacity
4. **Analyze why classifiers emerge in some models** - What's different about Llama-3.2?

## Files

```
two_stage_classifier/
├── EXPERIMENT.md       # This file
├── README.md           # Quick start
├── experiment.py       # Implementation
├── config.yaml         # Configuration
├── data/               # Generated data
├── checkpoints/        # Trained adapters and fused models
│   ├── stage1/
│   │   └── adapters/
│   └── stage2/
│       ├── fused_stage1/  # Fused stage 1 weights
│       └── adapters/      # Stage 2 LoRA
└── results/            # Run results (JSON)
```

## Running

```bash
lazarus experiment run two_stage_classifier
```
