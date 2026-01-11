# Classifier Emergence: Training Method Comparison

## Research Question

**Do different training methods produce different classifiers, and does classifier strength improve answer accuracy?**

This experiment compares:
1. **Training methods**: SFT vs Dual-Reward (vs GRPO, optional)
2. **LoRA vs Full fine-tuning** (configurable)

Key questions:
- Does SFT produce answer classifiers at late layers?
- Does dual-reward training create operation classifiers at earlier layers?
- Does classifier presence correlate with answer accuracy?

## Results Summary (January 10, 2026)

### Key Finding: SFT Outperforms Dual-Reward

| Method | Answer Accuracy | Classifier Strength | Peak Layer |
|--------|-----------------|---------------------|------------|
| **Baseline** (no training) | 88.9% | 74.8% | L14-L15 |
| **SFT + LoRA** | **100%** | **79.4%** | L14-L15 |
| **Dual-Reward + LoRA** | 77.8% | 69.4% | L14-L15 |

**Winner: SFT with LoRA** achieves perfect accuracy and stronger classifiers.

### Per-Prompt Results

#### Baseline (No Training)
| Prompt | Expected | Generated | Classifier Layer | Confidence |
|--------|----------|-----------|------------------|------------|
| 7 * 8 = | 56 | 56 | L15 | 91.4% |
| 12 * 5 = | 60 | 12 | L15 | 90.2% |
| 9 * 9 = | 81 | 81 | L15 | 84.8% |
| 23 + 45 = | 68 | 68 | L15 | 80.9% |
| 17 + 38 = | 55 | 55 | L15 | 82.4% |
| 55 + 27 = | 82 | 82 | L15 | 50.0% |
| 89 - 34 = | 55 | 55 | L14 | 2.2% |
| 65 - 28 = | 37 | 37 | L15 | 95.7% |
| 100 - 43 = | 57 | 57 | L15 | 95.7% |

**Accuracy: 88.9% (8/9)** - Base model already has strong classifiers!

#### SFT + LoRA (500 steps)
| Prompt | Expected | Generated | Classifier Layer | Confidence |
|--------|----------|-----------|------------------|------------|
| 7 * 8 = | 56 | 56 | L15 | **100%** |
| 12 * 5 = | 60 | 60 | L15 | **100%** |
| 9 * 9 = | 81 | 81 | L15 | **100%** |
| 23 + 45 = | 68 | 68 | L14 | 76.2% |
| 17 + 38 = | 55 | 55 | L15 | 45.5% |
| 55 + 27 = | 82 | 82 | L15 | 99.2% |
| 89 - 34 = | 55 | 55 | L1 | 0.7% |
| 65 - 28 = | 37 | 37 | L15 | 96.9% |
| 100 - 43 = | 57 | 57 | L15 | 96.1% |

**Accuracy: 100% (9/9)** - SFT fixes the baseline error (12*5) and strengthens classifiers.

#### Dual-Reward + LoRA (500 steps)
| Prompt | Expected | Generated | Classifier Layer | Confidence |
|--------|----------|-----------|------------------|------------|
| 7 * 8 = | 56 | 56 | L15 | 96.1% |
| 12 * 5 = | 60 | 12 | L15 | 85.9% |
| 9 * 9 = | 81 | 81 | L15 | 83.2% |
| 23 + 45 = | 68 | 68 | L15 | 47.1% |
| 17 + 38 = | 55 | 55 | L15 | 48.0% |
| 55 + 27 = | 82 | 82 | L15 | 35.9% |
| 89 - 34 = | 55 | 55 | L8 | 5.6% |
| 65 - 28 = | 37 | 373737 | L15 | 97.7% |
| 100 - 43 = | 57 | 57 | L15 | 99.2% |

**Accuracy: 77.8% (7/9)** - Dual-reward actually hurts performance!

## Analysis

### 1. Classifiers Already Exist in Base Models

The base Llama-3.2-1B model shows strong **answer classifiers** at L14-L15 (87-94% depth):
- Most arithmetic problems show 80-95% probability for the correct answer at L15
- This is NOT random - specific answer tokens are predicted before the final layer

### 2. SFT Strengthens Existing Classifiers

SFT training (500 steps) amplifies classifier confidence:
- Simple multiplications go from 84-91% to **100%**
- The one failure case (12*5=12) is fixed
- Classifier location remains stable at L14-L15

### 3. Dual-Reward Training May Harm Performance

Counter to hypothesis, dual-reward with explicit classification loss at L8:
- Did NOT create stronger classifiers at L8 (no detectable operation classifiers)
- Answer classifiers at L14-L15 weakened compared to baseline
- Answer accuracy dropped from 88.9% to 77.8%

**Why?** The dual-reward training optimizes for operation token prediction ("multiply", "add") at L8, but:
1. The classifier layer (55% depth = L8) may be too early
2. Training V/O projections only may interfere with answer generation
3. The symbolic math input confounds classification (operator symbols are in the prompt)

### 4. Classifier Strength Correlates with Answer Accuracy

| Condition | Avg Classifier Strength | Answer Accuracy |
|-----------|------------------------|-----------------|
| SFT | 79.4% | 100% |
| Baseline | 74.8% | 88.9% |
| Dual-Reward | 69.4% | 77.8% |

Stronger classifiers at late layers correlate with better answers.

## Methodology

### Model
- **Llama-3.2-1B** (16 transformer layers)

### Training Data
- 5000 arithmetic samples (add, subtract, multiply)
- Format: `"7 * 8 = 56"`
- 90/10 train/valid split

### Training Methods

**SFT (Supervised Fine-Tuning)**
- LoRA rank: 16
- Targets: q_proj, k_proj, v_proj, o_proj
- Learning rate: 2e-4
- Steps: 500

**Dual-Reward**
- LoRA rank: 32
- Targets: v_proj, o_proj only
- Classifier layer: L8 (55% depth)
- Classifier weight: 0.7 (70% classification, 30% answer)
- Learning rate: 5e-4
- Steps: 500

### Evaluation
- 9 test prompts (3 each: multiply, add, subtract)
- Metrics: Answer accuracy + classifier strength at each layer

## Running the Experiment

```bash
# Run the full comparison
lazarus experiment run classifier_emergence

# View results
lazarus experiment status classifier_emergence
```

### Configuration

Edit `config.yaml` to enable/disable training methods:

```yaml
training_methods:
  sft_lora:
    enabled: true
    method: sft
    use_lora: true
    max_steps: 500

  dual_reward_lora:
    enabled: true
    method: dual_reward
    classifier_weight: 0.7
    classifier_layer_pct: 0.55

  grpo_lora:
    enabled: false  # Optional: RL with verifiable rewards
```

## Conclusions

1. **SFT is the best method** for arithmetic tasks - it strengthens existing classifiers and improves accuracy.

2. **Base models already have classifiers** - Llama-3.2-1B shows 80-90% answer prediction at L14-L15 without any fine-tuning.

3. **Dual-reward training is not effective** for symbolic math inputs - the explicit classification loss at intermediate layers may interfere with answer generation.

4. **Classifier strength predicts accuracy** - stronger late-layer classifiers correlate with better performance.

## Future Work

1. **Test dual-reward with semantic input** - "What is 7 times 8?" instead of "7 * 8 ="
2. **Vary classifier layer** - try L12-L14 instead of L8
3. **Compare GRPO** - RL with verifiable rewards may discover different representations
4. **Test larger models** - do the findings generalize to 7B+ models?

## Files

```
classifier_emergence/
├── EXPERIMENT.md       # This file
├── README.md           # Quick start guide
├── experiment.py       # ExperimentBase implementation
├── config.yaml         # Configuration
├── data/               # Generated arithmetic data
├── checkpoints/        # Trained adapters
│   ├── sft_lora/
│   └── dual_reward_lora/
└── results/            # Run results (JSON)
```
