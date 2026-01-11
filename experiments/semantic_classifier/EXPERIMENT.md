# Semantic Classifier Experiment: Does Explicit Classification Help?

## Research Question

**When parsing is required, do explicit classifiers improve accuracy?**

Previous experiment (`classifier_emergence`) showed dual-reward failed on symbolic input (`7 * 8 =`).
Hypothesis: It failed because classification was trivial (operator visible in input).

This experiment tests semantic input where classification is **required**:
- Input: `"seven times eight"` (must infer operation)
- NOT: `"7 * 8 ="` (operator visible)

## Results Summary (January 10, 2026)

### Critical Finding: Dual-Reward Makes Things WORSE

| Method | Accuracy | Classifier Strength |
|--------|----------|---------------------|
| **Baseline** | **77.8%** | 0.4% |
| SFT + LoRA | 66.7% | 1.4% |
| Dual-Reward + LoRA | 33.3% | **60.1%** |

**The classifier emerged (60.1%) but accuracy collapsed (33.3%).**

### Per-Prompt Results

#### Baseline (No Training)
| Input | Expected | Generated | Correct |
|-------|----------|-----------|---------|
| seven times eight | 56 | 56 | Yes |
| twelve multiplied by five | 60 | 60 | Yes |
| the product of nine and nine | 81 | 81 | Yes |
| twenty three plus forty five | 68 | 68 | Yes |
| seventeen and thirty eight | 55 | 17 | No |
| the sum of fifty five and twenty seven | 82 | 82 | Yes |
| eighty nine minus thirty four | 55 | 55 | Yes |
| sixty five take away twenty eight | 37 | 37 | Yes |
| the difference between one hundred and forty three | 57 | 143 | No |

**Accuracy: 77.8% (7/9)** - Base model handles most semantic math!

#### SFT + LoRA (500 steps)
| Input | Expected | Generated | Correct |
|-------|----------|-----------|---------|
| seven times eight | 56 | 56 | Yes |
| twelve multiplied by five | 60 | 60 | Yes |
| the product of nine and nine | 81 | 81 | Yes |
| twenty three plus forty five | 68 | 68 | Yes |
| seventeen and thirty eight | 55 | 55 | Yes |
| the sum of fifty five and twenty seven | 82 | 72 | No |
| eighty nine minus thirty four | 55 | 15 | No |
| sixty five take away twenty eight | 37 | 17 | No |
| the difference between one hundred and forty three | 57 | 57 | Yes |

**Accuracy: 66.7% (6/9)** - SFT slightly hurt performance!

#### Dual-Reward + LoRA (500 steps)
| Input | Expected | Generated | Classifier |
|-------|----------|-----------|------------|
| seven times eight | 56 | 56 | 54.5% multiply |
| twelve multiplied by five | 60 | **50** | 71.9% multiply |
| the product of nine and nine | 81 | 81 | 75.4% multiply |
| twenty three plus forty five | 68 | **23** | 53.5% add |
| seventeen and thirty eight | 55 | **38** | 47.7% add |
| the sum of fifty five and twenty seven | 82 | **55** | 58.6% add |
| eighty nine minus thirty four | 55 | 55 | 64.8% subtract |
| sixty five take away twenty eight | 37 | **65** | 57.4% subtract |
| the difference between one hundred and forty three | 57 | **100** | 57.0% subtract |

**Accuracy: 33.3% (3/9)** - Classifier works but computation fails!

## Analysis

### 1. The Classifier Emerged Successfully

Dual-reward training created real operation classifiers at L8:
- Average 60.1% probability for correct operation token
- Up from 0.4% in baseline
- This proves the training objective works for classification

### 2. But Classification Didn't Help Computation

The model learned to classify but **forgot how to compute**:

```
"twelve multiplied by five"
  → Classifier: "multiply" (71.9% correct!)
  → Output: "50" (WRONG - should be 60)

"twenty three plus forty five"
  → Classifier: "add" (53.5% correct!)
  → Output: "23" (WRONG - should be 68)
```

Pattern: The model outputs the **first number** instead of computing.

### 3. Why This Happened

The dual-reward loss was:
```
total_loss = 0.7 * classifier_loss + 0.3 * answer_loss
```

**70% weight on classification starved the computation path.**

The model optimized for:
1. Predicting "multiply" at L8 ✓
2. Actually computing 12 * 5 = 60 ✗

These are **different skills** and the loss balance favored classification.

### 4. SFT Also Hurt (Slightly)

Even SFT dropped from 77.8% to 66.7%. Why?

The training data format was:
```
"seven times eight = 56"
```

This may have:
- Overfit to specific phrasings in training data
- Lost generalization to test phrasings
- Small model + 500 steps = unstable

## Key Insight: Classification ≠ Computation

```
CLASSIFICATION: "What operation is this?" → "multiply"
COMPUTATION:    "What is 12 * 5?"         → "60"

These are DIFFERENT capabilities.
Having one doesn't give you the other.
```

GPT-OSS's operation classifiers at L13 work because:
1. Classification **routes** to different computation circuits
2. The computation circuits were **already trained**
3. Classification is a **side effect**, not the goal

Our dual-reward approach:
1. Trained classification explicitly
2. Undertrained computation (only 30% of loss)
3. Got a classifier without a computer

## What Would Actually Work

### Option 1: Two-Stage Training
```
Stage 1: SFT on arithmetic (build computation skills)
Stage 2: Dual-reward to add classifiers (without hurting computation)
```

### Option 2: Better Loss Balance
```
# Early training: focus on answers
total_loss = 0.2 * classifier_loss + 0.8 * answer_loss

# Late training: add classifier pressure
total_loss = 0.5 * classifier_loss + 0.5 * answer_loss
```

### Option 3: Just Use SFT
The base model already has 77.8% accuracy. Classifiers emerge implicitly with scale.

## Conclusions

1. **Explicit classifiers don't help** - Dual-reward training creates classifiers but hurts accuracy (77.8% → 33.3%)

2. **Classification ≠ Computation** - Predicting "multiply" doesn't mean the model can compute 12*5

3. **Loss balance is critical** - 70% classifier weight starved the computation path

4. **Base models are surprisingly good** - Llama-3.2-1B handles 77.8% of semantic math without training

5. **SFT can hurt on small data** - 500 steps on semantic math slightly degraded performance

## Comparison with classifier_emergence

| Experiment | Input Type | Best Method | Best Accuracy |
|------------|------------|-------------|---------------|
| classifier_emergence | Symbolic (`7 * 8 =`) | SFT | 100% |
| semantic_classifier | Semantic (`seven times eight`) | Baseline | 77.8% |

On both input types:
- **SFT ≥ Baseline** for symbolic (where classification is trivial)
- **Baseline > SFT > Dual-Reward** for semantic (where parsing matters)

Dual-reward consistently underperforms, whether classification is needed or not.

## Files

```
semantic_classifier/
├── EXPERIMENT.md       # This file
├── README.md           # Quick start
├── experiment.py       # Implementation
├── config.yaml         # Configuration
├── data/               # Generated semantic data
├── checkpoints/        # Trained adapters
└── results/            # Run results (JSON)
```

## Running

```bash
lazarus experiment run semantic_classifier
```
