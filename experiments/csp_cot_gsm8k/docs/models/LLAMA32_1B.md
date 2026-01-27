# Llama-3.2-1B — GSM-8K YAML Trace Training

## Models Tested

| Model | Type | Parameters |
|-------|------|------------|
| meta-llama/Llama-3.2-1B | Base | 1.0B |
| meta-llama/Llama-3.2-1B-Instruct | Instruct | 1.0B |

**Training**: 6 unfrozen layers + lm_head

---

## Summary

Llama-3.2-1B was tested in both base and instruct variants to evaluate the impact of instruction-tuning on format learning and reasoning generalization.

**Key Finding**: Llama's instruction-tuning **helps** reasoning without hurting format learning — opposite of SmolLM2!

| Metric | Llama Base | Llama Instruct | SmolLM2 Instruct |
|--------|------------|----------------|------------------|
| SFT accuracy | 95% | **95%** | 65% |
| Parse rate | 93% | **100%** | 83% |
| Valid traces | 93% | **100%** | 83% |
| GSM-8K 30-sample | 7% | **17%** | 7% |

---

## Key Findings

### 1. Base Model Learns Format Well

Unlike SmolLM2-Instruct, Llama-3.2-1B base quickly adopts the YAML trace format:

| Phase | Llama-3.2-1B (base) | SmolLM2-1.7B (instruct) |
|-------|---------------------|------------------------|
| SFT accuracy | **95%** | 65% |
| SFT parse rate | **100%** | 95% |
| RL convergence | Fast (8/8 by iter 4) | Slow (5-6/8 throughout) |

The base model has weaker priors and doesn't fight the new format.

### 2. Cleaner Traces Than Instruct Models

Llama-3.2-1B produces structurally cleaner traces with fewer bugs:

| Issue | Llama-3.2-1B | SmolLM2-1.7B |
|-------|--------------|--------------|
| Invalid traces | 2/30 (7%) | 5/30 (17%) |
| Invented operations | Rare | Common |
| Repeated sub-traces | None | Yes |
| Variable overwriting | Rare | Common |

### 3. Same Accuracy Ceiling (7%)

Despite cleaner format, both models hit the same **7% GSM-8K accuracy**:

```
Llama-3.2-1B:  2/30 correct (7%)
SmolLM2-1.7B:  2/30 correct (7%)
```

**Implication**: The bottleneck is not format learning — it's reasoning generalization.

### 4. Failure Patterns

Llama-3.2-1B exhibits the same semantic failures as other models:

| Pattern | Example |
|---------|---------|
| Missing operations | Tip without adding to base (got 225, expected 18) |
| Wrong operation | Subtracted instead of added for interest (got 95, expected 106) |
| Incomplete computation | Missing final step in multi-step problems |
| Multi-entity confusion | "Half as many X as Y" relationships |
| Value extraction errors | Wrong numbers from problem text |

---

## GSM-8K 30-Sample Results

| Metric | Value |
|--------|-------|
| **Correct** | **2/30 (7%)** |
| Parsed | 28/30 (93%) |
| Valid traces | 28/30 (93%) |
| Wrong answer | 23 |
| Invalid trace | 2 |
| Wrong expert | 0 |

### Sample Failures

**Beanstalk (expected 10, got 9)**:
```yaml
- {op: compute, compute_op: mul, args: [start, multiplier], var: step1}  # 3×2=6
- {op: compute, compute_op: add, args: [step1, start], var: result}      # 6+3=9 (should add 4!)
```
Model adds `start` (3) instead of the growth amount (4).

**Interest (expected 106, got 95)**:
```yaml
- {op: compute, compute_op: sub, args: [start, rate], var: step1}  # 100-2=98 (should ADD!)
- {op: compute, compute_op: sub, args: [step1, time], var: result} # 98-3=95
```
Model subtracts instead of calculating compound interest.

**Milk calories (expected 48, got 16)**:
```yaml
- {op: compute, compute_op: mul, args: [amount, rate_decimal], var: result}  # 8×2=16
```
Missing the calories per ounce multiplication (×3).

---

## Training Results

### SFT Phase
```
Epoch 1: loss=0.0263, correct=95%, parsed=100%
```

Loss curve shows rapid convergence:
- Step 50: 0.0554
- Step 100: 0.0297
- Step 200: 0.0117
- Step 500: 0.0084
- Step 750: 0.0018

### RL Phase
```
RL iter  1: 8/8 correct (reward=1.00)
RL iter  5: 8/8 correct (eval: 19/20)
RL iter 10: 7/8 correct (eval: 19/20)
RL iter 20: 7/8 correct (eval: 19/20)
```

Consistently high performance throughout RL, reaching 95% eval accuracy.

---

## Comparison: Base vs Instruct

| Aspect | Base Model | Instruct Model |
|--------|------------|----------------|
| Format learning | Excellent | Poor |
| Trace structure | Clean | Buggy |
| GSM-8K accuracy | 7% | 7% |
| Reasoning | Limited | Limited |

**Conclusion**: Base models learn format better, but neither model type can reason about novel problems. The 7% ceiling appears to be a **reasoning limitation**, not a format limitation.

---

## Checkpoints

| Checkpoint | Accuracy | Notes |
|------------|----------|-------|
| `llama32_1b_base_run1_sft` | 95% | After SFT |
| `llama32_1b_base_run1` | 95% | After 20 RL iterations |

---

## Conclusions

### Base Model Advantage Confirmed

Llama-3.2-1B base learns the YAML format much better than SmolLM2-1.7B Instruct:
- 95% vs 65% SFT accuracy
- 93% vs 83% GSM-8K parse rate
- Cleaner traces with fewer structural bugs

### Reasoning Limitation Exposed

Both base and instruct models hit the same 7% GSM-8K ceiling. This suggests:

1. **Format is not the bottleneck** — Both can produce valid YAML
2. **Reasoning is the bottleneck** — Neither can generalize to novel problems
3. **Model size may matter** — 1-1.7B models may lack capacity for reasoning

---

## Run 22: Llama-3.2-1B-Instruct (Best Result!)

**Date**: 2026-01-26
**Model**: meta-llama/Llama-3.2-1B-Instruct

### Results

| Metric | Llama Instruct | Llama Base | Delta |
|--------|----------------|------------|-------|
| SFT accuracy | **95%** | 95% | = |
| Parse rate | **100%** | 93% | +7% |
| Valid traces | **100%** | 93% | +7% |
| GSM-8K | **17% (5/30)** | 7% (2/30) | **+10%** |

### Key Finding: Llama Instruction-Tuning Helps!

Unlike SmolLM2-Instruct which degraded both format learning AND reasoning, Llama-Instruct:

1. **Maintains format learning** — 95% SFT (same as base)
2. **Improves trace quality** — 100% valid vs 93% for base
3. **Improves reasoning** — 17% GSM-8K vs 7% for base

### Training Metrics

```
SFT: 95% correct, 100% parsed
RL: 8/8 correct for 18/20 iterations (near-perfect)
Eval: 19/20 correct throughout RL
```

### GSM-8K 30-Sample Breakdown

| Metric | Value |
|--------|-------|
| **Correct** | **5/30 (17%)** |
| Parsed | 30/30 (100%) |
| Valid traces | 30/30 (100%) |
| Wrong answer | 22 |
| Invalid trace | 0 |
| Wrong expert | 0 |

### Why Llama-Instruct Works Better

| Aspect | Llama-Instruct | SmolLM2-Instruct |
|--------|----------------|------------------|
| Instruction-tuning style | Compatible | Interferes |
| Format learning | Preserved | Degraded |
| Trace structure | Clean | Buggy |
| Reasoning transfer | Yes | No |

Llama's instruction-tuning appears to be more "general purpose" and doesn't create strong priors against new output formats. SmolLM2's instruction-tuning seems more aggressive, creating priors that actively resist new formats.

---

## Model Comparison Summary

| Model | Type | Format | Reasoning | Best For |
|-------|------|--------|-----------|----------|
| TinyLlama 1.1B | Light chat | ⭐⭐⭐ | ⭐ | Format learning |
| Llama-3.2-1B | Base | ⭐⭐⭐ | ⭐ | Clean traces |
| **Llama-3.2-1B-Instruct** | Instruct | ⭐⭐⭐ | ⭐⭐ | **Best overall** |
| SmolLM2-1.7B | Instruct | ⭐ | ⭐ | Avoid |

---

## Run 23: Llama-3.2-3B-Instruct

**Date**: 2026-01-26
**Model**: meta-llama/Llama-3.2-3B-Instruct (3x larger)

### Results

| Metric | 3B-Instruct | 1B-Instruct | Delta |
|--------|-------------|-------------|-------|
| SFT accuracy | **100%** | 95% | +5% |
| GSM-8K | **27% (8/30)** | 17% (5/30) | **+10%** |
| Parse rate | 97% | 100% | -3% |

### Key Finding: Sublinear Scaling

**3x parameters → only 1.6x performance**

The 3B model improves over 1B but with diminishing returns. Same failure patterns persist:
- Multi-entity confusion
- Missing final operations
- Wrong operation selection

### New Bugs in 3B

```yaml
# Expressions in init (invalid!)
{op: init, var: total, value: 4 * 18}

# Undefined variables
{op: compute, compute_op: mul, args: [count2, factor2], var: step2}
```

---

## Run 24-25: Layer Unfreezing Experiments

### Hypothesis

Unfreezing more layers would improve semantic understanding.

### Results

| Config | Layers Unfrozen | SFT | GSM-8K |
|--------|-----------------|-----|--------|
| 1B + 6 layers | 37% | 95% | **17%** |
| 1B + 8 layers | 50% | 95% | **17%** (no change!) |
| 1B + 16 layers (full) | 100% | 100% | **7%** (WORSE!) |

### Key Finding: Full Fine-Tune = Catastrophic Forgetting

The full fine-tune **overwrote** the model's base capabilities:

```yaml
# Partial fine-tune (correct):
{op: init, var: half, value: 2}

# Full fine-tune (wrong):
{op: init, var: half, value: 3}  # Why 3?!
```

The model memorized training patterns so hard it forgot basic math.

---

## Final Model Comparison

| Model | Size | Layers | GSM-8K | Notes |
|-------|------|--------|--------|-------|
| Llama-3.2-1B (base) | 1.0B | 6/16 | 7% | Clean format |
| Llama-3.2-1B-Instruct | 1.0B | 6/16 | **17%** | Best efficiency |
| Llama-3.2-1B-Instruct | 1.0B | 8/16 | 17% | No improvement |
| Llama-3.2-1B-Instruct | 1.0B | 16/16 | 7% | Catastrophic forgetting |
| **Llama-3.2-3B-Instruct** | 3.2B | 6/28 | **27%** | **Best overall** |

---

## Conclusions

### What We Learned

1. **Size helps (sublinearly)** — 3B gets 27% vs 1B's 17%
2. **Layer unfreezing doesn't help** — 8 layers = 6 layers
3. **Full fine-tune is catastrophic** — Goes from 17% to 7%
4. **The bottleneck is DATA DIVERSITY** — Not capacity or layers

### The Path Forward

The fix isn't more capacity or more trainable layers. It's **more diverse training patterns** that cover GSM-8K's linguistic variety.
