# TinyLlama 1.1B — GSM-8K YAML Trace Training

**Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
**Parameters**: 1.1B
**Type**: Base model with chat fine-tuning (light)
**Training**: 6 unfrozen layers + lm_head

---

## Summary

TinyLlama 1.1B is the primary model for YAML trace training experiments. It achieved strong results on training data and small GSM-8K probes, but failed to generalize to the full GSM-8K distribution.

| Metric | Best Result | Run |
|--------|-------------|-----|
| Training accuracy | 100% | Run 18, 19 |
| GSM-8K 10-sample | **90%** | Run 18 |
| GSM-8K 100-sample | **~2%** | Run 19 |
| Composition | 100% | Run 18 |
| Parse rate | 100% | All runs |

---

## Key Findings

### 1. Format Mastery Without Reasoning

The model achieves 100% parse rate on GSM-8K — every output is valid, parseable YAML with correct structure. But accuracy is ~2% because the model:

- **Pattern matches** instead of understanding problem semantics
- **Memorizes templates** rather than learning to reason
- **Extracts wrong values** from unfamiliar phrasings

### 2. The 10-Sample Trap

| Probe Size | Accuracy | Interpretation |
|------------|----------|----------------|
| 10 samples | 90% | "Almost solved!" |
| 100 samples | ~2% | "Completely broken" |

The 10-sample probe happened to match training templates. The remaining 1309 GSM-8K problems use different phrasings that the model cannot handle.

### 3. Composition Works Perfectly

Multi-expert traces (2-3 experts chained together) achieve 100% accuracy when the token limit is sufficient (750 tokens). The model correctly:
- Routes to multiple experts
- Wires `source: prev.result` between sub-traces
- Maintains YAML list structure

### 4. Base Model Advantage

TinyLlama's light chat tuning doesn't interfere with learning the YAML format. Unlike heavily instruction-tuned models, it has weak priors and easily adopts new structured output formats.

---

## Run History

| Run | Config | Training | GSM-8K | Key Finding |
|-----|--------|----------|--------|-------------|
| 1-5 | Format iterations | 75-95% | — | Template diversity hurts, abstract vars fail |
| 6 | Hybrid naming | 95% | 0% | All fixes validated, 100% valid traces |
| 7 | + Composition | 97% | 0% | Multi-expert traces work |
| 8 | + Interleaved init | 98% | 0% | (init\|compute)+ grammar |
| 9 | + Schema-based | 96% | 30% | First real progress |
| 10-13 | Pattern expansion | 90-94% | 30-50% | Variable naming critical |
| 14-17 | Cleanup + token fix | 85-100% | — | Token limit discovered |
| **18** | **max_tokens=750** | **96%** | **90%** | Best run |
| **19** | **3000 examples** | **100%** | **~2%** | 100-sample reveals truth |

---

## Architecture Insights

### What TinyLlama Learns Well

1. **YAML syntax** — Never produces malformed YAML after SFT
2. **Expert routing** — Correct expert selection 95%+ of the time
3. **Trace structure** — Correct step count and variable flow
4. **Template matching** — High accuracy on seen patterns

### What TinyLlama Fails At

1. **Novel phrasings** — Cannot map unseen question forms to known patterns
2. **Value extraction** — Confuses which numbers map to which variables
3. **Operation selection** — Picks wrong operations for unfamiliar problems
4. **Multi-entity reasoning** — "Twice as many X as Y" relationships

---

## Training Configuration

```bash
# Best configuration (Run 18)
python experiments/csp_cot_gsm8k/train_gsm8k_yaml.py \
    --n-train 1500 \
    --sft-epochs 1 \
    --rl-iters 10 \
    --eval-sample 50 \
    --max-tokens 750
```

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate (SFT) | 2e-5 | Adam optimizer |
| Learning rate (RL) | 5e-7 | REINFORCE |
| Batch size | 4 | SFT and RL |
| Unfrozen layers | 6 + lm_head | ~30% of model |
| Max sequence length | 1024 | Prompt + response |
| Max generation tokens | 750 | Critical for composition |

---

## Failure Analysis (100-Sample GSM-8K)

### Failure Categories

| Category | % | Example |
|----------|---|---------|
| Multi-entity confusion | 30% | "Bob has twice as many as Alice" |
| Value extraction | 25% | Wrong numbers from problem text |
| Multi-step truncation | 20% | 6-step problems get 2-3 steps |
| Rate/time confusion | 15% | Confuses rate, time, quantity |
| Decimal handling | 10% | 0.5 becomes 5 |

### Root Cause

The model learns a **lookup table** from question patterns to trace templates. When a question doesn't match a known pattern, it:
1. Picks the closest template (often wrong)
2. Extracts values that fit that template (often wrong)
3. Produces a valid trace with wrong answer

---

## Checkpoints

| Checkpoint | Run | Accuracy | Notes |
|------------|-----|----------|-------|
| `gsm8k_yaml_schema_run_18` | 18 | 96% train, 90% GSM-8K | Best overall |
| `gsm8k_yaml_schema_run_19` | 19 | 100% train, ~2% GSM-8K | More data, same issues |

---

## Conclusions

### TinyLlama 1.1B is Good For:
- Learning structured output formats (YAML traces)
- Multi-expert composition
- Pattern matching on seen templates

### TinyLlama 1.1B is Bad For:
- Generalizing to novel phrasings
- Mathematical reasoning beyond template lookup
- Value extraction from unfamiliar problem structures

### The Fundamental Limitation

**Format mastery ≠ Reasoning capability**

A 1.1B model can learn to produce perfectly structured YAML traces that parse and execute correctly. But it cannot learn to *reason* about math problems — it can only memorize patterns and hope new problems match.
