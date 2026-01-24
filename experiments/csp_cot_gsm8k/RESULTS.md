# GSM-8K YAML Trace Training Results

**Date**: 2026-01-24
**Model**: TinyLlama 1.1B (6 unfrozen layers + lm_head)
**Training**: 250 synthetic examples, 1 epoch SFT + 20 RL iterations

---

## Summary

| Metric | Value |
|--------|-------|
| Overall accuracy | **93%** (230/248) |
| Parse rate | 100% |
| Valid traces | 100% |
| Correct expert | 100% |

A 1B model can learn to emit structured YAML traces that an external solver executes deterministically. The model routes and structures; it does not compute.

---

## Results by Expert Type

| Expert | Accuracy | Count | Notes |
|--------|----------|-------|-------|
| arithmetic | 100% | 42/42 | Simplest format (init, compute, query) |
| entity_track | 97% | 102/105 | Transfer/consume patterns |
| rate_equation | 93% | 39/42 | Formula-based |
| percentage | 88% | 15/17 | percent_off / percent_increase |
| comparison | 76% | 32/42 | Complex simultaneous equations |

### Why Comparison is Weakest

Comparison traces are the most complex (8+ steps, simultaneous equations):

```yaml
expert: comparison
trace:
- {init: base, value: 120}
- formula: {var: person_a, expression: "base * 3"}
- formula: {var: person_b, expression: "base + 50"}
- compute: {op: sub, args: [person_a, person_b], var: difference}
- {query: difference}
```

The model must wire multiple variable references across formula and compute steps. Errors typically involve incorrect variable names in `args` lists.

---

## Training Progression

### Phase 1: SFT (1 epoch)

```
Epoch 1, batch 1/62: loss=4.521
Epoch 1, batch 10/62: loss=1.983
Epoch 1, batch 30/62: loss=0.892
Epoch 1, batch 62/62: loss=0.451

Post-SFT evaluation:
  Correct: 90%  Parsed: 100%  Valid: 100%
```

### Phase 2: RL (REINFORCE, 20 iterations)

```
RL iter 1:  reward=0.93  correct=93%
RL iter 5:  reward=0.93  correct=93%
RL iter 10: reward=0.93  correct=93%
RL iter 15: reward=0.93  correct=93%
RL iter 20: reward=0.93  correct=93%
```

RL provides marginal improvement (+3%) and stabilizes quickly. SFT does the heavy lifting.

---

## GSM-8K Generalization (10-sample probe)

```
Correct: 2/10 (20%)
Valid traces: 9/10 (90%)
Parse rate: 10/10 (100%)
```

### Failure Analysis

The model produces valid, well-structured traces for unseen GSM-8K problems, but often chooses wrong operations or variable wiring. Key issues:

1. **Number format**: GSM-8K uses commas in numbers (`80,000`) which breaks YAML parsing
2. **Multi-step reasoning**: Real problems often require 4-6 chained operations; training examples max at 3-4
3. **Operation selection**: Model picks plausible but incorrect ops (e.g., `add` instead of `mul`)

The 90% valid trace rate shows the model has learned the format; accuracy will improve with more diverse training data.

---

## Key Findings

### 1. System Prompt Length Matters

| System Prompt | Post-SFT Accuracy |
|--------------|-------------------|
| Verbose (450 tokens) | 70% |
| Minimal (1 line) | 90% |

The verbose prompt consumed context budget. For a fine-tuned model, a minimal expert list is sufficient:

```
You are a helpful assistant with access to the following experts: entity_track, arithmetic, rate_equation, comparison, percentage
```

### 2. max_len Truncation is Silent and Fatal

With `max_len=512`, the system prompt (~50 tokens) + question (~60 tokens) + target YAML (~200 tokens) fit, but only after reducing the system prompt. The original 450-token prompt left zero room for targets, causing 100% of training examples to be truncated. Result: 0% correct, 15% parsed.

Fix: `max_len=1024` accommodates all examples comfortably.

### 3. Synthetic Data is Sufficient

250 synthetic examples (no static/hand-crafted data) achieve 93% accuracy on the synthetic distribution. The `TraceGenerator` produces enough variety within each expert type.

### 4. Format Subset Confusion

Arithmetic traces (`init, compute, query`) are a strict subset of entity_track format. The model never confuses them because expert classification is learned from the question semantics, not trace structure.

---

## Graduated Reward Structure

```
1.0: Correct answer (trace executes to expected value)
0.7: Valid trace, wrong answer
0.5: Correct expert, trace execution error
0.3: Parsed YAML, wrong expert
0.0: Parse failure
```

After SFT, all examples score >= 0.7 (valid traces). RL optimizes from 0.7/1.0 mix toward 1.0.

---

## Comparison: Before and After Fixes

| Configuration | Parsed | Valid | Correct |
|---------------|--------|-------|---------|
| max_len=512, verbose prompt | 15% | 0% | 0% |
| max_len=1024, verbose prompt | 100% | 100% | 70% |
| max_len=1024, minimal prompt | 100% | 100% | 90% |
| + RL (20 iters) | 100% | 100% | 93% |

---

## Next Steps

1. **Scale training data** - 1000+ examples with more operation variety
2. **Handle number formats** - Preprocess commas in numbers for GSM-8K
3. **Improve comparison** - More diverse simultaneous equation patterns
4. **Longer chains** - Train on 5-6 step traces to match GSM-8K complexity
5. **New expert types** - Time, weather (minimal prompt makes this easy: just add to expert list)
