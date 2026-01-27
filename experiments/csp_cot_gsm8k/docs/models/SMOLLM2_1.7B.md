# SmolLM2-1.7B — GSM-8K YAML Trace Training

**Model**: HuggingFaceTB/SmolLM2-1.7B-Instruct
**Parameters**: 1.7B (55% larger than TinyLlama)
**Type**: Instruction-tuned model
**Training**: 6 unfrozen layers + lm_head

---

## Summary

SmolLM2-1.7B was tested to evaluate whether a larger, instruction-tuned model would generalize better to novel GSM-8K phrasings. **The hypothesis was rejected** — the larger model performed worse than TinyLlama 1.1B.

| Metric | SmolLM2-1.7B | TinyLlama 1.1B | Delta |
|--------|--------------|----------------|-------|
| SFT accuracy | 65% | 100% | **-35%** |
| Final training | 78% | 96% | **-18%** |
| GSM-8K 10-sample | 30% | 90% | **-60%** |
| GSM-8K 30-sample | **7%** | ~2% (100-sample) | +5% |
| Parse rate (30-sample) | 83% | 100% | **-17%** |
| Composition | 22% | 100% | **-78%** |

---

## Key Findings

### 1. Instruction-Tuning Hurts Format Learning

SmolLM2's instruction-tuning creates strong priors that resist learning the YAML trace format:

```
# Baseline output (before training):
<|entity_track|><|rate_equation|><|percentage|><|comparison|>...
```

The model initially tries to output special tokens rather than YAML — its chat training interferes with format learning.

### 2. Composition Catastrophic Failure (22%)

The most dramatic regression. Multi-expert traces require:
- YAML list-of-dicts format
- Cross-expert variable wiring (`source: prev.result`)
- 250-350 tokens of structured output

SmolLM2 struggles with all of these, producing malformed or incomplete traces.

### 3. Variable Overwriting Bug

SmolLM2 exhibits a structural error not seen in TinyLlama — **reinitializing computed variables**:

```yaml
- {op: compute, compute_op: mul, args: [start, factor], var: step1}  # step1 = 6
- {op: init, var: step1, value: 3}  # OVERWRITES step1 = 3 !!!
```

This destroys the computation and produces wrong answers.

### 4. Expert Boundary Violations

The model routes to arithmetic but emits percentage operations:

```yaml
expert: arithmetic
trace:
- {op: percent_of, base: distance, rate: 60, var: step1}  # WRONG EXPERT!
```

This cross-expert contamination causes invalid traces.

---

## GSM-8K 30-Sample Analysis

### Final Results

| Metric | Value |
|--------|-------|
| Correct | **2/30 (7%)** |
| Parsed | 25/30 (83%) |
| Valid traces | 25/30 (83%) |
| Wrong answer | 16 |
| Invalid trace | 5 |
| Wrong expert | 0 |

### Failure Patterns

#### 1. Missing Final Operations

The model extracts values correctly but doesn't complete the computation:

| Problem | Model Did | Missing Step | Expected |
|---------|-----------|--------------|----------|
| Pizza tip (20%) | 15 × 0.2 = 3 | + 15 | **18** |
| Typing average | 47+52+57 = 156 | / 3 | **52** |
| Milk calories | 8 × 2 = 16 | × 3 | **48** |

#### 2. Wrong Operation Selection

| Problem | Model Did | Should Do |
|---------|-----------|-----------|
| Bridge weight | 5000 + 3755 | 5000 - 3755 |
| Gum packs | 4 × 15 × 30 | (4 × 30) / 15 |
| Salary total | 5000 × 5000 | 5000 + 10000 |

#### 3. Multi-Entity Confusion

"Half as many X as Y and half as many Y as Z" problems completely fail:
- Robots/helmets/footballs: Expected 70, got 20.5
- Model cannot track multiple entity relationships

#### 4. Variable Overwriting

```yaml
# Pet store legs problem
- {op: init, var: items, value: 5}   # dogs
- {op: init, var: items, value: 2}   # cats (overwrites!)
- {op: init, var: items, value: 10}  # birds (overwrites!)
# Final items = 10, loses dog and cat counts
```

#### 5. Missing Query Step

Some traces are incomplete — no `{op: query}` at the end:

```yaml
- {op: compute, compute_op: add, args: [time, result], var: time}
# Missing: - {op: query, var: time}
```

#### 6. Repeated Sub-Traces (Catastrophic)

For "10 times a month" problems, model emits the SAME sub-trace 10 times:

```yaml
- expert: arithmetic
  trace: [{rate×time}]
- expert: arithmetic
  trace: [{rate×time}]  # REPEATED 10 TIMES!
...
```

Instead of computing once and multiplying by 10.

#### 7. Invented Operations

Model invents operations that don't exist:

```yaml
{op: add, args: [result1, result2], var: result}  # INVALID!
# Should be: {op: compute, compute_op: add, args: [...]}
```

#### 8. String Instead of Variable Reference

```yaml
{op: init, var: chinese, value: step1}  # 'step1' as STRING, not variable!
```

#### 9. Conflicting Init (source AND value)

```yaml
{op: init, var: remaining, source: step5, value: 20}  # Both source AND value??
```

---

## Comparison: TinyLlama vs SmolLM2

| Aspect | TinyLlama 1.1B | SmolLM2-1.7B |
|--------|----------------|--------------|
| Format learning | Excellent | Poor |
| Composition | 100% | 22% |
| Variable management | Clean | Overwrites computed vars |
| Expert routing | 95%+ correct | Boundary violations |
| Parse rate | 100% | 95% |
| Instruction-tuning | Light | Heavy (interferes) |

### Why TinyLlama Wins

1. **Weaker priors** — Base model easily adopts new format
2. **No chat interference** — Doesn't fight the YAML structure
3. **Consistent variable handling** — Never overwrites computed vars
4. **Clean expert boundaries** — Doesn't mix operation vocabularies

---

## Training Configuration

```bash
# Run 20 configuration
python experiments/csp_cot_gsm8k/train_gsm8k_yaml.py \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --n-train 3000 \
    --sft-epochs 1 \
    --rl-iters 20 \
    --max-tokens 750 \
    --save-checkpoint experiments/csp_cot_gsm8k/checkpoints/smollm2_1.7b_run1
```

### By Expert (Final Eval, 50 samples)

| Expert | Accuracy | Notes |
|--------|----------|-------|
| arithmetic | 81% (13/16) | Wrong operations |
| comparison | 100% (7/7) | Works well |
| composition | **22% (2/9)** | Catastrophic |
| entity_track | 92% (11/12) | Variable issues |
| percentage | 100% (3/3) | Small sample |
| rate_equation | 100% (3/3) | Small sample |

---

## Conclusions

### Bigger ≠ Better for Format Learning

The hypothesis that a larger model would generalize better was **rejected**. Instead:

1. **Instruction-tuning hurts** — Strong chat priors resist new formats
2. **Composition fails** — Complex structured output degrades
3. **New bugs emerge** — Variable overwriting, expert contamination

### When to Use SmolLM2-1.7B

- General chat and instruction-following tasks
- Tasks that align with its instruction-tuning
- NOT for learning new structured output formats

### When to Use TinyLlama 1.1B

- Learning new structured output formats (YAML, JSON, etc.)
- Tasks requiring format consistency
- Multi-expert composition traces

---

## Checkpoints

| Checkpoint | Accuracy | Notes |
|------------|----------|-------|
| `smollm2_1.7b_run1_sft` | 65% (SFT only) | Before RL |
| `smollm2_1.7b_run1` | 78% (final) | After 20 RL iterations |

---

## Next Steps

1. **Try SmolLM2-1.7B base model** (non-Instruct) — May have weaker priors
2. **Adjust learning rate** — Larger model may need different hyperparameters
3. **Chat template investigation** — May need custom template for YAML
4. **Compare with other 1.7B models** — Phi-2, StableLM, etc.
