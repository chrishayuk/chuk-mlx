# Next Experiments

**Date**: 2026-01-27
**Status**: Proposed experiments based on REVIEW.md findings

---

## Priority 1: Fix Evaluation (Immediate)

### Experiment A: True Baseline Measurement

**Problem**: We don't know real performance. The 10-sample was contaminated.

**Action**:
```bash
python train_gsm8k_yaml.py \
    --load-checkpoint checkpoints/llama32_3b_instruct_best \
    --eval-only \
    --use-hf \
    --eval-n 500 \
    --hf-only
```

**Metrics to capture**:
- Accuracy by problem length (short/medium/long)
- Accuracy by operation count (2-4 / 5-6 / 7+)
- Failure mode breakdown (parse / wrong expert / wrong answer)
- Per-expert accuracy on real problems

**Effort**: 1 hour
**Expected outcome**: True baseline number (likely 5-15% based on 100-sample result)

---

## Priority 2: Train on Real GSM-8K (High Impact)

### Experiment B: GSM-8K Training Split with Auto-Traces

**Hypothesis**: Training on real GSM-8K problems (not just synthetic) will dramatically improve generalization.

**Approach**:
1. Take GSM-8K training split (7473 problems)
2. Use LLM (Claude/GPT-4) to generate trace annotations
3. Filter to problems that verify correctly
4. Train on mix of synthetic + real

**Implementation**:
```python
# scripts/annotate_gsm8k.py
async def annotate_problem(problem: GSM8KProblem) -> dict | None:
    """Use Claude to generate trace for GSM-8K problem."""
    prompt = f"""
    Convert this math problem to a YAML trace.

    Question: {problem.question}
    Answer: {problem.answer}

    Output format:
    ```yaml
    expert: <arithmetic|entity_track|rate_equation|comparison|percentage>
    trace:
    - {{op: init, var: ..., value: ...}}
    - {{op: compute, compute_op: ..., args: [...], var: ...}}
    - {{op: query, var: result}}
    ```
    """
    # Call Claude API, verify trace, return if correct
```

**Training mix**:
- 50% annotated GSM-8K (real linguistic variety)
- 50% synthetic (structural coverage)

**Effort**: 1-2 days (annotation + training)
**Expected outcome**: 30-50% accuracy (2-3x improvement)

---

### Experiment C: Few-Shot In-Context Annotation

**Alternative to B**: Instead of pre-annotating, use few-shot prompting at inference.

**Approach**:
```python
def format_prompt_with_examples(question: str, similar_examples: list[dict]) -> str:
    """Include solved examples in prompt."""
    prompt = f"<|system|>\n{SYSTEM_PROMPT}</s>\n"

    # Add 3 similar solved examples
    for ex in similar_examples[:3]:
        prompt += f"<|user|>\n{ex['query']}</s>\n"
        prompt += f"<|assistant|>\n```yaml\n{format_trace(ex)}```</s>\n"

    # Add target question
    prompt += f"<|user|>\n{question}</s>\n<|assistant|>\n```yaml\n"
    return prompt
```

**Requires**: Embedding model for similarity search

**Effort**: 1 day
**Expected outcome**: 20-40% accuracy with good example selection

---

## Priority 3: Linguistic Diversity (Medium Impact)

### Experiment D: Template Augmentation

**Problem**: One template per pattern → memorization, not understanding.

**Approach**: Generate 5-10 paraphrases per pattern using LLM.

```python
# Before (1 template):
"A shirt costs ${price}. It's {rate}% off. What's the sale price?"

# After (5+ templates):
"A shirt costs ${price}. It's {rate}% off. What's the sale price?"
"The regular price of a shirt is ${price}. With a {rate}% discount, how much does it cost?"
"A ${price} shirt is on sale for {rate}% off. What do you pay?"
"There's a {rate}% off sale. A shirt normally costs ${price}. What's the discounted price?"
"You want to buy a shirt that costs ${price}. It's marked down {rate}%. Final price?"
```

**Implementation**:
```python
# scripts/augment_templates.py
def augment_template(template: str, n: int = 5) -> list[str]:
    """Use LLM to generate paraphrases preserving placeholders."""
    prompt = f"""
    Generate {n} paraphrases of this math problem template.
    Preserve all {{placeholders}} exactly.
    Vary sentence structure, vocabulary, and phrasing.

    Template: {template}
    """
    # Return list of paraphrased templates
```

**Effort**: 0.5 days
**Expected outcome**: 10-20% improvement on novel phrasings

---

### Experiment E: Word Number Training

**Problem**: GSM-8K uses "three" (48% of problems), training uses "3".

**Approach**: Add word→number preprocessing AND train with mixed representations.

```python
# 50% of training: "Alice has 5 apples"
# 50% of training: "Alice has five apples"
```

**Effort**: 0.5 days
**Expected outcome**: 5-10% improvement on word-number problems

---

## Priority 4: Architecture Changes (Exploratory)

### Experiment F: Hybrid CoT + Trace

**Hypothesis**: Natural language CoT activates reasoning; trace provides verification.

**Approach**: Two-stage generation:
1. Model generates natural language reasoning
2. Model converts reasoning to structured trace
3. Solver verifies trace

```yaml
# Stage 1 output (natural language):
reasoning: |
  Janet has 16 eggs. She eats 3 and bakes with 4.
  Remaining: 16 - 3 - 4 = 9 eggs
  She sells at $2 each: 9 × 2 = $18

# Stage 2 output (structured trace):
expert: arithmetic
trace:
- {op: init, var: eggs, value: 16}
- {op: init, var: eaten, value: 3}
- {op: init, var: baked, value: 4}
- {op: compute, compute_op: sub, args: [eggs, eaten], var: step1}
- {op: compute, compute_op: sub, args: [step1, baked], var: remaining}
- {op: init, var: price, value: 2}
- {op: compute, compute_op: mul, args: [remaining, price], var: result}
- {op: query, var: result}
```

**Effort**: 2-3 days
**Expected outcome**: Unknown, but may unlock pre-trained reasoning

---

### Experiment G: Iterative Refinement

**Hypothesis**: Let model see execution errors and self-correct.

**Approach**:
```
Turn 1: Model generates trace
        Solver executes → Error: "variable 'step2' not defined"
Turn 2: Model sees error, generates corrected trace
        Solver executes → Success
```

**Implementation**: Multi-turn generation with error feedback.

**Effort**: 2 days
**Expected outcome**: May help with complex traces, unknown magnitude

---

### Experiment H: Smaller Expert Granularity

**Hypothesis**: 5 experts is too coarse. Finer routing may help.

**Current**: arithmetic, entity_track, rate_equation, comparison, percentage

**Proposed**: Split arithmetic into sub-experts:
- `add_chain` — sequential additions
- `mul_chain` — sequential multiplications
- `mixed_chain` — interleaved operations
- `rate_time` — rate × time patterns

**Effort**: 1-2 days
**Expected outcome**: Better routing accuracy, unknown impact on final accuracy

---

## Priority 5: Scale Experiments (Resource-Intensive)

### Experiment I: More Training Data

**Current**: 1500-3000 examples
**Proposed**: 10,000-30,000 examples

**Approach**: Scale synthetic generation + GSM-8K annotation.

**Effort**: Compute-bound (training time)
**Expected outcome**: Diminishing returns unless diversity also increases

---

### Experiment J: Larger Models

**Current best**: Llama-3.2-3B (27%)
**Proposed**: Llama-3.2-8B or Llama-3.1-8B

**Effort**: 2-3x training time
**Expected outcome**: Maybe 35-45% (sublinear scaling observed)

---

## Recommended Experiment Order

| Order | Experiment | Effort | Expected Impact | Dependencies |
|-------|------------|--------|-----------------|--------------|
| 1 | A: True baseline | 1 hr | Establishes ground truth | None |
| 2 | D: Template augmentation | 0.5 days | 10-20% improvement | None |
| 3 | B: GSM-8K training split | 1-2 days | 2-3x improvement | Annotation script |
| 4 | E: Word numbers | 0.5 days | 5-10% improvement | None |
| 5 | C: Few-shot retrieval | 1 day | 20-40% improvement | Embedding model |
| 6 | F: Hybrid CoT + Trace | 2-3 days | Unknown (exploratory) | New training format |

---

## Success Criteria

| Milestone | Accuracy | Status |
|-----------|----------|--------|
| Current best | 27% | Achieved (Llama-3.2-3B) |
| **Target 1** | 40% | After experiments A-E |
| **Target 2** | 55% | After experiment B (real GSM-8K) |
| **Target 3** | 70% | With hybrid approach + scale |
| SOTA (fine-tuned 7B) | 70-80% | Comparison point |

---

## Quick Wins (Do This Week)

1. **Run experiment A** — Get true baseline on 500 samples
2. **Run experiment D** — Generate 5 paraphrases per template
3. **Start experiment B** — Write annotation script, annotate 1000 GSM-8K problems

These three experiments can run in parallel and will provide the foundation for everything else.
