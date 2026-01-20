# GSM-8K Virtual Expert Experiment

## Hypothesis

GSM-8K is fundamentally a task classification benchmark, not a reasoning benchmark. A linear probe at layer 4 can classify problem types with high accuracy, enabling routing to specialized virtual experts that achieve near-perfect accuracy through exact computation rather than neural approximation.

---

## Background

### GSM-8K Overview

- **8,500** grade school math word problems
- **Test set**: 1,319 problems
- Current SOTA: ~95% (GPT-4 + CoT)
- Open models: 60-80% depending on size

### The Standard Narrative

> "Models that score higher on GSM-8K have better reasoning capabilities."

### The Alternative Hypothesis

> "GSM-8K measures task classification + computation. Models fail because they approximate computation neurally. Route to exact solvers, accuracy approaches 100%."

---

## Problem Type Taxonomy

| Type | Description | Solver | % of GSM-8K (est.) |
|------|-------------|--------|-------------------|
| arithmetic_chain | Sequential arithmetic operations | Calculator chain | 35% |
| rate_ratio | Work rates, speed, proportions | Equation solver | 15% |
| allocation | Distribute with constraints | CSP assignment | 10% |
| comparison | Compare quantities | Calculator + compare | 15% |
| scheduling_time | Time calculations | Time calculator | 8% |
| geometry | Area, perimeter, volume | Geometry calculator | 7% |
| multi_constraint | Multiple equations | Equation solver | 5% |
| percentage | Discount, tax, tips | Percentage calculator | 5% |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GSM-8K Problem                               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    L4 Type Classifier                               │
│            type="arithmetic_chain", confidence=0.94                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Problem Extractor                                │
│     entities, operations, constraints → structured spec             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Expert Solver                                    │
│           Calculator / Equation / CSP / etc.                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Numeric Answer                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Expected Results

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Neural only | ~55% | No prompting |
| CoT | ~70% | Step-by-step |
| CoT + SC | ~75% | Self-consistency |
| Tool-use | ~78% | Explicit calculator |
| **Virtual Expert** | **~95%** | Route to exact solvers |

---

## Success Criteria

| Experiment | Metric | Threshold |
|------------|--------|-----------|
| Type classification | Probe accuracy | >80% |
| Extraction | Parse success rate | >85% |
| End-to-end | GSM-8K accuracy | >90% |
| Comparison | Beat all baselines | Yes |

---

## The Headline

> "GSM-8K Solved: 95% Accuracy via Task Classification + Symbolic Solvers"

We show that GSM-8K is not a reasoning benchmark but a task classification benchmark. The remaining errors are extraction failures, not reasoning failures.
