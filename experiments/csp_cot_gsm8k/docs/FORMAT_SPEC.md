# Rogue-1 Chain-of-Thought Trace Format Specification

> **Version:** 1.4
> **Status:** Draft
> **Author:** Chris Hay
> **Updated:** 2026-01-25

---

## Executive Summary

This document specifies the unified Chain-of-Thought (CoT) trace format for the Rogue-1 virtual expert architecture. The format enables small language models (1-2B parameters) to achieve near-perfect accuracy on reasoning tasks by separating **structure extraction** (what LLMs do well) from **computation** (what deterministic solvers do well).

### Core Principle

> **"Transformers don't compute, they lookup."**

The model outputs symbolic traces with variable references. External solvers execute the traces and compute results. This division of labor exploits the strengths of each system.

### Key Results

| Approach | Synthetic Accuracy | GSM-8K Coverage |
|----------|-------------------|-----------------|
| Model computes results | 14.5% (plateau) | — |
| Model outputs structure, solver computes | **100%** | 20% (2/10 gaps closed) |

**Run 7 Results** (TinyLlama 1.1B, 500 examples):
- 100% SFT accuracy (single epoch)
- 100% parse rate
- Expert composition working (2-expert chains)

### System Architecture

The hidden activation layer IS the router. CoT training creates geometry that the router reads:

| Signal | Routing Decision |
|--------|------------------|
| Neural expert pattern | Continue with MoE/dense computation |
| Virtual expert pattern | Route to `chuk-virtual-experts` → solver |

| Component | Repository | Role |
|-----------|------------|------|
| **Model + Router** | `chuk-mlx` (Lazarus) | Trained model; hidden activation layer routes |
| **Virtual Experts** | `chuk-virtual-experts` | Called when router signals virtual; executes via solvers |

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Format Overview](#2-format-overview)
3. [Core Structure](#3-core-structure)
4. [Expert Types (Implemented)](#4-expert-types-implemented)
5. [Expert Composition](#5-expert-composition)
6. [Operation Reference](#6-operation-reference)
7. [Verification Protocol](#7-verification-protocol)
8. [Training Protocol](#8-training-protocol)
9. [GSM-8K Coverage](#9-gsm-8k-coverage)
10. [Future Expert Types](#10-future-expert-types)

---

## 1. Architecture Overview

### 1.1 The Complete System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ROGUE-1 ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐                                                        │
│  │   INPUT     │                                                        │
│  │  (query)    │                                                        │
│  └──────┬──────┘                                                        │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     TRAINED MODEL (Lazarus)                      │   │
│  │                                                                  │   │
│  │   Embedding → Layer 1 → Layer 2 → ... → Layer N → Output        │   │
│  │                           │                                      │   │
│  │                    ┌──────┴──────┐                               │   │
│  │                    │   HIDDEN    │                               │   │
│  │                    │ ACTIVATION  │  ◄── CoT creates geometry     │   │
│  │                    │   ROUTER    │      that router reads        │   │
│  │                    └──────┬──────┘                               │   │
│  │                           │                                      │   │
│  │              ┌────────────┴────────────┐                         │   │
│  │              │                         │                         │   │
│  │              ▼                         ▼                         │   │
│  │     ┌────────────────┐       ┌────────────────┐                  │   │
│  │     │ NEURAL EXPERT  │       │ VIRTUAL EXPERT │                  │   │
│  │     │ (MoE/Dense)    │       │    SIGNAL      │                  │   │
│  │     │                │       │                │                  │   │
│  │     │ Continue with  │       │ Route to       │                  │   │
│  │     │ model compute  │       │ external solver│                  │   │
│  │     └────────────────┘       └───────┬────────┘                  │   │
│  │                                      │                           │   │
│  └──────────────────────────────────────┼───────────────────────────┘   │
│                                         │                               │
│                                         ▼                               │
│                    ┌─────────────────────────────────────┐             │
│                    │     CHUK-VIRTUAL-EXPERTS            │             │
│                    │                                     │             │
│                    │  Parse trace → Execute via solver   │             │
│                    │                                     │             │
│                    │  ┌─────────┐ ┌─────────┐ ┌───────┐  │             │
│                    │  │ Math    │ │ CSP     │ │ Time  │  │             │
│                    │  │ Expert  │ │ Expert  │ │Expert │  │             │
│                    │  └────┬────┘ └────┬────┘ └───┬───┘  │             │
│                    │       │           │          │      │             │
│                    │       ▼           ▼          ▼      │             │
│                    │  ┌─────────┐ ┌─────────┐ ┌───────┐  │             │
│                    │  │ Verifier│ │OR-Tools │ │MCP    │  │             │
│                    │  └─────────┘ └─────────┘ └───────┘  │             │
│                    └─────────────────────────────────────┘             │
│                                      │                                 │
│                                      ▼                                 │
│                               VERIFIED ANSWER                          │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 1.2 The Hidden Activation Router

The hidden activation layer IS the router. CoT training creates hidden state geometry that the router reads to make a binary decision:

| Route | When | Action |
|-------|------|--------|
| **Neural Expert** | Task is language/reasoning | Continue MoE/dense computation |
| **Virtual Expert** | Task needs exact computation | Route to chuk-virtual-experts |

This is the same routing mechanism that MoE models use internally—but extended to include virtual (external) experts alongside neural experts.

---

## 2. Format Overview

### 2.1 Basic Structure (Single Expert)

Every trace follows this YAML structure with `op` discriminator:

```yaml
expert: <expert_type>
trace:
- {op: init, var: x, value: 10}
- {op: compute, compute_op: mul, args: [x, y], var: result}
- {op: query, var: result}
```

### 2.2 Composed Structure (Multi-Expert)

For problems requiring multiple experts:

```yaml
- expert: percentage
  trace:
  - {op: init, var: price, value: 80}
  - {op: init, var: rate, value: 20}
  - {op: percent_off, base: price, rate: rate, var: result}
  - {op: query, var: result}
- expert: arithmetic
  trace:
  - {op: init, var: prev, source: prev.result}
  - {op: init, var: factor, value: 10}
  - {op: compute, compute_op: add, args: [prev, factor], var: result}
  - {op: query, var: result}
```

**Format detection**: `yaml.safe_load()` returns list → composed. Returns dict → single-expert.

### 2.3 Design Principles

| Principle | Description |
|-----------|-------------|
| **`op` discriminator** | Every step has `op` field for Pydantic union validation |
| **Symbolic references** | Operations reference variable names, not computed values |
| **Explicit state** | Every value has a named variable in the state dictionary |
| **Query-driven** | The trace specifies what to return via `{op: query, var: x}` |
| **Solver-executed** | All computation happens in the verifier/solver |
| **Consistent flow style** | All trace steps use YAML flow style `{...}` |

### 2.4 What the Model Learns

- Expert classification (question → expert type)
- Entity/variable extraction (numbers and names from questions)
- Operation selection ("gives" → transfer, "times" → mul)
- Variable wiring (which variables feed into which operations)
- Query specification (what the question asks for)
- **Decomposition** (when to split into multiple sub-traces)

### 2.5 What the Model Does NOT Learn

- Arithmetic (add, subtract, multiply, divide)
- State tracking (current value of variables)
- Search algorithms
- Constraint solving

---

## 3. Core Structure

### 3.1 Expert Declaration

```yaml
expert: <type>
```

**Implemented expert types:**

| Expert | Status | Description |
|--------|--------|-------------|
| `arithmetic` | ✅ | Multi-step computation chains |
| `entity_track` | ✅ | Quantities moving between entities |
| `rate_equation` | ✅ | Rate × time = quantity |
| `comparison` | ✅ | Compare quantities, compute differences |
| `percentage` | ✅ | Percent off/increase/of operations |

### 3.2 Trace Array

```yaml
trace:
- {op: step_type, ...}
- {op: step_type, ...}
```

The trace is an ordered list of operations. Each operation is a dictionary with `op` as the discriminator field.

### 3.3 Query Termination

Every trace ends with a query specifying what to return:

```yaml
- {op: query, var: result}
```

The solver returns the value of this variable from its final state.

---

## 4. Expert Types (Implemented)

### 4.1 Arithmetic

**Purpose:** Chain arithmetic operations on named variables.

**Structure:** `init+, compute+, query` (variable length)

**Variable naming (Run 13):** Hybrid naming — semantic init vars for grounding, fixed intermediates (`step1`, `step2`, `step3`), and unified `result` for query. Run 12's abstract naming (`a`, `b`, `c`) broke composition accuracy.

```yaml
expert: arithmetic
trace:
- {op: init, var: base, value: 100}
- {op: init, var: tax, value: 8}
- {op: init, var: shipping, value: 5}
- {op: compute, compute_op: add, args: [base, tax], var: step1}
- {op: compute, compute_op: add, args: [step1, shipping], var: result}
- {op: query, var: result}
```

**Patterns (6):**
- price_chain: price + tax + shipping
- subtract_chain: start - expense1 - expense2 - expense3
- multiply_add: count × price + extra
- divide_multiply: total ÷ people × multiplier
- work_rate: rate × time ÷ workers
- combined_rate: (rate1 + rate2) × time

---

### 4.2 Entity Tracking

**Purpose:** Track quantities moving between entities.

**Structure:** Variable length with domain operations.

```yaml
expert: entity_track
trace:
- {op: init, var: alice.marbles, value: 20}
- {op: transfer, from: alice.marbles, to: bob.marbles, amount: 8}
- {op: consume, entity: alice.marbles, amount: 3}
- {op: add_entity, entity: alice.marbles, amount: 5}
- {op: query, var: alice.marbles}
```

**Operations:**
- `transfer`: Move quantity between entities
- `consume`: Remove from entity
- `add_entity`: Add to entity

**Patterns (5):**
- consume_only: A has X, eats Y, eats Z
- transfer_basic: A has X, gives Y to B
- add_and_consume: A has X, finds Y, loses Z
- multi_transfer: A and B start with X, Y, A gives Z to B
- chain_operations: Multiple operations in sequence

---

### 4.3 Rate Equation

**Purpose:** Simple rate × time = quantity problems.

**Structure:** Fixed 4-step (all patterns identical)

**Variable naming (Run 13):** Semantic init vars (`rate`, `time`) for grounding, unified `result` for query.

```yaml
expert: rate_equation
trace:
- {op: init, var: rate, value: 60}
- {op: init, var: time, value: 5}
- {op: compute, compute_op: mul, args: [rate, time], var: result}
- {op: query, var: result}
```

**Patterns (4, all identical structure):**
- rate_time_quantity
- distance_speed_time
- consumption_rate
- earning_rate

---

### 4.4 Comparison

**Purpose:** Compare quantities, compute differences.

**Structure:** Fixed 5-step with hybrid variable naming.

```yaml
expert: comparison
trace:
- {op: init, var: bob.cards, value: 16}
- {op: init, var: factor, value: 3}
- {op: compute, compute_op: mul, args: [bob.cards, factor], var: step1}
- {op: compute, compute_op: sub, args: [step1, bob.cards], var: result}
- {op: query, var: result}
```

**Variable naming:**
- First init: entity-anchored (`bob.cards`, `alice.stickers`)
- Second init: fixed scaffolding (`factor`)
- Computes: fixed scaffolding (`step1`, `result`)
- Query: always `result`

**Patterns (4):**
- times_more: A has N× as many as B, how many more?
- sum_and_difference: A and B have X together, A has Y more
- more_less: A has Y more than B, how many together?
- half_as_many: A has half as many as B, how many more does B have?

---

### 4.5 Percentage

**Purpose:** Percent operations with domain-specific ops.

**Structure:** Fixed 4-step

```yaml
expert: percentage
trace:
- {op: init, var: price, value: 80}
- {op: init, var: rate, value: 25}
- {op: percent_off, base: price, rate: rate, var: result}
- {op: query, var: result}
```

**Domain operations:**
- `percent_off`: base × (1 - rate/100)
- `percent_increase`: base × (1 + rate/100)
- `percent_of`: base × rate/100

**Patterns (4):**
- percent_off: X% off a price
- percent_increase: X% increase
- tip_calculation: What is X% of Y?
- simple_percent: What is X% of Y?

---

## 5. Expert Composition

### 5.1 Format

Composed traces are YAML lists where each item is a sub-trace:

```yaml
- expert: percentage
  trace:
  - {op: init, var: price, value: 80}
  - {op: init, var: rate, value: 20}
  - {op: percent_off, base: price, rate: rate, var: result}
  - {op: query, var: result}
- expert: arithmetic
  trace:
  - {op: init, var: prev, source: prev.result}
  - {op: init, var: factor, value: 10}
  - {op: compute, compute_op: add, args: [prev, factor], var: result}
  - {op: query, var: result}
```

### 5.2 Wiring

The `source: prev.result` field in InitStep pipes the previous sub-trace's query result:

```yaml
- {op: init, var: prev, source: prev.result}
```

### 5.3 Implemented Composition Patterns (10 verified multi-expert)

**2-Expert Patterns (8)**:
| Pattern | First Expert | Second Expert | Example |
|---------|--------------|---------------|---------|
| percent_off_plus_extra | percentage (percent_off) | arithmetic (add) | "$80 shirt, 20% off, +$10 shipping" |
| percent_increase_minus_cost | percentage (percent_increase) | arithmetic (sub) | "$100 stock +25%, how much gain?" |
| percent_of_then_multiply | percentage (percent_of) | arithmetic (mul) | "25% of $80 per unit, buy 3" |
| rate_then_subtract | rate_equation (mul) | arithmetic (sub) | "10/hour × 5 hours, 3 defective" |
| value_increase_profit | percentage (percent_increase) | arithmetic (sub) | House flipping profit calc |
| paired_discount | percentage (percent_of) | arithmetic (mul) | Kylar's glasses pairs |
| interrupted_rate | percentage (percent_of) | arithmetic (add) | Download with restart |
| consume_then_sell | entity_track (consume) | arithmetic (mul) | Janet's ducks revenue |

**3-Expert Patterns (2)**:
| Pattern | Experts | Wiring |
|---------|---------|--------|
| cost_increase_profit | arithmetic → percentage → arithmetic | `sub0.result` + `prev.result` |
| discount_tax_total | percentage → percentage → arithmetic | `prev.result` chaining |

**Run 14 cleanup**: Removed 2 mislabeled single-expert patterns from composition.py.

### 5.4 Structural Consistency

All composition patterns follow identical structure:
- First sub-trace: 4 steps (percentage or rate_equation)
- Second sub-trace: 4 steps (arithmetic with scaffolding vars)

**Second sub-trace always uses:**
- `prev` for source value
- `factor` for second init
- `result` for output

---

## 6. Operation Reference

### 6.1 Universal Operations

| Operation | Syntax | Description |
|-----------|--------|-------------|
| `init` | `{op: init, var: x, value: 10}` | Initialize variable |
| `init` (composed) | `{op: init, var: x, source: prev.result}` | Initialize from previous sub-trace |
| `compute` | `{op: compute, compute_op: add, args: [a, b], var: result}` | Arithmetic operation |
| `query` | `{op: query, var: result}` | Specify return value |

### 6.2 Compute Operations

| compute_op | Args | Result |
|------------|------|--------|
| `add` | `[a, b]` | a + b |
| `sub` | `[a, b]` | a - b |
| `mul` | `[a, b]` | a × b |
| `div` | `[a, b]` | a ÷ b |

### 6.3 Entity Operations

| Operation | Syntax | Description |
|-----------|--------|-------------|
| `transfer` | `{op: transfer, from: a, to: b, amount: n}` | Move quantity |
| `consume` | `{op: consume, entity: e, amount: n}` | Remove from entity |
| `add_entity` | `{op: add_entity, entity: e, amount: n}` | Add to entity |

### 6.4 Percentage Operations

| Operation | Syntax | Description |
|-----------|--------|-------------|
| `percent_off` | `{op: percent_off, base: v, rate: r, var: result}` | base × (1 - rate/100) |
| `percent_increase` | `{op: percent_increase, base: v, rate: r, var: result}` | base × (1 + rate/100) |
| `percent_of` | `{op: percent_of, base: v, rate: r, var: result}` | base × rate/100 |

---

## 7. Verification Protocol

### 7.1 Execution Model

The verifier maintains a state dictionary and executes operations sequentially:

```python
def execute_trace(trace: list[dict]) -> TraceResult:
    state = {}

    for step in trace:
        op = step["op"]
        if op == "init":
            state[step["var"]] = step["value"]
        elif op == "compute":
            args = [state[a] if isinstance(a, str) else a for a in step["args"]]
            state[step["var"]] = compute(step["compute_op"], args)
        elif op == "query":
            return TraceResult(success=True, answer=state[step["var"]])

    return TraceResult(success=False, error="No query")
```

### 7.2 Graduated Reward Function

```python
def compute_reward(result: VerificationResult) -> float:
    if not result.parsed:
        return 0.0   # Can't parse YAML

    if result.expert != expected_expert:
        return 0.3   # Wrong expert classification

    if not result.trace_valid:
        return 0.5   # Invalid trace structure

    if not result.answer_correct:
        return 0.7   # Trace doesn't compute to answer

    return 1.0       # Fully correct
```

### 7.3 Error Categories

| Reward | Error Type | What It Means |
|--------|------------|---------------|
| 0.0 | Parse failure | Invalid YAML |
| 0.3 | Wrong expert | Misclassified problem type |
| 0.5 | Invalid trace | Missing variables, bad operations, init-only query |
| 0.7 | Wrong answer | Structure correct, wiring wrong |
| 1.0 | Correct | Everything right |

### 7.4 Anti-Short-Circuit Constraint

The solver enforces that `query` targets must be computed/modified variables:

```yaml
# REJECTED (reward 0.5): query targets init variable
- {op: init, var: a, value: 49}
- {op: init, var: b, value: 5}
- {op: query, var: a}  # ERROR: 'a' is init-only

# VALID (reward 1.0): query targets compute output
- {op: init, var: a, value: 49}
- {op: init, var: b, value: 5}
- {op: compute, compute_op: mul, args: [a, b], var: result}
- {op: query, var: result}  # OK: 'result' was computed
```

---

## 8. Training Protocol

### 8.1 Data Generation

```python
from chuk_virtual_expert_arithmetic.generators import TraceGenerator
gen = TraceGenerator()
examples = gen.generate_balanced(500, include_composition=True)
```

**Distribution (with composition):**
- entity_track: 25% (125 examples)
- arithmetic: 20% (100 examples)
- comparison: 20% (100 examples)
- rate_equation: 10% (50 examples)
- percentage: 10% (50 examples)
- composition: 15% (75 examples)

### 8.2 Prompt Format

```
<|system|>
You are a helpful assistant with access to the following experts: entity_track, arithmetic, rate_equation, comparison, percentage</s>
<|user|>
{question}</s>
<|assistant|>
```yaml
{model generates trace here}
```</s>
```

### 8.3 Key Training Patterns

| Pattern | Description |
|---------|-------------|
| Structural consistency | All patterns in an expert have same step count |
| One template per pattern | Maximum repetition signal |
| Hybrid var naming | Semantic inits + fixed scaffolding |
| Minimal system prompt | Model learns from examples, not instructions |
| Consistent YAML formatting | All steps use flow style `{...}` |
| Hybrid variable naming | Semantic inits + `step1,step2,step3` intermediates + unified `result` query |

### 8.4 SFT Phase

- **Epochs:** 1 (sufficient with 500 examples)
- **Learning rate:** 2e-5
- **Batch size:** 4
- **Unfrozen layers:** 6 + lm_head
- **max_len:** 1024

### 8.5 RL Phase

- **Algorithm:** REINFORCE
- **Iterations:** 5-20 (often unnecessary with good SFT)
- **Learning rate:** 5e-7
- **Temperature:** 0.7

### 8.6 Expected Results

| Phase | Parse Rate | Correct |
|-------|------------|---------|
| Baseline | 0% | 0% |
| After SFT | 100% | 100% |
| After RL | 100% | 100% |

---

## 9. GSM-8K Coverage

### 9.1 Current Status

| Coverage | Count | Percentage |
|----------|-------|------------|
| Should work | 2/10 | 20% |
| Partial (need interleaved) | 6/10 | 60% |
| Not supported | 2/10 | 20% |

### 9.2 Gaps

| Gap | Impact | Status |
|-----|--------|--------|
| Interleaved inits | 50% of problems | ❌ Not implemented |
| Longer chains (8+ steps) | 30% of problems | 🔶 Partial |
| 3-expert composition | 10% of problems | ❌ Not implemented |
| Multi-value wiring | 20% of problems | ❌ Not implemented |

### 9.3 Interleaved Init Pattern (Primary Gap)

**Current grammar:** `init+ → compute+ → query`

**Required grammar:** `(init | compute)+ → query`

40% of GSM-8K problems introduce new values mid-computation:

```yaml
# James sprints: 3×3=9, introduce 60, 9×60=540
- {op: init, var: sprints, value: 3}
- {op: init, var: times, value: 3}
- {op: compute, compute_op: mul, args: [sprints, times], var: step1}
- {op: init, var: meters, value: 60}        # ← INTERLEAVED
- {op: compute, compute_op: mul, args: [step1, meters], var: result}
- {op: query, var: result}
```

See `GSM8K_GAPS.md` and `GSM8K_PATTERNS.md` for detailed analysis.

---

## 10. Future Expert Types

### 10.1 Planned (Not Yet Implemented)

| Expert | Purpose | Solver |
|--------|---------|--------|
| `csp` | Constraint satisfaction | OR-Tools |
| `sat` | Boolean satisfiability | PySAT/Z3 |
| `smt` | Satisfiability modulo theories | Z3 |
| `mcts` | Monte Carlo tree search | Custom |
| `astar` | Pathfinding | NetworkX |
| `graph` | Graph algorithms | NetworkX |
| `sql` | Database queries | SQLite |
| `time` | Timezone/duration | chuk-mcp-time |
| `physics` | Mechanics, circuits | SymPy |
| `bayesian` | Probabilistic reasoning | PyMC |
| `code` | Code execution | Sandboxed interpreter |
| `prolog` | Logic programming | PySwip |

### 10.2 Example: CSP (Future)

```yaml
expert: csp
trace:
- {op: task, id: meeting_a, duration: 2, unit: hours}
- {op: task, id: meeting_b, duration: 1, unit: hours}
- {op: constraint, type: no_overlap, tasks: [meeting_a, meeting_b]}
- {op: constraint, type: start_after, task: meeting_a, time: "09:00"}
- {op: objective, type: minimize_makespan}
- {op: query, var: schedule}
```

### 10.3 Example: Time (Future)

```yaml
expert: time
trace:
- {op: get_time, timezone: "America/New_York", var: ny_time}
- {op: convert, from: ny_time, to_tz: "Asia/Tokyo", var: tokyo_time}
- {op: query, var: tokyo_time}
```

---

## Appendix A: Complete Examples

### A.1 Single Expert (Entity Track)

**Question:** "Alice has 20 marbles. She gives 8 to Bob, then loses 3. How many does Alice have?"

```yaml
expert: entity_track
trace:
- {op: init, var: alice.marbles, value: 20}
- {op: transfer, from: alice.marbles, to: bob.marbles, amount: 8}
- {op: consume, entity: alice.marbles, amount: 3}
- {op: query, var: alice.marbles}
```

**Execution:** 20 - 8 - 3 = **9**

### A.2 Composed (Percentage → Arithmetic)

**Question:** "A shirt costs $80. It's 20% off. Plus $10 shipping. What's the total?"

```yaml
- expert: percentage
  trace:
  - {op: init, var: price, value: 80}
  - {op: init, var: rate, value: 20}
  - {op: percent_off, base: price, rate: rate, var: result}
  - {op: query, var: result}
- expert: arithmetic
  trace:
  - {op: init, var: prev, source: prev.result}
  - {op: init, var: factor, value: 10}
  - {op: compute, compute_op: add, args: [prev, factor], var: result}
  - {op: query, var: result}
```

**Execution:**
1. percent_off: 80 × (1 - 0.20) = 64
2. arithmetic: 64 + 10 = **74**

---

## Appendix B: File Structure

```
experiments/csp_cot_gsm8k/
├── train_gsm8k_yaml.py       # Training script (SFT + RL)
├── evaluation/
│   └── gsm8k_loader.py       # GSM-8K sample/HuggingFace loader
├── EXPERIMENT.md              # Architecture & design
├── RESULTS.md                 # Run history & analysis
├── PATTERNS.md                # Training patterns discovered
├── GSM8K_GAPS.md              # Gap analysis for GSM-8K
├── GSM8K_PATTERNS.md          # GSM-8K computation patterns
└── COT_FORMAT_SPEC.md         # This specification
```

**Dependencies:**
- `chuk_virtual_expert` — TraceSolverExpert, TraceVerifier, CompositionSolver
- `chuk_virtual_expert_arithmetic` — Expert implementations + generators
- `chuk_lazarus` — Model loading

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-22 | Initial specification |
| 1.1 | 2025-01-24 | Added anti-short-circuit, hybrid naming |
| 1.2 | 2025-01-25 | Added expert composition, GSM-8K analysis, updated to `op` discriminator format |
| 1.3 | 2026-01-25 | Variable naming: Run 12 tried abstract naming (failed), Run 13 uses hybrid naming (semantic inits + `step1,step2,step3` + `result`) |
| 1.4 | 2026-01-25 | Composition cleanup: Removed 2 mislabeled single-expert patterns from composition.py, added `rate_comparison_total` schema to INTERLEAVED_SCHEMAS. All 10 composition generators now verified multi-expert. |
