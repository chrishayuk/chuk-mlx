# GSM-8K YAML Trace Training

## Thesis

LLMs should route and structure, not compute. A 1B model can learn to emit symbolic YAML traces that an external solver executes deterministically. The model never sees the answer — it wires the computation graph, the expert produces the result.

## Architecture

```
Math Problem (natural language)
       |
       v
+---------------------------+
|  Model (TinyLlama 1.1B)   |
|  - Classifies expert type  |
|  - Extracts quantities     |
|  - Wires computation graph |
|  - Cannot short-circuit    |
+---------------------------+
       |
       v  (YAML trace)
+---------------------------+
|  TraceSolverExpert         |
|  - Parses via Pydantic     |
|  - Validates query targets |
|  - Executes trace steps    |
|  - Returns answer          |
+---------------------------+
       |
       v
Answer + Verification Status
```

### Anti-Short-Circuit Constraint

The solver enforces that `query` targets must be computed/modified variables, never raw init values:

```yaml
# REJECTED (reward 0.5): query targets init variable
- {op: init, var: rate, value: 49}
- {op: init, var: time, value: 5}
- {op: query, var: rate}  # ERROR: 'rate' is init, not computed

# VALID (reward 1.0): query targets compute output
- {op: init, var: rate, value: 49}
- {op: init, var: time, value: 5}
- {op: compute, compute_op: mul, args: [rate, time], var: quantity}
- {op: query, var: quantity}  # OK: 'quantity' was computed
```

This makes it structurally impossible for the model to regurgitate extracted values. To produce a valid trace, it *must* emit compute steps and query their output.

## Trace Format

Every step has an `op` discriminator field matching the Pydantic `TraceStep` union:

```yaml
expert: entity_track
trace:
- {op: init, var: eggs, value: 16}
- {op: consume, entity: eggs, amount: 3}
- {op: consume, entity: eggs, amount: 4}
- op: compute
  compute_op: mul
  args: [eggs, 2]
  var: revenue
- {op: query, var: revenue}
```

The `op` field enables direct Pydantic validation — the verifier parses model output without any format conversion.

## Expert Types

| Expert | Role | Shape | Patterns | Operations |
|--------|------|-------|----------|------------|
| `rate_equation` | Single-step: rate x time | 4-step (fixed) | 4 | init, compute(mul), query |
| `arithmetic` | Multi-step chains | variable-length | 9 (6 seq + 3 interleaved) | (init\|compute)+, query |
| `comparison` | Two-quantity relationships | 5-step (fixed) | 4 | init(x), init(y), compute(z), compute(result), query(result) |
| `percentage` | Percent operations | 4-step (fixed) | 4 | init, domain_op, query |
| `entity_track` | Quantities moving between entities | variable | 5 | init, consume, transfer, compute, query |

### Expert Separation Principle

| Expert | Responsibility |
|--------|----------------|
| `rate_equation` | Single-step: rate x time, distance/speed/time |
| `arithmetic` | Multi-step chains: sequential + interleaved operations |

`rate_equation` is a clean 4-step pattern: init, init, compute(mul), query. Always. No variance.
`arithmetic` handles variable-length operation chains, including:
- Sequential: `init+ → compute+ → query` (price chains, divide-multiply)
- Interleaved: `(init|compute)+ → query` (parallel merge, chained multiply-sum)

## Structural Consistency Principle

**All patterns within an expert type must have the same trace shape.** A 1B model defaults to the majority template. Structural variance causes the model to emit the wrong number of steps or query the wrong variable.

### Comparison: Hybrid Variable Naming

All comparison patterns use entity-anchored first init + fixed scaffolding (factor, step1, result):

```
times_more:          init(bob.cards), init(factor), mul(bob.cards,factor→step1), sub(step1,bob.cards→result), query(result)
sum_and_difference:  init(total),     init(factor), add(total,factor→step1),     div(step1,2→result),         query(result)
more_less:           init(bob.cards), init(factor), add(bob.cards,factor→step1), add(step1,bob.cards→result), query(result)
half_as_many:        init(bob.cards), init(factor), div(bob.cards,factor→step1), sub(bob.cards,step1→result), query(result)
```

The entity-anchored init (`bob.cards`, `alice.stickers`) connects to question text. Fixed scaffolding (`factor`, `step1`, `result`) gives the model a reliable template. The discrimination task: given question text, pick which 2 ops to apply.

### Rate Equation: Uniform Shape

All 4 patterns are structurally identical:

```
rate_time_quantity:  init(rate), init(time), compute(mul→quantity), query(quantity)
distance_speed_time: init(speed), init(time), compute(mul→distance), query(distance)
consumption_rate:    init(rate), init(time), compute(mul→total),    query(total)
earning_rate:        init(rate), init(time), compute(mul→total),    query(total)
```

## Training Pipeline

### Data Generation

All training data is generated synthetically by `chuk_virtual_expert_arithmetic.generators`:

```python
from chuk_virtual_expert_arithmetic.generators import TraceGenerator
gen = TraceGenerator()
examples = gen.generate_balanced(
    500,
    include_composition=True,
    interleaved_ratio=0.5,  # 50% of arithmetic uses interleaved patterns
)
```

Distribution (Run 8, with interleaved):
- arithmetic: 30% (~150 examples @ n=500, 9 patterns, ~17/pattern)
- entity_track: 20% (~100 examples, 5 patterns, ~20/pattern)
- comparison: 15% (~75 examples, 4 patterns, ~19/pattern)
- composition: 15% (~75 examples, 4 patterns, ~19/pattern)
- percentage: 10% (~50 examples, 4 patterns, ~12/pattern)
- rate_equation: 10% (~50 examples, 4 identical patterns)

### Question Template Discipline

**One template per pattern.** Each pattern uses a single question form, varying only names/items/numbers. This gives the model maximum repetition signal per pattern.

Counter-example: Using 3 templates per pattern with 60 examples = ~5 examples per unique form. Too fragmented for a 1B model to learn the mapping.

### Serialization

Generators produce typed `TraceExample` objects with Pydantic `TraceStep` fields. The training script serializes each step individually to avoid discriminated union serialization issues:

```python
trace = [
    {k: v for k, v in step.model_dump(mode="json").items() if v is not None}
    for step in ex.trace
]
```

### Prompt Format

```
<|system|>
You are a helpful assistant with access to the following experts: entity_track, arithmetic, rate_equation, comparison, percentage</s>
<|user|>
{question}</s>
<|assistant|>
```yaml
{model generates YAML trace here}
```</s>
```

The system prompt is minimal — just lists available experts. The model learns trace format entirely from SFT examples.

### Phase 1: SFT (1 epoch)

- TinyLlama chat template
- 6 unfrozen layers + lm_head
- Adam, lr=2e-5, batch size 4
- max_len=1024 (accommodates prompt + full target)

### Phase 2: RL (REINFORCE, 20 iterations)

- Adam, lr=5e-7
- Batch size 8, temp=0.7
- Max 250 generated tokens
- Graduated reward from TraceSolverExpert:

```
1.0: Correct answer (trace executes to expected value)
0.7: Valid trace, wrong answer
0.5: Correct expert, trace execution error (includes init-only query)
0.3: Parsed YAML, wrong expert
0.0: Parse failure
```

Note: short-circuit traces (querying init vars) now receive 0.5 instead of 0.7, providing stronger negative signal during RL.

## Verification

The `TraceVerifier` (from `chuk_virtual_expert`) dispatches traces to the appropriate `TraceSolverExpert`:

1. Parse YAML via `yaml.safe_load`
2. Validate each step through Pydantic `TypeAdapter[TraceStep]` (discriminated on `op`)
3. Look up expert in registry
4. Execute trace steps maintaining state dict
5. **Enforce init-only constraint** — reject if query targets unmodified init variable
6. Compare computed answer to expected (tolerance=0.01)

Each expert handles domain-specific operations (transfer, consume, percent_of) while the base `TraceSolverExpert` handles common ones (init, compute, query).

## Key Design Decisions

1. **Model structures, expert computes** — The model never sees the answer. It wires the computation graph; the solver produces the result.
2. **Anti-short-circuit constraint** — Query targets must be computed variables. The model cannot regurgitate extracted values.
3. **Pydantic-native format** — Model output uses `op` discriminator field, enabling direct validation without format conversion.
4. **Structural consistency** — All patterns within an expert share the same step count, var names, and shape.
5. **One template per pattern** — Maximum repetition signal. Variety comes from numbers, not question structure.
6. **Hybrid var names** — Entity-anchored inits (semantic grounding) + fixed scaffolding (structural consistency).
7. **Expert separation** — Simple rate problems (4-step) vs. multi-step chains (arithmetic). No shape variance within an expert.
8. **Domain ops over manual compute** — Percentage uses `percent_of`/`percent_off`/`percent_increase` instead of mul/div chains.
9. **Minimal system prompt** — Model learns format from examples, not instructions.
10. **Synthetic data only** — No static JSON files. All training data generated by `TraceGenerator`.

## GSM-8K Coverage Analysis

### Current Coverage (10-sample probe)

| Coverage Level | Count | Description |
|---------------|-------|-------------|
| FULL | 2/10 | Existing pattern matches exactly |
| STRUCTURAL | 1/10 | Shape fits, new pattern variant needed |
| PARTIAL | 4/10 | Arithmetic handles shape, chains are longer |
| NONE | 3/10 | Requires multi-expert composition |

### Architectural Gaps

**1. Interleaved Init Pattern**
Current grammar: `init+ → compute+ → query`
Required grammar: `(init | compute)+ → query`

40% of GSM-8K problems introduce new quantities between compute steps. This changes the learning task from template filling to sequential decision-making about trace structure.

**2. Multi-Expert Composition**
30% of problems require crossing expert boundaries (e.g., percentage THEN arithmetic). The single-expert-per-trace format cannot represent these.

**3. Chain Length**
Training max is ~6 steps. GSM-8K median is 6-8 steps with some problems requiring 10+.

### Key Finding

The 0% accuracy with 100% valid traces confirms the architecture works — the model produces structurally valid output for any input. The failure is purely in wiring/semantics for unseen problem types, not in format or structure.

## Expert Composition (Implemented — Run 7)

### Architecture

```
Math Problem (natural language)
       |
       v
+---------------------------+
|  Model (TinyLlama 1.1B)   |
|  - Decomposes into steps   |
|  - Routes each sub-trace   |
|  - Wires outputs forward   |
+---------------------------+
       |
       v  (Composed YAML trace)
+---------------------------+
|  CompositionSolver         |
|  - Executes sub-traces     |
|  - Pipes prev.result       |
|  - Each expert stays clean |
+---------------------------+
       |
       v
Answer + Verification Status
```

### Format

Single-expert (dict — backward compatible):
```yaml
expert: percentage
trace:
- {op: init, var: price, value: 80}
- {op: init, var: rate, value: 20}
- {op: percent_off, base: price, rate: rate, var: result}
- {op: query, var: result}
```

Composed (list of sub-traces):
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

### Composition Patterns (4 total)

| Pattern | First Expert | Second Expert | Example |
|---------|--------------|---------------|---------|
| percent_off_plus_extra | percentage (percent_off) | arithmetic (add) | "$80 shirt, 20% off, +$10 shipping" |
| percent_increase_minus_cost | percentage (percent_increase) | arithmetic (sub) | "$100 stock +25%, how much gain?" |
| percent_of_then_multiply | percentage (percent_of) | arithmetic (mul) | "25% of $80 per unit, buy 3" |
| rate_then_subtract | rate_equation (mul) | arithmetic (sub) | "10/hour × 5 hours, 3 defective" |

All patterns have identical structure: 4-step first sub-trace + 4-step arithmetic sub-trace.

### Implementation Details

**InitStep `source` field** (`trace_models.py`):
```python
class InitStep(BaseTraceStep):
    op: Literal["init"] = "init"
    var: str
    value: float | int | str | dict[str, Any] = Field(default=0)
    source: str | None = Field(default=None)  # "prev.result" for composition
```

**CompositionSolver** (`composition_solver.py`):
- Executes sub-traces sequentially
- Resolves `source: prev.result` to actual value
- Returns final sub-trace's answer

**TraceVerifier extension** (`trace_verifier.py`):
- Detects composed format (YAML list vs dict)
- Routes to CompositionSolver for composed traces
- Same graduated reward scale

**Custom YAML formatter** (`train_gsm8k_yaml.py`):
- Forces consistent flow style `{...}` for all trace steps
- Avoids mixed block/flow styling that confused the model

### Critical Bugs Fixed

1. **`extract_yaml` only accepted dicts** — Composed traces rejected as parse failures
2. **Mixed YAML styling** — PyYAML auto-chose block style for compute with args array
3. **Semantic var names** — `sale`, `total`, `good` broke scaffolding pattern
4. **Structural inconsistency** — One pattern had 6-step arithmetic sub-trace

### Results

**Run 7**: 97% SFT accuracy, 100% parse rate with 249 examples (37 composed, 212 single). 0% GSM-8K (all valid traces, wrong answers).

### Design Principles

1. **Each sub-trace is a known template** — The model reuses patterns it already learned (levels 2-5)
2. **`source: prev.result` is the only new primitive** — Wires one expert's output to the next expert's input
3. **Expert boundaries stay clean** — No expert absorbs another's vocabulary
4. **Decomposition is the new learning task** — The model learns to split problems at expert boundaries
5. **Consistency is everything** — Same structure, same vars, same YAML style across all patterns

### Learning Hierarchy Extension

```
1. Format → 2. Routing → 3. Structure → 4. Wiring → 5. Discrimination → 6. Decomposition
```

Levels 1-6 validated (97% on synthetic with composition).

## Interleaved Init Patterns (Implemented — Run 8)

### The Grammar Gap

Run 7 GSM-8K evaluation revealed that 40% of problems introduce new quantities between compute steps:

```
Training grammar:  init+ → compute+ → query
GSM-8K requires:   (init | compute)+ → query
```

### New Patterns

| Generator | Structure | Example Problem |
|-----------|-----------|-----------------|
| `generate_interleaved_mul_mul` | init,init,compute,init,compute | "3 laps × 5 miles/day × 10 days" |
| `generate_parallel_merge` | init,init,compute,init,init,compute,compute | "(hens×per_hen)-(gift1+gift2)" |
| `generate_chained_mul_sum` | init,init,compute,init,compute,compute,compute | "city1×f1=city2, city2×f2=city3, sum" |

### Trace Example (parallel_merge)

```yaml
expert: arithmetic
trace:
- {op: init, var: hens, value: 4}
- {op: init, var: per_hen, value: 18}
- {op: compute, compute_op: mul, args: [hens, per_hen], var: produced}
- {op: init, var: gift1, value: 13}           # ← Interleaved init
- {op: init, var: gift2, value: 10}           # ← Interleaved init
- {op: compute, compute_op: add, args: [gift1, gift2], var: gifted}
- {op: compute, compute_op: sub, args: [produced, gifted], var: remaining}
- {op: query, var: remaining}
```

### Distribution

Updated `TraceGenerator.generate_balanced()`:
- Arithmetic: 20% → 30% (to accommodate 9 patterns)
- Added `interleaved_ratio` parameter (default 0.5) for sequential vs interleaved mix

### Pattern Count

| Category | Run 7 | Run 8 |
|----------|-------|-------|
| Sequential arithmetic | 6 | 6 |
| Interleaved arithmetic | 0 | 3 |
| Total patterns | 27 | **30** |
| GSM-8K coverage (est.) | ~20% | **~50%** |

### Learning Hierarchy Extension

```
1. Format → 2. Routing → 3. Structure → 4. Wiring → 5. Discrimination → 6. Decomposition → 7. Interleaving
```

Level 7 is the new challenge: deciding at each position whether to extract a new value or compute with existing ones.

## Files

```
experiments/csp_cot_gsm8k/
├── train_gsm8k_yaml.py       # Training script (SFT + RL)
├── evaluation/
│   └── gsm8k_loader.py       # GSM-8K sample/HuggingFace loader
├── COT_FORMAT_SPEC.md         # Rogue-1 trace format specification
├── EXPERIMENT.md              # This file
├── RESULTS.md                 # Results and analysis (Runs 1-7)
├── PATTERNS.md                # Training patterns (14 patterns + meta-pattern)
├── GSM8K_GAPS.md              # Gap analysis for GSM-8K generalization
└── GSM8K_PATTERNS.md          # GSM-8K computation patterns catalog
```

Dependencies:
- `chuk_virtual_expert` - TraceSolverExpert, TraceVerifier, ExpertRegistry, trace_models
- `chuk_virtual_expert_arithmetic` - Expert implementations + generators
- `chuk_lazarus` - Model loading

## Usage

```bash
# Standard training
python experiments/csp_cot_gsm8k/train_gsm8k_yaml.py

# Quick run (1 epoch SFT + 20 RL)
python experiments/csp_cot_gsm8k/train_gsm8k_yaml.py --sft-epochs 1 --rl-iters 20

# Save checkpoint
python experiments/csp_cot_gsm8k/train_gsm8k_yaml.py --save-checkpoint checkpoints/gsm8k

# Custom training size
python experiments/csp_cot_gsm8k/train_gsm8k_yaml.py --n-train 500

# Evaluate checkpoint on HuggingFace GSM-8K
python experiments/csp_cot_gsm8k/train_gsm8k_yaml.py --load-checkpoint checkpoints/gsm8k --eval-only --use-hf
```
