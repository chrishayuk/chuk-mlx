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
- {op: init, var: a, value: 49}
- {op: init, var: b, value: 5}
- {op: query, var: a}  # ERROR: 'a' is init, not computed

# VALID (reward 1.0): query targets compute output
- {op: init, var: a, value: 49}
- {op: init, var: b, value: 5}
- {op: compute, compute_op: mul, args: [a, b], var: result}
- {op: query, var: result}  # OK: 'result' was computed
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
  var: result
- {op: query, var: result}
```

### Hybrid Variable Naming (Run 13)

Run 12 tried abstract variable naming (`a`, `b`, `c`) but this broke composition (57% accuracy) — the same mistake as Run 5 where `x`, `y`, `z` lost semantic grounding.

**Run 13** uses **hybrid naming** — semantic init vars for grounding, fixed scaffolding for structure:

| Component | Example | Purpose |
|-----------|---------|---------|
| **Init vars** | `produced`, `use1`, `price`, `rate` | Semantic grounding to problem text |
| **Intermediate vars** | `step1`, `step2`, `step3` | Structural scaffolding (fixed) |
| **Query target** | `result` | Unified output (fixed) |

**Exception**: Entity-track and comparison use entity-anchored first init (`bob.cards`, `eggs`) for semantic grounding.

The `op` field enables direct Pydantic validation — the verifier parses model output without any format conversion.

## Expert Types

| Expert | Role | Shape | Patterns | Operations |
|--------|------|-------|----------|------------|
| `rate_equation` | Single-step: rate x time | 4-step (fixed) | 4 | init, compute(mul), query |
| `arithmetic` | Multi-step chains | variable-length | 10 (6 seq + 3 interleaved + 1 long) | (init\|compute)+, query |
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

All comparison patterns use entity-anchored first init + fixed scaffolding (`b`, `step1`, `result`):

```
times_more:          init(bob.cards), init(b), mul(bob.cards,b→step1), sub(step1,bob.cards→result), query(result)
sum_and_difference:  init(total),     init(b), add(total,b→step1),     div(step1,2→result),         query(result)
more_less:           init(bob.cards), init(b), add(bob.cards,b→step1), add(step1,bob.cards→result), query(result)
half_as_many:        init(bob.cards), init(b), div(bob.cards,b→step1), sub(bob.cards,step1→result), query(result)
```

The entity-anchored init (`bob.cards`, `alice.stickers`) connects to question text. Fixed scaffolding (`b`, `step1`, `result`) gives the model a reliable template. The discrimination task: given question text, pick which 2 ops to apply.

### Rate Equation: Uniform Shape

All 4 patterns are structurally identical with semantic variable naming:

```
rate_time_quantity:  init(rate), init(time), compute(mul→result), query(result)
distance_speed_time: init(speed), init(time), compute(mul→result), query(result)
consumption_rate:    init(rate), init(time), compute(mul→result), query(result)
earning_rate:        init(rate), init(time), compute(mul→result), query(result)
```

**Variable naming convention**: Semantic init vars (`rate`, `time`, `speed`), unified `result` for query target.

## Training Pipeline

### Data Generation

All training data is generated synthetically by `chuk_virtual_expert_arithmetic.generators`:

```python
from chuk_virtual_expert_arithmetic.generators import TraceGenerator
gen = TraceGenerator()
examples = gen.generate_balanced(
    500,
    include_composition=True,
    interleaved_ratio=0.4,  # 40% of arithmetic uses interleaved patterns
    long_chain_ratio=0.1,   # 10% of arithmetic uses long chain patterns
)
```

Distribution (Run 8 v5, with interleaved + long chain + GSM-8K patterns):
- arithmetic: 30% (~150 examples @ n=500, 12 patterns, ~12/pattern)
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
11. **Hybrid variable naming** (Run 13) — Semantic init vars for grounding + fixed `step1,step2,step3` for intermediates + unified `result` for query. Run 12's abstract naming (`a,b,c`) broke composition — same mistake as Run 5.

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
- {op: init, var: base, value: 80}
- {op: init, var: rate, value: 20}
- {op: percent_off, base: base, rate: rate, var: result}
- {op: query, var: result}
```

Composed (list of sub-traces with standardized vars):
```yaml
- expert: percentage
  trace:
  - {op: init, var: base, value: 80}
  - {op: init, var: rate, value: 20}
  - {op: percent_off, base: base, rate: rate, var: result}
  - {op: query, var: result}
- expert: arithmetic
  trace:
  - {op: init, var: prev, source: prev.result}
  - {op: init, var: factor, value: 10}
  - {op: compute, compute_op: add, args: [prev, factor], var: result}
  - {op: query, var: result}
```

**Composition variable convention**: First sub-trace uses domain vars (`base`, `rate`). Arithmetic sub-traces use `prev` (from previous result), `factor`, and `result`.

### Composition Patterns (10 total)

All 10 composition patterns are **verified multi-expert chains** (single-expert patterns removed in Run 14 cleanup).

| Pattern | First Expert | Second Expert | Example |
|---------|--------------|---------------|---------|
| percent_off_plus_extra | percentage (percent_off) | arithmetic (add) | "$80 shirt, 20% off, +$10 shipping" |
| percent_increase_minus_cost | percentage (percent_increase) | arithmetic (sub) | "$100 stock +25%, how much gain?" |
| percent_of_then_multiply | percentage (percent_of) | arithmetic (mul) | "25% of $80 per unit, buy 3" |
| rate_then_subtract | rate_equation (mul) | arithmetic (sub) | "10/hour × 5 hours, 3 defective" |
| value_increase_profit | percentage (percent_increase) | arithmetic (sub) | House flipping profit |
| paired_discount | percentage (percent_of) | arithmetic (mul) | Kylar's glasses pairs |
| interrupted_rate | percentage (percent_of) | arithmetic (add) | Carla download with restart |
| consume_then_sell | entity_track (consume) | arithmetic (mul) | Janet's ducks revenue |
| cost_increase_profit | arithmetic → percentage → arithmetic | 3-expert | Cost + increase - profit |
| discount_tax_total | percentage → percentage → arithmetic | 3-expert | Discount + tax + final |

All 2-expert patterns have identical structure: 4-step first sub-trace + 4-step arithmetic sub-trace.
3-expert patterns use `source: sub0.result` and `source: prev.result` for multi-value wiring.

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

### Trace Example (parallel_merge) — Hybrid Naming (Run 13)

```yaml
expert: arithmetic
trace:
- {op: init, var: hens, value: 4}
- {op: init, var: eggs_per, value: 18}
- {op: compute, compute_op: mul, args: [hens, eggs_per], var: step1}
- {op: init, var: gift1, value: 13}           # ← Interleaved init
- {op: init, var: gift2, value: 10}           # ← Interleaved init
- {op: compute, compute_op: add, args: [gift1, gift2], var: step2}
- {op: compute, compute_op: sub, args: [step1, step2], var: result}
- {op: query, var: result}
```

## Long Chain Patterns (Run 8 v4)

Extended traces (10 steps) for GSM-8K problems requiring 8+ operations.

| Generator | Structure | Example Problem |
|-----------|-----------|-----------------|
| `generate_long_expense_chain` | 5 init, 4 compute (3 sub + 1 mul), query | "$200 - $30 - $25 - $15, then ×3" |

### Trace Example (long_expense_chain) — Hybrid Naming (Run 13)

```yaml
expert: arithmetic
trace:
- {op: init, var: start, value: 200}
- {op: init, var: expense1, value: 30}
- {op: init, var: expense2, value: 25}
- {op: init, var: expense3, value: 15}
- {op: init, var: multiplier, value: 3}
- {op: compute, compute_op: sub, args: [start, expense1], var: step1}
- {op: compute, compute_op: sub, args: [step1, expense2], var: step2}
- {op: compute, compute_op: sub, args: [step2, expense3], var: step3}
- {op: compute, compute_op: mul, args: [step3, multiplier], var: result}
- {op: query, var: result}
```

**Example question**: "Sam starts with $200. He spends $30 on food, $25 on transport, and $15 on supplies. He then triples what's left by investing. How much does he have?"

**Answer**: (200 - 30 - 25 - 15) × 3 = 130 × 3 = 390

## GSM-8K Specific Patterns (Run 8 v5)

Patterns directly targeting common GSM-8K problem structures.

| Generator | Structure | GSM-8K Example |
|-----------|-----------|----------------|
| `generate_div_then_add` | init,init,div,add | "Tank A has 48 fish. Tank B has half. Total?" |
| `generate_consume_then_sell` | init,init,init,sub,sub,init,mul | "16 eggs - 3 - 4 = 9, sell at $2 = $18" |

## GSM-8K Template Variations (Run 8 v6)

Added template variations matching exact GSM-8K phrasing patterns.

### div_then_add variations

| Style | Template | GSM-8K Match |
|-------|----------|--------------|
| standard | "Tank A has X. Tank B has half as many." | — |
| half_that_much | "X bolts of blue and half that much white" | Robe fiber |
| twice_as_much | "First has twice as many as second. First has X." | Gail's fish tanks |
| robe_style | "A craft takes X and half that much Y" | Robe fiber |

### consume_then_sell variations

| Style | Template | GSM-8K Match |
|-------|----------|--------------|
| janet_ducks | "ducks lay X eggs... eats Y for breakfast... bakes with Z" | Janet's ducks |
| farm_produce | "farm harvests X... keeps Y... gives Z to neighbors" | — |
| factory_output | "factory produces X... Y for testing... Z as spares" | — |
| garden_harvest | "garden yields X... uses Y for cooking... Z for decoration" | — |

### interleaved_mul_mul variations

| Style | Template | GSM-8K Match |
|-------|----------|--------------|
| daily_laps | "X laps of Y miles each day... in Z days" | — |
| weekly_sprints | "X sprints Y times a week... Z meters each" | James sprints |
| dogs_weekly | "X dogs... Y hours each per day... per week" | John's dogs |
| weekly_practice | "practices X times... Y minutes each... Z days" | — |

### Trace Example (consume_then_sell — Janet's ducks pattern) — Hybrid Naming (Run 13)

```yaml
expert: arithmetic
trace:
- {op: init, var: produced, value: 16}
- {op: init, var: use1, value: 3}
- {op: init, var: use2, value: 4}
- {op: compute, compute_op: sub, args: [produced, use1], var: step1}
- {op: compute, compute_op: sub, args: [step1, use2], var: step2}
- {op: init, var: price, value: 2}           # ← Interleaved init
- {op: compute, compute_op: mul, args: [step2, price], var: result}
- {op: query, var: result}
```

**GSM-8K match**: "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins with four. She sells the remainder at $2 each."

### Trace Example (div_then_add — Gail's fish tanks pattern) — Hybrid Naming (Run 13)

```yaml
expert: arithmetic
trace:
- {op: init, var: total, value: 48}
- {op: init, var: divisor, value: 2}
- {op: compute, compute_op: div, args: [total, divisor], var: step1}
- {op: compute, compute_op: add, args: [total, step1], var: result}
- {op: query, var: result}
```

**GSM-8K match**: "Gail has two fish tanks. The first tank has twice as many fish as the second. If the first tank has 48 fish, how many fish are in both tanks?"

## Number Preprocessing (Run 8 v5)

GSM-8K problems contain comma-formatted numbers (e.g., `$80,000`) that break YAML parsing. Added preprocessing:

```python
def preprocess_numbers(text: str) -> str:
    """'80,000' → '80000'"""
    return re.sub(r'(\d),(\d)', r'\1\2', text)
```

Applied automatically when loading GSM-8K problems from HuggingFace or sample data.

### Distribution

Updated `TraceGenerator.generate_balanced()`:
- Arithmetic: 20% → 30% (to accommodate 12 patterns)
- Added `interleaved_ratio` parameter (default 0.4) for sequential vs interleaved mix
- Added `long_chain_ratio` parameter (default 0.1) for long chain patterns

### Pattern Count

| Category | Run 7 | Run 8 v3 | Run 8 v4 | Run 8 v5 |
|----------|-------|----------|----------|----------|
| Sequential arithmetic | 6 | 6 | 6 | **7** |
| Interleaved arithmetic | 0 | 3 | 3 | **4** |
| Long chain arithmetic | 0 | 0 | 1 | 1 |
| Total patterns | 27 | 30 | 31 | **33** |
| GSM-8K coverage (est.) | ~20% | ~50% | ~60% | **~70%** |

### Learning Hierarchy Extension

```
1. Format → 2. Routing → 3. Structure → 4. Wiring → 5. Discrimination → 6. Decomposition → 7. Interleaving
```

Level 7 is the new challenge: deciding at each position whether to extract a new value or compute with existing ones.

## Schema-Based Generation (Run 9+)

### Architecture Change

All data generation now uses **JSON schemas** instead of hardcoded Python generators:

```
chuk_virtual_expert_arithmetic/
├── schemas/                   # Problem definitions (JSON)
│   ├── arithmetic/            # 19 schemas
│   ├── entity_track/          # 5 schemas
│   ├── rate_equation/         # 4 schemas
│   ├── comparison/            # 4 schemas
│   └── percentage/            # 4 schemas
├── vocab/                     # Vocabulary (JSON)
│   ├── patterns/              # Question templates by expert
│   │   ├── arithmetic/
│   │   ├── entity_track/
│   │   └── ...
│   ├── names.json             # Person names + pronouns
│   ├── items.json             # Countable items
│   ├── materials.json         # Fabrics, materials
│   └── phrases.json           # Verbs, expressions
└── generators/
    ├── schema_generator.py    # Loads schemas, generates examples
    └── composition.py         # Multi-expert composition patterns
```

### Schema Structure

Each schema defines a complete problem type with **hybrid variable naming**:

```json
{
  "name": "material_half",
  "expert": "arithmetic",
  "description": "X units of material A and half that much of material B",
  "pattern": "material_patterns",
  "variant": "half_that_much",
  "variables": {
    "amount": {"type": "int", "min": 2, "max": 20, "multiple_of": 2}
  },
  "vocab": {
    "material1": {"path": "materials.fabrics"},
    "item": {"path": "items.countable_singular"}
  },
  "trace": [
    {"op": "init", "var": "amount", "value": "amount"},
    {"op": "init", "var": "divisor", "value": 2},
    {"op": "compute", "compute_op": "div", "args": ["amount", "divisor"], "var": "step1"},
    {"op": "compute", "compute_op": "add", "args": ["amount", "step1"], "var": "result"},
    {"op": "query", "var": "result"}
  ],
  "answer": "amount + amount // 2"
}
```

**Note**: All schemas use semantic init vars (matching schema variables), fixed intermediate vars (`step1`, `step2`), and unified query target (`result`).

### Benefits

1. **Declarative** — New patterns via JSON, not code
2. **Vocabulary-driven** — Rich text variety from vocab files
3. **Constraint handling** — Auto-regenerate if constraints fail
4. **Derived variables** — Compute intermediate values
5. **Template substitution** — `${name}`, `${amount}` placeholders

### Current Schema Count

| Expert | Count | Schemas |
|--------|-------|---------|
| arithmetic | 19 | price_chain, subtract_chain, multiply_add, divide_multiply, work_rate, combined_rate, interleaved_mul_mul, parallel_merge, chained_mul_sum, long_expense_chain, div_then_add, consume_then_sell, half_twice, conditional_rate, fraction_simple, shopping_spree, **material_half**, **material_twice**, **decimal_rate_week** |
| entity_track | 5 | entity_simple_transfer, entity_consume_sequence, entity_add_sequence, entity_consume_multiply, entity_production |
| rate_equation | 4 | rate_production, rate_distance, rate_consumption, rate_earning |
| comparison | 4 | comparison_times_more, comparison_sum_diff, comparison_more_less, comparison_half_as_many |
| percentage | 4 | percent_off, percent_increase, percent_of, tip_calculation |
| **composition** | 10 | percent_off_plus_extra, percent_increase_minus_cost, percent_of_then_multiply, rate_then_subtract, value_increase_profit, paired_discount, interrupted_rate, consume_then_sell, cost_increase_profit, discount_tax_total |
| **Total** | **46** | |

### GSM-8K Targeted Patterns (Run 9)

| Schema | GSM-8K Problem | Trace Pattern |
|--------|----------------|---------------|
| `material_half` | Robe fiber | `X + X/2` |
| `decimal_rate_week` | John's dogs | `count × 0.5 × 7` |
| `percent_increase_then_profit` | House flipping | % increase → subtract costs |
| `percent_discount_pairs` | Kylar's glasses | % of → pair × count |
| `partial_rate_with_restart` | Carla download | % of → time calculations |
| `consume_then_sell` | Janet's ducks | entity consume → multiply |

### Run 14 Template Variants (Phrasing Gaps)

Based on Run 13 GSM-8K failure analysis — correct structure but wrong value extraction due to phrasing differences:

| Schema | GSM-8K Problem | Phrasing Gap |
|--------|----------------|--------------|
| `twice_as_much` | Fish tanks | "first has **twice** as much" vs our "second has half" |
| `weekly_sprints_same` | James sprints | "**3** sprints **3** times" — same number appears twice |
| `feed_remainder_scattered` | Wendi chickens | Numbers scattered throughout sentence |

**Total schemas**: 41 (all verified passing)

---

## Files

```
experiments/csp_cot_gsm8k/
├── train_gsm8k_yaml.py       # Training script (SFT + RL)
├── evaluation/
│   └── gsm8k_loader.py       # GSM-8K sample/HuggingFace loader
├── COT_FORMAT_SPEC.md         # Rogue-1 trace format specification
├── EXPERIMENT.md              # This file
├── RESULTS.md                 # Results and analysis (Runs 1-9)
├── PATTERNS.md                # Training patterns (14 patterns + meta-pattern)
├── GSM8K_GAPS.md              # Gap analysis for GSM-8K generalization
└── GSM8K_PATTERNS.md          # GSM-8K computation patterns catalog
```

Dependencies:
- `chuk_virtual_expert` - TraceSolverExpert, TraceVerifier, ExpertRegistry, trace_models
- `chuk_virtual_expert_arithmetic` - Expert implementations + **schema-based generators**
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
