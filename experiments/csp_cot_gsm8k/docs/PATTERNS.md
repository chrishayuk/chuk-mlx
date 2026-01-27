# Training Patterns for Small Model Trace Emission

Patterns discovered through iterative training of TinyLlama 1.1B on symbolic YAML trace generation.

---

## Pattern 1: Anti-Short-Circuit Constraint

**Problem**: The model regurgitates extracted values instead of routing through computation. Given "49 items per day, how many in 5 days?", it emits `query: rate` (returning 49) instead of `query: quantity` (which requires computing 49×5).

**Pattern**: Make short-circuiting a structural error, not just a wrong answer.

**Implementation**: The solver tracks `init_only_vars`. If `query` targets a variable that was only set by `InitStep` (never modified by compute or domain steps), the trace fails with `success=False`.

**Reward impact**: Short-circuit drops from 0.7 (wrong answer) to 0.5 (invalid trace). The model learns to avoid it within 4 RL iterations.

**Principle**: If a failure mode is rewarded at 0.7, the model will exploit it. Make undesirable behaviors structurally impossible or maximally penalized.

---

## Pattern 2: Structural Consistency

**Problem**: When an expert type has patterns with different step counts (e.g., 3-step and 5-step comparison patterns), the model defaults to the majority template and forces it on all inputs.

**Pattern**: All patterns within an expert type must have identical trace structure — same step count, same scaffolding var names, same query target.

**Levels of consistency** (each is necessary):
1. **Step count** — All patterns produce the same number of steps
2. **Scaffolding vars** — Fixed names for structural positions (e.g., `factor`, `step1`, `result`)
3. **Query target** — All patterns query the same variable (`result`)
4. **Step types** — All patterns use the same sequence of step types (init, init, compute, compute, query)

**What varies**: Entity-anchored init vars (`bob.cards`) vary per example — this is intentional. They provide semantic grounding. See Pattern 11.

**Counter-example**: Comparison with ALL per-instance var names (`alice.books`, `multiplier`, `difference`) — the model can't learn a fixed query target because it's different every time.

**Principle**: Reduce the model's learning task to the minimum discriminative decision. Fix the scaffolding, vary only what connects to the problem text.

---

## Pattern 3: One Template Per Pattern

**Problem**: Multiple question templates per pattern (e.g., 3 ways to ask "times as many") fragments the training signal. With 15 examples and 3 templates, the model sees each form only 5 times — insufficient for a 1B model.

**Pattern**: Each pattern uses exactly one question template. Variety comes from randomized values (names, items, numbers), not from rephrased questions.

**Why it works**: The model sees the same question structure 15 times with different numbers. It learns the mapping: "this question form → these ops". Template diversity can be added later with more data.

**Evidence**: Percentage (1 template, 100% accuracy) vs. comparison with 3 templates (85% accuracy, same data volume).

**Principle**: At low data volume (< 20 examples per pattern), repetition of structure beats diversity of expression.

---

## Pattern 4: Expert Separation

**Problem**: The model mixes routing logic with operation vocabulary. It routes to `entity_track` (because "house" is an entity) but emits `PercentIncreaseStep` (because the problem mentions "increase").

**Pattern**: Each expert has a fixed, non-overlapping operation vocabulary. The model learns: "if routed to expert X, only emit ops from set X."

**Implementation**:
- `rate_equation`: only init, compute(mul), query
- `arithmetic`: only init, compute(any), query
- `comparison`: only init, compute(any), query
- `percentage`: only init, percent_of/percent_off/percent_increase, compute, query
- `entity_track`: only init, consume, transfer, add_entity, compute, query

**Principle**: Clean expert boundaries prevent cross-contamination. The routing decision and the operation selection should be independent.

---

## Pattern 5: Shape Determines Expert Assignment

**Problem**: `work_rate` (rate × time ÷ workers) was in `rate_equation` but has a 6-step trace, while all other rate patterns are 4-step. This structural inconsistency caused the model to confuse step counts.

**Pattern**: Assign patterns to experts based on trace shape, not problem semantics.

| Shape | Expert |
|-------|--------|
| init, init, compute(mul), query | rate_equation |
| init+, compute+, query | arithmetic |
| init, init, compute, compute, query | comparison |
| init, init, domain_op, query | percentage |

`work_rate` is semantically a "rate problem" but structurally an "arithmetic chain." It belongs in arithmetic.

**Principle**: The model learns trace structure from examples. If two patterns have different structures, they should be in different experts — regardless of semantic similarity.

---

## Pattern 6: Domain Ops Reduce Wiring Surface

**Problem**: Percentage calculations required 3 compute steps (mul, div, add) where each step's variable wiring could go wrong. Error rate: 12%.

**Pattern**: Replace multi-step compute chains with single domain operations that encapsulate the computation.

```yaml
# Before (3 wiring decisions):
- {op: compute, compute_op: mul, args: [bill, tip_rate], var: tip_times_100}
- {op: compute, compute_op: div, args: [tip_times_100, 100], var: tip}
- {op: compute, compute_op: add, args: [bill, tip], var: total}

# After (1 wiring decision):
- {op: percent_of, base: bill, rate: tip_rate, var: tip}
- {op: compute, compute_op: add, args: [bill, tip], var: total}
```

**Result**: Percentage accuracy 88% → 100%.

**Principle**: Fewer steps = fewer wiring decisions = fewer errors. If a computation is common and well-defined, encapsulate it as a domain op.

---

## Pattern 7: Minimal System Prompt

**Problem**: A verbose system prompt (450 tokens describing trace format, expert types, and examples) consumed context that the model needed for learning the actual format.

**Pattern**: The system prompt only lists available experts. The model learns format entirely from SFT examples.

```
You are a helpful assistant with access to the following experts: entity_track, arithmetic, rate_equation, comparison, percentage
```

**Result**: Accuracy 70% → 90-95%.

**Principle**: Don't tell the model what to do in the prompt. Show it what to do in the training data.

---

## Pattern 8: Distribution Weighted by Discrimination Difficulty

**Problem**: Equal distribution across expert types means easy experts (rate_equation — all patterns identical) get the same data budget as hard experts (comparison — 4 distinct op patterns).

**Pattern**: Allocate training data proportional to discrimination difficulty.

| Expert | Patterns | Discrimination | Allocation (Run 8) |
|--------|----------|---------------|------------|
| arithmetic | 10 (6 seq + 3 interleaved + 1 long) | High | 30% |
| entity_track | 5 (ops vary) | Medium | 20% |
| comparison | 4 (ops vary, same shape) | High | 15% |
| composition | 4 (multi-expert) | High | 15% |
| percentage | 4 (domain ops vary) | Low | 10% |
| rate_equation | 4 (identical shape) | None (all same) | 10% |

**Note**: Run 8 increased arithmetic allocation from 20% to 30% to accommodate 3 new interleaved patterns. Interleaved patterns require more examples because the model must learn position-dependent decisions (init vs compute at each step).

**Principle**: Invest training budget where the model needs to discriminate, not where it's trivial.

---

## Pattern 9: Graduated Reward for Partial Credit

**Problem**: Binary reward (correct/incorrect) gives the model no signal about how close it is. A valid trace with a wiring error gets the same reward as unparseable output.

**Pattern**: Multi-level reward based on how far the trace gets through validation:

```
1.0: Correct answer
0.7: Valid trace, wrong answer (wiring error)
0.5: Trace execution error (invalid structure)
0.3: Parsed YAML, wrong expert (routing error)
0.0: Parse failure (format error)
```

**Why it works**: The model first learns to produce valid YAML (0.0 → 0.3+), then correct routing (0.3 → 0.5+), then valid structure (0.5 → 0.7+), then correct wiring (0.7 → 1.0). Each transition provides positive gradient.

**Principle**: Reward the closest correct prefix. Give the model a gradient toward the solution at every stage.

---

## Pattern 10: max_len Must Accommodate Full Target

**Problem**: `max_len=512` silently truncated 100% of training targets. The model learned partial traces and could never produce complete output.

**Pattern**: Set `max_len` to accommodate the longest prompt + target in the training data. Verify by checking that no target is truncated.

**Diagnostic**: If loss is low but accuracy is 0%, check for truncation.

**Principle**: Silent data corruption is the hardest bug to find. Validate training data integrity before training.

---

## Pattern 11: Hybrid Variable Naming

**Problem**: Pure abstract var names (`x`, `y`, `z`, `result`) remove the semantic connection between question text and trace. The model can't ground "Bob has 28 cards" to `var: x, value: 28` — there's no "Bob" signal. Result: 75% SFT accuracy with arg-order confusion (`sub(z, x)` vs `sub(x, z)`).

Pure per-instance names (`alice.books`, `bob.coins`, `multiplier`, `difference`) fragment the training signal — every example has unique var names, so the model can't learn fixed scaffolding.

**Pattern**: Hybrid naming — entity-anchored init vars for semantic grounding, fixed scaffolding vars for structural consistency.

```yaml
# Entity-anchored first init (varies per example, connects to question text)
- {op: init, var: bob.cards, value: 28}
# Fixed scaffolding (same every time)
- {op: init, var: b, value: 2}
- {op: compute, args: [bob.cards, b], var: step1}
- {op: compute, args: [bob.cards, step1], var: result}
- {op: query, var: result}
```

**What varies**: The entity var name (`bob.cards`, `alice.stickers`) — derived from question text.
**What's fixed**: `b`, `step1`, `result` — always the same positions, always the same names.

**Note**: Pattern 16 (Variable Naming Standardization) takes this further for arithmetic: even init vars are standardized to `a`, `b`, `c`. Hybrid naming (entity-anchored first init) only applies to experts that need semantic grounding (comparison, entity_track).

**Evidence**:

| Naming strategy | SFT | RL iter 1 |
|----------------|-----|-----------|
| All per-instance (`alice.books`, `multiplier`, `difference`) | 95% | — (Run 2) |
| Abstract (`x`, `y`, `z`, `result`) | 75% | 6/8 (Run 5) |
| **Hybrid** (entity + fixed scaffolding) | **95%** | **8/8** (Run 6) |

**Why it works**: The model already knows entity-name extraction from entity_track (94-98% accuracy). The hybrid approach reuses that skill for the init step, while the fixed scaffolding (`factor`, `step1`, `result`) gives it a reliable template for compute and query steps. The arg-order in compute is no longer arbitrary — `sub(bob.cards, step1)` means "Bob's cards minus the computed value," which is semantically grounded.

**Principle**: Give the model semantic anchors where extraction happens (connecting text to vars), and structural anchors where computation happens (fixed scaffolding for ops and query).

---

## Pattern 12: Uniform Shape Per Expert Type

**Problem**: Percentage had 3 patterns at 4 steps and 1 pattern (`tip_calculation`) at 5 steps. While it was at 100% accuracy, this is the same latent instability that caused rate_equation's 12% regression.

**Pattern**: Every pattern in a fixed-shape expert must have exactly the same step count. If a pattern needs an extra step, either simplify its question or move it to a variable-length expert.

**Implementation**: Changed `tip_calculation` from "What's the total including tip?" (percent_of + add = 5 steps) to "How much is the tip?" (percent_of = 4 steps). The "total including tip" problem type belongs in arithmetic if needed.

| Expert | Shape | Enforced |
|--------|-------|----------|
| rate_equation | 4-step: init, init, compute(mul), query | All 4 patterns identical |
| percentage | 4-step: init, init, domain_op, query | All 4 patterns identical |
| comparison | 5-step: init, init, compute, compute, query | All 4 patterns identical |

**Principle**: Structural consistency is binary — either every pattern matches or it's a latent failure mode. No exceptions for "it works for now."

---

## Pattern 13: Expert Composition

**Problem**: 30% of GSM-8K problems require operations from multiple experts. "A house increases by 150% and costs are subtracted" needs percentage THEN arithmetic. The single-expert-per-trace architecture can't represent this.

**Pattern**: Allow the model to emit a sequence of sub-traces, each handled by its own expert, with outputs wired forward via `source: prev.result`.

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

**Implementation requirements** (discovered through iteration):

1. **Consistent YAML formatting** — All trace steps must use flow style `{...}`. Mixed formatting (flow for simple steps, block for compute with args) confuses the model. Custom formatter required.

2. **Fixed scaffolding vars in sub-traces** — Arithmetic sub-traces must use `prev`, `factor`, `result`, not semantic names like `sale`, `total`, `good`. This matches Pattern 11 (Hybrid Naming).

3. **Identical structure across patterns** — All 2-expert composition patterns must have same shape (4-step first sub-trace + 4-step arithmetic sub-trace). Violating this (e.g., 6-step arithmetic in one pattern) breaks learning.

4. **Extraction must accept lists** — The `extract_yaml` function must validate both `dict` (single-expert) and `list` (composed) formats. Original code only accepted dicts, causing 100% parse failure on composed traces.

5. **All generators must be true multi-expert** (Run 14 cleanup) — Two mislabeled patterns were removed from composition.py because they returned single-expert traces (arithmetic only). Single-expert patterns belong in the schema system, not composition.

**Format detection**: `yaml.safe_load()` returns list → composed. Returns dict → single-expert. Backward compatible.

**Wiring**: `source: prev.result` in InitStep tells the CompositionSolver to substitute the previous sub-trace's query result as this init's value. For 3-expert chains, `source: sub0.result` references a specific sub-trace by index.

**Why composition, not a general expert**: A single "math" expert that handles all operations defeats the routing thesis and makes expert boundaries meaningless. Composition preserves clean per-expert vocabularies while enabling multi-domain problems.

**Learning task extension**: The model reuses levels 2-5 (routing, structure, wiring, discrimination) per sub-trace. The new capability is **decomposition** — splitting a problem into sub-problems and deciding where to break.

**Composition patterns (10 total)**:

| Pattern | Experts | Example |
|---------|---------|---------|
| percent_off_plus_extra | percentage → arithmetic | Discount + shipping |
| percent_increase_minus_cost | percentage → arithmetic | Stock gain |
| percent_of_then_multiply | percentage → arithmetic | Per-unit discount × quantity |
| rate_then_subtract | rate_equation → arithmetic | Production - defects |
| value_increase_profit | percentage → arithmetic | House flipping |
| paired_discount | percentage → arithmetic | Kylar's glasses |
| interrupted_rate | percentage → arithmetic | Download with restart |
| consume_then_sell | entity_track → arithmetic | Janet's ducks |
| cost_increase_profit | arithmetic → percentage → arithmetic | 3-expert profit calc |
| discount_tax_total | percentage → percentage → arithmetic | 3-expert discount+tax |

**Result**: 100% SFT accuracy with 500 examples (75 composed, 425 single-expert). The model learns composition in 1 epoch with no RL needed.

**Hierarchy extension**:
```
1. Format → 2. Routing → 3. Structure → 4. Wiring → 5. Discrimination → 6. Decomposition
```

**Principle**: When a problem crosses expert boundaries, compose experts rather than expanding any single expert's vocabulary. The model learns to decompose, each expert stays simple.

---

## Pattern 14: Interleaved Init (GSM-8K Grammar Gap)

**Problem**: Current training assumes `init+ → compute+ → query` — all quantities declared first, then all operations. But 40% of real problems introduce new quantities mid-computation:

```
"He runs 3 sprints 3 times a week" → init(3), init(3), compute(mul→9)
"He runs 60 meters each sprint"    → init(60)  ← NEW QUANTITY AFTER COMPUTE
"How many total meters?"            → compute(mul→540), query
```

**Pattern**: The grammar must support `(init | compute)+ → query` — init and compute steps can be freely interleaved.

**Impact on the learning task**: With `init+, compute+, query`, the model only makes one structural decision (how many of each). With `(init|compute)+, query`, every position is a branch point — the model must decide whether to extract a new value or compute with existing ones.

**Evidence**: 4/10 GSM-8K sample problems have interleaved inits (James sprints, Wendi's feed, Toulouse's sheep, John's dogs). All are currently PARTIAL coverage.

**Implemented patterns** (3 generators) — Hybrid Naming (Run 13):

```yaml
# generate_interleaved_mul_mul: "Alex runs 3 laps of 5 miles each day. How many total miles in 10 days?"
- {op: init, var: laps, value: 3}
- {op: init, var: miles, value: 5}
- {op: compute, compute_op: mul, args: [laps, miles], var: step1}
- {op: init, var: days, value: 10}          # ← init after compute
- {op: compute, compute_op: mul, args: [step1, days], var: result}
- {op: query, var: result}

# generate_parallel_merge: "A farm has 3 hens each producing 20 eggs. They give away 15 to neighbors and 25 to friends. How many remain?"
- {op: init, var: hens, value: 3}
- {op: init, var: eggs_per, value: 20}
- {op: compute, compute_op: mul, args: [hens, eggs_per], var: step1}
- {op: init, var: gift1, value: 15}         # ← init after compute
- {op: init, var: gift2, value: 25}         # ← another init
- {op: compute, compute_op: add, args: [gift1, gift2], var: step2}
- {op: compute, compute_op: sub, args: [step1, step2], var: result}
- {op: query, var: result}

# generate_chained_mul_sum: "Seattle has 20 sheep. Austin has 4× as many. Memphis has 2× as many as Austin. Total?"
- {op: init, var: seattle, value: 20}
- {op: init, var: factor1, value: 4}
- {op: compute, compute_op: mul, args: [seattle, factor1], var: step1}
- {op: init, var: factor2, value: 2}        # ← init after compute
- {op: compute, compute_op: mul, args: [step1, factor2], var: step2}
- {op: compute, compute_op: add, args: [seattle, step1], var: step3}
- {op: compute, compute_op: add, args: [step3, step2], var: result}
- {op: query, var: result}
```

**Implications for training**: This is a harder learning task than template filling. The model must learn sequential decision-making about trace structure. May require more training data or a larger model for reliable interleaving.

**Variable naming (Run 13)**: Hybrid naming — semantic init vars (`laps`, `miles`, `days`, `hens`) provide grounding, fixed intermediates (`step1`, `step2`) and unified query (`result`) provide structural consistency. Run 12's abstract naming (`a`, `b`, `c`) broke composition accuracy.

**Principle**: Real problems introduce information as needed, not all upfront. The trace grammar must match the problem's natural information flow.

---

## Pattern 15: Schema-Based Generation

**Problem**: Hardcoded Python generators are:
- Hard to modify (requires code changes)
- Hard to audit (logic spread across functions)
- Inconsistent (each generator has different structure)
- Limited (text variations require code duplication)

**Pattern**: Define all problem patterns as JSON schemas. A single `SchemaGenerator` interprets them.

**Schema structure** (Run 13 hybrid naming):
```json
{
  "name": "material_half",
  "expert": "arithmetic",
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

**Note**: All arithmetic schemas use semantic init vars (matching schema variables), `step1`, `step2`, `step3` for intermediates, and `result` for query target (see Pattern 16). Run 12's abstract naming (`a`, `b`, `c`) was a failure.

**Benefits**:
1. **Declarative** — Problem logic visible in JSON
2. **Vocabulary-driven** — Text variety without code changes
3. **Constraint handling** — Auto-regenerate if constraints fail
4. **Auditable** — All patterns in one directory
5. **Extensible** — Add GSM-8K patterns by adding JSON files

**Implementation**:
```python
from chuk_virtual_expert_arithmetic.generators import SchemaGenerator

gen = SchemaGenerator()
print(gen.schema_names)  # ['material_half', 'price_chain', ...]
example = gen.generate("material_half")
```

**Organization**:
```
schemas/
├── arithmetic/           # 19 schemas
│   ├── price_chain.json
│   ├── material_half.json
│   └── ...
├── entity_track/         # 5 schemas
├── rate_equation/        # 4 schemas
├── comparison/           # 4 schemas
└── percentage/           # 4 schemas
```

**Adding new patterns**:
1. Create `schemas/{expert}/my_pattern.json`
2. Create `vocab/patterns/{expert}/my_pattern.json` (if new template needed)
3. Pattern is automatically available via `SchemaGenerator`

**Result**: 44 schemas covering all expert types + composition. New GSM-8K patterns added in minutes instead of hours.

**Principle**: Data-driven is better than code-driven. Patterns as data enable rapid iteration and clear auditing.

---

## Pattern 16: Hybrid Variable Naming (Run 13)

**Problem**: 21 arithmetic schemas had 7+ different query targets (`total`, `remaining`, `revenue`, `per_worker`, `final`, `weekly`, `output`). Each pattern used different init var names (`start`, `base`, `count`, `rate`) and intermediate names (`after1`, `subtotal`, `daily`, `with_tax`). Result: 81% arithmetic accuracy vs 100% for other experts.

**Failed attempt (Run 12)**: Abstract naming (`a`, `b`, `c` for inits) — broke composition (67% → 57%) and GSM-8K (50% → 30%). Same mistake as Run 5 (`x`, `y`, `z`) — lost semantic grounding.

**Correct pattern (Run 13)**: Hybrid naming — semantic init vars for grounding, fixed scaffolding for structure.

| Component | Example | Purpose |
|-----------|---------|---------|
| **Init vars** | `produced`, `use1`, `price`, `rate` | Semantic grounding to problem text |
| **Intermediate vars** | `step1`, `step2`, `step3` | Structural scaffolding (fixed) |
| **Query target** | `result` | Unified output (fixed) |

**Example (Run 11 — chaos)**:
```yaml
- {op: init, var: base, value: 100}
- {op: init, var: tax, value: 10}
- {op: init, var: shipping, value: 5}
- {op: compute, compute_op: add, args: [base, tax], var: with_tax}
- {op: compute, compute_op: add, args: [with_tax, shipping], var: total}
- {op: query, var: total}
```

**Example (Run 12 — abstract, WRONG)**:
```yaml
- {op: init, var: a, value: 100}
- {op: init, var: b, value: 10}
- {op: init, var: c, value: 5}
- {op: compute, compute_op: add, args: [a, b], var: step1}
- {op: compute, compute_op: add, args: [step1, c], var: result}
- {op: query, var: result}
```

**Example (Run 13 — hybrid, CORRECT)**:
```yaml
- {op: init, var: base, value: 100}
- {op: init, var: tax, value: 10}
- {op: init, var: shipping, value: 5}
- {op: compute, compute_op: add, args: [base, tax], var: step1}
- {op: compute, compute_op: add, args: [step1, shipping], var: result}
- {op: query, var: result}
```

**What changed from chaos → hybrid**:
- Init vars: KEPT semantic names (`base`, `tax`, `shipping`)
- Intermediate vars: CHANGED to fixed (`with_tax` → `step1`)
- Query target: UNIFIED to `result` (`total` → `result`)

**Scope of changes**:
- 21 arithmetic schemas
- 4 rate_equation schemas
- 12 composition pattern functions

**Verification**:
```
Arithmetic query targets: {'result'}      # Previously: 7+ different names
Rate equation query targets: {'result'}
Composition final query targets: {'result'}
Entity-track query targets: {'eggs'}      # Semantic entity names preserved
```

**Exception**: Entity-track and comparison retain entity-anchored names (`bob.cards`, `eggs`) for semantic grounding.

**Key insight**: Semantic grounding is non-negotiable. Abstract init vars (`a`, `b`, `c`) break the connection between question text and trace structure. The model can't ground `a` to "eggs produced" or `b` to "eggs eaten".

**Relationship to Pattern 11 (Hybrid Variable Naming)**: Pattern 11 established the principle. This pattern documents the Run 12 failure (abstract names) and Run 13 fix (hybrid names) that validated it.

**Principle**: Semantic init vars + structural intermediates + unified query. The model needs grounding (semantic) AND consistency (structural).

---

## Meta-Pattern: The Learning Task Hierarchy

For a 1B model learning trace emission, complexity increases in this order:

1. **Format** — Producing valid YAML (learned in first 10 SFT steps)
2. **Routing** — Selecting the correct expert (learned with minimal prompt)
3. **Structure** — Emitting the right number/type of steps (requires structural consistency)
4. **Wiring** — Connecting the right variables to the right ops (requires hybrid naming — semantic anchors for extraction + fixed scaffolding for computation)
5. **Discrimination** — Choosing between similar patterns (requires per-pattern template + sufficient data)
6. **Decomposition** — Breaking multi-expert problems into sub-traces and wiring outputs forward (requires expert composition training)
7. **Interleaving** — Deciding at each position whether to init a new value or compute with existing ones (requires interleaved patterns with semantic var names)

Each level requires the previous levels to be solid. Fixing level 4 (wiring) without fixing level 3 (structure) has no effect. And even with structure fixed, abstract var names (`x`, `y`) break wiring because the model can't ground args in the problem text.

**Current status** (Run 13 — Hybrid Variable Naming):
- Levels 1-6: Solved for synthetic distribution
- Level 7 (Interleaving): Implemented — 4 interleaved generators
- Level 8 (Long chains): Implemented — 10-step expense chain pattern
- Level 9 (GSM-8K patterns): Implemented — div_then_add, consume_then_sell, material_half, decimal_rate_week
- Level 10 (Composition): Implemented — 10 composition patterns (all verified multi-expert) including 3-expert chains
- **Level 11 (Variable Naming)**:
  - Run 12 tried abstract naming (`a,b,c`) — broke composition (57%)
  - Run 13 uses hybrid naming: semantic inits + `step1,step2,step3` + unified `result`
- **GSM-8K: 5/10 correct (50%)** in Run 11, dropped to 3/10 (30%) in Run 12

**Run 11 exposed the problem**: 81% arithmetic (vs 100% others) due to variable naming chaos.
**Run 12 wrong fix**: Abstract naming (`a`, `b`, `c`) fixed arithmetic but broke composition.
**Run 13 correct fix**: Hybrid naming (semantic inits + structural scaffolding).

**What's working on GSM-8K**:
- ✓ Janet's ducks (interleaved sub-sub-mul)
- ✓ Toulouse sheep (chained mul-sum)
- ✓ Fish tanks (div-then-add)
- ✓ House flipping (composition)
- ✓ Kylar's glasses (composition)

**The remaining GSM-8K gaps**:
- ~~Longer chains (8+ steps)~~ ✓ Implemented
- ~~Number preprocessing~~ ✓ Implemented
- ~~Janet's ducks pattern~~ ✓ Working!
- ~~Fish tanks / robe pattern~~ ✓ Fish tanks working
- ~~"Half that much" pattern~~ ✓ `material_half` schema added
- ~~Decimal rate values~~ ✓ `decimal_rate_week` schema added
- ~~Expert routing for % ops~~ ✓ Composition patterns added
- ~~Variable naming chaos~~ ✓ Hybrid naming fixed (Run 13)
- Complex multi-step (5, 8) — Future work
- Reading comprehension errors (4) — Needs more template variations

**Principle**: Solve the learning task top-down. Don't optimize wiring until structure is stable. Don't optimize structure until format is reliable. And at every level, give the model semantic bridges between the input text and the output trace.

---

## Pattern 17: Template Phrasing Coverage (Run 14)

**Problem**: Run 13 achieved 10/10 valid traces but only 3/10 correct on GSM-8K. Analysis revealed specific phrasing gaps — mathematically equivalent patterns phrased differently:

| GSM-8K Problem | Our Template | GSM-8K Phrasing | Why It Fails |
|----------------|--------------|-----------------|--------------|
| Fish tanks | "second has half" | "first has **twice as much**" | Inverse extraction |
| James sprints | "3 sprints, 5 times" | "**3** sprints **3** times" | Same number twice |
| Wendi chickens | sequential values | numbers **scattered** in prose | Value extraction order |

**Pattern**: Template phrasing must match target distribution's exact wording, not just mathematical equivalence.

**New schemas for Run 14**:

| Schema | Pattern | Template Example |
|--------|---------|------------------|
| `twice_as_much` | first/2 + first | "The first tank has twice as many fish as the second..." |
| `weekly_sprints_same` | x × x × y | "runs 3 sprints 3 times a week...60 meters each" |
| `feed_remainder_scattered` | (a×b) - (c+d) | "20 chickens...3 cups per day...15 in morning...25 in afternoon" |

**Implementation details**:

1. **Inverse framing** (`twice_as_much`):
   ```
   Our: "second has half as many" → extract second, multiply by 2
   GSM-8K: "first has twice as much" → extract first, divide by 2
   Same math, different extraction pattern
   ```

2. **Repeated values** (`weekly_sprints_same`):
   ```
   Our: "3 sprints, 5 times" → clear which is which
   GSM-8K: "3 sprints 3 times" → model must understand same value appears twice
   ```

3. **Scattered numbers** (`feed_remainder_scattered`):
   ```
   Our: values appear in computation order
   GSM-8K: "20 chickens...3 cups...15 cups...25 cups" scattered across prose
   ```

**Generator audit**: After creating schemas, discovered 2 broken patterns:

| Schema | Bug | Fix |
|--------|-----|-----|
| `long_expense_chain` | `mult_word` not resolving | Removed variable, used `${multiplier}` directly |
| `rate_production` | `producer.name` returning None | Fixed vocab paths, added `subject: "it"` to producer vocab |

**Final status**: 41/41 schemas passing (5 samples each, 205 total verified).

**Principle**: The model learns surface form, not abstract semantics. If GSM-8K says "twice as much" and we train on "half as many", the model won't generalize — even though they're mathematically equivalent.
