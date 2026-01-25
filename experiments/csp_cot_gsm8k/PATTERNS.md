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
| arithmetic | 9 (6 seq + 3 interleaved) | High | 30% |
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
- {op: init, var: factor, value: 2}
- {op: compute, args: [bob.cards, factor], var: step1}
- {op: compute, args: [bob.cards, step1], var: result}
- {op: query, var: result}
```

**What varies**: The entity var name (`bob.cards`, `alice.stickers`) — derived from question text.
**What's fixed**: `factor`, `step1`, `result` — always the same positions, always the same names.

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

3. **Identical structure across patterns** — All 4 composition patterns must have same shape (4-step first sub-trace + 4-step arithmetic sub-trace). Violating this (e.g., 6-step arithmetic in one pattern) breaks learning.

4. **Extraction must accept lists** — The `extract_yaml` function must validate both `dict` (single-expert) and `list` (composed) formats. Original code only accepted dicts, causing 100% parse failure on composed traces.

**Format detection**: `yaml.safe_load()` returns list → composed. Returns dict → single-expert. Backward compatible.

**Wiring**: `source: prev.result` in InitStep tells the CompositionSolver to substitute the previous sub-trace's query result as this init's value.

**Why composition, not a general expert**: A single "math" expert that handles all operations defeats the routing thesis and makes expert boundaries meaningless. Composition preserves clean per-expert vocabularies while enabling multi-domain problems.

**Learning task extension**: The model reuses levels 2-5 (routing, structure, wiring, discrimination) per sub-trace. The new capability is **decomposition** — splitting a problem into sub-problems and deciding where to break.

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

**Implemented patterns** (3 generators):

```yaml
# generate_interleaved_mul_mul: "Alex runs 3 laps of 5 miles each day. How many total miles in 10 days?"
- {op: init, var: sessions, value: 3}
- {op: init, var: per_session, value: 5}
- {op: compute, compute_op: mul, args: [sessions, per_session], var: daily}
- {op: init, var: days, value: 10}          # ← init after compute
- {op: compute, compute_op: mul, args: [daily, days], var: total}
- {op: query, var: total}

# generate_parallel_merge: "A farm has 3 hens each producing 20 eggs. They give away 15 to neighbors and 25 to friends. How many remain?"
- {op: init, var: hens, value: 3}
- {op: init, var: per_hen, value: 20}
- {op: compute, compute_op: mul, args: [hens, per_hen], var: produced}
- {op: init, var: gift1, value: 15}         # ← init after compute
- {op: init, var: gift2, value: 25}         # ← another init
- {op: compute, compute_op: add, args: [gift1, gift2], var: gifted}
- {op: compute, compute_op: sub, args: [produced, gifted], var: remaining}
- {op: query, var: remaining}

# generate_chained_mul_sum: "Seattle has 20 sheep. Austin has 4× as many. Memphis has 2× as many as Austin. Total?"
- {op: init, var: city1, value: 20}
- {op: init, var: factor1, value: 4}
- {op: compute, compute_op: mul, args: [city1, factor1], var: city2}
- {op: init, var: factor2, value: 2}        # ← init after compute
- {op: compute, compute_op: mul, args: [city2, factor2], var: city3}
- {op: compute, compute_op: add, args: [city1, city2], var: partial}
- {op: compute, compute_op: add, args: [partial, city3], var: total}
- {op: query, var: total}
```

**Implications for training**: This is a harder learning task than template filling. The model must learn sequential decision-making about trace structure. May require more training data or a larger model for reliable interleaving.

**Critical implementation detail**: Interleaved patterns must use **semantic var names** (`sessions`, `hens`, `city1`) not abstract names (`a`, `b`, `c`). Abstract names caused confusion because all 3 interleaved patterns shared the same var names, preventing discrimination. Semantic names ground each pattern in its question template.

**Principle**: Real problems introduce information as needed, not all upfront. The trace grammar must match the problem's natural information flow.

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

**Current status** (Run 8):
- Levels 1-6: Solved for synthetic distribution (97% accuracy, 100% parse rate)
- Level 7 (Interleaving): Implemented — 3 interleaved generators with semantic var names
- GSM-8K: 0% correct but 100% valid traces — model structures correctly but wires wrong for OOD problems

**The GSM-8K gap**: The synthetic distribution validates levels 1-7. Real problems additionally require:
- Longer chains (8+ steps) — GSM-8K median is 6-8 steps
- 3-expert composition — arithmetic→percentage→arithmetic chains
- Number preprocessing — commas in numbers (80,000 → 80000)

**Principle**: Solve the learning task top-down. Don't optimize wiring until structure is stable. Don't optimize structure until format is reliable. And at every level, give the model semantic bridges between the input text and the output trace.
