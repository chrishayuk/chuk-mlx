# GSM-8K Computation Patterns

A catalog of computation patterns found in GSM-8K problems, mapped to our expert system.

---

## Pattern Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Fully implemented and trained |
| 🔶 | Partially implemented (needs more training data) |
| ❌ | Not implemented |

---

## Single-Expert Patterns

### Arithmetic Patterns

#### ✅ Sequential Chain (init+ → compute+ → query)

**Structure**: All values declared first, then all operations.

```yaml
expert: arithmetic
trace:
- {op: init, var: a, value: N}
- {op: init, var: b, value: M}
- {op: init, var: c, value: K}
- {op: compute, compute_op: OP1, args: [a, b], var: step1}
- {op: compute, compute_op: OP2, args: [step1, c], var: result}
- {op: query, var: result}
```

**Examples**:
- Price + tax + shipping
- Start - expense1 - expense2
- Rate × time ÷ workers

**Status**: ✅ Implemented (6 patterns, 100 training examples)

---

#### ✅ Interleaved Chain ((init | compute)+ → query)

**Structure**: New values introduced between computations.

```yaml
expert: arithmetic
trace:
- {op: init, var: sessions, value: 3}
- {op: init, var: per_session, value: 5}
- {op: compute, compute_op: mul, args: [sessions, per_session], var: daily}
- {op: init, var: days, value: 10}           # ← Value introduced mid-chain
- {op: compute, compute_op: mul, args: [daily, days], var: total}
- {op: query, var: total}
```

**Example**: "Alex runs 3 laps of 5 miles each day. How many total miles in 10 days?"

**GSM-8K Analogs**:
- James sprints: sessions × per_session = daily, daily × days = total
- Toulouse's sheep: city1 × factor1 = city2, city2 × factor2 = city3, sum all

**Status**: ✅ Implemented (`generate_interleaved_mul_mul`, `generate_chained_mul_sum`)

---

#### ✅ Parallel Computation + Merge

**Structure**: Two independent sub-computations merged at the end.

```yaml
expert: arithmetic
trace:
- {op: init, var: hens, value: 3}
- {op: init, var: per_hen, value: 20}
- {op: compute, compute_op: mul, args: [hens, per_hen], var: produced}   # Branch 1
- {op: init, var: gift1, value: 15}
- {op: init, var: gift2, value: 25}
- {op: compute, compute_op: add, args: [gift1, gift2], var: gifted}      # Branch 2
- {op: compute, compute_op: sub, args: [produced, gifted], var: remaining}  # Merge
- {op: query, var: remaining}
```

**Example**: "A farm has 3 hens each producing 20 eggs. They give away 15 to neighbors and 25 to friends. How many eggs remain?"

**GSM-8K Analogs**:
- Wendi's chickens: (hens × per_hen) - (gift1 + gift2) = remaining

**Status**: ✅ Implemented (`generate_parallel_merge`)

---

#### 🔶 Long Chain (7+ steps)

**Structure**: Extended sequential chain with 4+ inits and 3+ computes.

```yaml
expert: arithmetic
trace:
- {op: init, var: a, value: 16}
- {op: init, var: b, value: 3}
- {op: init, var: c, value: 4}
- {op: init, var: d, value: 2}
- {op: compute, compute_op: sub, args: [a, b], var: step1}
- {op: compute, compute_op: sub, args: [step1, c], var: step2}
- {op: compute, compute_op: mul, args: [step2, d], var: result}
- {op: query, var: result}
```

**GSM-8K Examples**:
- Janet's eggs: 16-3-4=9, 9×2=18 (4 inits, 3 computes)

**Status**: 🔶 Partially implemented (trained up to ~6 steps)
**Fix**: Generate longer chain training examples

---

#### ✅ Divide Then Operate

**Structure**: Division followed by another operation.

```yaml
expert: arithmetic
trace:
- {op: init, var: total, value: 100}
- {op: init, var: divisor, value: 4}
- {op: init, var: multiplier, value: 3}
- {op: compute, compute_op: div, args: [total, divisor], var: step1}
- {op: compute, compute_op: mul, args: [step1, multiplier], var: result}
- {op: query, var: result}
```

**Status**: ✅ Implemented (divide_multiply pattern)

---

### Rate Equation Patterns

#### ✅ Rate × Time = Quantity

**Structure**: Simple multiplication with domain semantics.

```yaml
expert: rate_equation
trace:
- {op: init, var: rate, value: 6}
- {op: init, var: time, value: 5}
- {op: compute, compute_op: mul, args: [rate, time], var: quantity}
- {op: query, var: quantity}
```

**GSM-8K Examples**:
- Bakery: 6 loaves/hour × 5 hours = 30 loaves

**Status**: ✅ Implemented (4 patterns, all identical structure)

---

### Comparison Patterns

#### ✅ Times More/Less

**Structure**: Multiply then subtract to find difference.

```yaml
expert: comparison
trace:
- {op: init, var: bob.cards, value: 16}
- {op: init, var: factor, value: 2}
- {op: compute, compute_op: mul, args: [bob.cards, factor], var: step1}
- {op: compute, compute_op: sub, args: [step1, bob.cards], var: result}
- {op: query, var: result}
```

**Status**: ✅ Implemented

---

#### ✅ Half As Many

**Structure**: Divide then find difference or sum.

```yaml
expert: comparison
trace:
- {op: init, var: first, value: 48}
- {op: init, var: factor, value: 2}
- {op: compute, compute_op: div, args: [first, factor], var: step1}
- {op: compute, compute_op: add, args: [first, step1], var: result}
- {op: query, var: result}
```

**GSM-8K Examples**:
- Fish tanks: 48/2=24, 48+24=72

**Status**: ✅ Implemented

---

#### ✅ Chained Comparisons

**Structure**: Multiple comparison relationships in sequence.

```yaml
expert: arithmetic  # Too complex for comparison expert
trace:
- {op: init, var: city1, value: 20}
- {op: init, var: factor1, value: 4}
- {op: compute, compute_op: mul, args: [city1, factor1], var: city2}
- {op: init, var: factor2, value: 2}                    # ← Interleaved
- {op: compute, compute_op: mul, args: [city2, factor2], var: city3}
- {op: compute, compute_op: add, args: [city1, city2], var: partial}
- {op: compute, compute_op: add, args: [partial, city3], var: total}
- {op: query, var: total}
```

**Example**: "Seattle has 20 sheep. Austin has 4 times as many as Seattle. Memphis has 2 times as many as Austin. How many sheep in total?"

**GSM-8K Analogs**:
- Toulouse's sheep: city1 × factor1 = city2, city2 × factor2 = city3, sum all

**Status**: ✅ Implemented (`generate_chained_mul_sum`)

---

### Percentage Patterns

#### ✅ Percent Of

**Structure**: Calculate X% of Y.

```yaml
expert: percentage
trace:
- {op: init, var: whole, value: 100}
- {op: init, var: rate, value: 25}
- {op: percent_of, base: whole, rate: rate, var: result}
- {op: query, var: result}
```

**Status**: ✅ Implemented

---

#### ✅ Percent Off (Discount)

**Structure**: Calculate price after X% discount.

```yaml
expert: percentage
trace:
- {op: init, var: price, value: 80}
- {op: init, var: rate, value: 20}
- {op: percent_off, base: price, rate: rate, var: result}
- {op: query, var: result}
```

**Status**: ✅ Implemented

---

#### ✅ Percent Increase

**Structure**: Calculate value after X% increase.

```yaml
expert: percentage
trace:
- {op: init, var: base, value: 100}
- {op: init, var: rate, value: 50}
- {op: percent_increase, base: base, rate: rate, var: result}
- {op: query, var: result}
```

**Status**: ✅ Implemented

---

### Entity Track Patterns

#### ✅ Consume/Transfer

**Structure**: Quantities moving between entities.

```yaml
expert: entity_track
trace:
- {op: init, var: eggs, value: 16}
- {op: consume, entity: eggs, amount: 3}
- {op: consume, entity: eggs, amount: 4}
- {op: compute, compute_op: mul, args: [eggs, 2], var: revenue}
- {op: query, var: revenue}
```

**Status**: ✅ Implemented (5 patterns)

---

## Multi-Expert Composition Patterns

### ✅ Percentage → Arithmetic (2-expert)

**Structure**: Percentage calculation feeds into arithmetic operation.

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

**Implemented Patterns**:
| Pattern | First Expert | Operation | Second Expert | Operation |
|---------|--------------|-----------|---------------|-----------|
| percent_off_plus_extra | percentage | percent_off | arithmetic | add |
| percent_increase_minus_cost | percentage | percent_increase | arithmetic | sub |
| percent_of_then_multiply | percentage | percent_of | arithmetic | mul |
| rate_then_subtract | rate_equation | mul | arithmetic | sub |

**Status**: ✅ Implemented (4 patterns, 75 training examples)

---

### ❌ Arithmetic → Percentage → Arithmetic (3-expert)

**Structure**: Three experts in sequence.

```yaml
# Sub-trace 1: Calculate cost
- expert: arithmetic
  trace:
  - {op: init, var: price, value: 80000}
  - {op: init, var: repairs, value: 50000}
  - {op: compute, compute_op: add, args: [price, repairs], var: result}
  - {op: query, var: result}

# Sub-trace 2: Calculate increase
- expert: percentage
  trace:
  - {op: init, var: base, value: 80000}
  - {op: init, var: rate, value: 150}
  - {op: percent_increase, base: base, rate: rate, var: result}
  - {op: query, var: result}

# Sub-trace 3: Calculate profit
- expert: arithmetic
  trace:
  - {op: init, var: new_value, source: prev.result}
  - {op: init, var: cost, source: sub1.result}    # ← Multi-value wiring
  - {op: compute, compute_op: sub, args: [new_value, cost], var: result}
  - {op: query, var: result}
```

**GSM-8K Examples**:
- House flipping: cost + repairs, value × 1.5, new_value - cost

**Status**: ❌ Not implemented
**Blocked by**: 3-expert chains + multi-value wiring
**Fix**: Extend CompositionSolver for named result references

---

### 🔶 Percentage → Complex Arithmetic

**Structure**: Percentage feeds into interleaved arithmetic.

```yaml
- expert: percentage
  trace:
  - {op: init, var: price, value: 5}
  - {op: init, var: rate, value: 60}
  - {op: percent_of, base: price, rate: rate, var: result}
  - {op: query, var: result}
- expert: arithmetic
  trace:
  - {op: init, var: discounted, source: prev.result}
  - {op: init, var: full, value: 5}
  - {op: compute, compute_op: add, args: [full, discounted], var: pair}
  - {op: init, var: total, value: 16}               # ← Interleaved
  - {op: init, var: size, value: 2}                 # ← Interleaved
  - {op: compute, compute_op: div, args: [total, size], var: pairs}
  - {op: compute, compute_op: mul, args: [pairs, pair], var: result}
  - {op: query, var: result}
```

**GSM-8K Examples**:
- Kylar's glasses: 60% of $5, then complex pricing calculation

**Status**: 🔶 Partially unblocked (interleaved patterns now available)
**Remaining**: Add composition generator with interleaved second sub-trace

---

## Implementation Summary

### By Pattern Type

| Category | Total | ✅ Done | 🔶 Partial | ❌ Missing |
|----------|-------|---------|------------|------------|
| Sequential arithmetic | 6 | 6 | 0 | 0 |
| Interleaved arithmetic | 3 | 3 | 0 | 0 |
| Rate equation | 4 | 4 | 0 | 0 |
| Comparison | 4 | 4 | 0 | 0 |
| Percentage | 4 | 4 | 0 | 0 |
| Entity track | 5 | 5 | 0 | 0 |
| 2-expert composition | 4 | 4 | 0 | 0 |
| 3-expert composition | 2 | 0 | 0 | 2 |
| Complex composition | 2 | 0 | 1 | 1 |
| **Total** | **34** | **30** | **1** | **3** |

### By Implementation Status

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Fully implemented | 30 | 88% |
| 🔶 Needs more data | 1 | 3% |
| ❌ Not implemented | 3 | 9% |

---

## Priority Implementation Roadmap

### ✅ Phase 1: Interleaved Inits (COMPLETE)

Implemented 3 new arithmetic generators:

```python
def generate_interleaved_mul_mul():
    """a × b = step1, introduce c, step1 × c = result"""

def generate_parallel_merge():
    """(a × b) - (c + d) with interleaved inits"""

def generate_chained_mul_sum():
    """a × factor1 = b, b × factor2 = c, sum(a,b,c)"""
```

**Status**: ✅ Complete
**Impact**: +3 patterns, ~50% GSM-8K coverage

### Phase 2: Longer Chains (Unlocks 1 pattern)

```python
# Extended arithmetic patterns
def generate_long_chain():
    """4+ inits, 3+ computes"""
```

**Effort**: 1-2 hours
**Impact**: +1 pattern, ~70% GSM-8K coverage

### Phase 3: 3-Expert Composition (Unlocks 2 patterns)

```python
# Extended composition
def generate_three_expert():
    """arithmetic → percentage → arithmetic"""
```

**Effort**: 2-3 hours
**Impact**: +2 patterns, ~80% GSM-8K coverage

### Phase 4: Multi-Value Wiring (Unlocks 2 patterns)

```python
# Named result references
source: sub1.result  # Reference first sub-trace
source: sub2.result  # Reference second sub-trace
```

**Effort**: 4-6 hours
**Impact**: +2 patterns, ~90% GSM-8K coverage

---

## Coverage Projection

| Phase | Patterns | GSM-8K Coverage |
|-------|----------|-----------------|
| Run 7 (composition) | 27 | ~20% |
| ✅ Run 8 (+ interleaved) | 30 | ~50% |
| + Longer chains | 31 | ~70% |
| + 3-expert | 33 | ~80% |
| + Multi-value | 34 | ~90% |
