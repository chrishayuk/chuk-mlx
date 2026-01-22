# CoT Standardization Extended: YAML Trace Format

## Research Question

**Can we extend the standardized CoT format to full YAML traces with verifiable reasoning steps?**

Building on `cot_standardization` (100% accuracy with simple `expert: spec -> result`), this experiment tests a richer trace format that enables step-by-step verification.

---

## Format Comparison

### Simple Format (cot_standardization)
```
word_problem: eggs:16 | -3 -4 *2 -> 18
```

### Extended Format (this experiment)
```yaml
expert: entity_track
trace:
  - {init: eggs, value: 16}
  - {consume: {entity: eggs, amount: 3}}
  - {consume: {entity: eggs, amount: 4}}
  - {state: {eggs: 9}}
  - {compute: {op: mul, args: [9, 2], result: 18}}
answer: 18
```

---

## Why Extended Format?

| Feature | Simple | Extended |
|---------|--------|----------|
| Expert routing | ✓ | ✓ |
| Final answer | ✓ | ✓ |
| Step-by-step trace | ✗ | ✓ |
| Verifiable reasoning | ✗ | ✓ |
| Error localization | ✗ | ✓ |
| Intermediate states | ✗ | ✓ |

**Key benefit:** If trace is valid, answer is correct. 100% error detection.

---

## Expert Types

| Expert | Pattern | Example |
|--------|---------|---------|
| `entity_track` | Quantities moving between entities | "Bob gives 8 marbles to Carol" |
| `arithmetic` | Chain of operations | "Price + tax + shipping" |
| `rate_equation` | Rate × time = quantity | "20 pages/min for 7 minutes" |
| `comparison` | Compute then compare | "4 times as many, how many more?" |
| `allocation` | Distribute with constraints | "Split $60, X gets twice Y" |
| `percentage` | Percent increase/decrease | "20% off $50" |

---

## Trace Schemas

### Entity Tracking

```yaml
expert: entity_track
trace:
  - {init: <entity>, value: <n>}
  - {transfer: {from: <e1>, to: <e2>, amount: <n>}}
  - {consume: {entity: <e>, amount: <n>}}
  - {state: {<entity>: <value>, ...}}
  - {compute: {op: <op>, args: [...], result: <n>}}
  - {query: <entity>}
answer: <n>
```

### Arithmetic

```yaml
expert: arithmetic
trace:
  - {init: <var>, value: <n>}
  - {compute: {op: <add|sub|mul|div>, args: [...], var: <name>, result: <n>}}
  - {query: <var>}
answer: <n>
```

### Rate/Equation

```yaml
expert: rate_equation
trace:
  - {given: {rate: <n>, unit: "<unit>", time: <n>}}
  - {formula: "<equation>"}
  - {compute: {op: mul, args: [...], var: <name>, result: <n>}}
  - {query: <var>}
answer: <n>
```

### Comparison

```yaml
expert: comparison
trace:
  - {init: <entity>, value: <n>}
  - {compute: {op: mul, args: [...], var: <name>, result: <n>}}
  - {compare: {op: sub, args: [...], var: difference, result: <n>}}
  - {query: difference}
answer: <n>
```

### Percentage

```yaml
expert: percentage
trace:
  - {init: <var>, value: <n>}
  - {percent_off: {base: <var>, rate: <n>, result: <n>}}
  - {query: <var>}
answer: <n>
```

---

## Training Protocol

### Phase 1: Generate Synthetic Data

```python
# Distribution
data = {
    "entity_track": 100,  # 40%
    "arithmetic": 40,     # 15%
    "rate_equation": 40,  # 15%
    "comparison": 40,     # 15%
    "allocation": 20,     # 10%
    "percentage": 15,     # 5%
}
# Total: ~255 examples
```

### Phase 2: Minimal SFT (1 epoch)

Same as `cot_standardization`:
- Native chat template
- 6 unfrozen layers
- System prompt with format + examples

### Phase 3: RL with Verified Rewards

```python
def compute_reward(output: str, gold: dict) -> float:
    try:
        parsed = yaml.safe_load(output)
    except:
        return 0.0  # Parse failure

    if parsed.get("expert") != gold["expert"]:
        return 0.3  # Wrong expert

    if not verify_trace(parsed.get("trace", [])):
        return 0.5  # Invalid trace

    if parsed.get("answer") != gold["answer"]:
        return 0.7  # Valid trace, wrong answer

    return 1.0  # Correct
```

---

## Trace Verifier

The key differentiator: traces can be replayed to verify correctness.

```python
def verify_trace(trace: list) -> bool:
    """Replay trace and verify each step."""
    state = {}

    for step in trace:
        if "init" in step:
            state[step["init"]] = step["value"]

        elif "transfer" in step:
            t = step["transfer"]
            if state.get(t["from"], 0) < t["amount"]:
                return False  # Invalid transfer
            state[t["from"]] -= t["amount"]
            state[t["to"]] = state.get(t["to"], 0) + t["amount"]

        elif "consume" in step:
            c = step["consume"]
            if state.get(c["entity"], 0) < c["amount"]:
                return False  # Invalid consume
            state[c["entity"]] -= c["amount"]

        elif "compute" in step:
            c = step["compute"]
            args = [state.get(a, a) for a in c["args"]]
            expected = compute(c["op"], args)
            if abs(expected - c["result"]) > 0.01:
                return False  # Computation error
            if "var" in c:
                state[c["var"]] = c["result"]

        elif "state" in step:
            # Verify state matches
            for k, v in step["state"].items():
                if abs(state.get(k, 0) - v) > 0.01:
                    return False

    return True
```

---

## Success Criteria

| Metric | Target |
|--------|--------|
| YAML parse rate | >95% |
| Expert accuracy | >90% |
| Trace validity | >90% |
| Answer accuracy | >85% |
| Error detection | 100% (invalid trace = wrong answer) |

---

## Files

```
experiments/cot_standardization_extended/
├── EXPERIMENT.md           # This file
├── config.yaml             # Training config
├── data/
│   ├── generators/
│   │   ├── entity_track.py
│   │   ├── arithmetic.py
│   │   ├── rate_equation.py
│   │   ├── comparison.py
│   │   └── percentage.py
│   └── generate.py         # Main generator
├── trace/
│   ├── schema.py           # Trace dataclasses
│   └── verifier.py         # Replay verification
├── training/
│   └── train.py            # SFT + RL training
├── evaluation/
│   └── eval.py             # Evaluation script
└── results/
```

---

## Comparison to cot_standardization

| Aspect | cot_standardization | extended |
|--------|---------------------|----------|
| Format | `expert: spec -> result` | YAML with trace |
| Verifiable | No | Yes |
| Complexity | Low | Higher |
| Token count | ~20 | ~100-200 |
| Error detection | Parse only | Full replay |

---

## Hypothesis

If `cot_standardization` achieved 100% with simple format, extended format should achieve >85% with the added benefit of verifiable reasoning.

The trade-off: more tokens, more structure, but guaranteed correctness when trace is valid.
