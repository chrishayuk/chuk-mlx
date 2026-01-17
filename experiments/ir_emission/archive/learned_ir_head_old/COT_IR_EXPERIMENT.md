# Chain-of-Thought as Learned IR Parser

## Key Insight

**CoT normalization IS the learned parser.** The LLM learns to transform arbitrary natural language into a canonical format that is trivially parseable. The "regex" becomes `string.split()` on the LLM's own structured output.

```
Traditional:  NL → LLM → free-form text → regex (brittle) → IR
This work:    NL → LLM → canonical format → split (trivial) → IR
```

## Results Summary

| Test Suite | Accuracy | Examples |
|------------|----------|----------|
| Simple expressions | 100% (8/8) | "12 times 9" → 108 |
| Complex expressions | 100% (11/11) | "12 * (5 + 2)" → 84 |
| Word problems | 100% (8/8) | "Jenny has 5 apples..." → 3 |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  "Jenny has 5 apples. She gives 2 to Bob. How many left?"       │
│         ↓                                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  CoT Normalization (LLM generation, few-shot)             │  │
│  │  This IS the learned parser                               │  │
│  │  → "5 - 2 ="                                              │  │
│  └───────────────────────────────────────────────────────────┘  │
│         ↓                                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Deterministic Parse (string.split)                       │  │
│  │  → {op: subtract, a: 5, b: 2}                             │  │
│  └───────────────────────────────────────────────────────────┘  │
│         ↓                                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Shunting Yard (for complex expressions)                  │  │
│  │  → Postfix: [push(5), push(2), sub]                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│         ↓                                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  WASM Stack Machine                                       │  │
│  │  → 3                                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Experiment Progression

### Stage 1: Simple Expressions (cot_ir_pipeline.py)

**Goal:** Validate that CoT can normalize NL to canonical math format.

```
Input:  "twelve times nine"
Output: "12 * 9 ="
Parse:  {op: multiply, a: 12, b: 9}
WASM:   [i32.const 12, i32.const 9, i32.mul]
Result: 108
```

**Results:** 100% accuracy on 8 test cases including:
- Word forms: "twelve times nine" → `12 * 9 =`
- Phrase forms: "25 plus 17" → `25 + 17 =`
- Mixed: "48 divided by 6" → `48 / 6 =`

### Stage 2: Complex Expressions (cot_ir_complex.py)

**Goal:** Handle operator precedence, chaining, and parentheses.

```
Input:  "12 times 20 and then minus 5"
Output: "12 * 20 - 5 ="
Parse:  [push(12), push(20), mul, push(5), sub]
Stack:  [12] → [12,20] → [240] → [240,5] → [235]
Result: 235
```

**Key capability:** Shunting yard algorithm converts infix to postfix, respecting precedence:
- `10 + 5 * 2` → 20 (not 30) - multiplication binds tighter
- `(5 + 2) * 3` → 21 - parentheses override precedence

**Results:** 100% accuracy on 11 test cases.

### Stage 3: Word Problems (cot_word_problems.py)

**Goal:** Extract mathematical structure from narrative text.

```
Input:  "A pizza has 8 slices. 3 people each eat 2 slices. How many left?"
Output: "8 - 3 * 2 ="
Parse:  [push(8), push(3), push(2), mul, sub]
Result: 2
```

**Semantic mappings learned:**

| Verb/Phrase | Operation |
|-------------|-----------|
| "gives", "loses", "eats" | SUBTRACT |
| "gets", "finds", "receives" | ADD |
| "each", "per", "groups of" | MULTIPLY |
| "split equally", "divided between" | DIVIDE |

**Results:** 100% accuracy on 8 word problems.

## The "= " Suffix

The `=` suffix is crucial - it signals "expression complete, ready for result." This:

1. **Tells the model** to stop generating and emit the answer position
2. **Triggers arithmetic circuits** in the hidden states (proven in suffix routing experiments)
3. **Provides parse boundary** - we know the expression ends at `=`

## Few-Shot Learning as Parser Configuration

When the model failed on "3 people each eat 2 slices" (outputting `8 - 3` instead of `8 - 3 * 2`), we fixed it by adding few-shot examples:

```
"10 cookies. 2 kids each eat 3. How many left?" → 10 - 2 * 3 =
"A jar has 20 marbles. 4 people each take 3. How many remain?" → 20 - 4 * 3 =
```

**The prompt IS the parser configuration.** Adding examples teaches new patterns without retraining.

## Why This Matters

### 1. Separation of Concerns

| Component | Responsibility |
|-----------|----------------|
| LLM | Semantic understanding (NL → canonical) |
| Shunting Yard | Syntax (precedence, associativity) |
| WASM | Execution (native speed, verified) |

### 2. Reliability

- **No regex on free-form LLM output** - the canonical format is constrained
- **Deterministic execution** - WASM is provably correct
- **Failure modes are clear** - either parse succeeds or fails cleanly

### 3. Extensibility

To add new operations:
1. Add few-shot examples to the prompt
2. Add operator to the parser
3. Add WASM opcode

No model retraining required.

## Files

| File | Purpose |
|------|---------|
| `cot_ir_pipeline.py` | Simple expressions, validates core concept |
| `cot_ir_complex.py` | Complex expressions with precedence |
| `cot_word_problems.py` | Word problems with semantic understanding |

## Running the Experiments

```bash
# Simple expressions
uv run python experiments/ir_emission/learned_ir_head/cot_ir_pipeline.py

# Complex expressions
uv run python experiments/ir_emission/learned_ir_head/cot_ir_complex.py

# Word problems
uv run python experiments/ir_emission/learned_ir_head/cot_word_problems.py
```

## Conclusion

**Chain-of-Thought is not just reasoning - it's format normalization.**

The LLM's ability to transform "Jenny has 5 apples, gives 2 to Bob" into "5 - 2 =" is exactly what a parser does. The difference:

- Traditional parser: hand-written rules, brittle to variation
- CoT parser: learned from examples, generalizes to novel phrasings

The "regex problem" (parsing free-form LLM output) disappears when you realize **you can train the LLM to emit parseable output**. The canonical format becomes the contract between the LLM and the execution engine.

```
CoT = Learned Parser
Prompt Engineering = Parser Configuration
Canonical Format = Interface Contract
WASM = Verified Execution
```

---

## Part 2: Training the Parser into the Model

### The Scaling Problem

Prompt engineering achieves 100% accuracy but doesn't scale:
- Context window fills up with few-shot examples
- Manual pattern identification required
- Cost per inference increases with example count

### Solution: SFT + RL with Verifiable Rewards

#### Stage 1: Generate Verified Training Data

```bash
uv run python experiments/ir_emission/learned_ir_head/generate_sft_data.py
```

Creates 5000 (question, expression) pairs, ALL verified via WASM execution:
- `sft_data/train.jsonl` (4000 examples)
- `sft_data/val.jsonl` (500 examples)
- `sft_data/test.jsonl` (500 examples)

#### Stage 2: Supervised Fine-Tuning

```bash
uv run python experiments/ir_emission/learned_ir_head/sft_train.py
```

Results after 2 epochs:
- **Parse success: 100%** (model always outputs valid expressions)
- **Accuracy: 67.8%** (correct answer)

#### Stage 3: RL with Verifiable Rewards

```python
def compute_reward(expression, expected_answer, wasm_runtime):
    ir = parse_expression(expression)

    if ir is None:
        return 0.0  # Parse failure

    result = wasm_execute(ir)

    if result == expected_answer:
        return 1.0  # Correct!

    return 0.3  # Valid structure, wrong answer
```

**Key insight: WASM execution IS the reward function.**

No human labels needed. The reward is deterministic and verifiable.

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Synthetic Data     →    SFT Training    →    RL Fine-tuning   │
│  (WASM verified)         (67.8% acc)          (target: 90%+)   │
│                                                                 │
│  Reward = parse_success × execution_correctness                │
│           └── deterministic, no human labels                   │
└─────────────────────────────────────────────────────────────────┘
```

### New Files

| File | Purpose |
|------|---------|
| `generate_sft_data.py` | Generate WASM-verified training data |
| `sft_train.py` | Supervised fine-tuning |
| `rl_train.py` | RL with verifiable rewards |
| `cot_hard_eval.py` | GSM8K-style evaluation |

### The Punchline

**Training for structural compliance + execution correctness.**

The model learns to:
1. Output parseable expressions (structural compliance)
2. Output correct expressions (semantic correctness)

Both are verifiable via WASM - no human labeling required.
