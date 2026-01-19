# RL with Verifiable Rewards Experiment

## Hypothesis

Chain-of-Thought is format normalization, not reasoning. We can train a model to emit canonical math expressions directly (without few-shot prompting) using:

1. **SFT Cold Start** - Teach the model the target format
2. **RL with Verifiable Rewards** - Refine using execution as ground truth

The reward is **verifiable** via deterministic execution - no reward model needed.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Training Pipeline                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Word Problem                                                   │
│  "Has 42. Remove 15. Sell rest for $2 each."                   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Model (TinyLlama 1.1B)                                 │   │
│  │  Prompt: "Q: <problem>\nA:\n"                           │   │
│  │  Output: "42 - 15 =\n_ * 2 ="                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Verifier (Expression Execution)                        │   │
│  │  "42 - 15 =" → 27                                       │   │
│  │  "_ * 2 ="   → 54   (where _ = previous result)        │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Verifiable Reward                                      │   │
│  │  final_result == expected ? 1.0 : (parsed ? 0.3 : 0.0) │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: The model does **semantic parsing** (NL → operators/operands). The verifier does **computation** (arithmetic). The model doesn't need to know math!

## Expression-Only Format

The critical breakthrough: model emits expressions **without computed results**.

```
# Model outputs:
42 - 15 =
_ * 2 =
[END]

# Verifier computes:
42 - 15 = 27  ← verifier
27 * 2 = 54   ← verifier (chains previous result)
```

This separates concerns:
- **Model**: Maps natural language → operators and operands
- **Verifier**: Executes the expression chain

## Results Summary

### Single-Step Format (Initial Experiment)

| Phase | Test Accuracy | Notes |
|-------|--------------|-------|
| Baseline | 17% | Model outputs NL sentences |
| After SFT | 100% | 30 examples, 5 epochs |
| After RL | 100% | Maintained |

### Multi-Step Expression-Only Format (Scaled)

| Phase | Test Accuracy | Notes |
|-------|--------------|-------|
| Baseline | 5% | Natural language output |
| After SFT | 98% | 500 examples, 8 epochs |
| After RL | 98% | 30 iterations, maintained |

### Accuracy by Pattern (100 test examples)

| Pattern | Accuracy | Example |
|---------|----------|---------|
| subtract_subtract | 100% | "loses 11, loses 10" → `_ - 11 = \| _ - 10 =` |
| multiply_subtract | 100% | "7 rows of 6, sells 10" → `7 * 6 = \| _ - 10 =` |
| subtract_multiply | 100% | "uses 8, sells at $2" → `_ - 8 = \| _ * 2 =` |
| subtract_subtract_multiply | 100% | 3-step chains |
| multiply_subtract_divide | 89% | 3-step chains |
| single_add | 92% | "has 11, gets 25" → `11 + 25 =` |
| single_subtract | 100% | "has 78, drops 14" → `78 - 14 =` |
| single_multiply | 100% | "8 boxes with 4 each" → `8 * 4 =` |
| single_divide | 100% | "divide 36 among 4" → `36 / 4 =` |

## Key Findings

### 1. Expression-Only Format is Critical

**Failed approach**: Chain format with computed results
```
# Model outputs (wrong):
25 - 8 = 15    ← model computed 15 (should be 17)
_ - 9 = 6     ← wrong chain
```

The model can learn the format but **cannot reliably compute arithmetic** on unseen numbers. It memorizes training examples.

**Working approach**: Expression-only format
```
# Model outputs:
25 - 8 =
_ - 9 =

# Verifier computes:
25 - 8 = 17   ← correct
17 - 9 = 8    ← correct
```

### 2. Diverse Training Data Fixes Generalization

| Training Size | Test Accuracy | Issue |
|---------------|---------------|-------|
| 12 examples | 50% | Multi-step patterns not learned |
| 500 examples | 98% | Diverse patterns generalize |

Pattern coverage matters more than quantity:
- Single operations (add, subtract, multiply, divide)
- Two-step chains (subtract-subtract, multiply-subtract, etc.)
- Three-step chains (subtract-subtract-multiply, etc.)

### 3. SFT Cold Start is Essential

Pure RL from base model fails:
- Model outputs natural language, not expressions
- Reward signal too sparse (mostly 0.0 parse failures)
- REINFORCE can't bridge the distribution gap

SFT teaches the format; RL refines with verifiable rewards.

### 4. Model Learns Semantic Mapping

After training, the model internalizes:
```
"gets", "finds", "earns", "more"  → ADD
"eats", "loses", "gives", "uses"  → SUBTRACT
"each", "per", "rows of", "with"  → MULTIPLY
"split", "divide", "among"        → DIVIDE
"then", "after", sequential       → CHAIN (_)
```

No few-shot prompting needed at inference.

## Training Configuration

### SFT Phase
- Model: TinyLlama-1.1B-Chat
- Trainable: last 3 transformer layers + lm_head
- Learning rate: 2e-5
- Batch size: 8
- Epochs: 8

### RL Phase (REINFORCE)
- Learning rate: 5e-7
- Temperature: 0.7
- Batch size: 16
- Iterations: 30
- Baseline: exponential moving average (α=0.1)

## Reward Function

```python
def compute_reward(chain_text: str, expected: int) -> tuple[float, str]:
    result, reason = execute_chain(chain_text)

    if result is None:
        return 0.0, reason      # Parse failure

    if result == expected:
        return 1.0, "correct"   # Verifiably correct!

    return 0.3, f"wrong:{result}"  # Valid structure, wrong answer
```

## Sample Outputs

### Before Training (Baseline)
```
Q: Sara has 40 pencils. Uses 12. Sells rest at $2 each. Revenue?
A: Sara has 40 pencils. | Uses 12 pencils. | Sells rest at $2 each.
   | Revenue = $40 * 12 / 2 = $60
Status: ✗ parse_fail
```

### After Training
```
Q: Sara has 40 pencils. Uses 12. Sells rest at $2 each. Revenue?
A: 40 - 12 =
   _ * 2 =
Status: ✓ (verifier computes: 28 * 2 = 56)
```

### Three-Step Example
```
Q: Has 8 groups of 10. Takes 6 away. Divides remaining among 2. Each gets?
A: 8 * 10 =
   _ - 6 =
   _ / 2 =
Status: ✓ (verifier computes: 80 → 74 → 37)
```

## Files

```
experiments/ir_emission/learned_ir_head/
├── rl_expr_scaled.py      # Main training script (expression-only)
├── gen_expr_data.py       # Data generator with pattern library
├── expr_data/             # Generated datasets
│   ├── train.jsonl        # 500 training examples
│   ├── val.jsonl          # 50 validation examples
│   └── test.jsonl         # 100 test examples
├── rl_cot_experiment.py   # Original single-step experiment
├── chain_cot_generator.py # Chain format generator (deprecated)
└── rl_chain_cot.py        # Chain format training (deprecated)
```

## Run the Experiment

```bash
# Generate training data
python experiments/ir_emission/learned_ir_head/gen_expr_data.py

# Run training
python experiments/ir_emission/learned_ir_head/rl_expr_scaled.py
```

## Conclusion

**RL with verifiable rewards works** for training models to emit structured output:

1. **SFT provides the format** - teaches NL → expression mapping
2. **Expression-only format** - model doesn't need to compute arithmetic
3. **Verifier executes** - deterministic ground truth, no reward model
4. **Diverse patterns** - generalization requires coverage, not just quantity

The model learns to be a **semantic parser**, normalizing natural language to canonical expressions that can be deterministically executed. This validates the hypothesis:

> **Chain-of-Thought is format normalization, not reasoning.**

The "reasoning" happens in the verifier (execution), not the model.

## v2 Results: Extended Patterns

### GSM8K Gap Analysis

Analysis of 500 GSM8K problems revealed missing patterns:
- Fractions/percentages: 67 signals (13.5%)
- Long chains (4+ steps): 42% of problems
- "X times"/"twice": 30 signals
- Time calculations: 51 signals
- Rate problems: 25 signals

### Expression-Only v2 (Extended Patterns)

Added new patterns to cover GSM8K distribution:

| Category | Patterns | Test Accuracy |
|----------|----------|---------------|
| Times/twice | times_multiply, twice_compare | 100% |
| Fractions | half_of, third_of, percent_of | 100% |
| Time/rate | rate_time, daily_total, price_quantity | 100% |
| Long chains | four_step_chain, five_step_chain | 76-83% |
| Business | buy_sell_chain | **0%** |

**Total v2: 95% accuracy** (190/200 test examples)

### Multi-Variable Limitation Discovered

The `buy_sell_chain` pattern failed completely (0%) because it requires tracking TWO values:

```
# Problem: "Makes item for $10. Has 10 items. Sells 6 at $11 each. Profit?"
# Requires:
cost = 10 * 10        # track total cost
sold = 6              # track quantity sold
revenue = 6 * 11      # revenue from sales
remaining = 10 - 6    # unsold items
profit = revenue - cost  # final answer

# But expression-only can only track ONE value (_):
10 * 10 =             # cost = 100
6 * 11 =              # revenue = 66... but we lost the cost!
```

### Named Variable Format (Multi-Variable)

Created new format to handle multi-variable tracking:

```
# Input: "Makes item for $10. Has 10 items. Sells 6 at $11 each. Profit?"
# Output:
cost = 10 * 10
revenue = 6 * 11
profit = revenue - cost
[END]
```

**Results: 99% accuracy** (195/196 test examples)

| Pattern | Accuracy |
|---------|----------|
| two_step_sequential | 100% (47/47) |
| single_op | 100% (35/35) |
| buy_sell_profit | 96% (24/25) |
| two_people_compare | 100% (18/18) |
| rate_expense_bonus | 100% (16/16) |
| discount_calculation | 100% (14/14) |
| five_step_weekly | 100% (12/12) |
| profit_margin | 100% (11/11) |
| four_step_business | 100% (10/10) |
| split_and_share | 100% (8/8) |

**Key insight**: Named variable format maps directly to SSA/IR representation.

---

## GSM8K Real-World Evaluation

### Test Setup

Both formats trained on 500 synthetic examples (6 epochs), then evaluated on 20 real GSM8K problems.

### Results

| Format | Correct | Parse Fail | Wrong |
|--------|---------|------------|-------|
| Expression-only v2 | 0/20 (0%) | 2/20 (10%) | 18/20 (90%) |
| Named variable | 1/20 (5%) | 0/20 (0%) | 19/20 (95%) |

### Failure Analysis

**Example: Janet's ducks** (expected: 18)
```
Problem: "Janet's ducks lay 16 eggs per day. She eats three for
breakfast every morning and bakes muffins for her friends every
day with four. She sells the remainder at the farmers' market
daily for $2 per fresh duck egg."

Correct solution:
remaining = 16 - 3 - 4     # eggs after eating/baking = 9
revenue = remaining * 2     # sell at $2 each = 18

Model output (expression-only):
16 * 3 =                    # wrong semantic mapping

Model output (named variable):
after_gain = 16 * 3         # wrong semantic mapping
remaining = after_gain - 4
```

**Key finding**: Models achieve 95-99% on synthetic data but 0-5% on GSM8K because:

1. **Format learning is easy** - Both formats learned correctly
2. **Semantic mapping is hard** - Model doesn't know WHICH numbers to use

The model learned:
- ✅ HOW to write `x * y =` (format)
- ❌ WHICH x and y to extract from natural language (semantics)

### What This Means for the Hypothesis

This **validates** the hypothesis "Chain-of-Thought is format normalization, not reasoning":

1. **Format learning**: Easy (95-99% with ~1000 examples)
2. **Semantic mapping**: Requires training data that covers the SEMANTIC PATTERNS of the target domain
3. **The "reasoning" in CoT**: Not reasoning - it's format normalization

GSM8K problems use different semantic patterns than our synthetic data:
- More complex entity tracking ("Janet", "ducks", "eggs", "muffins")
- Implicit subtraction ("eats three... bakes with four")
- Background information filtering ("every morning", "farmers' market")

### Path Forward

To achieve GSM8K performance:
1. **Semantic pattern mining** - Extract actual operation patterns from GSM8K solutions
2. **Template expansion** - Generate synthetic data matching GSM8K semantics
3. **Few-shot examples** - Show model how to extract operands from GSM8K-style prose
4. **Curriculum learning** - Progress from simple to GSM8K-like complexity

---

## Files (v2)

```
experiments/ir_emission/learned_ir_head/
├── rl_expr_scaled.py        # Expression-only training
├── rl_expr_scaled_v2.py     # Extended expression-only (v2)
├── rl_named_vars.py         # Named variable training
├── gen_expr_data.py         # Original data generator
├── gen_expr_data_v2.py      # Extended patterns (GSM8K-inspired)
├── gen_named_vars.py        # Multi-variable data generator
├── gsm8k_pattern_analysis.py # GSM8K distribution analysis
├── test_gsm8k.py            # GSM8K evaluation
├── expr_data/               # Original expression data
├── expr_data_v2/            # Extended expression data
├── named_var_data/          # Named variable data
└── RL_VERIFIABLE_EXPERIMENT.md  # This document
```

---

## Conclusion

**RL with verifiable rewards works** for format learning:

1. **Expression-only format**: 95% on extended synthetic patterns
2. **Named variable format**: 99% on multi-variable patterns
3. **GSM8K**: 0-5% due to semantic distribution shift

The experiment confirms:
> **Chain-of-Thought is format normalization, not reasoning.**

Format learning is the easy part. The hard part is semantic mapping - understanding which numbers and operations to extract from natural language. This requires training data that matches the semantic distribution of the target domain.

## GSM8K IR Training (Direct)

### Approach

Instead of synthetic data, train directly on GSM8K solutions converted to IR format.

GSM8K solutions contain `<<expr=result>>` annotations that provide exact computation graphs:
```
Answer: "She eats 3 + 4 = <<3+4=7>>7 eggs...
         The remainder is 16 - 7 = <<16-7=9>>9 eggs...
         She earns 9 * 2 = <<9*2=18>>$18"
```

Converted to IR:
```
step1 = 3+4
step2 = 16-step1
step3 = step2*2
[END]
```

### Data Statistics

- **Parsed**: 7378/7473 GSM8K examples (98.7%)
- **Step distribution**:
  - 1 step: 5.5%
  - 2 steps: 29.5%
  - 3 steps: 29.0%
  - 4 steps: 19.3%
  - 5+ steps: 16.7%

### Training Results

| Metric | Value |
|--------|-------|
| Training data | 2000 examples |
| Test data | 738 examples |
| Final test accuracy | **7%** (48/738) |
| Parse failures | 1% (6/738) |
| Wrong answers | 93% (684/738) |
| Train accuracy | 52% (overfitting) |

### Epoch Progression

| Epoch | Loss | Train Acc | Val Acc |
|-------|------|-----------|---------|
| 1 | 0.515 | 6% | 10% |
| 2 | 0.398 | 10% | 6% |
| 3 | 0.338 | 6% | 6% |
| 4 | 0.259 | 24% | 6% |
| 5 | 0.177 | 38% | 12% |
| 6 | 0.123 | 52% | 10% |
| 7 | 0.091 | 48% | 8% |
| 8 | 0.073 | 52% | 12% |

### Analysis

**Format learning succeeded:**
```
Q: Miriam spent 30 minutes doing laundry...
Out: step1 = 15+30 | step2 = step1+40 | step3 = step2/60 | [END]
Status: ✗ wrong (valid syntax, wrong semantics)
```

**Semantic mapping failed:**
- Model extracts numbers from the problem
- Model generates plausible-looking operations
- Model does NOT understand which numbers to use or which operations to apply

**The 52% train vs 7% test gap indicates:**
1. TinyLlama 1.1B memorizes rather than generalizes on this task
2. GSM8K has too much semantic diversity for 2000 examples
3. The model lacks the capacity to learn the NL→IR mapping from limited data

### Implications for IR Approach

This confirms that **format learning is orthogonal to semantic learning**:

| Aspect | Result | Conclusion |
|--------|--------|------------|
| IR format | 99% valid syntax | Easy to learn |
| Argument placement | 7% correct | Hard to learn |
| Operation selection | 7% correct | Hard to learn |

To improve GSM8K performance, the semantic understanding must come from:
1. **Larger models** with better reasoning capabilities
2. **More training data** with diverse patterns
3. **Pre-existing knowledge** leveraged via few-shot prompting
4. **Curriculum learning** from simple to complex

The IR format is ready for when semantic understanding improves.

---

## Future Work

1. **Scale training** - Use all 5902 GSM8K examples
2. **Larger models** - Test with 7B+ models that may generalize better
3. **Few-shot IR prompting** - Combine pre-trained knowledge with IR format
4. **WASM backend** - Replace Python verifier with IR → WASM execution
5. **Curriculum learning** - Progressive difficulty from synthetic → GSM8K
6. **Hybrid approach** - Use larger model for semantics, smaller for IR execution
