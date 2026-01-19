# RL with Verifiable Rewards

## Research Question

**Can we train a model to emit canonical math expressions (without few-shot prompting) using WASM execution as the reward signal?**

## Summary

**Yes, on synthetic patterns. Semantic gap blocks GSM8K.**

| Phase | Synthetic Accuracy | GSM8K Accuracy |
|-------|-------------------|----------------|
| Baseline | 5% | ~0% |
| After SFT | 98% | 7% |
| After RL | 98% | ~5% |

The model learns to map semantic patterns to expressions on synthetic data, but struggles with the semantic diversity of real GSM8K problems.

---

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
│  │  Verifier (WASM Execution)                              │   │
│  │  "42 - 15 =" → 27                                       │   │
│  │  "_ * 2 ="   → 54   (where _ = previous result)        │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Verifiable Reward                                      │   │
│  │  result == expected ? 1.0 : (parsed ? 0.3 : 0.0)       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: The model does **semantic parsing** (NL → operators/operands). The verifier does **computation** (arithmetic). No reward model needed—execution is ground truth.

---

## Expression-Only Format

**Critical breakthrough**: Model emits expressions **without computed results**.

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
- **Verifier**: Executes the expression chain deterministically

---

## Results

### Synthetic Patterns (98% accuracy)

| Pattern | Accuracy | Example |
|---------|----------|---------|
| subtract_subtract | 100% | "loses 11, loses 10" → `_ - 11 = \| _ - 10 =` |
| multiply_subtract | 100% | "7 rows of 6, sells 10" → `7 * 6 = \| _ - 10 =` |
| subtract_multiply | 100% | "uses 8, sells at $2" → `_ - 8 = \| _ * 2 =` |
| single_add | 92% | "has 11, gets 25" → `11 + 25 =` |
| single_subtract | 100% | "has 78, drops 14" → `78 - 14 =` |
| single_multiply | 100% | "8 boxes with 4 each" → `8 * 4 =` |

### GSM8K (7% accuracy)

The semantic gap is the blocker:
- Training data: "has X, loses Y" → `X - Y =`
- GSM8K: "If she originally had X and Y were eaten..." (different phrasing)

---

## Key Findings

### 1. Expression-Only Format is Critical

**Failed approach**: Chain format with computed results
```
# Model outputs (wrong):
25 - 8 = 15    ← model computed 15 (should be 17)
_ - 9 = 6     ← wrong chain
```

The model **cannot reliably compute arithmetic** on unseen numbers. It memorizes training examples.

**Working approach**: Expression-only format
```
# Model outputs:
25 - 8 =
_ - 9 =

# Verifier computes:
25 - 8 = 17   ← correct
17 - 9 = 8    ← correct
```

### 2. SFT Cold Start is Essential

Pure RL from base model fails:
- Model outputs natural language, not expressions
- Reward signal too sparse (mostly 0.0 parse failures)
- REINFORCE can't bridge the distribution gap

SFT teaches the format; RL refines with verifiable rewards.

### 3. Model Learns Semantic Mapping

After training, the model internalizes:
```
"gets", "finds", "earns", "more"  → ADD
"eats", "loses", "gives", "uses"  → SUBTRACT
"each", "per", "rows of", "with"  → MULTIPLY
"split", "divide", "among"        → DIVIDE
```

### 4. Semantic Gap is the Blocker

Format learning works (95-99% on synthetic). Semantic mapping is hard:
- Synthetic: Controlled vocabulary, clear patterns
- GSM8K: Diverse phrasing, implicit operations, multi-step reasoning

---

## Running the Experiment

```bash
# Generate verified training data
python experiments/ir_emission/rl_verifiable_rewards/generate_data.py

# SFT training
python experiments/ir_emission/rl_verifiable_rewards/sft_train.py

# RL fine-tuning (long-running)
python experiments/ir_emission/rl_verifiable_rewards/rl_train.py

# Evaluation on GSM8K
python experiments/ir_emission/rl_verifiable_rewards/eval_gsm8k.py
```

---

## Configuration

See `config.yaml`:

```yaml
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

training:
  sft_examples: 500
  sft_epochs: 8
  rl_iterations: 30

reward:
  correct: 1.0
  parsed_but_wrong: 0.3
  parse_failure: 0.0

format:
  expression_only: true  # Critical!
  use_underscore_chain: true
```

---

## Files

```
rl_verifiable_rewards/
├── EXPERIMENT.md           # This file
├── config.yaml             # Configuration
├── generate_data.py        # Create verified training data
├── sft_train.py            # Supervised fine-tuning
├── rl_train.py             # REINFORCE with verifiable rewards
├── rl_verifiable.py        # Verifiable reward computation
└── eval_gsm8k.py           # GSM8K evaluation
```

---

## Status

**Work in Progress.** Format learning is solved; semantic generalization is not.

**Solved:**
- ✓ Expression-only format (no model arithmetic)
- ✓ WASM verification as reward signal
- ✓ SFT + RL training pipeline
- ✓ 98% on synthetic multi-step patterns

**Unsolved:**
- ✗ GSM8K accuracy (7%)
- ✗ Semantic diversity handling
- ✗ Implicit operation detection

---

## Implications

### For Verifiable Rewards

WASM execution provides **ground truth reward** without:
- Human labeling
- Reward models
- Approximations

The reward is deterministic and verifiable.

### For the Semantic Gap

The bottleneck is not format or computation—it's **semantic understanding**:
- Mapping diverse NL to canonical operations
- Handling implicit quantities
- Multi-step reasoning chains

This is the frontier for future work.

---

## Conclusion

**Format learning works. Semantic transfer is hard.**

The expression-only format + WASM verification achieves near-perfect accuracy on controlled synthetic data. The gap to real-world benchmarks (GSM8K) is semantic, not computational.

```
WASM = Perfect Verifier
Model = Semantic Parser (needs more diverse training)
Gap = NL variation, not arithmetic capability
```
