# Suffix Routing: Training-Dependent Behavior

## Research Question

**Is suffix routing (e.g., `= ` triggering numeric output) a property of the transformer architecture, or is it learned during training?**

## Summary

**Suffix routing is learned from training data, not architectural.**

| Model | Size | Instruction-Tuned | Suffix Routing |
|-------|------|-------------------|----------------|
| GPT-2 | 124M | No | None |
| GPT-2 Medium | 355M | No | None |
| TinyLlama-Chat | 1.1B | Yes | Strong |
| GPT-OSS | 20B | Yes | Strong |

Architecture (dense vs MoE, GELU vs SwiGLU, learned pos vs RoPE) is irrelevant. **Only instruction tuning creates routing.**

---

## Background

Previous experiments with TinyLlama-Chat revealed that the `= ` suffix activates arithmetic circuits regardless of input content:

```
15 > 10 =  → 5    (subtraction, not comparison)
foo bar =  → 1    (numeric output from gibberish)
```

This raised a fundamental question: Is this routing behavior inherent to transformers, or is it learned?

---

## Methodology

### Models Tested

| Model | Size | Type | Architecture |
|-------|------|------|--------------|
| TinyLlama-1.1B-Chat | 1.1B | Instruction-tuned | Llama (RoPE, SwiGLU) |
| GPT-2 | 124M | Base | Original GPT (learned pos, GELU) |
| GPT-2 Medium | 355M | Base | Original GPT |
| GPT-OSS | 20B | Instruction-tuned | MoE (32 experts, RoPE) |

### Test Cases

**Suffix Swap Test**: Same content, different suffixes
```
15 > 10 =    → Expected: numeric (if routing exists)
15 > 10 is   → Expected: boolean (if routing exists)
```

**Garbage Input Test**: Invalid content, valid suffix
```
foo bar =    → Tests if suffix alone controls output
15 10 =      → No operator
=            → Just the suffix
```

**Inversion Test**: Semantically identical, syntactically different
```
15 > 10 =    → "15 is greater than 10"
10 < 15 =    → "10 is less than 15" (same meaning!)
```

---

## Results

### TinyLlama 1.1B (Instruction-Tuned)

| Expression | Prediction | Analysis |
|------------|------------|----------|
| `15 > 10 =` | `5` | Subtraction circuit activated |
| `15 > 10 is` | `1` | Boolean circuit activated |
| `foo bar =` | `1` | Numeric from gibberish |
| `=` | `0` | Suffix alone → numeric |
| `10 < 15 =` | `9` | Different pattern, different output |

**Verdict**: Strong suffix routing. The `= ` suffix triggers arithmetic regardless of content.

### GPT-2 124M (Base Model)

| Expression | Prediction | Analysis |
|------------|------------|----------|
| `15 > 10 =` | `????` | Random Unicode |
| `15 > 10 is` | `iced` | Language completion |
| `foo bar =` | `` | Empty/unknown |
| `=` | `` | No pattern |

**Verdict**: No suffix routing. Outputs are chaotic language completions.

### GPT-2 355M (Base Model)

| Expression | Prediction | Analysis |
|------------|------------|----------|
| `15 > 10 =` | `????` | Random Unicode |
| `15 > 10 is` | `~~` | Random symbols |
| `foo bar =` | `~~` | No pattern |

**Verdict**: No suffix routing. Larger size doesn't help—the pattern isn't in the training.

### GPT-OSS 20B (Instruction-Tuned MoE)

| Expression | Prediction | Analysis |
|------------|------------|----------|
| `15 > 10 =` | `5` | Subtraction circuit (same as TinyLlama) |
| `15 > 10 is` | `1` | Boolean circuit |
| `foo bar =` | `1` | Numeric from gibberish |
| `2 + 2 =` | `4` | Correct arithmetic |
| `15 - 10 =` | `5` | Correct arithmetic |

**Verdict**: Strong suffix routing, identical pattern to TinyLlama despite completely different architecture.

---

## Deep Dive: GPT-OSS Routing

Additional probing revealed:

```
[Basic Arithmetic]
2 + 2 =         → 4     (conf: 15.43%)
5 + 3 =         → 8     (conf: 57.03%)
10 - 7 =        → 3     (conf: 80.08%)
15 - 10 =       → 5     (conf: 85.16%)

[Comparisons with '=' suffix]
15 > 10 =       → 5     (subtraction interpretation)
10 > 15 =       → 0     (clips negative to zero)
5 > 5 =         → 0     (zero difference)

[Boolean 'is' suffix]
15 > 10 is      → 1     (True)
10 > 15 is      → 0     (False)
True is         → 1
False is        → 0
```

The model correctly distinguishes:
- `= ` suffix → Arithmetic mode (even `>` becomes subtraction)
- `is ` suffix → Boolean mode (proper True/False)

---

## Key Findings

### 1. Suffix Routing is NOT Architectural

| Architecture | Base Model | Instruction-Tuned |
|--------------|------------|-------------------|
| GPT (2019) | No routing | N/A |
| Llama (2023) | N/A | Strong routing |
| MoE (2024) | N/A | Strong routing |

Different architectures show identical routing behavior **if and only if** they've been instruction-tuned.

### 2. Routing is LEARNED from Training Data

The pattern emerges from training data containing:
- Math problems: `2 + 2 = 4`
- Equations: `x = 5`
- Comparisons as equations: `result = True`

Instruction tuning reinforces: "When you see `= `, output a number or value."

### 3. Model Size Doesn't Create Routing

GPT-2 Medium (355M) shows no more routing than GPT-2 (124M). The pattern must be in the training data—you can't scale your way to it.

### 4. The Routing is Syntactic, Not Semantic

Both instruction-tuned models interpret `15 > 10 = ` as subtraction, not comparison. They learned a surface pattern:

```
Pattern learned: "A > B = " → output (A - B)
NOT: "A > B = " → evaluate comparison → output boolean
```

---

## Running the Experiment

```bash
# Multi-model comparison
python experiments/ir_emission/suffix_routing/multi_model_comparison.py

# Deep dive into GPT-OSS
python experiments/ir_emission/suffix_routing/probe_gpt_oss.py

# Layer-by-layer analysis
python experiments/ir_emission/suffix_routing/probe_layers.py
```

---

## Configuration

See `config.yaml`:

```yaml
models:
  tinyllama:
    name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    type: instruct
  gpt2:
    name: gpt2
    type: base
  gpt2_medium:
    name: gpt2-medium
    type: base

test_cases:
  suffix_swap:
    - "15 > 10 ="
    - "15 > 10 is"
  garbage:
    - "foo bar ="
    - "="
  arithmetic:
    - "2 + 2 ="
    - "5 + 3 ="
```

---

## Files

```
suffix_routing/
├── EXPERIMENT.md              # This file
├── config.yaml                # Configuration
├── multi_model_comparison.py  # Main comparison script
├── probe_gpt_oss.py          # Detailed GPT-OSS analysis
├── probe_layers.py           # Layer-by-layer analysis
└── MULTI_MODEL_ROUTING.md    # Additional routing documentation
```

---

## Implications

### For LLM Understanding

Models don't "understand" that `= ` means "equals" in a mathematical sense. They've learned a statistical association:
- Training data has patterns like `2 + 2 = 4`
- Model learns: after `= `, output digits
- This applies even to nonsense like `foo bar =`

### For System Design

If you want reliable arithmetic:
- Don't rely on suffix routing (it's syntactic, not semantic)
- Use the neural compiler approach: normalize → classify → execute
- The model's job is semantic understanding, not computation

### For the "LLMs Don't Reason" Thesis

This experiment provides direct evidence:
1. **Routing is learned, not inherent**: Base models don't route
2. **Routing is syntactic**: `= ` activates circuits regardless of semantic validity
3. **Architecture is irrelevant**: Only training data matters

---

## Conclusion

Suffix routing is a **learned behavior from instruction tuning**, not an architectural property of transformers. Models learn statistical associations between surface patterns (like `= `) and output types (numbers) from training data.

The same suffix that makes TinyLlama output `5` for `15 > 10 = ` makes GPT-2 output gibberish—because GPT-2 never learned that `= ` means "output a number."

**Bottom line**: If you want a model to "understand" math, you don't need a better architecture. You need training data that teaches the right patterns. And even then, it's still pattern matching—just more sophisticated pattern matching.
