# CoT Correlation Experiment

## Research Question

**Does L13 vocabulary signal predict/gate CoT generation?**

Hypothesis: GPT-OSS L13 vocabulary alignment ("multiply", "add") might gate whether the model generates CoT reasoning vs direct answers.

## Critical Finding (January 10, 2026)

### GPT-OSS-20B Does NOT Have L13 Vocabulary Classifiers

```
Tested prompts at L13:
  "7 * 8 = "              | multiply: 1.06e-09 (essentially 0%)
  "What is 7 times 8?"    | multiply: 7.10e-09
  "Calculate: 7 * 8"      | multiply: 1.90e-08
  "23 + 45 = "            | add: 2.31e-06
  "What is 23 plus 45?"   | add: 6.08e-09
```

**The "multiply/add" tokens never reach significant probability at L13 in this checkpoint.**

### What the Model Does Show

At L13, the top tokens are:
- `<|endoftext|>` (41-97%)
- Special tokens
- Sometimes "sum" for addition (12%)
- Digits appear at later layers (L18+)

### Implications

1. **GPT-OSS papers may have used a different methodology** or checkpoint
2. **Vocabulary alignment is NOT universal** even at 20B scale
3. **The two-layer routing hypothesis cannot be tested** on this model
4. **Our earlier findings still hold**: Task info exists early (L4 probe = 100%) but is NOT vocabulary-aligned

## Generation Behavior

Despite lacking vocab alignment, the model generates correctly:

```
"7 * 8 = "           → "56" (direct)
"What is 7 times 8?" → "7 times 8 is 56." (verbalized)
```

So the model CAN distinguish between formats - it just doesn't use vocabulary-aligned representations at L13.

## Comparison with Llama-3.2-1B

| Measure | Llama-3.2-1B | GPT-OSS-20B |
|---------|--------------|-------------|
| L4 probe accuracy | 100% | Not tested |
| L13/14 vocab alignment | 0% | 0% |
| L15 vocab alignment | ~75% | N/A |
| CoT generation | No | Yes (format-dependent) |

## Possible Explanations

### 1. Different Checkpoint/Version
The GPT-OSS papers may have analyzed an internal version with different training. The HuggingFace release might be a different checkpoint.

### 2. MoE Changes Representations
GPT-OSS uses Mixture of Experts. The MoE routing might handle task classification differently than vocabulary projection.

### 3. Papers Measured Differently
The original analysis may have:
- Used different prompts
- Measured at different positions
- Applied different normalization
- Used internal tooling

### 4. Vocabulary Alignment is Emergent
It may only emerge under specific conditions:
- RLHF training
- Constitutional AI
- Specific fine-tuning

## What We Learned

1. **Cannot reproduce GPT-OSS L13 vocab classifiers** on HuggingFace checkpoint
2. **Vocabulary alignment is not automatic** at 20B scale
3. **Task routing must use learned projections**, not vocabulary lookup
4. **The probe experiment findings generalize**: Task info exists but is non-vocab-aligned

## Future Work

1. Test with different GPT-OSS checkpoints if available
2. Check if vocab alignment exists at OTHER layers
3. Test with MoE-specific introspection (router activations)
4. Train vocab alignment explicitly via dual-reward (as in classifier_emergence)

## Files

```
cot_correlation/
├── EXPERIMENT.md       # This file
├── config.yaml         # Configuration
└── experiment.py       # Implementation (for reference)
```

## Running

```bash
lazarus experiment run cot_correlation
```

Note: Experiment may not produce meaningful correlation results since GPT-OSS lacks the expected L13 signal.

## Key Takeaway

**GPT-OSS's L13 vocabulary classifiers (as described in papers) are not present in the HuggingFace checkpoint.** Use learned routing projections (like linear probes) rather than expecting vocabulary alignment.
