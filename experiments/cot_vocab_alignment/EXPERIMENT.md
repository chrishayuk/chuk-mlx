# CoT Vocabulary Alignment Experiment

## Research Question

**Does Chain-of-Thought (CoT) training create vocabulary-aligned classifiers?**

GPT-OSS shows "multiply", "add", "subtract" tokens at L13 with 30-50% probability. One hypothesis is that this emerges from CoT training where operation words appear in the output.

## Results Summary (January 10, 2026)

### Finding: CoT Training Does NOT Create Vocabulary Alignment

| Stage | Accuracy | Max Vocab Alignment |
|-------|----------|---------------------|
| Baseline | 88.9% | **0.0%** |
| Direct SFT | 66.7% | 0.2% |
| CoT SFT | 55.6% | **0.0%** |

**CoT training did not create vocabulary-aligned classifiers at any layer.**

### Per-Layer Vocabulary Alignment

| Layer | Baseline | Direct SFT | CoT SFT |
|-------|----------|------------|---------|
| L4 (25%) | 0.0% | 0.0% | 0.0% |
| L8 (50%) | 0.0% | 0.2% | 0.0% |
| L12 (75%) | 0.0% | 0.0% | 0.0% |
| L13 (81%) | 0.0% | 0.0% | 0.0% |
| L15 (94%) | 0.0% | 0.0% | 0.0% |

## Analysis

### 1. No Vocabulary Alignment at Any Layer

Despite training on CoT format where "multiply", "add", "subtract" appear in the output, these tokens never reached significant probability at intermediate layers.

### 2. Training Hurt Accuracy

| Stage | Accuracy | Change |
|-------|----------|--------|
| Baseline | 88.9% | - |
| Direct SFT | 66.7% | -22.2% |
| CoT SFT | 55.6% | -33.3% |

Both training formats made the model WORSE. This suggests:
- Small model (1B) may be sensitive to fine-tuning
- 500 steps on 3000 samples may cause overfitting
- The CoT format may confuse the model

### 3. CoT Format Requires More Processing

The CoT format adds a parsing step:
```
Input:  "7 * 8 = "
Output: "multiply: 56"
        ^^^^^^^^^^
        Model must generate operation word THEN answer
```

This may be harder than direct answer generation for a 1B model.

## Why GPT-OSS Has Vocabulary Alignment

Since CoT training alone doesn't create vocabulary alignment, GPT-OSS must use something else:

### Hypothesis 1: Scale Creates Redundancy
```
1B params:  Task info encoded efficiently (one subspace)
20B params: Task info encoded redundantly (including vocab-aligned)

More capacity → more representations → some naturally align with vocabulary
```

### Hypothesis 2: MoE Architecture
```
GPT-OSS has Mixture of Experts
Router must make DISCRETE decisions
Discrete decisions → vocabulary-like representations

Dense models don't need discrete routing
→ No pressure for vocabulary alignment
```

### Hypothesis 3: Explicit Training Objective
```
OpenAI may have explicitly trained for vocabulary classifiers:
  L_total = L_answer + λ * L_classifier

Where L_classifier rewards "multiply" at L13 for multiplication problems.
```

### Hypothesis 4: RLHF/Constitutional AI
```
RLHF training may create vocabulary alignment:
- Human feedback rewards clear reasoning
- Clear reasoning often uses operation words
- Model learns to "think" in vocabulary tokens
```

## What We Learned

1. **CoT training alone does NOT create vocabulary alignment** at intermediate layers

2. **500 steps of CoT hurts accuracy** (88.9% → 55.6%) on a 1B model

3. **GPT-OSS vocabulary classifiers require something else** - scale, MoE, or explicit training

4. **Vocabulary alignment may be emergent at scale** rather than trained

## Comparison with probe_classifier

| Experiment | What it tests | Result |
|------------|---------------|--------|
| probe_classifier | Is task info encoded? | YES (100% at L4) |
| cot_vocab_alignment | Is task info vocab-aligned? | NO (0% at all layers) |

Task information exists but is NOT vocabulary-aligned in Llama-3.2-1B.

## Implications

### For Virtual Expert Architecture

Since vocabulary alignment doesn't naturally emerge from training:

1. **Use learned routing projections** (like linear probes)
2. **Don't rely on vocabulary lookup** (logit lens approach)
3. **Train routing matrices** to read the task subspace

### For Understanding GPT-OSS

GPT-OSS's vocabulary classifiers are likely:
- An artifact of scale (20B >> 1B)
- Related to MoE router training
- Or explicitly trained

They're probably NOT:
- A natural consequence of CoT training
- Something small models can easily reproduce

## Training Details

### Data Formats

**Direct format:**
```
"7 * 8 = 56"
```

**CoT format:**
```
"7 * 8 = multiply: 56"
```

### Training Config

```yaml
max_steps: 500
batch_size: 4
learning_rate: 0.0002
lora:
  rank: 16
  alpha: 32.0
  targets: [q_proj, k_proj, v_proj, o_proj]
```

## Files

```
cot_vocab_alignment/
├── EXPERIMENT.md       # This file
├── config.yaml         # Configuration
├── experiment.py       # Implementation
├── data/
│   ├── train_direct.jsonl
│   ├── train_cot.jsonl
│   ├── valid_direct.jsonl
│   └── valid_cot.jsonl
├── checkpoints/
│   ├── sft_direct/
│   └── sft_cot/
└── results/
```

## Running

```bash
lazarus experiment run cot_vocab_alignment
```

## Conclusion

**CoT training does not create vocabulary-aligned classifiers.**

GPT-OSS's L13 classifiers are likely emergent from scale or MoE architecture, not from CoT training. For virtual expert architectures on smaller models, use **learned routing projections** rather than vocabulary lookup.
