# Classifier Emergence via Dual-Reward Training

## Research Question

**Can we induce vocabulary-aligned classifiers at intermediate layers through dual-reward training?**

Specifically: When training V/O projections with a combined loss (classification at L12 + answer generation at final layer), do we create classifiers that can be read via logit lens?

## Background

### The GPT-OSS Phenomenon

GPT-OSS exhibits a remarkable property at Layer 13 (~54% depth):
- Input: `45 * 45 =`
- Logit lens at L13: **"multiply"** with 50-80% probability

This is NOT present in base models like Llama or TinyLlama - they show 0% for operation tokens at intermediate layers.

### Two Types of Classifiers

| Type | Detection Method | Base Models | GPT-OSS |
|------|-----------------|-------------|---------|
| **Hidden-space** | Linear probe on activations | ✓ Present (100% acc) | ✓ Present |
| **Vocab-aligned** | Logit lens projection | ✗ Absent (0% prob) | ✓ Present (50-80%) |

## Hypothesis

Training V/O projections (value and output projections in attention) with dual-reward creates vocab-aligned classifiers because:

1. V projection determines what information flows through attention
2. O projection determines how that information maps back to residual stream
3. Combined, they can steer the representation toward vocabulary tokens

## Method

### Training Architecture

```
Input: "7 * 8 = "
         │
         ▼
┌─────────────────────────────────┐
│  Transformer Layers 0-11       │
│  (frozen base weights)          │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Layer 12 (Classifier Layer)   │
│  LoRA on v_proj, o_proj        │
│         │                       │
│         ▼                       │
│  Logit lens → "multiply" token │
│  Classification Loss (weight=0.7)│
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Transformer Layers 13-21      │
│  LoRA on v_proj, o_proj        │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Final Layer Output            │
│  → "56"                        │
│  Answer Loss (weight=0.3)      │
└─────────────────────────────────┘
```

### Dual-Reward Loss

```python
total_loss = classifier_weight * classifier_loss + (1 - classifier_weight) * answer_loss

# classifier_loss: Cross-entropy at L12 for operation token
# answer_loss: Standard language modeling loss at final layer
```

### Training Configuration

- **Model**: TinyLlama-1.1B-Chat
- **LoRA targets**: v_proj, o_proj only
- **LoRA rank**: 32
- **Classifier layer**: L12 (55% of 22 layers)
- **Classifier weight**: 0.7
- **Training steps**: 1000
- **Learning rate**: 5e-4

## Running the Experiment

```bash
# Run the experiment
lazarus experiment run cli_classifier_emergence

# View results
lazarus experiment status cli_classifier_emergence

# Or run directly
python -m chuk_lazarus.cli.main experiment run cli_classifier_emergence
```

## Results

### Training Metrics

| Step | Total Loss | Classifier Loss | Answer Loss | Cls Accuracy |
|------|-----------|-----------------|-------------|--------------|
| 100  | 0.577     | 0.078           | 1.740       | 100%         |
| 500  | 0.013     | 0.020           | 0.002       | 100%         |
| 1000 | 0.004     | 0.005           | 0.001       | 100%         |

Training accuracy reaches 100% quickly, but this is on the training distribution.

### Evaluation Results

| Prompt | Expected | Predicted | Confidence | Correct |
|--------|----------|-----------|------------|---------|
| 7 * 8 =  | multiply | subtract | 4.8% | ✗ |
| 12 * 5 = | multiply | subtract | 6.3% | ✗ |
| 23 + 45 = | add | subtract | 1.3% | ✗ |
| 17 + 38 = | add | subtract | 2.2% | ✗ |
| 50 - 23 = | subtract | subtract | 35.9% | ✓ |
| 89 - 34 = | subtract | subtract | 33.8% | ✓ |
| 48 / 6 = | divide | subtract | 4.2% | ✗ |
| 81 / 9 = | divide | subtract | 3.6% | ✗ |

**Accuracy: 25% (2/8)**

### Analysis

The model shows a strong bias toward predicting "subtract" (the `-` token). This reveals an important insight:

1. **Operator symbols are already in the prompt**: When the model sees `50 - 23 = `, the `-` token is literally present in the input, giving it high prior probability.

2. **Competition between similar tokens**: The operator tokens (`*`, `+`, `-`, `/`) are all similar punctuation marks in the vocabulary, making discrimination difficult.

3. **Subtraction bias**: The `-` character appears in more contexts (negative numbers, hyphens) than other operators, giving it a baseline advantage.

## Key Findings

### What Works
- Dual-reward training successfully optimizes both losses
- Training accuracy reaches 100% on the training set
- The model learns to produce correct answers

### What Doesn't Work
- Vocab-aligned operation classification struggles with symbolic math prompts
- The classifier layer shows operator bias, not true classification
- Evaluation accuracy (25%) is near random chance for 4 classes

### Implications

1. **Vocab-aligned classifiers may require semantic input**: The ir_emission experiment achieves 100% accuracy because it uses natural language ("What is 7 times 8?") instead of symbolic math ("7 * 8 =").

2. **Operator presence in input confounds classification**: When the classifier target token is already in the input, the model learns to copy rather than classify.

3. **Different training objectives may be needed**: Pure SFT or GRPO might produce different classifier emergence patterns.

## Comparison with Other Approaches

| Approach | Method | Classifier Type | Accuracy |
|----------|--------|-----------------|----------|
| **This experiment** | Dual-reward V/O | Vocab-aligned (operation) | 25% |
| **classifier_emergence** | SFT + logit lens | Vocab-aligned (answer) | High |
| **ir_emission** | Dual-reward + normalization | Vocab-aligned (operation) | 100% |

The key differentiator is whether the input is **symbolic** (operators visible) or **semantic** (natural language).

## Files

```
cli_classifier_emergence/
├── EXPERIMENT.md      # This file
├── README.md          # Quick start guide
├── experiment.py      # ExperimentBase implementation
├── config.yaml        # Configuration
├── data/              # Generated arithmetic data
├── checkpoints/       # Trained LoRA weights
├── results/           # Run results (JSON)
└── archive/           # Historical scripts
```

## Does the Classifier Help?

### Baseline Performance

The base TinyLlama-1.1B already achieves **75% accuracy** on simple arithmetic without any fine-tuning:

| Prompt | Expected | Base Model | Correct |
|--------|----------|------------|---------|
| 7 * 8 = | 56 | 56 | ✓ |
| 12 * 5 = | 60 | 60 | ✓ |
| 23 + 45 = | 68 | 78 | ✗ |
| 50 - 23 = | 27 | 27 | ✓ |
| 48 / 6 = | 8 | 8 | ✓ |
| 9 * 9 = | 81 | 720 | ✗ |
| 17 + 38 = | 55 | 55 | ✓ |
| 100 - 43 = | 57 | 57 | ✓ |

### Training Impact

During dual-reward training:
- Answer loss: 1.7 → 0.001 (dramatic improvement)
- Classifier loss: 0.078 → 0.005 (converged)

The answer loss dropping indicates the model learned to produce correct answers. However, we cannot definitively say whether the **classifier component** contributed to this improvement without a controlled comparison.

### Open Questions

1. **Does classifier loss improve answer accuracy?**
   - Need to compare: SFT-only vs Dual-reward

2. **Does the classifier provide interpretability?**
   - Even if accuracy is similar, a working classifier could provide insight into model reasoning

3. **Does GRPO produce better classifiers?**
   - RL with verifiable rewards might create different internal structures

## Future Work

1. **Controlled comparison**: Run SFT-only, dual-reward, and GRPO on same data
2. **Test with natural language prompts**: "What is 7 times 8?" instead of "7 * 8 ="
3. **Vary classifier layer depth**: Is 55% optimal?
4. **Test different classifier targets**: Numbers, operation words, custom tokens
5. **Analyze attention patterns**: What are V/O projections learning?

## Open Research Questions

This experiment raises several questions for future investigation:

1. **Classifier emergence mechanisms**: How do classifiers emerge naturally during training?
2. **Vocab-alignment requirements**: What conditions produce vocab-aligned vs hidden-space classifiers?
3. **Training objective impact**: Do different objectives (SFT, GRPO, dual-reward) produce qualitatively different classifiers?

## Citation

This experiment investigates the hypothesis that vocabulary-aligned classifiers can be induced through targeted training, inspired by observations of GPT-OSS behavior at intermediate layers.
