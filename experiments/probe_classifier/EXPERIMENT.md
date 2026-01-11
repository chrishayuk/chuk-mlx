# Probe Classifier Experiment: Task Information at Intermediate Layers

## Research Question

**Is task information encoded at intermediate layers, even if not vocabulary-aligned?**

Previous experiments showed:
- Vocabulary-aligned classifiers (logit lens) show ~75% confidence at L14-L15
- Dual-reward training to create vocabulary classifiers HURTS accuracy
- But does the model "know" the task type earlier, just in a different representation?

## Results Summary (January 10, 2026)

### Critical Finding: 100% Probe Accuracy from L4 Onwards

#### Symbolic Input (`7 * 8 =`)

| Layer | Depth | Test Accuracy |
|-------|-------|---------------|
| L4 | 25% | **100%** |
| L5 | 35% | **100%** |
| L7 | 45% | **100%** |
| L8 | 55% | **100%** |
| L10 | 65% | **100%** |
| L12 | 75% | **100%** |
| L13 | 85% | **100%** |
| L15 | 95% | **100%** |

#### Semantic Input (`seven times eight`)

| Layer | Depth | Test Accuracy |
|-------|-------|---------------|
| L4 | 25% | **100%** |
| L8 | 55% | **100%** |
| L12 | 75% | **100%** |
| L15 | 95% | **100%** |

#### TinyLlama-1.1B (22 layers)

| Layer | Depth | Test Accuracy |
|-------|-------|---------------|
| L3 | 15% | **100%** |
| L5 | 25% | **100%** |
| L7 | 35% | **100%** |
| L9 | 45% | **100%** |
| L12 | 55% | **100%** |
| L14 | 65% | **100%** |
| L16 | 75% | **100%** |
| L18 | 85% | **100%** |
| L20 | 95% | **100%** |

**A simple linear probe achieves perfect task classification at every layer tested, across BOTH models and input formats.**

## Methodology

### Linear Probe

```python
class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        self.linear = nn.Linear(input_dim, num_classes)

    def __call__(self, x):
        return self.linear(x)
```

The probe is a single linear layer:
- Input: Hidden state at layer L (2048 dims for Llama-3.2-1B)
- Output: 3-class logits (multiply, add, subtract)
- Training: 100 epochs, Adam optimizer, cross-entropy loss

### Dataset

- 2000 arithmetic samples (balanced across operations)
- 80/20 train/test split
- Format: `"7 * 8 = "` with task label

### Model

- **Llama-3.2-1B** (16 transformer layers)
- Hidden dimension: 2048

## Analysis

### 1. Task Information Emerges Early

Task classification is perfect from L4 (25% depth). This means:
- The model "knows" the operation type after just 4 layers
- This information persists through all subsequent layers
- No additional training is needed to extract it

### 2. Vocabulary Alignment is Unnecessary

Logit lens (vocabulary projection) showed ~75% classifier confidence at L14-L15. But probing shows 100% accuracy. The difference:

| Method | L4 Accuracy | L15 Accuracy |
|--------|-------------|--------------|
| Logit Lens (vocab-aligned) | ~10% | ~75% |
| Linear Probe | **100%** | **100%** |

The hidden states encode task info in a **non-vocabulary-aligned** representation that a linear probe can extract.

### 3. Why Dual-Reward Failed

Dual-reward training tried to force vocabulary alignment:
```
hidden_state → embed_weight.T → logits → "multiply" token
```

This failed because:
1. Task info already exists in a different subspace
2. Forcing vocabulary alignment distorts the representation
3. The "classifier" objective conflicts with the existing encoding

### 4. Routing is Viable via Learned Projections

Instead of vocabulary lookup, use learned routing:
```
hidden_state → W_route → task_weights
```

Where `W_route` is a learned [num_tasks, hidden_dim] matrix, similar to the probe.

## Implications for Virtual Expert Architecture

### Original Approach (Vocabulary Classifiers)
```
Forward pass:
  1. Get hidden state at L8
  2. Project to vocabulary: h @ embed.T
  3. Read "multiply"/"add" token probabilities
  4. Route to corresponding expert

Problem: Vocabulary tokens may not align with task concepts
```

### New Approach (Learned Routing)
```
Forward pass:
  1. Get hidden state at L4 (or any layer after L4)
  2. Apply learned routing: h @ W_route.T
  3. Softmax to get expert weights
  4. Route to corresponding expert

Advantage: Routing matrix learns the actual task subspace
```

### Training the Routing Matrix

Option 1: **Probe-style supervised learning**
```python
# Extract hidden states for labeled data
hidden_states = get_layer_outputs(model, prompts, layer=4)
task_labels = get_task_labels(prompts)

# Train routing matrix
W_route = train_linear_classifier(hidden_states, task_labels)
```

Option 2: **End-to-end with task loss**
```python
# Route based on hidden state
weights = softmax(hidden @ W_route.T)
# Combine expert outputs
output = sum(weights[i] * experts[i](hidden) for i in range(num_experts))
# Train with task-specific loss
loss = task_loss(output, target)
```

## Comparison with GPT-OSS

GPT-OSS shows vocabulary-aligned classifiers at L13 (54% depth). Our findings suggest:

1. **Vocabulary alignment may be emergent, not required** - The task info exists earlier in a different form

2. **Scale matters** - GPT-OSS (20B) may develop vocabulary alignment naturally; smaller models may not

3. **Routing can work without vocabulary** - A learned projection is sufficient for task routing

## Conclusions

1. **Task information is encoded early** - Perfect classification from L4 (25% depth)

2. **Vocabulary alignment is not needed** - Linear probes extract task info that logit lens misses

3. **Dual-reward was solving the wrong problem** - Forcing vocabulary alignment when task info already exists

4. **Routing should use learned projections** - Not vocabulary lookup

5. **Virtual expert architecture is viable** - Route at L4+ using trained routing matrices

## Cross-Experiment Summary

| Experiment | Question | Answer |
|------------|----------|--------|
| classifier_emergence | SFT or dual-reward? | SFT (100% accuracy) |
| semantic_classifier | Do classifiers help? | No - they hurt (33% accuracy) |
| two_stage_classifier | Can we preserve computation? | Yes, with low LR |
| **probe_classifier** | Is task info encoded? | **YES - 100% at L4** |
| cot_vocab_alignment | Does CoT create vocab alignment? | No (0% at all layers) |

### The Complete Picture

```
LLAMA-3.2-1B (16 layers):
  Task info at L4 (25%):  YES (100% linear probe)
  Vocabulary-aligned:     NO  (0% logit lens)

TINYLLAMA-1.1B (22 layers):
  Task info at L3 (15%):  YES (100% linear probe)
  Vocabulary-aligned:     NO  (0% logit lens)

GPT-OSS-20B:
  Task info at L13:       YES
  Vocabulary-aligned:     YES (30-50% logit lens)

FINDING:
  Task info emerges VERY EARLY (15-25% depth) across architectures.
  Vocabulary alignment is NOT present in 1B models.
  Scale (20B) or MoE may create vocabulary alignment.
```

### Practical Implication

**Use learned routing projections, not vocabulary lookup:**

```python
# BAD: Vocabulary lookup (doesn't work on 1B)
task_prob = softmax(hidden @ embed.T)["multiply"]

# GOOD: Learned projection (works on all scales)
task_weights = softmax(hidden @ W_route.T)
```

## Future Work

1. ~~**Test on semantic input**~~ - ✅ DONE: 100% accuracy on semantic too!

2. ~~**Test if CoT creates vocab alignment**~~ - ✅ DONE: NO, it doesn't

3. **Multi-layer routing** - Does routing at different layers give different behavior?

4. **End-to-end training** - Train routing matrix jointly with expert adapters

5. **Cross-task generalization** - Does a routing matrix trained on arithmetic transfer to other tasks?

6. **GPT-OSS causality test** - Is L13 vocab classifier causal or epiphenomenal?

## Virtual Expert Router Implementation

Based on these findings, here's the practical routing architecture:

```python
class VirtualExpertRouter:
    """
    Route using learned projections at L4, not vocabulary lookup.
    Works on any model size because task info is non-vocab-aligned.
    """

    def __init__(self, model, routing_layer: int = 4):
        self.model = model
        self.layer = routing_layer
        self.W_route = None  # [num_tasks, hidden_dim]

    def train(self, examples: list[tuple[str, str]]):
        """Train routing projection from (prompt, task_label) pairs."""
        hiddens = []
        labels = []

        for prompt, task in examples:
            h = self.get_layer_output(prompt, self.layer)
            hiddens.append(h)
            labels.append(task)

        # Train linear classifier (logistic regression or similar)
        self.W_route = fit_linear_classifier(hiddens, labels)

    def route(self, prompt: str) -> dict[str, float]:
        """Get task weights from hidden state."""
        h = self.get_layer_output(prompt, self.layer)
        logits = h @ self.W_route.T
        return softmax(logits)

    def execute_with_experts(self, prompt: str, experts: dict):
        """Route to appropriate expert based on task."""
        weights = self.route(prompt)
        task = max(weights, key=weights.get)
        return experts[task](prompt)
```

This works because:
1. **Task info at L4**: 100% probe accuracy (this experiment)
2. **No vocab alignment needed**: Learned projection reads the task subspace
3. **Works on both formats**: Symbolic and semantic input both work
4. **Doesn't break computation**: Unlike dual-reward training

## Files

```
probe_classifier/
├── EXPERIMENT.md       # This file
├── README.md           # Quick start
├── experiment.py       # Implementation
├── config.yaml         # Configuration
├── data/               # Train/test data
│   ├── train.jsonl
│   └── test.jsonl
└── results/            # Run results (JSON)
```

## Running

```bash
lazarus experiment run probe_classifier
```

## Key Takeaway

**Don't force vocabulary alignment. The model already knows the task - just learn to read it.**
