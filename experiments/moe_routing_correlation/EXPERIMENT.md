# MoE Routing Correlation: Does MoE Architecture Force Vocabulary Alignment?

## Research Question

**Does the MoE routing mechanism create pressure for vocabulary-aligned task representations to emerge at intermediate layers?**

GPT-OSS reportedly shows vocabulary classifiers at L13 (~54% depth) where operation tokens ("multiply", "add") appear with 50-80% probability via logit lens. Dense models like Llama-3.2-1B show ~0%.

**Hypothesis**: MoE routing requires discrete decisions. Unlike dense models where task information can exist in arbitrary subspaces, MoE models must make explicit routing choices. This architectural pressure may force vocabulary-aligned representations to emerge naturally - no special training objective needed.

## Results Summary (January 11, 2026)

### Vocabulary Alignment (Logit Lens)

| Model | Type | L4 | L6 | L8 | L10 | L12 | L14 |
|-------|------|-----|-----|-----|------|------|------|
| **OLMoE-1B-7B** | MoE (64 experts) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Llama-3.2-1B** | Dense | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

**Neither model shows vocabulary-aligned classifiers at any intermediate layer.**

### Linear Probe (Learned Subspace)

| Model | Type | L4 (25%) | L8 (50%) |
|-------|------|----------|----------|
| **OLMoE-1B-7B** | MoE (64 experts) | **100%** | **100%** |
| **Llama-3.2-1B** | Dense | **100%** | **100%** |

**Both models encode task information perfectly in a learned subspace.**

### Task Accuracy (Answer Generation)

| Model | Accuracy | Notes |
|-------|----------|-------|
| **OLMoE-1B-7B** | 66.7% | Some arithmetic errors |
| **Llama-3.2-1B** | 100% | All correct |

## Conclusion

### Hypothesis: REJECTED

**MoE architecture alone does NOT create vocabulary-aligned classifiers.**

Both OLMoE (MoE) and Llama (dense) show:
- 0% vocabulary alignment at all intermediate layers
- 100% probe accuracy at L4 onwards

The MoE routing mechanism does not force vocabulary alignment. Task information exists in a learned subspace in both architectures.

## Implications

### 1. GPT-OSS's L13 Classifiers Are Not From MoE Architecture

If MoE architecture forced vocabulary alignment, OLMoE would show it. It doesn't. GPT-OSS's vocabulary classifiers must come from:
- **Explicit training objective** - A loss term that rewards vocabulary alignment
- **Scale effects** - 20B parameters may exhibit emergent properties 1B doesn't
- **Different pretraining data** - Training data composition matters
- **Unknown architectural differences** - Something beyond standard MoE

### 2. Task Information Encoding Is Architecture-Agnostic

Both MoE and dense models encode task information the same way:
- Early layers (L4, 25% depth): 100% probe accuracy
- Non-vocabulary-aligned: 0% logit lens
- Linear probe extracts it perfectly

The "what operation is this?" question is answered identically by both architectures.

### 3. Routing Should Use Learned Projections

For virtual expert routing:

```python
# DON'T: Vocabulary lookup (doesn't work)
task_prob = softmax(hidden @ embed.T)["multiply"]  # 0% on both architectures

# DO: Learned projection (works on both)
task_weights = softmax(hidden @ W_route.T)  # 100% on both architectures
```

## Methodology

### Models Tested

| Model | Architecture | Experts | Active | Params | Layers |
|-------|-------------|---------|--------|--------|--------|
| OLMoE-1B-7B-0924 | MoE | 64 | 8 | 7B total, 1B active | 16 |
| Llama-3.2-1B | Dense | N/A | N/A | 1.2B | 16 |

### Test Prompts

```yaml
addition:
  - "5 + 3 = "
  - "12 + 7 = "
  - "45 + 23 = "
subtraction:
  - "10 - 4 = "
  - "25 - 8 = "
  - "100 - 37 = "
multiplication:
  - "6 * 7 = "
  - "8 * 9 = "
  - "12 * 11 = "
division:
  - "20 / 4 = "
  - "36 / 6 = "
  - "100 / 5 = "
```

### Logit Lens Method

```python
# At each layer, project hidden state to vocabulary
h_normed = layer_norm(hidden_state)
logits = h_normed @ embed_weight.T
probs = softmax(logits)

# Check probability of task tokens
task_prob = max(probs["add"], probs["plus"], probs["addition"], ...)
```

### Linear Probe Method

```python
# Train simple linear classifier on hidden states
probe = Linear(hidden_dim, num_classes=3)  # multiply, add, subtract
probe.train(hidden_states, task_labels, epochs=50)
accuracy = probe.evaluate(test_hidden, test_labels)
```

## Detailed Results

### OLMoE-1B-7B (MoE Model)

```
Layer-by-layer vocabulary alignment:
  L4:   0.0%
  L6:   0.0%
  L8:   0.0%
  L10:  0.0%
  L12:  0.0%
  L14:  0.0%

Linear probe accuracy:
  L4 (25%):  100%
  L8 (50%):  100%

Task accuracy: 66.7% (8/12 correct)
```

### Llama-3.2-1B (Dense Model)

```
Layer-by-layer vocabulary alignment:
  L4:   0.0%
  L6:   0.0%
  L8:   0.0%
  L10:  0.0%
  L12:  0.0%
  L14:  0.0%

Linear probe accuracy:
  L4 (25%):  100%
  L8 (50%):  100%

Task accuracy: 100% (12/12 correct)
```

### Comparison

```
                    OLMoE (MoE)    Llama (Dense)    Delta
Vocab alignment:       0.0%           0.0%          0.0%
Probe accuracy L4:   100.0%         100.0%          0.0%
Probe accuracy L8:   100.0%         100.0%          0.0%
Task accuracy:        66.7%         100.0%        -33.3%
```

## Analysis

### Why OLMoE Has Lower Task Accuracy

OLMoE shows 66.7% accuracy vs Llama's 100%. This is NOT because of vocabulary alignment. Possible reasons:

1. **Training data focus**: OLMoE may be trained on more diverse text, less pure arithmetic
2. **Active parameter count**: 1B active vs 1.2B total
3. **Expert routing noise**: For simple symbolic tasks, routing may add unnecessary complexity

### The Vocabulary Alignment Mystery

If neither MoE nor standard pretraining creates vocabulary alignment, how does GPT-OSS get it?

| Hypothesis | Evidence For | Evidence Against |
|------------|-------------|------------------|
| Scale (20B) | GPT-OSS is 20B | Would need to test 20B MoE |
| Explicit training | Would explain consistency | Not documented |
| MoE architecture | Has routing decisions | OLMoE doesn't show it |
| Emergent property | Larger models have more | No clear threshold |

### Cross-Experiment Summary

| Experiment | Question | Answer |
|------------|----------|--------|
| classifier_emergence | SFT or dual-reward? | SFT (100% accuracy) |
| semantic_classifier | Do explicit classifiers help? | No - they hurt (33%) |
| probe_classifier | Is task info encoded? | YES - 100% at L4 |
| cot_vocab_alignment | Does CoT create vocab alignment? | No (0% at all layers) |
| cot_correlation | Does GPT-OSS HF checkpoint have L13 classifiers? | No (~0%) |
| **moe_routing_correlation** | Does MoE architecture create vocab alignment? | **No (0% for both)** |

## The Complete Picture

```
TASK INFORMATION ENCODING:

┌─────────────────────────────────────────────────────────────┐
│                    Where is task info?                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   OLMoE (MoE):     [===TASK INFO===]───────────────────────▶│
│                    L4 (25%)        100% probe, 0% vocab      │
│                                                              │
│   Llama (Dense):   [===TASK INFO===]───────────────────────▶│
│                    L4 (25%)        100% probe, 0% vocab      │
│                                                              │
│   GPT-OSS (20B):   [===TASK INFO===]──[VOCAB ALIGNED]──────▶│
│                    Early?          L13 (54%)                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘

FINDING:
  - Task info emerges early (L4) in ALL architectures
  - Vocabulary alignment is NOT from MoE architecture
  - Vocabulary alignment is NOT from CoT training
  - GPT-OSS vocabulary alignment source: UNKNOWN
```

## Practical Recommendations

### For Virtual Expert Routing

```python
class VirtualExpertRouter:
    """
    Works on BOTH MoE and dense architectures.
    Uses learned projections, not vocabulary lookup.
    """

    def __init__(self, model, routing_layer: int = 4):
        self.model = model
        self.layer = routing_layer
        self.W_route = None  # [num_tasks, hidden_dim]

    def train(self, examples: list[tuple[str, str]]):
        """Train from (prompt, task_label) pairs."""
        hiddens = self.extract_hidden_states(examples, self.layer)
        labels = [task for _, task in examples]
        self.W_route = fit_linear_classifier(hiddens, labels)

    def route(self, prompt: str) -> dict[str, float]:
        """Get task weights from hidden state."""
        h = self.get_layer_output(prompt, self.layer)
        logits = h @ self.W_route.T
        return softmax(logits)
```

### Why This Works

1. **Task info exists at L4**: 100% probe accuracy on both architectures
2. **Architecture-agnostic**: Same approach works for MoE and dense
3. **No vocabulary dependency**: Learned projection reads the actual task subspace
4. **Doesn't require GPT-OSS's mystery training**: Works with standard pretrained models

## Files

```
moe_routing_correlation/
├── EXPERIMENT.md       # This file
├── README.md           # Quick start guide
├── experiment.py       # ExperimentBase implementation
├── config.yaml         # Configuration
└── results/            # Run results (JSON)
    └── run_20260111_*.json
```

## Running

```bash
# Via framework
lazarus experiment run moe_routing_correlation

# Direct
python experiments/moe_routing_correlation/experiment.py
```

## Model Requirements

- **OLMoE-1B-7B**: ~14GB download (now supported in framework)
- **Llama-3.2-1B**: ~2.5GB download

## Key Takeaway

**MoE architecture is not the source of vocabulary-aligned classifiers. Use learned routing projections - they work on any architecture.**
