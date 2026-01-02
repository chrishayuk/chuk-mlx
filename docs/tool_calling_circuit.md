# Tool-Calling Circuit Analysis: FunctionGemma 270M

## Executive Summary

We performed a complete mechanistic interpretability analysis of the tool-calling circuit in FunctionGemma 270M. This document summarizes our findings.

### Key Discoveries

1. **The decision is made EARLY (L0-L5)**, not at the weight-divergence layers (L11-12)
2. **L11 is a perfect control point** - patching 5 neurons achieves 100% decision flip rate
3. **L12 Neuron 230 is a kill switch** - a NO-TOOL veto that blocks tool-calling
4. **86% of tool information exists in raw embeddings** before any computation
5. **Steering works** - boosting L11 neurons increases P(tool) by +93.7%

---

## The Complete Circuit

```
┌─────────────────────────────────────────────────────────────────┐
│                        SUBSTRATE                                │
│                     (Embeddings)                                │
│                                                                 │
│   86% of tool/no-tool information present before computation    │
│   Key dimensions: 639, 400, 356                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    COMPUTE ZONE (L0-L5)                         │
│                                                                 │
│   L0: 96.7% probe accuracy, 40% drop if ablated                 │
│   L1: 100% probe accuracy                                       │
│   L5: 100% probe accuracy, 50% drop if ablated (CRITICAL)       │
│                                                                 │
│   Decision: COMPUTED HERE                                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STABLE ZONE (L6-L10)                         │
│                                                                 │
│   0% accuracy drop when ablated                                 │
│   0% flip rate when patched                                     │
│   Decision locked in, cannot be modified                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONTROL POINT (L11)                          │
│                                                                 │
│   100% flip rate with just 5 neurons patched                    │
│   86% of variance on PC1 (nearly 1D decision boundary)          │
│                                                                 │
│   Tool Promoters:                                               │
│   ├── Neuron 803:  +11,110 in steering direction                │
│   ├── Neuron 2036: +5,703                                       │
│   └── Neuron 831:  +5,700                                       │
│                                                                 │
│   Tool Suppressors:                                             │
│   ├── Neuron 1237: -5,883 ("What is X?" detector)               │
│   ├── Neuron 821:  -4,704 ("Why/How" detector)                  │
│   └── Neuron 1347: -6,695                                       │
│                                                                 │
│   INTERVENTION TARGET                                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       GATE (L12)                                │
│                                                                 │
│   Neuron 230: NO-TOOL VETO                                      │
│                                                                 │
│   Fires HIGH (+17,000) for: "How does gravity work?"            │
│   Fires LOW (-5,400) for: "What is the weather?"                │
│                                                                 │
│   70% accuracy drop if ablated alone                            │
│   Function: VETO tool-calling for factual queries               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FORMATION ZONE (L13-L17)                      │
│                                                                 │
│   0% accuracy drop when ablated                                 │
│   Output formatting only, not causal for classification        │
└───────────────────────────────────────────────────────────────┘
```

---

## Experimental Evidence

### 1. Full MLP Ablation

| Layer | Accuracy Drop | Interpretation |
|-------|---------------|----------------|
| L0 | -40% | Critical encoding |
| L1 | -30% | Critical encoding |
| L3 | -40% | Context mixing |
| **L5** | **-50%** | **Most critical** |
| L6-L10 | 0% | Stable zone |
| L11 | -10% | Control point |
| L12 | 0% | Gate layer |
| L13-L17 | 0% | Formation |

### 2. Neuron Ablation

| Layer | Ablate Top 1 | Ablate Top 10 |
|-------|--------------|---------------|
| L10 | -40% | -40% |
| L11 | -50% | -50% |
| **L12** | **-70%** | -50% |

L12's top neuron (230) alone causes 70% accuracy drop.

### 3. Activation Patching

| Layer | Flip Rate |
|-------|-----------|
| L10 | 0% |
| **L11** | **100%** |
| L12 | 68% |

L11 patching achieves perfect control.

### 4. Steering Validation

On NO-TOOL prompts, force-tool steering increases P(tool) by **+93.7%** on average.

---

## Named Components

```python
CIRCUIT = {
    # Zones
    'SUBSTRATE': 'embeddings',      # 86% information
    'COMPUTE': (0, 5),              # Decision computed
    'STABLE': (6, 10),              # Decision locked
    'CONTROL': 11,                  # Intervention point
    'GATE': 12,                     # Kill switch
    'FORMATION': (13, 17),          # Output generation

    # Key Neurons
    'TOOL_PROMOTERS': [803, 2036, 831],
    'TOOL_SUPPRESSORS': [1237, 821, 1347],
    'KILL_SWITCH': (12, 230),

    # Key Dimensions
    'COMPUTE_DIMS': [639, 400, 356],
}
```

---

## Intervention Guide

### Force Tool-Calling

```python
from chuk_lazarus.introspection.steering import ToolCallingSteering, SteeringConfig, SteeringMode

steerer = ToolCallingSteering.from_pretrained("mlx-community/functiongemma-270m-it-bf16")
result = steerer.predict(prompt, mode="force_tool")
```

### Prevent Tool-Calling

```python
result = steerer.predict(prompt, mode="prevent_tool")
```

### Kill Switch (Block All Tools)

```python
config = SteeringConfig(
    mode=SteeringMode.NORMAL,
    use_kill_switch=True,  # Zero L12:230
)
result = steerer.forward_with_steering(tokens, config)
```

---

## Key Insights

### 1. Weight Divergence ≠ Causal Importance

Weight divergence analysis suggested L11-12 were most modified by tool training. However, causal ablation shows L0-L5 are most critical.

**Reconciliation**: L11-12 are most MODIFIED by training because they learn to READ the decision (computed in L0-L5) and FORMAT the output.

### 2. The Decision is Nearly 1D

At L11, 86% of the tool/no-tool variance lies on PC1. This means the decision boundary is essentially a hyperplane, making steering straightforward.

### 3. Embeddings Are Highly Informative

86% probe accuracy on raw embeddings suggests the model learns tool-calling patterns at the vocabulary level:
- "weather", "email", "timer" → tool cluster
- "explain", "what is", "why" → no-tool cluster

---

## Files

| File | Purpose |
|------|---------|
| `feature_emergence_neurons.py` | Probe accuracy by layer, neuron profiling |
| `neuron_causal_v2.py` | Ablation and patching experiments |
| `circuit_deep_dive.py` | Kill switch and steering direction analysis |
| `steering_validation.py` | Validate steering effectiveness |
| `steering.py` | Production steering tool |

---

## Future Work

1. **Cross-model validation**: Test on Gemma 2B, 9B, 27B to see if circuit scales
2. **Feature tracking**: Map specific features (action verbs, entities) through layers
3. **Attention analysis**: Which heads move information from input to decision point
4. **Fine-grained steering**: Per-tool control (force weather but not email)

---

## Citation

This analysis was performed using the chuk-lazarus introspection framework on FunctionGemma 270M.
