# Attention-as-Router: Is Routing Universal Across MoE Architectures?

## Research Question

Earlier experiments on GPT-OSS showed:
- 96% of router signal comes from attention output
- Token embedding contributes only 4%
- Context determines routing, not token identity

**Does attention dominate routing universally across MoE architectures?**

## Why This Matters

| If Attention Dominates Universally | If Architectures Differ |
|------------------------------------|-------------------------|
| Universal finding | Architecture-specific finding |
| Router is always redundant | Router matters in some architectures |
| Can eliminate router in all MoE | Need architecture-specific optimization |
| Attention-gated-subspace applies everywhere | Different optimization per type |

## Hypothesis Space

### H1: Attention Dominates Universally
```
All MoE architectures:
  - Router reads attention output
  - Token embedding is minor signal
  - Routing decision made by attention, router just reads it

Criteria: All models show >85% attention contribution
Implication: Router is architectural redundancy in ALL MoE
```

### H2: Some Architectures Use Token Embedding More
```
Some MoE: Attention dominates (96%)
Others: More balanced (50-80% attention)

Criteria: Some architectures show 50-80% attention contribution
Implication: Orthogonal experts need token-level routing
```

### H3: Routing Is Fundamentally Architecture-Dependent
```
Some MoE: Attention dominates (96%)
Others: Token embedding dominates (>50%)

Criteria: Some architectures show <50% attention contribution
Implication: Can't generalize findings across types
            "MoE" is not one architecture, it's many
```

## Method

### Step 1: Decompose Router Input

At each MoE layer, the router receives a hidden state:
```
hidden_L = embed + attention_contributions + previous_mlp_contributions
```

Simplified decomposition:
```
hidden_L ≈ embed + attention_delta

Where:
  embed = original token embedding
  attention_delta = hidden_L - embed (everything prior layers added)
```

### Step 2: Project to Router Space

```python
def decompose_router_signal(model, tokens, layer_idx):
    # Get token embeddings
    embed = model.embed_tokens(tokens)

    # Get hidden state at router input
    hidden = forward_to_layer(model, tokens, layer_idx)

    # Attention contribution
    attention_delta = hidden - embed

    # Get router weights
    router_weights = model.layers[layer_idx].moe.router.weight

    # Project each component to router space
    router_from_embed = embed @ router_weights.T
    router_from_attention = attention_delta @ router_weights.T

    # Measure magnitudes
    embed_contribution = norm(router_from_embed)
    attention_contribution = norm(router_from_attention)

    # Ratio
    attention_ratio = attention_contribution / (embed_contribution + attention_contribution)

    return attention_ratio
```

### Step 3: Test Context Sensitivity

Does the same token route to different experts based on context?

```python
contexts = [
    ("111 127", "127"),   # Number after numbers
    ("abc 127", "127"),   # Number after letters
    ("The number 127", "127"),  # Number after words
    ("= 127", "127"),     # Number after operator
]

# If same token -> same expert: Token embedding dominates
# If same token -> different experts: Context (attention) dominates
```

## Running the Experiment

```bash
# Run on OLMoE (True MoE)
lazarus experiment run attention_router

# Run with custom models
lazarus experiment run attention_router \
    --parameter models='["allenai/OLMoE-1B-7B-0924"]'

# Run with specific layers
lazarus experiment run attention_router \
    --parameter layers='[4, 8, 12]'
```

## Expected Results

### If H1 Confirmed (Attention Universal) ✓ ACTUAL RESULT

```
Model               | Attention Ratio | Context Sensitive
--------------------|-----------------|------------------
GPT-OSS             | 96%             | Yes
OLMoE               | 89-98%          | Yes (78%)
```

### If H2 Confirmed (Some More Balanced)

```
Model               | Attention Ratio | Context Sensitive
--------------------|-----------------|------------------
GPT-OSS             | 96%             | Yes
OLMoE               | 50-80%          | Partial
```

### If H3 Confirmed (Architecture-Dependent)

```
Model               | Attention Ratio | Context Sensitive
--------------------|-----------------|------------------
GPT-OSS             | 96%             | Yes
OLMoE               | <50%            | No
```

## Output Files

Results are saved to `results/` directory:
- `results_YYYYMMDD_HHMMSS.json`: Full experiment results
- Contains decomposition data, context sensitivity tests, and hypothesis evaluation

## Connection to Other Findings

| Finding | If H1 (Confirmed) | If H2/H3 |
|---------|-------------------|----------|
| Attention dominance | Universal across MoE | Architecture-specific |
| Cold experts | May apply to both | May differ by type |
| Workhorse concentration | Test on both architectures | May differ |

**Result**: H1 was confirmed. Attention dominates routing universally (89-98% at middle/late layers).

## References

- GPT-OSS MoE analysis (previous experiments)
- OLMoE: Open-Source MoE from Allen AI
- Attention-gated-subspace hypothesis
