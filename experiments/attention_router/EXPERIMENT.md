# Attention-as-Router: True MoE vs Pseudo-MoE

## Research Question

Earlier experiments on GPT-OSS (Pseudo-MoE) showed:
- 96% of router signal comes from attention output
- Token embedding contributes only 4%
- Context determines routing, not token identity

**Does attention dominate routing in True MoE (OLMoE) as well?**

## Why This Matters

| If Attention Dominates Both | If True MoE Is Different |
|-----------------------------|--------------------------|
| Universal finding | Architecture-specific finding |
| Router is always redundant | Router matters in True MoE |
| Can eliminate router in all MoE | Only simplify Pseudo-MoE |
| Attention-gated-subspace applies everywhere | Different optimization per type |

## Hypothesis Space

### H1: Attention Dominates Universally
```
Both True MoE and Pseudo-MoE:
  - Router reads attention output
  - Token embedding is minor signal
  - Routing decision made by attention, router just reads it

Criteria: All models show >85% attention contribution
Implication: Router is architectural redundancy in ALL MoE
```

### H2: True MoE Uses Token Embedding More
```
Pseudo-MoE: Attention dominates (96%)
True MoE: More balanced (50-80% attention)

Criteria: True MoE shows 50-80% attention contribution
Implication: Orthogonal experts need token-level routing
            Pseudo-MoE can simplify; True MoE router has purpose
```

### H3: True MoE Routing Is Fundamentally Different
```
Pseudo-MoE: Attention dominates (96%)
True MoE: Token embedding dominates (>50%)

Criteria: True MoE shows <50% attention contribution
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

### If H1 Confirmed (Attention Universal)

```
Model               | Attention Ratio | Context Sensitive
--------------------|-----------------|------------------
GPT-OSS (Pseudo)    | 96%             | Yes
OLMoE (True)        | 90%+            | Yes
```

### If H2 Confirmed (True MoE More Balanced)

```
Model               | Attention Ratio | Context Sensitive
--------------------|-----------------|------------------
GPT-OSS (Pseudo)    | 96%             | Yes
OLMoE (True)        | 50-80%          | Partial
```

### If H3 Confirmed (Fundamentally Different)

```
Model               | Attention Ratio | Context Sensitive
--------------------|-----------------|------------------
GPT-OSS (Pseudo)    | 96%             | Yes
OLMoE (True)        | <50%            | No
```

## Output Files

Results are saved to `results/` directory:
- `results_YYYYMMDD_HHMMSS.json`: Full experiment results
- Contains decomposition data, context sensitivity tests, and hypothesis evaluation

## Connection to Other Findings

| Finding | If H1 | If H2/H3 |
|---------|-------|----------|
| Gate rank 1 (GPT-OSS) | Pseudo-MoE specific | Pseudo-MoE specific |
| 56% cold experts | May apply to both | May differ by type |
| 6x compression | Pseudo-MoE only | Pseudo-MoE only |
| Workhorse concentration | Test on True MoE | May differ |

## References

- GPT-OSS MoE analysis (previous experiments)
- OLMoE: Open-Source MoE from Allen AI
- Attention-gated-subspace hypothesis
