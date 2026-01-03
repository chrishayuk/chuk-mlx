# Early Layer Information Encoding Analysis

## Executive Summary

A key discovery about how transformer models encode arithmetic information: **representations that appear nearly identical (cosine similarity ~0.997) actually contain fully separable information encoded in orthogonal subspaces**.

This resolves an apparent paradox: how can the model distinguish `2*3=` from `2+3=` when their internal representations are 99.7% similar?

---

## The Orthogonal Subspaces Paradox

### Observation

At layer 0 (immediately after the first transformer block):

| Metric | Value |
|--------|-------|
| Cosine similarity between `2*3=` and `2+3=` | 0.998 |
| Cosine similarity between `2*3=` and `2-3=` | 0.999 |
| Operation type classification accuracy | 100% |
| Operand A extraction R² | 0.998 |
| Operand B extraction R² | 1.000 |
| Answer extraction R² | 0.983 |

### The Paradox

How can representations be 99.8% similar yet contain completely different information?

### Resolution

**Information is encoded in orthogonal directions, not as distinct clusters.**

Imagine a 3072-dimensional space (Gemma's hidden dimension). The representations for all arithmetic expressions occupy a tiny subspace—they're all "arithmetic-like" and hence similar. But *within* that subspace, different pieces of information (operation type, operand values, answer) are encoded along orthogonal directions.

```
High-dimensional space visualization:

     Operation direction
           ↑
           |  * * *     (multiplication cluster)
           | + + +      (addition cluster)
           |_ _ _ _ _ → Operand A direction
          /
         /
        ↓
    Answer direction
```

A linear probe finds these orthogonal directions and extracts the information, even though the overall vectors point in nearly the same direction.

---

## Experimental Results

### Part 1: Representation Similarity

Tested on 108 prompts (digits 2-7, operations *, +, -).

**Cross-expression similarity at '=' position:**

| Layer | 2*3= vs 2+3= | 2*3= vs 2-3= | 2+3= vs 2-3= |
|-------|--------------|--------------|--------------|
| L0 | 0.998 | 0.999 | 0.999 |
| L1 | 0.998 | 0.999 | 0.999 |
| L2 | 0.997 | 0.999 | 0.998 |
| L4 | 0.994 | 0.996 | 0.995 |
| L8 | 0.982 | 0.988 | 0.986 |

**Key insight**: Representations remain highly similar through early layers, only gradually differentiating.

### Part 2: Information Extractability

What can a linear probe extract at each layer?

| Layer | Op Acc | A R² | B R² | Answer R² |
|-------|--------|------|------|-----------|
| L0 | 100% | 0.998 | 1.000 | 0.983 |
| L1 | 100% | 1.000 | 1.000 | 0.996 |
| L2 | 100% | 1.000 | 1.000 | 0.999 |
| L4 | 100% | 1.000 | 1.000 | 1.000 |
| L8 | 100% | 1.000 | 1.000 | 1.000 |

**Key insight**: All information is extractable from layer 0. By layer 2, answer extraction is perfect (R² = 0.999).

---

## Implications

### 1. Computation Happens Earlier Than Expected

Traditional interpretability assumes computation flows through layers. But if the answer is extractable at L0 with R² = 0.98, most "computation" may happen in:
- The embedding layer
- The first attention + MLP block (L0)

Later layers may primarily handle:
- Output formatting
- Confidence calibration
- Edge case handling

### 2. The "Lookup Table" is in the Weights

The near-perfect extraction at L0 suggests the model has essentially memorized a lookup table in its weights. The first layer reads the operands and operation, then immediately activates the corresponding answer direction.

This is consistent with findings from:
- Commutativity test: 2×3 and 3×2 have 0.999 similarity (same lookup entry)
- OOD test: Model fails on numbers outside training range (no entry in table)

### 3. Linear Probes Find What Forward Pass Cannot

The forward pass produces a single output token. A linear probe can extract *multiple* pieces of information simultaneously because it searches for specific directions.

This means:
- The model "knows" more than it outputs
- Information is preserved but not necessarily used
- Residual stream is information-rich, not computation-rich

---

## Methodology

### Linear Probe Training

For classification (operation type):
```python
from sklearn.linear_model import LogisticRegression
probe = LogisticRegression(max_iter=1000)
probe.fit(activations, labels)
accuracy = probe.score(activations, labels)
```

For regression (operands, answer):
```python
from sklearn.linear_model import Ridge
probe = Ridge(alpha=1.0)
probe.fit(activations, values)
r_squared = probe.score(activations, values)
```

### Activation Collection

Activations collected at the '=' token position (last token before generation):
```python
# Hook captures hidden states after each layer
activations[layer] = hidden_states[0, -1, :]  # Shape: (hidden_dim,)
```

### Similarity Computation

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## CLI Command

```bash
# Basic analysis
lazarus introspect early-layers -m mlx-community/gemma-3-4b-it-bf16

# Analyze specific layers
lazarus introspect early-layers -m model --layers 0,1,2,4,8,12,16,20

# Include position-wise analysis
lazarus introspect early-layers -m model --analyze-positions

# Test specific operations and digit range
lazarus introspect early-layers -m model --operations "*,+,-,/" --digits 2-9

# Save results
lazarus introspect early-layers -m model --output early_layers.json
```

---

## Connection to Other Findings

| Finding | Connection |
|---------|------------|
| Task type 100% from embeddings | Token identity (*, +, =) encodes task |
| Commutativity 0.999 | Lookup table uses canonical form |
| Late layers dispensable | Computation done by L2 |
| 6-phase architecture | Phases may be redundant/formatting |
| Component redundancy | Multiple neurons encode same info |

---

## Future Directions

1. **Dimensionality analysis**: How many dimensions encode each piece of information?
2. **Causal intervention**: Can we edit the "answer direction" to change output?
3. **Cross-model comparison**: Do GPT, Llama, Qwen use similar encoding?
4. **Training dynamics**: When do these directions form during training?

---

## Conclusion

The "orthogonal subspaces" finding fundamentally changes how we interpret transformer representations:

> **High cosine similarity ≠ Same information**
>
> Information can be encoded in orthogonal directions within a high-similarity manifold.

This explains why:
- Models can distinguish expressions that look "the same" to cosine similarity
- Linear probes are so effective at extracting information
- The residual stream preserves multiple pieces of information simultaneously

The transformer is not computing in the traditional sense—it's activating pre-learned directions that encode the answer, operands, and operation type in orthogonal subspaces from the very first layer.
