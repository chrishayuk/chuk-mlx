# lazarus introspect operand-directions

Analyze how operands are encoded in activation space (A_d and B_d directions).

## Synopsis

```bash
lazarus introspect operand-directions -m MODEL [OPTIONS]
```

## Description

The `operand-directions` command extracts and analyzes operand directions to understand how the model encodes the first operand (A) and second operand (B) in arithmetic expressions.

Key questions answered:
- Are A and B encoded in orthogonal subspaces?
- Does digit identity dominate position (A=3 similar to B=3)?
- Is encoding compositional or holistic?

## Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | HuggingFace model ID or local path (required) |
| `-l, --layers LAYERS` | Layers to analyze (comma-separated, default: 25%,50%,60%,75%) |
| `--operation OP` | Operation symbol: *, +, - (default: *) |
| `--digits RANGE` | Digit range (e.g., "2-9" or "2,3,5,7") |
| `-o, --output FILE` | Save results to JSON file |

## Examples

### Basic Operand Analysis

```bash
lazarus introspect operand-directions -m openai/gpt-oss-20b
```

### Specific Layers

```bash
lazarus introspect operand-directions \
    -m model \
    --layers 4,8,12,16,20
```

### Addition Operation

```bash
lazarus introspect operand-directions \
    -m model \
    --operation "+" \
    --digits 1-9
```

### Custom Digit Range

```bash
lazarus introspect operand-directions \
    -m model \
    --digits 2,3,5,7 \
    -o prime_operands.json
```

## Output

### Per-Layer Analysis

```
======================================================================
LAYER 12
======================================================================

Extracting A_d directions (B fixed at 5)...
Extracting B_d directions (A fixed at 5)...

--- Orthogonality Analysis ---
A_i vs A_j (diff first operands): 0.234 ± 0.089
B_i vs B_j (diff second operands): 0.256 ± 0.092
A_i vs B_j (cross A/B, diff digits): 0.123 ± 0.067
A_i vs B_i (same digit, diff role): 0.789 ± 0.045

--- Interpretation ---
Distinct operand directions (compositional encoding)
A and B subspaces are orthogonal
```

### Summary Across Layers

```
======================================================================
SUMMARY ACROSS LAYERS
======================================================================
Layer    A vs A       B vs B       A vs B (cross)  A vs B (same)
--------------------------------------------------------------------
L6       0.234        0.256        0.123           0.789
L12      0.198        0.212        0.098           0.823
L18      0.156        0.178        0.067           0.856
L24      0.134        0.145        0.045           0.878
```

## Interpretation Guide

### Similarity Metrics

| Metric | Meaning | Ideal for Compositional |
|--------|---------|------------------------|
| A_i vs A_j | Different first operands | Low (<0.5) |
| B_i vs B_j | Different second operands | Low (<0.5) |
| A_i vs B_j | Cross position, diff digits | Low (<0.3) |
| A_i vs B_i | Same digit, diff position | Variable |

### Encoding Types

**Compositional Encoding** (good for generalization):
- Low A vs A and B vs B (distinct operand representations)
- Low A vs B cross (orthogonal position subspaces)
- Model can recombine operands flexibly

**Holistic Encoding** (memorization-like):
- High A vs A and B vs B (operands look similar)
- High overlap between A and B spaces
- Model treats each expression as unique fact

**Digit-Dominant Encoding**:
- High A_i vs B_i (same digit similar regardless of position)
- Digit identity matters more than position
- May struggle with order-dependent operations

## Use Cases

### Understanding Arithmetic Structure

```bash
# Compare multiplication vs addition encoding
lazarus introspect operand-directions \
    -m model \
    --operation "*" \
    -o mult_operands.json

lazarus introspect operand-directions \
    -m model \
    --operation "+" \
    -o add_operands.json

# Multiplication may show more holistic encoding
# Addition may show more compositional encoding
```

### Layer-by-Layer Evolution

```bash
# Track how encoding develops through layers
lazarus introspect operand-directions \
    -m model \
    --layers 0,2,4,6,8,10,12,14,16,18,20,22,24 \
    -o encoding_evolution.json
```

### Model Comparison

```bash
# Compare encoding structure across model sizes
for size in 1b 3b 7b 20b; do
    lazarus introspect operand-directions \
        -m model-${size} \
        -o operand_${size}.json
done
```

## Theoretical Background

### The Operand Subspace Hypothesis

If arithmetic is compositional, we expect:
1. **Separate A and B subspaces**: A_d ⊥ B_d
2. **Distinct digit directions**: A_3 ≠ A_7
3. **Position-invariant digits**: A_3 similar to B_3 (same digit)

### What the Metrics Reveal

**Low A_i vs A_j** (< 0.5):
- Different first operands have distinct representations
- Model can distinguish 3*5 from 7*5

**Low A_i vs B_j cross** (< 0.3):
- A and B live in orthogonal subspaces
- Position information is preserved

**High A_i vs B_i same** (> 0.7):
- Digit identity shared across positions
- "3" looks similar whether first or second operand

### Implications for Intervention

Understanding operand encoding helps with:
- Targeted activation patching
- Steering specific operand representations
- Predicting generalization patterns

## Saved Output Format

```json
{
  "model": "openai/gpt-oss-20b",
  "operation": "*",
  "digits": [2, 3, 4, 5, 6, 7, 8, 9],
  "layers": [6, 12, 18, 24],
  "results_by_layer": {
    "12": {
      "a_vs_a_mean": 0.234,
      "a_vs_a_std": 0.089,
      "b_vs_b_mean": 0.256,
      "b_vs_b_std": 0.092,
      "a_vs_b_cross_mean": 0.123,
      "a_vs_b_cross_std": 0.067,
      "a_vs_b_same_mean": 0.789,
      "a_vs_b_same_std": 0.045
    }
  }
}
```

## See Also

- [introspect directions](introspect-directions.md) - Compare multiple direction vectors
- [introspect neurons](introspect-neurons.md) - Analyze individual neurons
- [introspect commutativity](introspect-commutativity.md) - Test A*B = B*A
- [introspect early-layers](introspect-early-layers.md) - Information extractability
- [Introspection Overview](../introspection.md) - Full module documentation
