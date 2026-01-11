# lazarus introspect commutativity

Test whether the model's internal representations respect commutativity (A*B = B*A).

## Synopsis

```bash
lazarus introspect commutativity -m MODEL [OPTIONS]
```

## Description

The `commutativity` command tests whether the model represents commutative pairs (like 3*7 and 7*3) with similar internal representations. This reveals:

1. **Algorithmic vs lookup** - High similarity suggests memorized facts (lookup table)
2. **Compositional structure** - Low similarity suggests actual computation
3. **Layer-specific patterns** - Where does commutativity emerge?

A model using pure memorization would show near-identical representations for A*B and B*A, while an algorithmic model might represent them differently despite producing the same answer.

## Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | HuggingFace model ID or local path (required) |
| `-l, --layer N` | Layer to analyze (default: auto-select ~50% depth) |
| `--pairs PAIRS` | Explicit pairs to test (format: "2*3,3*2\|7*8,8*7") |
| `-o, --output FILE` | Save results to JSON file |

## Examples

### Basic Commutativity Test

Test default multiplication pairs:

```bash
lazarus introspect commutativity -m openai/gpt-oss-20b
```

### Specific Layer Analysis

Focus on a particular layer:

```bash
lazarus introspect commutativity \
    -m model \
    -l 15
```

### Custom Pairs

Test specific commutative pairs:

```bash
lazarus introspect commutativity \
    -m model \
    --pairs "2*7,7*2|3*8,8*3|4*9,9*4|5*6,6*5"
```

### Save Results

```bash
lazarus introspect commutativity \
    -m model \
    -o commutativity_results.json
```

## Output

### Pair-by-Pair Results

```
Analyzing at layer 12
Testing 8 commutative pairs

Pair A       Pair B       Cosine Sim
----------------------------------------
2*7=         7*2=         0.998234
3*8=         8*3=         0.997891
4*9=         9*4=         0.996543
5*6=         6*5=         0.999012
6*7=         7*6=         0.998765
7*8=         8*7=         0.997234
8*9=         9*8=         0.996789
3*9=         9*3=         0.998456
```

### Summary Statistics

```
==================================================
COMMUTATIVITY ANALYSIS
==================================================
Mean similarity: 0.997866
Std similarity:  0.000892
Min similarity:  0.996543
Max similarity:  0.999012

[VERY HIGH] Representations are nearly identical for
commutative pairs. This suggests a lookup table structure
rather than algorithmic computation.
```

## Interpretation Levels

| Similarity | Level | Interpretation |
|------------|-------|----------------|
| > 0.99 | VERY HIGH | Lookup table / memorized facts |
| 0.95 - 0.99 | HIGH | Mostly memorized with some structure |
| 0.80 - 0.95 | MODERATE | Mix of memorization and computation |
| < 0.80 | LOW | Likely algorithmic computation |

## Use Cases

### Model Comparison

Compare memorization vs computation across models:

```bash
for model in gpt2 llama gemma; do
    echo "=== $model ==="
    lazarus introspect commutativity -m $model
done
```

### Layer-by-Layer Analysis

Find where commutativity emerges:

```bash
for layer in 0 4 8 12 16 20 24; do
    echo "=== Layer $layer ==="
    lazarus introspect commutativity -m model -l $layer
done
```

### Operation Comparison

Test commutativity for different operations:

```bash
# Multiplication (should be commutative)
lazarus introspect commutativity -m model \
    --pairs "3*7,7*3|4*8,8*4"

# Addition (also commutative)
lazarus introspect commutativity -m model \
    --pairs "3+7,7+3|4+8,8+4"

# Subtraction (NOT commutative - expect low similarity)
lazarus introspect commutativity -m model \
    --pairs "7-3,3-7|8-4,4-8"
```

## Theoretical Background

### Lookup Table Hypothesis

If a model stores multiplication facts as a lookup table:
- `3*7` and `7*3` would map to the same memory location
- Internal representations should be nearly identical
- Cosine similarity > 0.99

### Algorithmic Hypothesis

If a model computes multiplication algorithmically:
- `3*7` processes 3 rows of 7
- `7*3` processes 7 rows of 3
- Different intermediate representations
- Lower cosine similarity

### Empirical Findings

Most small models (< 3B parameters) show:
- Very high commutativity similarity (> 0.99)
- Consistent across layers
- Suggests predominantly lookup-based retrieval

## Saved Output Format

```json
{
  "model_id": "openai/gpt-oss-20b",
  "layer": 12,
  "num_pairs": 8,
  "mean_similarity": 0.997866,
  "std_similarity": 0.000892,
  "min_similarity": 0.996543,
  "max_similarity": 0.999012,
  "level": "very_high",
  "interpretation": "Representations are nearly identical...",
  "pairs": [
    {
      "prompt_a": "2*7=",
      "prompt_b": "7*2=",
      "similarity": 0.998234
    }
  ]
}
```

## See Also

- [introspect patch](introspect-patch.md) - Activation patching experiments
- [introspect arithmetic](introspect-arithmetic.md) - Systematic arithmetic testing
- [introspect operand-directions](introspect-operand-directions.md) - Operand encoding analysis
- [Introspection Overview](../introspection.md) - Full module documentation
