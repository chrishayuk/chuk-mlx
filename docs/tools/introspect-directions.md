# lazarus introspect directions

Compare multiple direction vectors for orthogonality to verify independent features.

## Synopsis

```bash
lazarus introspect directions FILE1.npz FILE2.npz [FILE3.npz ...] [OPTIONS]
```

## Description

The `directions` command compares saved direction vectors to check if they represent independent features in activation space. This is crucial for validating mechanistic interpretability findings:

1. **Orthogonal directions** (cosine ~ 0) indicate truly independent features
2. **Correlated directions** may represent overlapping or redundant concepts
3. **Aligned directions** suggest the same underlying feature with different labels

## Options

| Option | Description |
|--------|-------------|
| `FILE1.npz FILE2.npz ...` | Direction files to compare (from `introspect probe --save-direction`) |
| `--threshold FLOAT` | Cosine similarity threshold for "orthogonal" (default: 0.1) |
| `-o, --output FILE` | Save similarity matrix to JSON file |

## Examples

### Compare Two Directions

Check if difficulty and uncertainty are independent:

```bash
lazarus introspect directions \
    difficulty.npz \
    uncertainty.npz
```

### Compare Multiple Directions

Test orthogonality across all L15 dimensions:

```bash
lazarus introspect directions \
    difficulty.npz \
    operation.npz \
    format.npz \
    certainty.npz \
    magnitude.npz
```

### Custom Threshold

Use stricter orthogonality threshold:

```bash
lazarus introspect directions \
    difficulty.npz uncertainty.npz \
    --threshold 0.05
```

### Save Results

Export similarity matrix for further analysis:

```bash
lazarus introspect directions \
    difficulty.npz operation.npz format.npz \
    --output similarity_matrix.json
```

## Output

### Cosine Similarity Matrix

```
================================================================================
COSINE SIMILARITY MATRIX
================================================================================
(Threshold for 'orthogonal': |cos| < 0.1)

                  easy→hard  add→mult  no_space→space
easy→hard             1.000     0.087          -0.023
add→mult              0.087     1.000           0.041
no_space→space       -0.023     0.041           1.000
```

### Orthogonality Heatmap

```
================================================================================
ORTHOGONALITY HEATMAP
================================================================================
(■ = aligned, ▓ = correlated, ▒ = weak, ░ = near-orthogonal, · = orthogonal)

                easy→h  add→mu  no_spa
easy→hard           ■       ·       ·
add→mult            ·       ■       ·
no_space→space      ·       ·       ■
```

### Summary

```
================================================================================
SUMMARY
================================================================================

Total pairs: 3
Orthogonal (|cos| < 0.1): 3
Correlated (0.1 <= |cos| <= 0.5): 0
Aligned (|cos| > 0.5): 0

Orthogonal pairs (independent dimensions):
  no_space→space ⊥ easy→hard (cos = -0.023)
  add→mult ⊥ no_space→space (cos = +0.041)
  easy→hard ⊥ add→mult (cos = +0.087)

Mean |cosine similarity|: 0.050
Assessment: Directions are largely ORTHOGONAL (independent features)
```

## Interpreting Results

| Pattern | Interpretation |
|---------|----------------|
| cos ~ 0 | Independent features, can be used together |
| cos > 0.5 | Highly correlated, may be redundant |
| cos < -0.5 | Anti-correlated (opposite ends of same axis) |
| Mixed results | Some features overlap, some independent |

### What Orthogonality Tells Us

- **Orthogonal dimensions** can be used for independent steering
- **Correlated dimensions** may interfere when steering simultaneously
- **Aligned dimensions** suggest you've found the same feature twice

### Dimension Mismatch Warning

If directions come from different models or layers, dimensions won't match:

```
WARNING: Dimension mismatch: [2880, 4096]
  Directions from different models/layers may not be comparable
```

Only compare directions from the same model and layer.

## Workflow Example

Complete workflow for finding and validating L15 dimensions:

```bash
# 1. Extract difficulty direction
lazarus introspect probe \
    -m openai/gpt-oss-20b \
    --class-a "47*47=|67*83=" --label-a hard \
    --class-b "2+2=|5+5=" --label-b easy \
    --layer 15 \
    --save-direction difficulty.npz

# 2. Extract operation direction
lazarus introspect probe \
    -m openai/gpt-oss-20b \
    --class-a "2+2=|3+5=|10+20=" --label-a add \
    --class-b "2*2=|3*5=|10*20=" --label-b mult \
    --layer 15 \
    --save-direction operation.npz

# 3. Extract format direction
lazarus introspect probe \
    -m openai/gpt-oss-20b \
    --class-a "2+2= |5+5= " --label-a space \
    --class-b "2+2=|5+5=" --label-b no_space \
    --layer 15 \
    --save-direction format.npz

# 4. Check orthogonality
lazarus introspect directions \
    difficulty.npz operation.npz format.npz
```

## Saved Output Format

The JSON output contains:

```json
{
  "files": ["difficulty.npz", "operation.npz", "format.npz"],
  "names": ["easy→hard", "add→mult", "no_space→space"],
  "metadata": [
    {
      "file": "difficulty.npz",
      "name": "easy→hard",
      "layer": 15,
      "method": "difference",
      "accuracy": 1.0,
      "dim": 2880
    }
  ],
  "similarity_matrix": [
    [1.0, 0.087, -0.023],
    [0.087, 1.0, 0.041],
    [-0.023, 0.041, 1.0]
  ],
  "threshold": 0.1,
  "pairs": [
    {"a": "easy→hard", "b": "add→mult", "cosine": 0.087, "orthogonal": true},
    {"a": "easy→hard", "b": "no_space→space", "cosine": -0.023, "orthogonal": true},
    {"a": "add→mult", "b": "no_space→space", "cosine": 0.041, "orthogonal": true}
  ]
}
```

## Python API

```python
import numpy as np

# Load directions
d1 = np.load("difficulty.npz")["direction"]
d2 = np.load("operation.npz")["direction"]

# Normalize
d1_norm = d1 / np.linalg.norm(d1)
d2_norm = d2 / np.linalg.norm(d2)

# Cosine similarity
cos_sim = np.dot(d1_norm, d2_norm)
print(f"Cosine similarity: {cos_sim:.3f}")

if abs(cos_sim) < 0.1:
    print("Directions are orthogonal (independent features)")
```

## See Also

- [introspect probe](introspect-probe.md) - Extract directions from contrastive prompts
- [introspect steer](introspect-steer.md) - Apply steering directions
- [introspect neurons](introspect-neurons.md) - Analyze individual neurons
- [Introspection Overview](../introspection.md) - Full module documentation
