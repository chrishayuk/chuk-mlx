# introspect layer

Analyze what specific layers do with representation similarity.

## Usage

```bash
lazarus introspect layer -m MODEL -p PROMPTS [OPTIONS]
```

## Description

This tool helps answer questions about layer function:
- Do similar inputs cluster together at layer N?
- Where do working/broken prompts diverge?
- At which layer does format (trailing space) affect computation?

It computes cosine similarity between hidden states at specific layers, optionally clustering by labels.

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--model` | `-m` | Model name or HuggingFace ID (required) |
| `--prompts` | `-p` | Prompts to analyze, pipe-separated or @file.txt (required) |
| `--labels` | `-l` | Labels for clustering (comma-separated) |
| `--layers` | | Layers to analyze (comma-separated). Default: auto |
| `--attention` | `-a` | Also analyze attention patterns |
| `--output` | `-o` | Save results to JSON file |

## Examples

### Basic similarity analysis

```bash
lazarus introspect layer -m gemma-3-4b -p "The cat sat|The dog sat|A random thing"
```

### Format sensitivity (working vs broken)

```bash
lazarus introspect layer \
    -m mlx-community/gemma-3-4b-it-bf16 \
    -p "100 - 37 = |100 - 37 =|50 + 25 = |50 + 25 =" \
    -l working,broken,working,broken \
    --layers 2,4,6,8
```

This shows whether working prompts (with trailing space) cluster separately from broken prompts at each layer.

### With attention analysis

```bash
lazarus introspect layer \
    -m model \
    -p "prompt1|prompt2|prompt3" \
    --attention
```

### Load prompts from file

```bash
# prompts.txt (one per line)
lazarus introspect layer -m model -p @prompts.txt -l label1,label2,label3
```

## Output

The tool outputs:

1. **Similarity Matrix** - Cosine similarity between all prompt pairs at each layer
2. **Clustering Analysis** (if labels provided):
   - Within-cluster similarity (how similar prompts with same label are)
   - Between-cluster similarity (how similar prompts with different labels are)
   - Separation score (within - between; higher = better separation)
3. **Interpretation** - Whether the layer distinguishes between groups

## Example Output

```
=== Layer 4 Similarity Matrix ===

                             0     1     2     3
100 - 37 =          [working] 1.00  0.89  0.95  0.88
100 - 37 =          [broken]  0.89  1.00  0.87  0.96
50 + 25 =           [working] 0.95  0.87  1.00  0.86
50 + 25 =           [broken]  0.88  0.96  0.86  1.00

--- Clustering Analysis ---
Within-cluster similarity:
  working: 0.9500
  broken: 0.9600
Between-cluster similarity:
  working <-> broken: 0.8750
Separation score: 0.0850

=== Format Sensitivity Summary ===

Layer 4:
  Within 'working': 0.9500
  Within 'broken': 0.9600
  Between 'working' <-> 'broken': 0.8750
  Separation score: 0.0850
  -> Layer 4 DOES distinguish between groups
```

## Python API

```python
from chuk_lazarus.introspection import LayerAnalyzer

analyzer = LayerAnalyzer.from_pretrained("mlx-community/gemma-3-4b-it-bf16")

prompts = [
    "100 - 37 = ",   # working
    "100 - 37 =",    # broken
    "50 + 25 = ",    # working
    "50 + 25 =",     # broken
]
labels = ["working", "broken", "working", "broken"]

result = analyzer.analyze_representations(
    prompts=prompts,
    layers=[2, 4, 6, 8],
    labels=labels,
)

# Print similarity matrices
for layer in result.layers:
    analyzer.print_similarity_matrix(result, layer)

# Access clusters
for layer_idx, cluster in result.clusters.items():
    print(f"Layer {layer_idx}: separation = {cluster.separation_score:.4f}")
```

## Use Cases

1. **Format sensitivity research** - Find which layers are affected by trailing whitespace
2. **Prompt clustering** - See if semantically similar prompts cluster in representation space
3. **Debugging model behavior** - Understand where predictions diverge for similar inputs
4. **Feature emergence** - Track when representations start distinguishing categories

## See Also

- [introspect-format-sensitivity.md](introspect-format-sensitivity.md) - Quick format sensitivity check
- [introspect-analyze.md](introspect-analyze.md) - Logit lens analysis
- [introspect-ablate.md](introspect-ablate.md) - Ablation studies
