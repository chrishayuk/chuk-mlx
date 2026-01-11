# lazarus introspect activation-cluster

Visualize how different prompt types cluster in activation space using PCA.

## Synopsis

```bash
lazarus introspect activation-cluster -m MODEL [OPTIONS]
```

## Description

The `activation-cluster` command projects hidden states to 2D using PCA to visualize whether different prompt types form distinct clusters. This reveals:

1. **Task separation** - Do math prompts cluster separately from language prompts?
2. **Difficulty encoding** - Do easy and hard problems form different clusters?
3. **Format sensitivity** - Does trailing space affect clustering?

## Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | HuggingFace model ID or local path (required) |
| `-l, --layer N` | Layer(s) to analyze (comma-separated, default: 50% depth) |
| `--class-a PROMPTS` | Class A prompts (pipe-separated or @file.txt) |
| `--class-b PROMPTS` | Class B prompts (pipe-separated or @file.txt) |
| `--label-a LABEL` | Label for class A (default: 'class_a') |
| `--label-b LABEL` | Label for class B (default: 'class_b') |
| `--prompts PROMPTS` | Multi-class: prompts for one class (repeatable) |
| `--label LABEL` | Multi-class: label for preceding --prompts (repeatable) |
| `--save-plot FILE` | Save matplotlib plot to PNG file |

## Examples

### Two-Class Clustering (Legacy Syntax)

Compare math vs language prompts:

```bash
lazarus introspect activation-cluster \
    -m openai/gpt-oss-20b \
    --class-a "2+2=|5*5=|10-3=" \
    --label-a math \
    --class-b "Hello world|The cat sat|Once upon" \
    --label-b language \
    -l 12
```

### Multi-Class Clustering

Compare easy, medium, and hard arithmetic:

```bash
lazarus introspect activation-cluster \
    -m model \
    --prompts "2+2=|3+3=|5+5=" --label easy \
    --prompts "45*45=|67+89=" --label medium \
    --prompts "97*89=|67*83=" --label hard \
    -l 15
```

### Multiple Layers

Analyze clustering across multiple layers:

```bash
lazarus introspect activation-cluster \
    -m model \
    --class-a "47*47=|67*83=" --label-a hard \
    --class-b "2+2=|5+5=" --label-b easy \
    -l 8,12,16,20 \
    --save-plot clusters.png
```

### Use Prompts from Files

```bash
echo -e "47*47=\n67*83=\n97*89=" > hard.txt
echo -e "2+2=\n5+5=\n3*3=" > easy.txt

lazarus introspect activation-cluster \
    -m model \
    --class-a @hard.txt --label-a hard \
    --class-b @easy.txt --label-b easy
```

## Output

### Cluster Statistics

```
======================================================================
ACTIVATION CLUSTERS AT LAYER 15
======================================================================
PCA explained variance: 45.2% + 23.1%

Cluster separations:
  hard <-> easy: 156.34

Label           Count    Center (PC1, PC2)
--------------------------------------------------
hard            5        (89.23, -45.67)
easy            5        (-67.11, 23.45)
```

### ASCII Scatter Plot

```
======================================================================
SCATTER PLOT (ASCII) - Layer 15
======================================================================

                                   H
                              H        H
                         H
                                   H


          E
     E         E
               E   E


  Legend: H=hard, E=easy
```

### Matplotlib Plot

When using `--save-plot`, a publication-quality scatter plot is saved with:
- Color-coded points for each class
- Cluster centers marked with X
- Legend with sample counts
- Grid overlay

## Interpreting Results

| Pattern | Interpretation |
|---------|----------------|
| High separation, distinct clusters | Model encodes task difference at this layer |
| Overlapping clusters | Layer doesn't distinguish these tasks |
| High PCA variance (>70%) | Most information captured in 2D |
| Low PCA variance (<30%) | Clustering in higher dimensions |

## Use Cases

### Finding Task Representation Layers

```bash
# Check multiple layers to find where tasks separate
for layer in 4 8 12 16 20 24; do
    echo "=== Layer $layer ==="
    lazarus introspect activation-cluster \
        -m model \
        --class-a "math prompts" --label-a math \
        --class-b "language prompts" --label-b language \
        -l $layer
done
```

### Difficulty Stratification

Visualize how difficulty levels separate:

```bash
lazarus introspect activation-cluster \
    -m model \
    --prompts "2+2=|3*3=" --label trivial \
    --prompts "45+67=|23*4=" --label moderate \
    --prompts "97*89=|67*83=" --label hard \
    -l 15 \
    --save-plot difficulty_clusters.png
```

## See Also

- [introspect probe](introspect-probe.md) - Train linear probes on activations
- [introspect layer](introspect-layer.md) - Layer representation similarity
- [introspect embedding](introspect-embedding.md) - Embedding-level analysis
- [Introspection Overview](../introspection.md) - Full module documentation
