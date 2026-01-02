# lazarus introspect neurons

Analyze individual neuron activations across prompts.

## Synopsis

```bash
lazarus introspect neurons -m MODEL -l LAYER --prompts PROMPTS [OPTIONS]
```

## Description

The `neurons` command shows how specific neurons fire across different prompts. This is useful for:

1. **Understanding probe results** - After training a probe, see how the top neurons actually behave
2. **Finding feature detectors** - Identify neurons that detect specific patterns (difficulty, topic, etc.)
3. **Validating interpretability hypotheses** - Confirm that neurons encode what you think they encode

## Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | HuggingFace model ID or local path (required) |
| `-l, --layer N` | Layer to analyze (required) |
| `-p, --prompts PROMPTS` | Prompts to analyze (pipe-separated or @file.txt) (required) |
| `-n, --neurons NEURONS` | Neuron indices (comma-separated, e.g., '808,1190') |
| `--from-direction FILE` | Load top neurons from saved direction .npz file |
| `--top-k N` | Number of top neurons when using --from-direction (default: 10) |
| `--labels LABELS` | Labels for prompts (pipe-separated, same order) |
| `-o, --output FILE` | Save results to JSON file |

## Examples

### Analyze Specific Neurons

Analyze key neurons from a previous probe:

```bash
lazarus introspect neurons \
    -m openai/gpt-oss-20b \
    -l 15 \
    --prompts "2+2=|10*10=|45*45=|47*47=|67*83=" \
    --neurons 808,1190,1168,891 \
    --labels "trivial|trivial|medium|hard|hard"
```

### Load Neurons from Saved Direction

After running `introspect probe --save-direction`, load top neurons:

```bash
# First, train probe and save direction
lazarus introspect probe \
    -m model \
    --class-a "47*47=|67*83=" --label-a hard \
    --class-b "2+2=|5+5=" --label-b easy \
    --save-direction difficulty.npz

# Then analyze those neurons
lazarus introspect neurons \
    -m model -l 15 \
    --prompts "2+2=|10*10=|45*45=|47*47=|67*83=|97*89=" \
    --from-direction difficulty.npz \
    --top-k 6
```

### Use Prompts from File

```bash
echo -e "2+2=\n10*10=\n45*45=\n47*47=\n67*83=" > prompts.txt
lazarus introspect neurons \
    -m model -l 15 \
    --prompts @prompts.txt \
    --neurons 808,1190
```

## Output

### Numeric Activation Map

```
================================================================================
NEURON ACTIVATION MAP AT LAYER 15
================================================================================
Prompt               | N 808 | N1190 | N1168 | N 891 | Label
----------------------------------------------------------------------
2+2=                 |   -156 |   +213 |   -198 |   +260 | trivial
10*10=               |     +2 |   +180 |   -119 |   +195 | trivial
45*45=               |   +232 |    +86 |    -28 |   +141 | medium
47*47=               |   +326 |     -2 |    +26 |    +77 | hard
67*83=               |   +168 |    +63 |     -7 |   +110 | hard
```

### ASCII Heatmap

```
================================================================================
ASCII HEATMAP (░ = low, ▒ = medium, ▓ = high, █ = max)
================================================================================
Prompt               | N 808 | N1190 | N1168 | N 891 |
------------------------------------------------------
2+2=                 |        |  ▓▓▓▓  |        |  ▓▓▓▓  | trivial
10*10=               |  ░░░░  |  ▒▒▒▒  |        |  ▓▓▓▓  | trivial
45*45=               |  ▓▓▓▓  |  ▒▒▒▒  |  ░░░░  |  ▒▒▒▒  | medium
47*47=               |  ████  |  ░░░░  |  ░░░░  |  ▒▒▒▒  | hard
67*83=               |  ▒▒▒▒  |  ░░░░  |  ░░░░  |  ▒▒▒▒  | hard
```

### Neuron Statistics

```
================================================================================
NEURON STATISTICS
================================================================================
Neuron  808: min= -156.0, max= +326.0, mean= +128.1, std= 146.3
Neuron 1190: min=   -1.8, max= +213.0, mean= +101.0, std=  71.1
```

### Label Correlation

```
================================================================================
LABEL CORRELATION
================================================================================

hard:
  Neuron  808: mean= +228.8  ← fires high for hard problems
  Neuron 1190: mean=  +40.7  ← fires low for hard problems

trivial:
  Neuron  808: mean=  -40.7  ← fires low for easy problems
  Neuron 1190: mean= +186.7  ← fires high for easy problems
```

## Interpreting Results

### Feature Detector Neurons

Look for neurons with large activation differences across categories:

| Pattern | Interpretation |
|---------|----------------|
| N808: trivial=-40, hard=+229 | Difficulty/uncertainty detector |
| N1190: trivial=+187, hard=+41 | "Easy problem" detector |
| High std, wide min/max range | Encodes a discriminative feature |
| Similar across all prompts | Not task-specific |

### Using with Direction Files

When loading from `--from-direction`, the neuron weights tell you the direction:
- **Positive weight** → Higher activation = more like class A (positive class)
- **Negative weight** → Higher activation = more like class B (negative class)

## Workflow Example

Complete workflow for understanding arithmetic difficulty:

```bash
# 1. Train probe to find direction
lazarus introspect probe \
    -m openai/gpt-oss-20b \
    --class-a "47*47=|67*83=|53*59=" --label-a hard \
    --class-b "2+2=|5+5=|3*3=" --label-b easy \
    --layer 15 \
    --save-direction difficulty.npz

# 2. Analyze top neurons
lazarus introspect neurons \
    -m openai/gpt-oss-20b -l 15 \
    --prompts "2+2=|10*10=|45*45=|47*47=|67*83=|97*89=" \
    --from-direction difficulty.npz \
    --top-k 6 \
    --labels "trivial|trivial|medium|hard|hard|hard"

# 3. Save detailed results
lazarus introspect neurons \
    -m openai/gpt-oss-20b -l 15 \
    --prompts "2+2=|47*47=" \
    --neurons 808,1190 \
    --output neuron_analysis.json
```

## Saved Output Format

The JSON output contains:

```json
{
  "model_id": "openai/gpt-oss-20b",
  "layer": 15,
  "neurons": [808, 1190],
  "prompts": ["2+2=", "47*47="],
  "labels": ["easy", "hard"],
  "activations": [[-156.0, 213.0], [326.0, -2.0]],
  "neuron_weights": {"808": 0.2017, "1190": -0.107},
  "stats": {
    "808": {"min": -156.0, "max": 326.0, "mean": 85.0, "std": 241.0},
    "1190": {"min": -2.0, "max": 213.0, "mean": 105.5, "std": 107.5}
  }
}
```

## See Also

- [introspect probe](introspect-probe.md) - Train probes and extract directions
- [introspect ablate](introspect-ablate.md) - Ablation studies
- [introspect steer](introspect-steer.md) - Apply steering directions
- [Introspection Overview](../introspection.md) - Full module documentation
