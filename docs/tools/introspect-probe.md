# lazarus introspect probe

Train linear probes to find classification layers and extract steering directions.

## Synopsis

```bash
lazarus introspect probe -m MODEL --class-a PROMPTS --class-b PROMPTS [OPTIONS]
```

## Description

The `probe` command trains logistic regression probes at each layer to discover where the model separates different types of prompts in activation space. This reveals:

1. **Which layers classify tasks** (e.g., math vs language)
2. **The "direction" in activation space** that separates the classes
3. **Potential steering vectors** for modifying model behavior

## Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | HuggingFace model ID or local path (required) |
| `--class-a PROMPTS` | Class A prompts (pipe-separated or @file.txt) (required) |
| `--class-b PROMPTS` | Class B prompts (pipe-separated or @file.txt) (required) |
| `--label-a LABEL` | Label for class A (default: 'class_a') |
| `--label-b LABEL` | Label for class B (default: 'class_b') |
| `-l, --layer N` | Specific layer to probe (default: find best layer) |
| `--method {logistic,difference}` | Direction extraction method (default: logistic) |
| `--save-direction FILE` | Save direction vector to .npz file |
| `-t, --test PROMPTS` | Test prompts to classify after training |
| `-o, --output FILE` | Save results to JSON file |

## Direction Extraction Methods

| Method | Description |
|--------|-------------|
| `logistic` | Use logistic regression weights (default) |
| `difference` | Use difference of means between classes (simpler, normalized) |

## Examples

### Find Difficulty Classification Layer

Train probes to distinguish easy from hard arithmetic:

```bash
lazarus introspect probe \
    -m openai/gpt-oss-20b \
    --class-a "47*47=|67*83=|53*59=|71*73=|97*89=" \
    --label-a hard \
    --class-b "2+2=|5+5=|3*3=|10*10=|7+8=" \
    --label-b easy
```

### Extract and Save Direction

Extract a difficulty direction at layer 15 and save it:

```bash
lazarus introspect probe \
    -m openai/gpt-oss-20b \
    --class-a "47*47=|67*83=|53*59=" \
    --label-a hard \
    --class-b "2+2=|5+5=|3*3=" \
    --label-b easy \
    --layer 15 \
    --method difference \
    --save-direction difficulty_direction.npz
```

### Test Classification

Train probe and test on new prompts:

```bash
lazarus introspect probe \
    -m model \
    --class-a "How to hack...|Bypass security..." \
    --label-a harmful \
    --class-b "Write a poem|Tell me a joke" \
    --label-b safe \
    --test "Make a bomb|Hello world|Explain chemistry"
```

### Use Prompts from Files

For larger datasets, use @file.txt syntax:

```bash
# Create files
echo -e "47*47=\n67*83=\n53*59=" > hard_prompts.txt
echo -e "2+2=\n5+5=\n3*3=" > easy_prompts.txt

lazarus introspect probe \
    -m model \
    --class-a @hard_prompts.txt --label-a hard \
    --class-b @easy_prompts.txt --label-b easy \
    --save-direction difficulty.npz
```

## Output

```
Loading model: openai/gpt-oss-20b
  Layers: 24

Class A (hard): 6 prompts
Class B (easy): 6 prompts

Collecting activations...
Training probes at each layer...

======================================================================
PROBE ACCURACY BY LAYER (hard vs easy)
======================================================================
Layer    Accuracy     Std        Bar
----------------------------------------------------------------------
  L0     0.900        0.200     #############################################
  L1     1.000        0.000     ##################################################
  ...
  L15    1.000        0.000     ################################################## ← SELECTED
  ...
----------------------------------------------------------------------

Selected layer: L15 (accuracy: 100.0%)

Direction method: difference of means (normalized)

Projection statistics:
  hard: +423.09 ± 59.58
  easy: -908.36 ± 153.82
  Separation: 1331.45

Top 10 neurons for hard detection:
  Neuron 808: weight 0.2017
  Neuron 1190: weight -0.1070
  ...

======================================================================
TEST PREDICTIONS
======================================================================
  45*45=                                   → hard (81.4%)
  100-37=                                  → easy (100.0%)
  What is 7^13?                            → hard (99.9%)

Direction vector saved to: difficulty_direction.npz
  Shape: (2880,)
  Layer: 15
  Use with: lazarus introspect steer -d difficulty_direction.npz ...
```

## Use Cases

### Uncertainty Detection

Find a direction that separates "certain" from "uncertain" prompts:

```bash
lazarus introspect probe \
    -m model \
    --class-a "47*47=|Who won 1923 Nobel?|What is pi^10?" \
    --label-a uncertain \
    --class-b "2+2=|Capital of France?|Water freezes at" \
    --label-b certain \
    --save-direction uncertainty_direction.npz
```

Then use during inference to detect potential hallucinations.

### Task Classification

Find which layers distinguish task types:

```bash
lazarus introspect probe \
    -m model \
    --class-a "2+2=|45*45=|sqrt(16)" \
    --label-a math \
    --class-b "Write a poem|Tell a story|Summarize" \
    --label-b creative
```

### Refusal Detection

Find the refusal direction:

```bash
lazarus introspect probe \
    -m model \
    --class-a "How to hack|Make explosives|Bypass security" \
    --label-a refused \
    --class-b "Write code|Explain physics|Help with homework" \
    --label-b allowed \
    --save-direction refusal_direction.npz
```

## Saved Direction Format

The `.npz` file contains:

| Key | Description |
|-----|-------------|
| `direction` | Normalized direction vector (shape: hidden_size) |
| `layer` | Layer the direction was extracted from |
| `label_positive` | Label for positive class (class A) |
| `label_negative` | Label for negative class (class B) |
| `model_id` | Model the direction was trained on |
| `method` | Extraction method used |
| `accuracy` | Probe accuracy at this layer |
| `separation` | Distance between class centroids |
| `class_a_mean_projection` | Mean projection of class A |
| `class_b_mean_projection` | Mean projection of class B |

## Python API

```python
import numpy as np
from chuk_lazarus.introspection import ModelHooks, CaptureConfig, PositionSelection

# Load saved direction
data = np.load("difficulty_direction.npz")
direction = data["direction"]
layer = int(data["layer"])

# Use for steering (add to activations at layer)
def apply_steering(model, prompt, direction, layer, scale=1.0):
    hooks = ModelHooks(model)
    # ... hook setup to inject direction at layer ...
```

## See Also

- [introspect steer](introspect-steer.md) - Apply steering directions
- [introspect ablate](introspect-ablate.md) - Ablation studies
- [introspect cluster](introspect-cluster.md) - Visualize activation clusters
- [Introspection Overview](../introspection.md) - Full module documentation
