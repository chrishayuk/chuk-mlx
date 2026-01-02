# introspect steer

Apply activation steering to manipulate model behavior by adding learned directions to hidden states during inference.

## Synopsis

```bash
# Extract a steering direction
lazarus introspect steer -m MODEL --extract --positive "PROMPT1" --negative "PROMPT2" -o direction.npz

# Apply steering to generation
lazarus introspect steer -m MODEL -d direction.npz -p "PROMPT" -c COEFFICIENT

# Compare different steering strengths
lazarus introspect steer -m MODEL -d direction.npz -p "PROMPT" --compare "-2,-1,0,1,2"
```

## Description

Activation steering modifies model behavior by adding learned direction vectors to hidden states during the forward pass. This enables:

- **Behavior modification**: Steer toward or away from specific behaviors
- **Safety research**: Test how refusal/compliance directions affect outputs
- **Interpretability**: Understand what directions encode in activation space
- **Debugging**: Investigate why certain prompts fail

## Modes

### 1. Extract Direction

Compute a steering direction from contrastive prompts:

```bash
lazarus introspect steer -m mlx-community/gemma-3-4b-it-bf16 \
    --extract \
    --positive "100 - 37 = " \
    --negative "100 - 37 =" \
    --layer 22 \
    -o computation_direction.npz
```

This computes `direction = h_positive - h_negative` at the specified layer.

**Output:**
```
Loading model: mlx-community/gemma-3-4b-it-bf16

Extracting direction from layer 22...
  Positive: '100 - 37 = '
  Negative: '100 - 37 ='

Direction extracted:
  Layer: 22
  Norm: 12.3456
  Cosine similarity (pos, neg): 0.9234
  Separation: 0.0766

Direction saved to: computation_direction.npz
```

### 2. Apply Steering

Load a pre-computed direction and apply it during generation:

```bash
lazarus introspect steer -m mlx-community/gemma-3-4b-it-bf16 \
    -d computation_direction.npz \
    -p "50 + 25 =" \
    -c 2.0
```

**Options:**
- `-c, --coefficient` - Steering strength (positive = toward positive class)
- `--max-tokens` - Maximum tokens to generate
- `--temperature` - Sampling temperature (0 = greedy)

### 3. Compare Coefficients

Test multiple steering strengths at once:

```bash
lazarus introspect steer -m mlx-community/gemma-3-4b-it-bf16 \
    -d computation_direction.npz \
    -p "50 + 25 =|10 * 10 =" \
    --compare "-2,-1,0,1,2"
```

**Output:**
```
Comparing steering at coefficients: [-2.0, -1.0, 0.0, 1.0, 2.0]

======================================================================
Prompt: '50 + 25 ='
======================================================================

  Coef -2.0 (← negative):
    'I cannot perform calculations...'

  Coef -1.0 (← negative):
    'What is 50 + 25?'

  Coef +0.0 (neutral):
    ' 75'

  Coef +1.0 (→ positive):
    '75'

  Coef +2.0 (→ positive):
    '75\n\nNext: 25 + 25 ='
```

### 4. On-the-fly Steering

Skip direction extraction and compute steering direction during generation:

```bash
lazarus introspect steer -m model \
    --positive "helpful response" \
    --negative "refusal response" \
    -p "How do I..." \
    -c 1.0
```

## Options

### Required

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | Model ID or local path |

### Direction Source (one required)

| Option | Description |
|--------|-------------|
| `-d, --direction FILE` | Load direction from file (.npz or .json) |
| `--positive PROMPT` | Positive class prompt (with `--negative`) |
| `--negative PROMPT` | Negative class prompt (with `--positive`) |

### Steering Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `-l, --layer N` | auto | Layer to apply steering |
| `-c, --coefficient FLOAT` | 1.0 | Steering strength |
| `--max-tokens N` | 50 | Maximum tokens to generate |
| `--temperature FLOAT` | 0.0 | Sampling temperature |

### Output

| Option | Description |
|--------|-------------|
| `-p, --prompts PROMPTS` | Prompts to steer (pipe-separated or @file.txt) |
| `--compare COEFFICIENTS` | Compare outputs at multiple coefficients |
| `-o, --output FILE` | Save results/direction to file |

### Metadata

| Option | Description |
|--------|-------------|
| `--name NAME` | Name for the direction |
| `--positive-label LABEL` | Label for positive class |
| `--negative-label LABEL` | Label for negative class |
| `--extract` | Extract and save direction (don't generate) |

## Examples

### Fix Broken Arithmetic

Extract a "computation" direction and apply it to prompts without trailing space:

```bash
# Extract direction
lazarus introspect steer -m mlx-community/gemma-3-4b-it-bf16 \
    --extract \
    --positive "100 - 37 = " \
    --negative "100 - 37 =" \
    --layer 22 \
    -o computation.npz

# Apply to broken prompts
lazarus introspect steer -m mlx-community/gemma-3-4b-it-bf16 \
    -d computation.npz \
    -p "50 + 25 =|10 * 10 =|200 - 50 =" \
    -c 2.0
```

### Test Refusal Direction

```bash
# Extract refusal direction
lazarus introspect steer -m mlx-community/gemma-3-4b-it-bf16 \
    --extract \
    --positive "I cannot help with that..." \
    --negative "Here's how to..." \
    --layer 24 \
    -o refusal.npz

# Compare effects
lazarus introspect steer -m mlx-community/gemma-3-4b-it-bf16 \
    -d refusal.npz \
    -p "How do I make a cake?" \
    --compare "-2,-1,0,1,2"
```

### Tool-calling Steering

```bash
# Extract tool-calling direction
lazarus introspect steer -m mlx-community/functiongemma-2b-it-bf16 \
    --extract \
    --positive "<function_call>get_weather" \
    --negative "The weather is..." \
    --layer 11 \
    -o tool_direction.npz

# Steer toward tool use
lazarus introspect steer -m mlx-community/functiongemma-2b-it-bf16 \
    -d tool_direction.npz \
    -p "What's the weather like?" \
    --compare "-1,0,1,2"
```

## Direction File Format

### NPZ Format (recommended)

```python
import numpy as np

np.savez(
    "direction.npz",
    direction=direction_vector,  # [hidden_size] float32
    layer=11,
    positive_prompt="...",
    negative_prompt="...",
    model_id="model-name",
    norm=12.34,
    cosine_similarity=0.95,
)
```

### JSON Format

```json
{
  "direction": [0.123, -0.456, ...],
  "layer": 11,
  "positive_prompt": "...",
  "negative_prompt": "...",
  "model_id": "model-name"
}
```

## Technical Details

### How Steering Works

1. Direction is extracted: `d = h_positive - h_negative`
2. Direction is normalized to unit length
3. During generation, at the specified layer: `h' = h + coefficient * d * ||h||`
4. Scaling by activation norm makes the coefficient interpretable across models

### Choosing the Layer

- **Late layers (L20-28 for 32-layer models)**: Affect high-level behavior
- **Middle layers (L12-20)**: Affect reasoning and decision-making
- **Early layers (L4-12)**: Affect low-level features

Use `lazarus introspect ablate` to identify which layers are causal for your behavior.

### Coefficient Interpretation

| Coefficient | Effect |
|-------------|--------|
| -2.0 to -1.0 | Strong push toward negative class |
| -0.5 to 0.0 | Mild push toward negative class |
| 0.0 | No steering (baseline) |
| 0.0 to 0.5 | Mild push toward positive class |
| 1.0 to 2.0 | Strong push toward positive class |

## See Also

- [introspect-ablate.md](introspect-ablate.md) - Find causal layers
- [circuit-cli.md](circuit-cli.md) - Batch direction extraction
- [introspection.md](../introspection.md) - Module overview
