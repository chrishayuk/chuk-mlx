# lazarus introspect ablate

Run ablation studies to identify causal circuits in the model.

## Synopsis

```bash
lazarus introspect ablate -m MODEL -p PROMPT -c CRITERION [OPTIONS]
```

## Description

The `ablate` command zeros out model components (MLP or attention) at specific layers to identify which are **causal** for specific behaviors. If ablating layer N breaks behavior X, then layer N is causally involved in X.

Supports two modes:
1. **Sweep mode** (default): Test each layer independently
2. **Multi mode** (`--multi`): Ablate all specified layers together

## Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | HuggingFace model ID or local path (required) |
| `-p, --prompt PROMPT` | Prompt to test (required) |
| `-c, --criterion CRITERION` | Behavior to test for (required) |
| `--component {mlp,attention,both}` | Component to ablate (default: mlp) |
| `--layers L1,L2,...` | Specific layers to test (default: all). Supports ranges like `20-23` |
| `--max-tokens N` | Max tokens to generate (default: 60) |
| `--multi` | Ablate all specified layers together (default: sweep each individually) |
| `--raw` | Use raw prompt without chat template |
| `--prompts` | Multiple prompts with expected outputs (pipe-separated) |
| `-v, --verbose` | Show full generated outputs |
| `-o, --output FILE` | Save results to JSON |

## Built-in Criteria

| Criterion | Detects |
|-----------|---------|
| `function_call` | Tool/function call patterns (`<function_call>`, `get_weather(`, etc.) |
| `sorry` | Apologies ("sorry", "apologize") |
| `positive` | Positive sentiment ("great", "good", "excellent") |
| `negative` | Negative sentiment ("bad", "terrible", "awful") |
| `refusal` | Refusal patterns ("cannot", "can't", "won't") |
| `<any string>` | Substring match |

## Examples

### Basic Layer Sweep

Sweep layers 20-23 individually on an arithmetic prompt:

```bash
lazarus introspect ablate \
    -m openai/gpt-oss-20b \
    -p "45 * 45 = " \
    -c "2025" \
    --layers 20-23 \
    --max-tokens 15
```

### Multi-Layer Ablation

Ablate L22+L23 together (tests if the **combination** is causal):

```bash
lazarus introspect ablate \
    -m openai/gpt-oss-20b \
    -p "45 * 45 = " \
    -c "2025" \
    --layers 22,23 \
    --multi \
    --max-tokens 15
```

### Multi-Prompt Difficulty Gradient

Test multiple prompts with different expected answers to find differential causality:

```bash
lazarus introspect ablate \
    -m openai/gpt-oss-20b \
    -p "x" -c "x" \
    --prompts "10 * 10 = :100|45 * 45 = :2025|47 * 47 = :2209" \
    --layers "20-23" \
    --max-tokens 15
```

Output:
```
======================================================================
MULTI-PROMPT ABLATION TEST
======================================================================
Ablation             | 10 * 10 =          | 45 * 45 =          | 47 * 47 =
-----------------------------------------------------------------------------------
None (baseline)      | Y 100              | Y 2025             | Y 2209
L20                  | Y 100              | Y 2025             | Y 2209
L21                  | Y 100              | Y 2025             | Y 2209
L22                  | Y 100              | Y 2025             | N 2,209 ← BROKEN
L23                  | Y 100              | Y 2025             | Y 2209
```

### Combined Multi-Prompt + Multi-Layer

Test multiple prompts with combined layer ablation:

```bash
lazarus introspect ablate \
    -m openai/gpt-oss-20b \
    -p "x" -c "x" \
    --prompts "10 * 10 = :100|45 * 45 = :2025|47 * 47 = :2209" \
    --layers "22,23" \
    --multi \
    --max-tokens 15
```

### Find Tool-Calling Layers

```bash
lazarus introspect ablate \
    -m google/gemma-3-1b-it \
    -p "What's the weather in Paris?" \
    -c function_call \
    --component mlp
```

### Ablate Attention Instead of MLP

```bash
lazarus introspect ablate \
    -m model \
    -p "Tell me a joke" \
    -c positive \
    --component attention
```

### Raw Mode (Skip Chat Template)

```bash
lazarus introspect ablate \
    -m model \
    -p "2 + 2 = " \
    -c "4" \
    --raw
```

## Output

### Sweep Mode (default)

```
Loading model: google/gemma-3-1b-it
Sweeping layers individually: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
Component: mlp
Mode: CHAT

Running ablation sweep...

=== Ablation Study Results ===
Criterion: function_call
Component: mlp

Layer | Output (truncated)                              | Criterion Met
------|------------------------------------------------|---------------
Base  | {"name": "get_weather", "parameters": {"loc... | Yes
  0   | {"name": "get_weather", "parameters": {"loc... | Yes
  ...
 10   | {"name": "get_weather", "parameters": {"loc... | Yes
 11   | I don't have access to real-time weather...   | No  ***
 12   | I'm sorry, I can't check the weather...       | No  ***
 13   | {"name": "get_weather", "parameters": {"loc... | Yes
  ...

Critical layers (ablation breaks criterion): [11, 12]
```

### Multi Mode (--multi)

```
Loading model: openai/gpt-oss-20b
Ablating layers together: [22, 23]
Component: mlp
Mode: CHAT

Ablating layers [22, 23] together...

============================================================
Prompt: 45 * 45 =
Criterion: 2025
Layers ablated: [22, 23]
============================================================

Original output (PASS):
  2025

So, the result of 45 * 45 is

Ablated output (FAIL):
  2.025e6, which is 2025000.

=> CAUSAL: Ablating [22, 23] breaks the criterion
```

## Understanding Results

### Single-Layer vs Multi-Layer Ablation

| Finding | Interpretation |
|---------|----------------|
| Layer N breaks behavior | N is causally necessary |
| No single layer breaks behavior | Distributed/redundant circuit |
| L22+L23 together breaks behavior | These layers work as a unit |
| L22 alone breaks hard problems only | L22 gates "hard problem" pathway |

### Differential Causality (Difficulty Gradient)

When testing prompts of varying difficulty:

| Result | Interpretation |
|--------|----------------|
| Trivial works, hard breaks | Layer gates access to complex computation |
| All break | Layer is load-bearing for the entire capability |
| None break | Layer is epiphenomenal (correlate, not cause) |
| Intermediate answer appears | Layer controls "keep computing" signal |

### Example: Arithmetic Circuit Discovery

Testing GPT-OSS 20B on multiplication:

| Ablation | 10×10 (trivial) | 45×45 (factorizable) | 47×47 (prime) |
|----------|-----------------|----------------------|---------------|
| None | ✓ 100 | ✓ 2025 | ✓ 2209 |
| L22 only | ✓ 100 | ✓ 2025 | ✗ 2,209 |
| L22,23 | ✓ 100 | ✗ 2.025e6 | ✗ 2,209 |
| L21,22,23 | ✓ 1000 (wrong!) | ✗ 1.5×45 | ✗ 2.47×47 |

**Interpretation:**
- **L22** = "hard problem gate" (only breaks prime multiplication)
- **L22-23** = full arithmetic gate (breaks non-trivial multiplication)
- **L21** = magnitude circuit (10×10 → 1000 when ablated)

## How It Works

1. **Baseline**: Generate output without any ablation
2. **For each layer (or layer set)**: Zero out the MLP (or attention) weights, generate output
3. **Evaluate**: Check if the criterion is still met
4. **Report**: Layers where ablation breaks the criterion are "causal"

## Components

| Component | What It Ablates |
|-----------|-----------------|
| `mlp` | Feed-forward network (the "knowledge" part) |
| `attention` | Self-attention mechanism (the "routing" part) |
| `both` | Both MLP and attention |

## Use Cases

### Finding Arithmetic Circuits

Identify which layers are responsible for different arithmetic operations:

```bash
lazarus introspect ablate \
    -m model \
    -p "x" -c "x" \
    --prompts "5+5=:10|23*23=:529|47*47=:2209" \
    --layers 15-23
```

### Finding Tool-Calling Circuits

Identify which layers are responsible for tool/function calling:

```bash
lazarus introspect ablate -m functiongemma -p "Get weather" -c function_call
```

### Understanding Refusal Circuits

Find which layers cause the model to refuse requests:

```bash
lazarus introspect ablate -m model -p "How to hack..." -c refusal
```

### Sentiment Control

Find which layers control sentiment:

```bash
lazarus introspect ablate -m model -p "The movie was" -c positive
```

## Python API

```python
from chuk_lazarus.introspection import AblationStudy, AblationConfig, ComponentType

study = AblationStudy.from_pretrained(model_id)

# Single-layer sweep
def criterion(output: str) -> bool:
    return "2025" in output

result = study.run_layer_sweep(
    prompt="45 * 45 = ",
    criterion=criterion,
    component=ComponentType.MLP,
    layers=range(20, 24),
)

study.print_sweep_summary(result)

# Multi-layer ablation
config = AblationConfig(max_new_tokens=15)

original = study.ablate_and_generate("45 * 45 = ", layers=[], config=config)
ablated = study.ablate_and_generate("45 * 45 = ", layers=[22, 23], config=config)

print(f"Original: {original}")
print(f"Ablated L22+L23: {ablated}")
print(f"Criterion met: {'2025' in ablated}")
```

### Multi-Prompt Analysis

```python
from chuk_lazarus.introspection import AblationStudy, AblationConfig

study = AblationStudy.from_pretrained("openai/gpt-oss-20b")
config = AblationConfig(max_new_tokens=15)

test_cases = [
    ("10 * 10 = ", "100", "trivial"),
    ("45 * 45 = ", "2025", "factorizable"),
    ("47 * 47 = ", "2209", "prime"),
]

layer_combos = [
    [],           # baseline
    [22],         # L22 only
    [22, 23],     # L22+L23
    [21, 22, 23], # L21-23
]

for layers in layer_combos:
    print(f"\n{'Baseline' if not layers else f'L{layers}'}")
    for prompt, expected, difficulty in test_cases:
        out = study.ablate_and_generate(prompt, layers=layers, config=config)
        correct = expected in out
        print(f"  {difficulty}: {'✓' if correct else '✗'} {out.strip()[:30]}")
```

## See Also

- [introspect analyze](introspect-analyze.md) - Logit lens analysis
- [introspect steer](introspect-steer.md) - Activation steering
- [introspect weight-diff](introspect-weight-diff.md) - Weight comparison
- [Introspection Overview](../introspection.md) - Full module documentation
