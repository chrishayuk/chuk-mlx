# lazarus introspect patch

Perform activation patching to test causal relationships between prompts.

## Synopsis

```bash
lazarus introspect patch -m MODEL --source PROMPT --target PROMPT [OPTIONS]
```

## Description

The `patch` command performs activation patching: transferring activations from a source prompt to a target prompt at specific layers. This is a causal intervention technique that tests whether activations from one prompt can transfer computation to another.

For example, patching activations from "7*8=" into "7+8=" at the computation layer should cause the model to output "56" instead of "15".

This is useful for:
- Identifying which layers encode "computation" vs "operands"
- Testing cross-operation transfer
- Finding the causal layer for answer production
- Understanding information flow

## Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | HuggingFace model ID or local path (required) |
| `--source PROMPT` | Source prompt to patch FROM (required) |
| `--target PROMPT` | Target prompt to patch INTO (required) |
| `-l, --layer N` | Layer to patch (default: sweep all layers) |
| `--layers RANGE` | Layer range to sweep (e.g., "8-16") |
| `--position POS` | Token position to patch: last, all, specific index |
| `-o, --output FILE` | Save results to JSON file |

## Examples

### Cross-Operation Patching

Transfer multiplication computation to addition:

```bash
lazarus introspect patch \
    -m openai/gpt-oss-20b \
    --source "7*8=" \
    --target "7+8="
```

### Specific Layer

Patch at a known computation layer:

```bash
lazarus introspect patch \
    -m model \
    --source "7*8=" \
    --target "7+8=" \
    --layer 15
```

### Layer Sweep

Find which layer causes the transfer:

```bash
lazarus introspect patch \
    -m model \
    --source "7*8=" \
    --target "7+8=" \
    --layers 8-20 \
    -o patch_sweep.json
```

### Operand Patching

Test if operand encoding transfers:

```bash
# Patch first operand: does 3*8 become 7*8?
lazarus introspect patch \
    -m model \
    --source "7*8=" \
    --target "3*8=" \
    --position 0
```

## Output

### Single Layer Patching

```
======================================================================
ACTIVATION PATCHING
======================================================================
Source: 7*8= (expected: 56)
Target: 7+8= (expected: 15)
Patching at layer 15

Before patching:
  Target output: 15 (correct for 7+8)

After patching:
  Target output: 56 (source answer transferred!)

Effect: TRANSFER SUCCESS
  Computation from source replaced target computation
```

### Layer Sweep

```
======================================================================
LAYER SWEEP: 7*8= → 7+8=
======================================================================
Layer    Target Output    Effect
------------------------------------------
L8       15               No effect
L10      15               No effect
L12      15               No effect
L14      45               Partial transfer
L15      56               FULL TRANSFER
L16      56               FULL TRANSFER
L18      56               FULL TRANSFER
L20      15               Effect fades

Critical layer: 15 (first full transfer)
```

### Effect Classification

| Effect | Meaning |
|--------|---------|
| NO EFFECT | Target output unchanged |
| PARTIAL TRANSFER | Output changed but not to source answer |
| FULL TRANSFER | Output matches source answer |
| CORRUPTION | Output is neither source nor target answer |

## Use Cases

### Finding Computation Layers

```bash
# Sweep to find where computation happens
lazarus introspect patch \
    -m model \
    --source "47*47=" \
    --target "47+47=" \
    --layers 0-24 \
    -o computation_layers.json

# The first layer showing FULL TRANSFER is the computation layer
```

### Testing Operand Independence

```bash
# Does changing first operand change computation?
lazarus introspect patch \
    -m model \
    --source "9*8=" \
    --target "2*8=" \
    --layer 15

# If transfer works, first operand affects computation at this layer
```

### Cross-Task Transfer

```bash
# Can we transfer across very different tasks?
lazarus introspect patch \
    -m model \
    --source "The capital of France is Paris" \
    --target "The capital of Germany is" \
    --layer 12
```

### Difficulty-Based Analysis

```bash
# Compare patching for easy vs hard problems
lazarus introspect patch \
    -m model \
    --source "2*2=" --target "2+2=" \
    -o easy_patch.json

lazarus introspect patch \
    -m model \
    --source "47*47=" --target "47+47=" \
    -o hard_patch.json
```

## Theoretical Background

### What Patching Tests

Activation patching answers: "If I replace the internal state of prompt A with prompt B at layer L, does the output change?"

**If output changes to B's answer:**
- Layer L contains causal information for the answer
- B's computation can override A's

**If output stays as A's answer:**
- Layer L doesn't contain decisive information
- Or information is redundant across layers

### Patching Positions

| Position | What it tests |
|----------|---------------|
| `last` | Final token position (default, most common) |
| `all` | All positions (tests full representation) |
| `0`, `1`, etc. | Specific token positions |

### Interpreting Layer Effects

```
Layers 0-4:   No effect (embedding, early processing)
Layers 5-10:  Partial effects (operand encoding)
Layers 11-16: Full transfer (computation layers)
Layers 17-24: Effects fade (output formatting)
```

## Saved Output Format

```json
{
  "model_id": "openai/gpt-oss-20b",
  "source_prompt": "7*8=",
  "source_expected": "56",
  "target_prompt": "7+8=",
  "target_expected": "15",
  "position": "last",
  "results": [
    {
      "layer": 15,
      "original_output": "15",
      "patched_output": "56",
      "effect": "full_transfer",
      "source_answer_found": true
    }
  ],
  "critical_layer": 15,
  "summary": {
    "first_effect_layer": 12,
    "full_transfer_layer": 15,
    "effect_fade_layer": 20
  }
}
```

## See Also

- [introspect commutativity](introspect-commutativity.md) - Test A*B = B*A
- [introspect arithmetic](introspect-arithmetic.md) - Systematic arithmetic testing
- [introspect ablate](introspect-ablate.md) - Ablation studies
- [Introspection Overview](../introspection.md) - Full module documentation
