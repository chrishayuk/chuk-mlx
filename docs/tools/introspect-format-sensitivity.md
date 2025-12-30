# introspect format-sensitivity

Quick format sensitivity check (trailing space vs no space).

## Usage

```bash
lazarus introspect format-sensitivity -m MODEL -p PROMPTS [OPTIONS]
```

## Description

This is a convenience wrapper around `introspect layer` that automatically:
1. Takes base prompts (without trailing space)
2. Creates working variants (with trailing space) and broken variants (without)
3. Runs representation analysis with working/broken labels
4. Reports which layers show separation

This is useful for researching how models handle format differences, particularly the trailing space issue discovered in arithmetic prompts.

## Background

Some models behave differently depending on trailing whitespace:
- `"156 + 287 = "` (with space) -> correctly answers "443"
- `"156 + 287 ="` (no space) -> refuses or gives wrong answer

This tool helps identify which layers are responsible for this difference.

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--model` | `-m` | Model name or HuggingFace ID (required) |
| `--prompts` | `-p` | Base prompts (pipe-separated or @file.txt). Trailing space added/removed automatically (required) |
| `--layers` | | Layers to analyze (comma-separated). Default: auto |
| `--summary-only` | `-s` | Only show summary (skip detailed output) |

## Examples

### Quick format check

```bash
lazarus introspect format-sensitivity \
    -m mlx-community/gemma-3-4b-it-bf16 \
    -p "156 + 287 =|100 - 37 =|50 + 25 ="
```

### Summary only

```bash
lazarus introspect format-sensitivity \
    -m model \
    -p "prompt1|prompt2" \
    --summary-only
```

### Specific layers

```bash
lazarus introspect format-sensitivity \
    -m model \
    -p "prompt1|prompt2" \
    --layers 2,4,6,8,10,12
```

## Output

```
Format sensitivity analysis for mlx-community/gemma-3-4b-it-bf16
Testing 3 prompts with/without trailing space

=== Where Format Matters ===
  Layer 0: separation = 0.0012
  Layer 4: separation = 0.0523 ★
  Layer 8: separation = 0.0891 ★
  Layer 12: separation = 0.0234
  Layer 16: separation = 0.0156
  Layer 20: separation = 0.0089
```

The ★ marker indicates layers with significant separation (> 0.02).

## Python API

```python
from chuk_lazarus.introspection import analyze_format_sensitivity

result = analyze_format_sensitivity(
    model_id="mlx-community/gemma-3-4b-it-bf16",
    base_prompts=[
        "156 + 287 =",
        "100 - 37 =",
        "50 + 25 =",
    ],
    layers=[2, 4, 6, 8, 10, 12],
)

# Find where format matters most
for layer_idx in result.layers:
    if result.clusters and layer_idx in result.clusters:
        sep = result.clusters[layer_idx].separation_score
        if sep > 0.02:
            print(f"Layer {layer_idx}: format matters! (sep={sep:.4f})")
```

## Interpretation

- **High separation score** (> 0.02): The layer distinguishes between working and broken formats
- **Low separation score**: The layer treats both formats similarly

Layers with high separation are candidates for:
- Being where "format sensitivity" is computed
- Potential targets for activation steering to fix broken prompts

## See Also

- [introspect-layer.md](introspect-layer.md) - General layer representation analysis
- [introspect-analyze.md](introspect-analyze.md) - Logit lens analysis
- `docs/gemma_alignment_circuits.md` - Gemma circuit findings
