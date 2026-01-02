# lazarus introspect weight-diff

Compare weight differences between base and fine-tuned models.

## Synopsis

```bash
lazarus introspect weight-diff -b BASE -f FINETUNED [-o OUTPUT]
```

## Description

The `weight-diff` command computes per-layer, per-component relative weight differences between a base model and a fine-tuned version. This reveals where fine-tuning had the most impact.

## Options

| Option | Description |
|--------|-------------|
| `-b, --base MODEL` | Base model (required) |
| `-f, --finetuned MODEL` | Fine-tuned model (required) |
| `-o, --output FILE` | Save results to JSON |

## Examples

### Compare Base vs Fine-tuned

```bash
lazarus introspect weight-diff \
    -b google/gemma-3-1b \
    -f google/gemma-3-1b-it
```

### Save Results

```bash
lazarus introspect weight-diff \
    -b base-model \
    -f finetuned-model \
    -o weight_diff.json
```

## Output

```
Loading base model: google/gemma-3-1b
Loading fine-tuned model: google/gemma-3-1b-it
Detected model family: gemma

Comparing 16 layers...

Layer    Component    Rel. Diff
-----------------------------------
0        mlp_down        0.000123
0        attn_o          0.000089
1        mlp_down        0.000145
1        attn_o          0.000112
...
10       mlp_down        0.023456 ***
10       attn_o          0.015678 ***
11       mlp_down        0.045678 ***
11       attn_o          0.034567 ***
...

Top 5 divergent components:
  Layer 11 mlp_down: 0.045678
  Layer 11 attn_o: 0.034567
  Layer 10 mlp_down: 0.023456
  Layer 10 attn_o: 0.015678
  Layer 12 mlp_down: 0.012345
```

## Interpreting Results

- **Relative Diff**: L2 norm of weight difference / L2 norm of base weight
- **`***` marker**: Indicates divergence > 0.1 (10%)
- **Top divergent**: Layers/components with largest relative changes

## Use Cases

### Finding Fine-tuning Focus

See which layers were most modified during fine-tuning:

```bash
lazarus introspect weight-diff -b base -f finetuned
```

### Comparing LoRA Merged Models

After merging LoRA weights:

```bash
lazarus introspect weight-diff -b original -f lora-merged
```

## See Also

- [introspect activation-diff](introspect-activation-diff.md) - Compare activations
- [introspect compare](introspect-compare.md) - Compare predictions
