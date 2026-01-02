# lazarus introspect compare

Compare how two models process the same prompt.

## Synopsis

```bash
lazarus introspect compare -m1 MODEL1 -m2 MODEL2 -p PROMPT [OPTIONS]
```

## Description

The `compare` command runs logit lens analysis on two models with the same prompt, showing side-by-side how their predictions differ. Useful for comparing base vs fine-tuned models.

## Options

| Option | Description |
|--------|-------------|
| `-m1, --model1 MODEL` | First model (required) |
| `-m2, --model2 MODEL` | Second model (required) |
| `-p, --prompt PROMPT` | Prompt to analyze (required) |
| `--top-k N` | Show top N predictions (default: 5) |
| `--track TOKENS` | Comma-separated tokens to track |

## Examples

### Compare Base vs Fine-tuned

```bash
lazarus introspect compare \
    -m1 google/gemma-3-1b \
    -m2 google/gemma-3-1b-it \
    -p "The answer is"
```

### Track Token Evolution

```bash
lazarus introspect compare \
    -m1 base-model \
    -m2 finetuned-model \
    -p "What's the weather?" \
    --track "get_weather,weather"
```

## Output

```
Loading: google/gemma-3-1b
Loading: google/gemma-3-1b-it

======================================================================
Prompt: 'The answer is'
======================================================================

=== Final Predictions ===
Model                                    Top Token        Prob
-----------------------------------------------------------------
google/gemma-3-1b                        42               0.1234
google/gemma-3-1b-it                     :                0.0891

=== Token Evolution Comparison ===

Token '42':
  google/gemma-3-1b                    : emerges at layer 12, final prob 0.1234
  google/gemma-3-1b-it                 : emerges at layer 18, final prob 0.0456
```

## Use Cases

### Understanding Fine-tuning Effects

Compare where predictions diverge between base and fine-tuned:

```bash
lazarus introspect compare \
    -m1 google/gemma-3-270m-it \
    -m2 google/functiongemma-270m-it \
    -p "Get the weather" \
    --track "get_weather,function"
```

### Comparing Model Sizes

See how different model sizes process the same prompt:

```bash
lazarus introspect compare \
    -m1 TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    -m2 meta-llama/Llama-3.2-3B-Instruct \
    -p "The capital of France is"
```

## Python API

```python
from chuk_lazarus.introspection import ModelAnalyzer, AnalysisConfig

models = ["model1", "model2"]
results = {}

for model_id in models:
    async with ModelAnalyzer.from_pretrained(model_id) as analyzer:
        config = AnalysisConfig(track_tokens=["token1", "token2"])
        result = await analyzer.analyze(prompt, config)
        results[model_id] = result

# Compare emergence layers
for model_id, result in results.items():
    for evo in result.token_evolutions:
        print(f"{model_id}: '{evo.token}' emerges at layer {evo.emergence_layer}")
```

## See Also

- [introspect analyze](introspect-analyze.md) - Single model analysis
- [introspect weight-diff](introspect-weight-diff.md) - Weight comparison
- [introspect activation-diff](introspect-activation-diff.md) - Activation comparison
