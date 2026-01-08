# lazarus introspect logit-lens

Apply logit lens analysis to check if classifiers project to vocabulary tokens.

## Synopsis

```bash
lazarus introspect logit-lens -m MODEL --prompts PROMPTS [OPTIONS]
```

## Description

The `logit-lens` command projects hidden states at intermediate layers through the unembedding matrix to see which vocabulary tokens emerge. This checks whether internal representations are "vocabulary-mappable" - i.e., can be interpreted as specific tokens.

**Important**: Logit lens often gives **false negatives** for operation classifiers! Models have classifiers that exist in hidden state space but don't project to vocabulary tokens. Use `introspect classifier` for accurate detection.

## Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | HuggingFace model ID or local path (required) |
| `-p, --prompts PROMPTS` | Prompts to analyze (pipe-separated or @file.txt) (required) |
| `-l, --layer N` | Specific layer to analyze (default: 55% depth) |
| `-t, --targets TOKEN` | Target tokens to check probability for (repeatable) |
| `-o, --output FILE` | Save results to JSON file |

### Prompts Format

```
--prompts "prompt1|prompt2|prompt3"
```

Or load from file:
```
--prompts @prompts.txt
```

## Examples

### Check for Operation Classifiers

```bash
lazarus introspect logit-lens -m meta-llama/Llama-3.2-1B \
  --prompts "7 * 8 = |12 * 5 = |23 + 45 = |17 + 38 = " \
  --targets "multiply" \
  --targets "add" \
  --targets "subtract" \
  --targets "divide"
```

### Check Specific Layer

```bash
lazarus introspect logit-lens -m meta-llama/Llama-3.2-1B \
  --prompts "7 * 8 = |23 + 45 = " \
  --layer 10 \
  --targets "multiply" \
  --targets "add"
```

### Save Results

```bash
lazarus introspect logit-lens -m model \
  --prompts "7 * 8 = |23 + 45 = |50 - 23 = |48 / 6 = " \
  --targets "multiply" --targets "add" --targets "subtract" --targets "divide" \
  --output results/logit_lens.json
```

## Output

```
Loading model: meta-llama/Llama-3.2-1B
  Layers: 16
  Target layer: L8 (50% depth)

Analyzing 4 prompts at layer L8
Target tokens: ['multiply', 'add', 'subtract', 'divide']

================================================================================
Prompt                    Top Token           Prob Target Probs
--------------------------------------------------------------------------------
  7 * 8 =                 ' palindrome'     2.04%  multiply:0.0% | add:0.0% | subtract:0.0% | divide:0.0%
  12 * 5 =                ' palindrome'     1.86%  multiply:0.0% | add:0.0% | subtract:0.0% | divide:0.0%
  23 + 45 =               'orex'            1.70%  multiply:0.0% | add:0.0% | subtract:0.0% | divide:0.0%
  50 - 23 =               'ặn'              5.22%  multiply:0.0% | add:0.0% | subtract:0.0% | divide:0.0%
--------------------------------------------------------------------------------

SUMMARY: Checking for classifier tokens at L8
  multiply: 0/4 prompts have this as top token
  add: 0/4 prompts have this as top token
  subtract: 0/4 prompts have this as top token
  divide: 0/4 prompts have this as top token
```

## Interpreting Results

### 0% for all target tokens (common case)
- **Does NOT mean classifiers don't exist!**
- Classifiers exist in hidden state space but don't map to vocabulary
- Use `introspect classifier` with linear probes for accurate detection

### Target tokens appearing with high probability
- Classifier has been trained to project to vocabulary
- This happens after dual-reward training or specific fine-tuning

### Top tokens are random/garbage (e.g., 'palindrome', 'orex', 'MZQ')
- Normal for untrained models at intermediate layers
- These are artifacts of projecting internal representations to vocabulary space

## Why Logit Lens Often Fails

Logit lens assumes that intermediate representations can be meaningfully projected to vocabulary space. This works for:
- Next-token predictions at late layers
- Some syntactic features

But it fails for **abstract task classifiers** because:
1. Operations like "multiply" exist as directions in hidden space
2. These directions are NOT aligned with vocabulary embeddings
3. The model never needs to OUTPUT "multiply" - it just needs to route computation

## JSON Output Format

```json
{
  "model": "meta-llama/Llama-3.2-1B",
  "layer": 8,
  "num_layers": 16,
  "results": [
    {
      "prompt": "7 * 8 = ",
      "top_token": " palindrome",
      "top_prob": 0.0204,
      "target_probs": {
        "multiply": 0.0,
        "add": 0.0,
        "subtract": 0.0,
        "divide": 0.0
      }
    },
    ...
  ]
}
```

## Use Cases

### Verify Dual-Reward Training

After training with `introspect dual-reward`, use logit-lens to verify that classifiers now project to vocabulary:

```bash
# Before training: 0% target tokens
lazarus introspect logit-lens -m model --prompts "7 * 8 = " --targets "multiply"

# After training: should see high probability for "multiply"
lazarus introspect logit-lens -m model --adapter checkpoint --prompts "7 * 8 = " --targets "multiply"
```

### Baseline Measurement

Establish baseline before any training:

```bash
lazarus introspect logit-lens -m meta-llama/Llama-3.2-1B \
  --prompts "7 * 8 = |23 + 45 = |50 - 23 = |48 / 6 = " \
  --targets "multiply" --targets "add" --targets "subtract" --targets "divide" \
  --output baseline_logit_lens.json
```

## See Also

- [introspect classifier](introspect-classifier.md) - Multi-class linear probes (accurate detection)
- [introspect dual-reward](introspect-dual-reward.md) - Train vocabulary projection
- [introspect analyze](introspect-analyze.md) - General logit lens analysis
- [Introspection Overview](../introspection.md) - Full module documentation
