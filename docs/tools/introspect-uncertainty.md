# lazarus introspect uncertainty

Detect model uncertainty using hidden state geometry.

## Synopsis

```bash
lazarus introspect uncertainty -m MODEL --prompts PROMPTS [OPTIONS]
```

## Description

The `uncertainty` command uses hidden state distance to "compute center" vs "refusal center" to predict whether the model is confident about an answer before generation.

This geometric approach:
1. Calibrates on known working and broken prompts
2. Computes centroids for each category
3. Classifies new prompts by distance to each centroid

Unlike entropy-based uncertainty, this method works at a specific layer before the full forward pass completes.

## Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | HuggingFace model ID or local path (required) |
| `-p, --prompts PROMPTS` | Test prompts (pipe-separated or @file.txt) (required) |
| `-l, --layer N` | Detection layer (default: 70% of depth) |
| `--working PROMPTS` | Calibration prompts that work (comma-separated) |
| `--broken PROMPTS` | Calibration prompts that fail (comma-separated) |
| `-o, --output FILE` | Save results to JSON file |

## Examples

### Basic Uncertainty Detection

```bash
lazarus introspect uncertainty \
    -m openai/gpt-oss-20b \
    --prompts "2+2=|47*47=|What is pi^10?"
```

### Custom Calibration

```bash
lazarus introspect uncertainty \
    -m model \
    --prompts "test prompts here" \
    --working "100 - 37 = ,50 + 25 = ,10 * 10 = " \
    --broken "100 - 37 =,50 + 25 =,10 * 10 ="
```

### Specific Layer

```bash
lazarus introspect uncertainty \
    -m model \
    --prompts @test_prompts.txt \
    --layer 18 \
    -o uncertainty_results.json
```

### Format Sensitivity Testing

```bash
# Compare with/without trailing space
lazarus introspect uncertainty \
    -m model \
    --prompts "100-37=|100-37= |47*47=|47*47= "
```

## Output

### Calibration Phase

```
Loading model: openai/gpt-oss-20b
  Layers: 24
  Detection layer: 17

Calibrating on 5 working + 5 broken examples...
  Compute-Refusal separation: 1234
  Calibration complete!
```

### Detection Results

```
================================================================================
UNCERTAINTY DETECTION RESULTS
================================================================================
Prompt                         Score   Prediction   ->Compute  ->Refusal
--------------------------------------------------------------------------------
100 - 37 =                      +856   CONFIDENT        1023       1879
100 - 37 =                      -234   UNCERTAIN        1567       1333
47*47=                          -156   UNCERTAIN        1489       1333
47*47= (with space)             +678   CONFIDENT        1123       1801
What is pi to 100 digits?       -890   UNCERTAIN        1890       1000
2+2=                            +1023  CONFIDENT         890       1913
--------------------------------------------------------------------------------
Summary: 3 confident, 3 uncertain
```

## Interpretation

### Score Meaning

| Score | Prediction | Meaning |
|-------|------------|---------|
| Positive (> 0) | CONFIDENT | Closer to "compute center" |
| Negative (< 0) | UNCERTAIN | Closer to "refusal center" |
| Large magnitude | Strong signal | Clear classification |
| Small magnitude | Weak signal | Borderline case |

### Distance Interpretation

- **->Compute**: Distance to centroid of working examples
- **->Refusal**: Distance to centroid of broken examples
- **Score** = Refusal distance - Compute distance

## Use Cases

### Pre-Generation Filtering

```bash
# Check uncertainty before expensive generation
lazarus introspect uncertainty \
    -m model \
    --prompts "complex question here"

# If UNCERTAIN, might want to:
# - Use chain-of-thought prompting
# - Request more context
# - Flag for human review
```

### Format Optimization

```bash
# Find which format the model prefers
lazarus introspect uncertainty \
    -m model \
    --prompts "100-37=|100-37 =|100 - 37=|100 - 37 ="

# The most CONFIDENT format is likely to produce best results
```

### Batch Quality Prediction

```bash
# Score a batch of prompts for expected quality
lazarus introspect uncertainty \
    -m model \
    --prompts @batch_prompts.txt \
    -o uncertainty_scores.json

# Sort by score to prioritize reliable results
```

### Model Comparison

```bash
# Compare uncertainty patterns across models
for model in small medium large; do
    lazarus introspect uncertainty \
        -m $model \
        --prompts "47*47=|67*83=|97*89=" \
        -o ${model}_uncertainty.json
done
```

## Theoretical Background

### Geometric Uncertainty

Traditional uncertainty measures (entropy, perplexity) require a full forward pass. Geometric uncertainty works differently:

1. **Working prompts** form a cluster in activation space
2. **Broken prompts** form a different cluster
3. New prompts are classified by proximity

### Why It Works

At ~70% network depth:
- Model has processed input and formed intent
- "Confident" states cluster together
- "Uncertain" states cluster separately
- Distance captures this separation

### Calibration Requirements

For best results, calibration prompts should:
- Be similar in structure to test prompts
- Include clear examples of both success and failure
- Cover the range of expected inputs

### Separation Score

The calibration reports "Compute-Refusal separation" - the distance between centroids. Higher separation means:
- Clearer distinction between states
- More reliable predictions
- Better calibration quality

## Saved Output Format

```json
{
  "model_id": "openai/gpt-oss-20b",
  "detection_layer": 17,
  "separation": 1234.5,
  "results": [
    {
      "prompt": "100 - 37 = ",
      "score": 856.3,
      "prediction": "CONFIDENT",
      "dist_to_compute": 1023.4,
      "dist_to_refusal": 1879.7
    },
    {
      "prompt": "100 - 37 =",
      "score": -234.1,
      "prediction": "UNCERTAIN",
      "dist_to_compute": 1567.2,
      "dist_to_refusal": 1333.1
    }
  ]
}
```

## See Also

- [introspect metacognitive](introspect-metacognitive.md) - Strategy detection (DIRECT vs CoT)
- [introspect probe](introspect-probe.md) - Train classification probes
- [introspect analyze](introspect-analyze.md) - Layer-by-layer analysis
- [Introspection Overview](../introspection.md) - Full module documentation
