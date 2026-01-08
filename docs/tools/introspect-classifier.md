# lazarus introspect classifier

Train multi-class linear probes to detect operation classifiers at all layers.

## Synopsis

```bash
lazarus introspect classifier -m MODEL --classes LABEL:PROMPTS [OPTIONS]
```

## Description

The `classifier` command trains multi-class logistic regression probes at each layer to detect if the model has internal representations that distinguish between different types of operations (e.g., multiply, add, subtract, divide).

This is the **correct way** to detect operation classifiers, as they exist in hidden state space but typically don't map to vocabulary tokens (which logit lens would miss).

## Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | HuggingFace model ID or local path (required) |
| `-c, --classes LABEL:PROMPTS` | Class definition (repeatable, required) |
| `-t, --test PROMPTS` | Test prompts for predictions |
| `-o, --output FILE` | Save results to JSON file |

### Class Definition Format

```
--classes "label:prompt1|prompt2|prompt3"
```

- `label`: The class name (e.g., "multiply", "add")
- `prompt1|prompt2|...`: Pipe-separated list of prompts for this class
- Use `@file.txt` to load prompts from a file (one per line)

## Examples

### Detect Arithmetic Operation Classifiers

```bash
lazarus introspect classifier -m meta-llama/Llama-3.2-1B \
  --classes "multiply:7 * 8 = |12 * 5 = |3 * 9 = |6 * 7 = " \
  --classes "add:23 + 45 = |17 + 38 = |11 + 22 = |5 + 9 = " \
  --classes "subtract:50 - 23 = |89 - 34 = |77 - 11 = |40 - 15 = " \
  --classes "divide:48 / 6 = |81 / 9 = |36 / 4 = |24 / 3 = " \
  --test "11 * 12 = |6 * 9 = |13 + 14 = |25 + 17 = "
```

### Save Results to JSON

```bash
lazarus introspect classifier -m meta-llama/Llama-3.2-1B \
  --classes "multiply:7 * 8 = |12 * 5 = " \
  --classes "add:23 + 45 = |17 + 38 = " \
  --output results/llama_classifier.json
```

### Load Prompts from Files

```bash
# Create files
echo -e "7 * 8 = \n12 * 5 = \n3 * 9 = " > multiply.txt
echo -e "23 + 45 = \n17 + 38 = " > add.txt

lazarus introspect classifier -m model \
  --classes "multiply:@multiply.txt" \
  --classes "add:@add.txt"
```

## Output

```
Loading model: meta-llama/Llama-3.2-1B
  Layers: 16

Classes defined: 4
  multiply: 4 prompts
  add: 4 prompts
  subtract: 4 prompts
  divide: 4 prompts

Collecting activations...
Training multi-class probes at each layer...

======================================================================
MULTI-CLASS PROBE ACCURACY (4 classes)
======================================================================
Layer    Accuracy     Std        Bar
----------------------------------------------------------------------
  L0     1.000        0.000      ################################################## <- BEST
  L1     1.000        0.000      ##################################################
  L2     1.000        0.000      ##################################################
  ...
  L15    1.000        0.000      ##################################################
----------------------------------------------------------------------

Best layer: L0 (accuracy: 100.0%)

======================================================================
TEST PREDICTIONS
======================================================================
  11 * 12 =                                -> multiply (30.4%)
  6 * 9 =                                  -> multiply (31.5%)
  13 + 14 =                                -> add (27.6%)
  25 + 17 =                                -> add (27.6%)
```

## Interpreting Results

### High accuracy at ALL layers (e.g., Llama 3.2)
- Model has strong, persistent operation classifiers
- Classification signal established at embedding level
- Preserved throughout all transformer layers

### High accuracy at early layers, dropping at mid-layers (e.g., Granite)
- Model has classifiers but signal gets mixed during computation
- May indicate different architectural processing patterns

### Low accuracy everywhere
- Model may not have developed operation classifiers
- Try different prompt formats

## JSON Output Format

```json
{
  "model": "meta-llama/Llama-3.2-1B",
  "num_layers": 16,
  "classes": {
    "multiply": ["7 * 8 = ", "12 * 5 = ", "3 * 9 = ", "6 * 7 = "],
    "add": ["23 + 45 = ", "17 + 38 = ", "11 + 22 = ", "5 + 9 = "],
    ...
  },
  "layer_results": [
    {"layer": 0, "accuracy": 1.0, "std": 0.0},
    {"layer": 1, "accuracy": 1.0, "std": 0.0},
    ...
  ],
  "best_layer": 0,
  "best_accuracy": 1.0
}
```

## Key Findings

1. **Logit lens gives false negatives** - Classifiers exist but don't map to vocabulary tokens
2. **Linear probes reveal the truth** - 100% accuracy proves classifiers exist
3. **Classification is layer-agnostic** for Llama models - signal uniform across all layers

## See Also

- [introspect logit-lens](introspect-logit-lens.md) - Check vocabulary-mappable classifiers
- [introspect dual-reward](introspect-dual-reward.md) - Train vocabulary projection
- [introspect probe](introspect-probe.md) - Binary classification probes
- [Introspection Overview](../introspection.md) - Full module documentation
