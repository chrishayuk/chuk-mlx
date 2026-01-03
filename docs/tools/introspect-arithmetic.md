# lazarus introspect arithmetic

Run systematic arithmetic studies to find answer emergence layers.

## Synopsis

```bash
lazarus introspect arithmetic -m MODEL [OPTIONS]
```

## Description

The `arithmetic` command runs a comprehensive suite of arithmetic tests across all model layers to discover:

1. **Emergence layers** - At which layer does the correct answer first appear?
2. **Difficulty patterns** - Do harder problems emerge later?
3. **Operation differences** - Do multiplication and addition emerge at different layers?
4. **Magnitude effects** - Do larger numbers require more computation?

## Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | HuggingFace model ID or local path (required) |
| `--hard-only` | Only test hard problems (multi-digit multiplication) |
| `--easy-only` | Only test easy problems (single-digit, small results) |
| `--quick` | Run reduced test set (every 3rd test) |
| `--raw` | Skip chat template, use raw prompts |
| `-o, --output FILE` | Save results to JSON file |

## Examples

### Full Arithmetic Study

Run complete test suite across all operations:

```bash
lazarus introspect arithmetic -m openai/gpt-oss-20b
```

### Hard Problems Only

Focus on challenging multi-digit multiplication:

```bash
lazarus introspect arithmetic \
    -m openai/gpt-oss-20b \
    --hard-only \
    -o hard_results.json
```

### Quick Scan

Fast overview with reduced test set:

```bash
lazarus introspect arithmetic \
    -m model \
    --quick
```

## Output

### Test Progress

```
Loading model: openai/gpt-oss-20b
Model: openai/gpt-oss-20b
  Layers: 24
  Mode: CHAT

Running 156 arithmetic tests...

Problem          Expected  First@   Final@   Correct?
------------------------------------------------------
2+2=             4         L4       L24      yes
47*47=           2209      L18      L24      yes
100-37=          63        L8       L24      yes
```

### Summary Statistics

```
======================================================================
EMERGENCE LAYER ANALYSIS
======================================================================

By Operation:
  add: mean emergence L6.2, 100% accuracy
  mul: mean emergence L14.8, 95% accuracy
  sub: mean emergence L7.1, 98% accuracy
  div: mean emergence L12.3, 85% accuracy

By Difficulty:
  easy: mean emergence L5.1, 100% accuracy
  medium: mean emergence L10.4, 97% accuracy
  hard: mean emergence L16.2, 82% accuracy

By Magnitude:
  <10: mean emergence L4.2
  10-100: mean emergence L8.5
  100-1000: mean emergence L12.1
  >1000: mean emergence L17.3
```

### Interpretation

```
======================================================================
KEY FINDINGS
======================================================================

1. Addition emerges 8 layers before multiplication
2. Magnitude correlates with emergence layer (r=0.78)
3. Hard problems require 3x more layers than easy ones
4. 15% of hard multiplications never produce correct answer
```

## Test Categories

| Difficulty | Description | Examples |
|------------|-------------|----------|
| Easy | Single digit, small result | 2+2=, 3*3= |
| Medium | Multi-digit, moderate result | 45+67=, 12*8= |
| Hard | Multi-digit multiplication | 47*47=, 67*83= |

## Use Cases

### Comparing Models

```bash
# Test emergence on different model sizes
for model in small-model medium-model large-model; do
    lazarus introspect arithmetic -m $model -o ${model}_arithmetic.json
done
```

### Finding Computation Layers

Identify which layers perform actual arithmetic:

```bash
lazarus introspect arithmetic \
    -m model \
    --hard-only \
    -o emergence.json

# Then use the emergence layer for patching experiments
lazarus introspect patch \
    -m model \
    --source "7*8=" --target "7+8=" \
    --layer $(cat emergence.json | jq '.mean_emergence_layer')
```

## Saved Output Format

```json
{
  "model_id": "openai/gpt-oss-20b",
  "total_tests": 156,
  "by_operation": {
    "add": {"count": 36, "mean_emergence": 6.2, "accuracy": 1.0},
    "mul": {"count": 64, "mean_emergence": 14.8, "accuracy": 0.95}
  },
  "by_difficulty": {
    "easy": {"count": 50, "mean_emergence": 5.1, "accuracy": 1.0},
    "hard": {"count": 40, "mean_emergence": 16.2, "accuracy": 0.82}
  },
  "results": [
    {
      "prompt": "47*47=",
      "expected": "2209",
      "emergence_layer": 18,
      "final_correct": true,
      "difficulty": "hard",
      "operation": "mul"
    }
  ]
}
```

## See Also

- [introspect analyze](introspect-analyze.md) - General layer-by-layer analysis
- [introspect patch](introspect-patch.md) - Causal intervention experiments
- [introspect commutativity](introspect-commutativity.md) - Test A*B = B*A
- [Introspection Overview](../introspection.md) - Full module documentation
