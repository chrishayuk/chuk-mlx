# lazarus introspect early-layers

Analyze what information is encoded in early transformer layers.

## Synopsis

```bash
lazarus introspect early-layers -m MODEL [OPTIONS]
```

## Description

The `early-layers` command probes what the model has "computed" at each early layer by testing whether linear probes can extract:

- **Operation type** (*, +, -)
- **Operand values** (A and B)
- **The final answer**

Key insight: Even when hidden states look similar (high cosine similarity), information can be encoded in orthogonal subspaces. This command reveals when different pieces of information become linearly extractable.

## Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | HuggingFace model ID or local path (required) |
| `-l, --layers LAYERS` | Layers to analyze (comma-separated, default: 0,1,2,4,8) |
| `--operations OPS` | Operations to test (comma-separated, default: *,+,-) |
| `--digits RANGE` | Digit range (e.g., "2-8" or "2,3,5,7") |
| `--analyze-positions` | Include position-wise analysis |
| `-o, --output FILE` | Save results to JSON file |

## Examples

### Basic Early Layer Analysis

```bash
lazarus introspect early-layers -m openai/gpt-oss-20b
```

### Custom Layer Range

Focus on specific layers:

```bash
lazarus introspect early-layers \
    -m model \
    --layers 0,1,2,3,4,6,8,12
```

### Specific Operations

Test only multiplication:

```bash
lazarus introspect early-layers \
    -m model \
    --operations "*" \
    --digits 2-9
```

### Position Analysis

Include token position analysis:

```bash
lazarus introspect early-layers \
    -m model \
    --analyze-positions \
    -o early_layer_analysis.json
```

## Output

### Part 1: Representation Similarity

```
======================================================================
PART 1: REPRESENTATION SIMILARITY
======================================================================
How similar are different expressions at the '=' position?

Sample expressions: ['2*3=', '2+3=', '2-3=']

Layer    2*3= vs 2+3=        2*3= vs 2-3=
-------------------------------------------------
L0       0.9823              0.9756
L1       0.9712              0.9689
L2       0.9534              0.9501
L4       0.8923              0.8867
L8       0.7234              0.7156
```

### Part 2: Information Extractability

```
======================================================================
PART 2: INFORMATION EXTRACTABILITY (Linear Probes)
======================================================================
What can a linear probe extract at each layer?

Layer    Op Acc       A R2         B R2         Answer R2
--------------------------------------------------------
L0       85.2%        0.923        0.918        0.234
L1       97.3%        0.956        0.951        0.456
L2       100.0%       0.978        0.972        0.678
L4       100.0%       0.989        0.985        0.891
L8       100.0%       0.995        0.993        0.978
```

### Part 3: Position-wise Analysis (if enabled)

```
======================================================================
PART 3: POSITION-WISE ANALYSIS
======================================================================

Sample: '2*3=' -> ['2', '*', '3', '=']

Layer 0 - position similarities:
              '2'       '*'       '3'       '='
'2'         1.000     0.234     0.456     0.123
'*'         0.234     1.000     0.345     0.234
'3'         0.456     0.345     1.000     0.234
'='         0.123     0.234     0.234     1.000
```

### Interpretation

```
======================================================================
INTERPRETATION
======================================================================

Answer becomes extractable (R2 > 0.95) at layer 4
! Computation mostly complete by layer 4 (R2 = 0.891)
  -> Later layers may be formatting/output, not computation

! PARADOX at layer 0:
  - Representations look similar (avg cosine = 0.975)
  - But answer is extractable (R2 = 0.234)
  -> Information encoded in ORTHOGONAL subspaces
```

## Key Insights

### The Similarity-Extractability Paradox

High cosine similarity doesn't mean information is absent:

| Layer | Cosine Sim | Answer R2 | What it means |
|-------|------------|-----------|---------------|
| L0 | 0.98 | 0.23 | Similar overall, but answer partially encoded |
| L4 | 0.89 | 0.89 | More distinct, answer well encoded |
| L8 | 0.72 | 0.98 | Very distinct, answer fully computed |

### Information Timeline

Typical pattern for arithmetic:

1. **L0-L1**: Operation type becomes classifiable (100%)
2. **L1-L2**: Operands fully extractable (R2 > 0.95)
3. **L4-L8**: Answer emerges (R2 > 0.95)
4. **L8+**: Answer formatting and output preparation

## Use Cases

### Finding Computation Layers

```bash
lazarus introspect early-layers \
    -m model \
    --layers 0,2,4,6,8,10,12,14,16 \
    --operations "*" \
    -o computation_layers.json
```

### Comparing Model Architectures

```bash
for model in gemma llama qwen; do
    lazarus introspect early-layers \
        -m $model \
        -o ${model}_early.json
done
```

### Understanding Information Flow

```bash
# Detailed analysis with positions
lazarus introspect early-layers \
    -m model \
    --layers 0,1,2,3,4 \
    --analyze-positions \
    -o information_flow.json
```

## Saved Output Format

```json
{
  "model": "openai/gpt-oss-20b",
  "layers": [0, 1, 2, 4, 8],
  "operations": ["*", "+", "-"],
  "digits": [2, 3, 4, 5, 6, 7],
  "num_prompts": 108,
  "similarity_results": {
    "0": [0.982, 0.976],
    "4": [0.892, 0.887]
  },
  "probe_results": {
    "0": {"op_accuracy": 0.852, "a_r2": 0.923, "b_r2": 0.918, "answer_r2": 0.234},
    "4": {"op_accuracy": 1.0, "a_r2": 0.989, "b_r2": 0.985, "answer_r2": 0.891}
  }
}
```

## See Also

- [introspect embedding](introspect-embedding.md) - Embedding-level analysis
- [introspect layer](introspect-layer.md) - Layer representation similarity
- [introspect probe](introspect-probe.md) - Linear probing for classification
- [Introspection Overview](../introspection.md) - Full module documentation
