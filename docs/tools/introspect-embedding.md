# lazarus introspect embedding

Analyze what information is encoded at the embedding level before any layer computation.

## Synopsis

```bash
lazarus introspect embedding -m MODEL [OPTIONS]
```

## Description

The `embedding` command tests the RLVF backprop hypothesis: if RLVF gradients backprop to embeddings, we should find task-relevant information already encoded in the raw embeddings before any transformer layer computation.

Tests performed:
1. **Task type detection** - Can we classify arithmetic vs language from embeddings alone?
2. **Operation type detection** - Can we distinguish multiplication from addition?
3. **Answer correlation** - Is the numerical answer encoded in embeddings?

## Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | HuggingFace model ID or local path (required) |
| `-l, --layers LAYERS` | Layers to compare against (comma-separated, default: 0,1,2) |
| `--operation OP` | Operation to test: *, +, mult, add, all (default: all) |
| `-o, --output FILE` | Save results to JSON file |

## Examples

### Basic Embedding Analysis

```bash
lazarus introspect embedding -m openai/gpt-oss-20b
```

### Specific Operation

Focus on multiplication:

```bash
lazarus introspect embedding \
    -m model \
    --operation mult
```

### Extended Layer Comparison

Compare embeddings against more layers:

```bash
lazarus introspect embedding \
    -m model \
    --layers 0,1,2,4,8,12 \
    -o embedding_analysis.json
```

## Output

### Test 1: Task Type Detection

```
======================================================================
TEST 1: TASK TYPE DETECTION
======================================================================
Task type from embeddings: 100.0%
Task type after L0: 100.0%
Task type after L1: 100.0%
Task type after L2: 100.0%
```

### Test 2: Answer Correlation

```
======================================================================
TEST 2: ANSWER CORRELATION (arithmetic only)
======================================================================
Answer R2 from embeddings: 0.023
Answer R2 after L0: 0.156
Answer R2 after L1: 0.423
Answer R2 after L2: 0.678
```

### Test 3: Embedding Similarity

```
======================================================================
TEST 3: EMBEDDING SIMILARITY ANALYSIS
======================================================================
Within arithmetic similarity: 0.8234
Within language similarity: 0.7891
Between task similarity: 0.4567
```

### Interpretation

```
======================================================================
INTERPRETATION
======================================================================
Task type is BAKED INTO embeddings (100% detection)
  -> Consistent with RLVF backprop hypothesis

Answer NOT in embeddings (requires computation)
  -> Actual arithmetic happens in layers, not embeddings
```

## Key Findings

### What's in Embeddings vs Layers

| Information | Embeddings | After Layers | Interpretation |
|-------------|------------|--------------|----------------|
| Task type | 100% | 100% | Pre-computed via RLVF |
| Operation | ~95% | 100% | Mostly in embeddings |
| Operands | ~90% | 100% | Encoded in embeddings |
| Answer | <10% | >95% | Computed in layers |

### The RLVF Hypothesis

RLVF (Reinforcement Learning from Verifiable Feedback) may cause gradients to flow back to embeddings during training, "baking in" task-relevant information:

1. Model learns to distinguish task types early
2. Embedding layer learns task-specific representations
3. This reduces computation needed in transformer layers
4. Explains why task classification works at L0

## Use Cases

### Testing RLVF Impact

```bash
# Compare base model vs RLVF-tuned model
lazarus introspect embedding -m base-model -o base.json
lazarus introspect embedding -m rlvf-tuned-model -o tuned.json

# Tuned model should show higher task detection at embedding level
```

### Understanding Representation Timeline

```bash
# Combine with early-layers for full picture
lazarus introspect embedding -m model -o embeddings.json
lazarus introspect early-layers -m model -o early.json

# Embeddings: task type, operands
# Early layers: answer computation
```

### Cross-Task Analysis

```bash
# Test what information is pre-encoded for different task types
lazarus introspect embedding \
    -m model \
    --operation mult \
    -o mult_embeddings.json

lazarus introspect embedding \
    -m model \
    --operation add \
    -o add_embeddings.json
```

## Theoretical Background

### Why Check Embeddings?

Traditional view: Embeddings just encode tokens
New view: RLVF training may add task-relevant structure

If task type is 100% classifiable from embeddings:
- Information is pre-computed before any attention
- Suggests learned "routing" at embedding level
- Model "knows" what kind of problem before processing

### Similarity Patterns

| Pattern | Meaning |
|---------|---------|
| High within-task similarity | Task-specific embedding clusters |
| Low between-task similarity | Clear task separation |
| Arithmetic ≠ Language | Different representational structure |

## Saved Output Format

```json
{
  "model": "openai/gpt-oss-20b",
  "num_arith_prompts": 72,
  "num_lang_prompts": 8,
  "layers_analyzed": [0, 1, 2],
  "results": {
    "task_from_embedding": 1.0,
    "task_after_L0": 1.0,
    "task_after_L1": 1.0,
    "answer_r2_embedding": 0.023,
    "answer_r2_L0": 0.156,
    "answer_r2_L1": 0.423,
    "within_arith_sim": 0.8234,
    "within_lang_sim": 0.7891,
    "between_task_sim": 0.4567
  }
}
```

## See Also

- [introspect early-layers](introspect-early-layers.md) - Early layer computation analysis
- [introspect layer](introspect-layer.md) - Layer representation similarity
- [introspect probe](introspect-probe.md) - Linear probing for classification
- [Introspection Overview](../introspection.md) - Full module documentation
