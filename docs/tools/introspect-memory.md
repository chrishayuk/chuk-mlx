# lazarus introspect memory / memory-inject

Analyze and manipulate how facts are stored in model memory.

## Synopsis

```bash
# Analyze memory structure
lazarus introspect memory -m MODEL [OPTIONS]

# Inject facts into memory
lazarus introspect memory-inject -m MODEL --fact FACT [OPTIONS]
```

## Description

### introspect memory

Analyzes how facts are stored in model memory by examining neighborhood activation patterns - what other facts co-activate when retrieving a specific fact.

Reveals:
- **Memory organization** (row vs column based, clusters)
- **Asymmetry** (A->B vs B->A retrieval differences)
- **Attractor nodes** (frequently co-activated facts)
- **Difficulty patterns** (which facts are hardest)

### introspect memory-inject

Injects new facts into the model's memory by finding and modifying the key-value associations in MLP layers.

## Options

### memory

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | HuggingFace model ID or local path (required) |
| `--facts TYPE` | Fact type: multiplication, addition, capitals (default: multiplication) |
| `-l, --layer N` | Layer to analyze (default: auto-select) |
| `--top-k N` | Number of top neighbors to show (default: 5) |
| `-o, --output FILE` | Save results to JSON file |

### memory-inject

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | HuggingFace model ID or local path (required) |
| `--fact QUERY=ANSWER` | Fact to inject (e.g., "7*8=42") |
| `-l, --layer N` | Layer to inject at (default: auto-select) |
| `--strength FLOAT` | Injection strength (default: 1.0) |
| `--test PROMPTS` | Test prompts after injection |
| `-o, --output FILE` | Save results to JSON file |

## Examples

### Analyze Multiplication Memory

```bash
lazarus introspect memory \
    -m openai/gpt-oss-20b \
    --facts multiplication
```

### Analyze Capital Cities

```bash
lazarus introspect memory \
    -m model \
    --facts capitals \
    -l 12 \
    -o capital_memory.json
```

### Inject a Fake Fact

```bash
lazarus introspect memory-inject \
    -m model \
    --fact "7*8=42" \
    --test "7*8=|8*7=|7*9=" \
    -o injection_result.json
```

### Compare Row vs Column Organization

```bash
lazarus introspect memory \
    -m model \
    --facts multiplication \
    --top-k 10 \
    -o memory_structure.json
```

## Output (memory)

### Neighborhood Analysis

```
======================================================================
MEMORY NEIGHBORHOOD ANALYSIS
======================================================================
Analyzing 64 multiplication facts at layer 12

Query: 7*8=
Expected: 56
Top 5 neighbors:
  1. 7*9= (sim: 0.923) [same row]
  2. 7*7= (sim: 0.912) [same row]
  3. 8*7= (sim: 0.908) [commutative]
  4. 7*6= (sim: 0.901) [same row]
  5. 6*8= (sim: 0.856) [same column]
```

### Organization Statistics

```
======================================================================
MEMORY ORGANIZATION
======================================================================
Same-row similarity:    0.912 ± 0.034
Same-column similarity: 0.834 ± 0.056
Cross similarity:       0.567 ± 0.123

Organization: ROW-DOMINANT
  Facts cluster by first operand (7*2, 7*3, 7*4...)
  rather than by second operand (...*8)
```

### Asymmetry Analysis

```
======================================================================
ASYMMETRY ANALYSIS
======================================================================
A*B → B*A similarity: 0.908 ± 0.023
B*A → A*B similarity: 0.907 ± 0.024

Asymmetry score: 0.001 (symmetric)
```

### Attractor Nodes

```
======================================================================
ATTRACTOR NODES
======================================================================
Facts that frequently appear as neighbors:

  5*5= appears in 23/64 neighborhoods (attractor strength: 0.359)
  2*2= appears in 19/64 neighborhoods (attractor strength: 0.297)
  10*10= appears in 18/64 neighborhoods (attractor strength: 0.281)
```

## Output (memory-inject)

```
======================================================================
FACT INJECTION
======================================================================
Injecting: 7*8=42 at layer 12

Before injection:
  7*8= → 56 (correct)
  8*7= → 56 (correct)

After injection:
  7*8= → 42 (injected)
  8*7= → 56 (unchanged - not commutative injection)

Injection strength: 1.0
Affected layers: [12]
```

## Use Cases

### Understanding Memory Structure

```bash
# Compare organization across models
for model in small medium large; do
    lazarus introspect memory \
        -m $model \
        --facts multiplication \
        -o ${model}_memory.json
done
```

### Finding Difficult Facts

```bash
# Facts with unusual neighborhood patterns may be harder
lazarus introspect memory \
    -m model \
    --facts multiplication \
    -o memory.json

# Check which facts have low self-similarity or many distant neighbors
```

### Testing Memory Manipulation

```bash
# Inject and test a counterfactual
lazarus introspect memory-inject \
    -m model \
    --fact "France capital=Berlin" \
    --test "The capital of France is|Paris is the capital of|Berlin is in"
```

## Theoretical Background

### Row vs Column Organization

For multiplication facts, the model might organize by:
- **Row**: 7*2, 7*3, 7*4... cluster together (first operand)
- **Column**: 2*7, 3*7, 4*7... cluster together (second operand)
- **Neither**: Facts organized by answer or other features

### Attractor Nodes

Some facts appear frequently in neighborhoods because:
- They're "central" in representation space
- They share features with many other facts
- They may be retrieval anchors

### Memory Injection Theory

MLP layers store key-value associations. Injection works by:
1. Finding the key vector for the query
2. Modifying the value to produce desired output
3. Strength controls how much original value is preserved

## Saved Output Format (memory)

```json
{
  "model_id": "openai/gpt-oss-20b",
  "layer": 12,
  "fact_type": "multiplication",
  "num_facts": 64,
  "organization": {
    "type": "row-dominant",
    "same_row_sim": 0.912,
    "same_col_sim": 0.834,
    "cross_sim": 0.567
  },
  "asymmetry": 0.001,
  "attractors": [
    {"fact": "5*5=", "strength": 0.359},
    {"fact": "2*2=", "strength": 0.297}
  ],
  "neighborhoods": {
    "7*8=": {
      "expected": "56",
      "neighbors": [
        {"fact": "7*9=", "similarity": 0.923, "relation": "same_row"}
      ]
    }
  }
}
```

## See Also

- [introspect commutativity](introspect-commutativity.md) - Test A*B = B*A
- [introspect patch](introspect-patch.md) - Activation patching
- [introspect arithmetic](introspect-arithmetic.md) - Arithmetic testing
- [Introspection Overview](../introspection.md) - Full module documentation
