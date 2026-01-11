# lazarus introspect metacognitive

Detect whether the model will use direct computation or chain-of-thought reasoning.

## Synopsis

```bash
lazarus introspect metacognitive -m MODEL --problems PROBLEMS [OPTIONS]
```

## Description

The `metacognitive` command probes the model's "decision layer" (typically ~70% through the network) to detect the model's strategy before generation:

- **Direct computation**: Decision layer predicts a digit → answer comes immediately
- **Chain-of-thought (CoT)**: Decision layer predicts ' ', 'To', 'Let' → reasoning first

The key insight is that token IDENTITY at the decision layer reveals the model's strategy, not just confidence. A digit token means "I know the answer", while a non-digit means "I need to think about this".

## Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | HuggingFace model ID or local path (required) |
| `-p, --problems PROBLEMS` | Problems to analyze (pipe-separated or @file.txt) (required) |
| `--decision-layer N` | Layer to probe (default: 70% of depth) |
| `--generate` | Generate random arithmetic problems |
| `--num-problems N` | Number of problems to generate (default: 20) |
| `--seed N` | Random seed for generation |
| `--raw` | Skip chat template |
| `-o, --output FILE` | Save results to JSON file |

## Examples

### Basic Strategy Detection

```bash
lazarus introspect metacognitive \
    -m openai/gpt-oss-20b \
    --problems "2+2=|47*47=|What is 7^13?"
```

### Generate Random Problems

```bash
lazarus introspect metacognitive \
    -m model \
    --generate \
    --num-problems 50 \
    --seed 42
```

### Custom Decision Layer

```bash
lazarus introspect metacognitive \
    -m model \
    --problems "67*83=|97*89=" \
    --decision-layer 18
```

### From File

```bash
echo -e "2+2=\n47*47=\n67*83=\n97*89=" > problems.txt
lazarus introspect metacognitive \
    -m model \
    --problems @problems.txt \
    -o strategy_analysis.json
```

## Output

### Problem-by-Problem Analysis

```
Loading model: openai/gpt-oss-20b
Model: openai/gpt-oss-20b
  Layers: 24
  Decision layer: 17 (~71% depth)
  Mode: CHAT

Analyzing 5 problems...

==========================================================================================
Prompt                    L17 Top      Prob   Strategy     Digit? Match?
------------------------------------------------------------------------------------------
2+2=                      '4'          0.98   DIRECT       yes    [correct]
10*10=                    '1'          0.95   DIRECT       yes    [correct]
47*47=                    ' '          0.67   CoT          no     [unknown]
67*83=                    'To'         0.54   CoT          no     [unknown]
What is pi^10?            'I'          0.89   CoT          no     [unknown]
```

### Strategy Distribution

```
======================================================================
STRATEGY DISTRIBUTION
======================================================================
  DIRECT: 2 (40.0%)
  CoT:    3 (60.0%)

Direct answer accuracy: 2/2 (100.0%)
```

### Confidence Analysis

```
======================================================================
CONFIDENCE ANALYSIS
======================================================================
  DIRECT avg confidence: 0.965
  CoT avg confidence:    0.700
```

### Pattern Analysis

```
======================================================================
PATTERN ANALYSIS (Multiplication)
======================================================================
  Multiplication: 1 direct, 2 CoT
  Squares (n*n): 1/1 direct (47*47 uses CoT despite being square)
```

## Strategy Tokens

| Decision Layer Token | Strategy | Interpretation |
|---------------------|----------|----------------|
| `'0'`-`'9'` | DIRECT | Model will output answer immediately |
| `' '` (space) | CoT | Model will add thinking space |
| `'To'`, `'Let'`, `'First'` | CoT | Model will reason step-by-step |
| `'I'`, `'Well'` | CoT | Model will use conversational reasoning |

## Use Cases

### Difficulty Threshold Detection

Find where the model switches from direct to CoT:

```bash
# Test range of difficulties
lazarus introspect metacognitive \
    -m model \
    --problems "2*2=|5*5=|10*10=|15*15=|20*20=|30*30=|50*50=|99*99="
```

### Model Comparison

Compare strategy patterns across models:

```bash
for model in small medium large; do
    lazarus introspect metacognitive \
        -m $model \
        --generate --num-problems 100 \
        -o ${model}_strategy.json
done
```

### Predict Generation Mode

Before expensive generation, check what strategy the model will use:

```bash
# Quick strategy check
lazarus introspect metacognitive \
    -m model \
    --problems "your complex question here"

# If CoT, expect longer generation with reasoning
# If DIRECT, expect short answer
```

## Theoretical Background

### The Decision Layer

At approximately 70% network depth, the model has:
1. Processed all input information
2. Decided on output strategy
3. Not yet committed to specific tokens

This layer reveals the model's "metacognitive" decision about HOW to answer.

### Strategy Selection

The model appears to use uncertainty to select strategy:
- **High certainty** → Direct answer (digit token)
- **Low certainty** → Chain-of-thought (reasoning tokens)

This is adaptive: easy problems get fast answers, hard problems get careful reasoning.

### Why Token Identity Matters

Unlike entropy or probability, token IDENTITY is categorical:
- A digit means "I know a specific number"
- A space/word means "I need to think"

This binary signal is robust even when probabilities are similar.

## Saved Output Format

```json
{
  "model": "openai/gpt-oss-20b",
  "decision_layer": 17,
  "total_problems": 5,
  "direct_count": 2,
  "cot_count": 3,
  "results": [
    {
      "problem": "2+2=",
      "expected": "4",
      "generated": "4",
      "decision_layer": 17,
      "decision_token": "4",
      "decision_prob": 0.98,
      "strategy": "DIRECT",
      "is_digit": true,
      "correct_start": true,
      "final_token": "4",
      "final_prob": 0.99
    }
  ]
}
```

## See Also

- [introspect uncertainty](introspect-uncertainty.md) - Uncertainty detection via geometry
- [introspect analyze](introspect-analyze.md) - Layer-by-layer analysis
- [introspect arithmetic](introspect-arithmetic.md) - Systematic arithmetic testing
- [Introspection Overview](../introspection.md) - Full module documentation
