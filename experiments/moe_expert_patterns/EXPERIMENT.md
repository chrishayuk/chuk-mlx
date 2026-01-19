# MoE Expert Specialization Analysis

## Research Question

**Do experts in Mixture-of-Experts (MoE) models specialize for specific linguistic patterns and semantic relationships?**

This experiment analyzes expert routing patterns to understand how MoE models allocate their experts across different types of tokens, syntactic structures, and semantic relationships.

## Background

### Mixture of Experts

MoE models route different tokens to different "expert" sub-networks. Unlike dense models where all parameters process all inputs, MoE models selectively activate only a subset of experts per token.

Key questions:
- Do experts specialize by token type (numbers, keywords, punctuation)?
- Do experts handle specific semantic patterns (synonyms, antonyms, analogies)?
- How does specialization evolve across layers?

### Token Type Classification

The experiment classifies tokens into categories:

| Type | Description | Examples |
|------|-------------|----------|
| NUM | Numbers | "42", "3.14" |
| KW | Keywords | "def", "class", "function" |
| OP | Operators | "+", "-", "==", "+=" |
| BR | Brackets | "(", ")", "[", "]" |
| PN | Punctuation | ".", ",", ";" |
| FW | Function words | "the", "a", "is", "of" |
| CAP | Capitalized | "Janet", "Paris" |
| CW | Content words | "apple", "happy" |

## Running the Experiment

```bash
# Run via framework
lazarus experiment run expert_analysis

# View results
lazarus experiment status expert_analysis

# Or run directly (model override supported)
python experiments/expert_analysis/experiment.py [model_name]
```

**Note**: This experiment requires an MoE model like Mixtral-8x7B (~90GB).

## Configuration

See `config.yaml` for parameters:

```yaml
model: mistralai/Mixtral-8x7B-v0.1

analyses:
  - pattern_summary     # Token type trigram patterns
  - semantic_patterns   # Semantic relationship patterns
  - sequence_patterns   # N-gram sequence patterns
  - combined_analysis   # Combined pattern analysis

parameters:
  num_prompts: 100              # Prompts to analyze
  min_activity: 5               # Min expert activations to count
  specialist_threshold: 0.15    # Concentration for "specialist"
```

## Method

### Pattern Analysis

The experiment uses **trigram patterns** to characterize expert specialization:

```
Token sequence:  "x" "+" "3"
Types:           VAR  OP  NUM
Trigrams:        ^→VAR→OP  VAR→OP→NUM  OP→NUM→$
```

For each expert, we count which trigrams it handles and compute **concentration** - how much of an expert's activity is on its top pattern.

### Semantic Analysis

Semantic prompts test expert handling of linguistic relationships:

```yaml
synonyms:
  - "Happy means the same as joyful."
  - "Big is similar to large."

antonyms:
  - "Hot is the opposite of cold."
  - "Up is contrary to down."

analogies:
  - "King is to queen as man is to woman."
  - "Hot is to cold as up is to down."

arithmetic:
  - "5 + 3 = 8"
  - "10 - 4 = 6"
```

### Output

The experiment identifies **specialist experts**:

```json
{
  "layer": 12,
  "expert": 3,
  "top_pattern": "NUM→OP→NUM",
  "concentration": 0.45,
  "examples": ["+", "*", "="]
}
```

High concentration = expert specializes on that pattern.

## Expected Results

Based on prior MoE research, we expect to find:

1. **Arithmetic experts**: Specialists for NUM→OP→NUM patterns
2. **Code structure experts**: Specialists for keywords and brackets
3. **Punctuation experts**: Handling sentence boundaries
4. **Position experts**: Start-of-sequence (^→) vs end-of-sequence (→$)

## Files

```
expert_analysis/
├── EXPERIMENT.md      # This file
├── README.md          # Quick start guide
├── experiment.py      # ExperimentBase implementation
├── config.yaml        # Configuration
├── analyses/          # Analysis output files
└── results/           # Experiment results
```

## Analysis Types

### 1. Pattern Summary (`pattern_summary`)

Finds the most specialized experts by token-type trigram:
- Computes trigram counts per expert
- Ranks experts by concentration (top pattern / total)
- Returns top 50 specialists

### 2. Semantic Patterns (`semantic_patterns`)

Analyzes expert handling of semantic relationships:
- Synonyms: ADJ→SYN patterns
- Antonyms: ADJ→ANT patterns
- Analogies: →AS→ and →TO→ patterns
- Arithmetic: NUM→OP patterns

### 3. Sequence Patterns (`sequence_patterns`)

Studies n-gram and positional patterns:
- Start-of-sequence specialists
- End-of-sequence specialists
- N-gram repetition patterns

### 4. Combined Analysis (`combined_analysis`)

Multi-aspect analysis combining all the above.

## Interpretation

### Specialist Score

An expert with concentration 0.45 on "NUM→OP→NUM" means:
- 45% of its activations involve that specific pattern
- This is significantly above random (~3% for diverse trigrams)
- The expert has **specialized** for arithmetic

### Layer Evolution

Early layers often show:
- Position-based specialists (start/end tokens)
- Surface-level pattern matching

Later layers often show:
- Semantic specialists (synonym/antonym handling)
- Complex syntactic patterns

## Requirements

- MoE model (e.g., Mixtral-8x7B-v0.1)
- ~90GB disk space for model weights
- MLX with MoE support

## Future Directions

1. **Cross-model comparison**: Compare specialization across different MoE architectures
2. **Fine-tuning impact**: How does training change expert specialization?
3. **Task-specific routing**: Do experts specialize differently for different tasks?
4. **Activation statistics**: Beyond routing, analyze what experts compute
5. **Expert pruning**: Identify and remove non-specialist experts
