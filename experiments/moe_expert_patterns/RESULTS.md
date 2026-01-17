# MoE Expert Specialization Patterns: Results

## Summary

Analysis of OLMoE-1B-7B (64 experts, 16 MoE layers) reveals **clear expert specialization patterns** based on token type trigrams. Experts show measurable concentration on specific linguistic patterns, with some achieving >70% specialization.

## Model Analyzed

- **Model**: allenai/OLMoE-1B-7B-0924
- **Experts**: 64 per layer
- **MoE Layers**: 16
- **Analysis**: Token-type trigram patterns

## Key Findings

### Top Pattern Specialists

| Layer | Expert | Top Pattern | Concentration | Examples |
|-------|--------|-------------|---------------|----------|
| 15 | 14 | ^→FW→CW | **73%** | "The" (start of sequence) |
| 6 | 35 | PN→NUM→PN | **56%** | Numbers in punctuation context |
| 9 | 2 | FW→CW→FW | **50%** | Content words in function word context |
| 11 | 1 | OP→NUM→OP | ~50% | Numbers between operators |

### Pattern Type Legend

- **FW**: Function words (the, a, is, of)
- **CW**: Content words (apple, happy, run)
- **NUM**: Numbers (42, 3.14)
- **PN**: Punctuation (., ,, ;)
- **OP**: Operators (+, -, =)
- **BR**: Brackets ((), [], {})
- **CAP**: Capitalized words (Paris, Janet)
- **KW**: Keywords (def, class, if)
- **^**: Start of sequence
- **$**: End of sequence

### Layer-wise Patterns

**Early Layers (0-5)**:
- Broader pattern coverage
- Less specialization
- Position-based patterns emerging

**Middle Layers (6-10)**:
- Strongest specialization observed
- Numeric/arithmetic experts (PN→NUM→PN, OP→NUM→OP)
- Function word context experts

**Late Layers (11-15)**:
- Sequence position specialists (^→FW→CW for start)
- Semantic context patterns
- High concentration specialists (>70%)

## Interpretation

### Confirmed Predictions

1. **Arithmetic experts**: Layer 6, Expert 35 specializes in PN→NUM→PN (numbers in punctuation context)
2. **Position specialists**: Layer 15, Expert 14 specializes in ^→FW→CW (sequence starts)
3. **Layer evolution**: Specialization increases with depth

### Expert Specialization Distribution

Most experts show low specialization (random ~3%), but a subset achieves significant concentration:
- 1 expert with >70% concentration
- ~5 experts with >50% concentration
- ~15 experts with >30% concentration

This matches the "workhorse" pattern observed in other MoE research - a small number of experts handle most specialized patterns.

## Methodology

### Token Type Classification

Tokens are classified into 10 categories based on:
- Lexical features (numbers, punctuation, operators)
- Syntactic role (keywords, function words)
- Case (capitalized)
- Position (sequence boundaries)

### Trigram Analysis

For each token position, we compute the trigram:
```
prev_type → current_type → next_type
```

Example for sequence "x + 3":
- x: VAR→OP (^→VAR→OP)
- +: VAR→OP→NUM
- 3: OP→NUM→$

### Concentration Metric

For each expert, concentration = (top pattern count) / (total activations)

High concentration (>30%) indicates specialization.
Random baseline is ~3% (1/33 common trigrams).

## Semantic Patterns

### Analyzed Relationships

| Category | Example Prompts | Top Experts |
|----------|-----------------|-------------|
| Synonyms | "Happy means joyful" | L8-E42, L11-E17 |
| Antonyms | "Hot is opposite of cold" | L9-E31, L12-E5 |
| Arithmetic | "5 + 3 = 8" | L6-E35, L11-E1 |
| Analogies | "King is to queen as man is to woman" | L10-E22, L14-E8 |

Experts show measurable specialization for semantic relationships, though weaker than syntactic patterns.

## Recommendations

### For Expert Pruning

Experts with <5% concentration on any pattern may be candidates for pruning - they act as generalists with no clear specialization.

### For Expert Merging

Experts with similar top patterns (e.g., both handling NUM contexts) could potentially be merged.

### For Routing Optimization

Understanding which experts handle which patterns enables:
- Predictive routing based on token type
- Expert preloading for known input types
- Specialized expert caching strategies

## Files

- `experiment.py`: Analysis implementation
- `config.yaml`: Configuration (model, analysis types)
- `EXPERIMENT.md`: Research methodology
- `RESULTS.md`: This file
- `results/`: JSON result files

## Reproducing

```bash
# Via framework
lazarus experiment run moe_expert_patterns

# View results
lazarus experiment status moe_expert_patterns
```

## Future Work

1. **Cross-model comparison**: Compare OLMoE patterns to Mixtral-8x7B
2. **Fine-tuning impact**: How does task-specific training change specialization?
3. **Expert output analysis**: Beyond routing, what do specialist experts compute?
4. **Quantitative metrics**: Develop specialization score for entire model
