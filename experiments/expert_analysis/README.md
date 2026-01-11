# Expert Analysis Experiment

MoE (Mixture of Experts) expert specialization analysis.

## Overview

This experiment analyzes which experts in Mixture-of-Experts models are activated for different types of semantic tasks. It studies expert routing patterns to understand specialization in MoE architectures.

## Running the Experiment

```bash
# Run via framework (requires Mixtral model ~90GB)
lazarus experiment run expert_analysis

# View results
lazarus experiment status expert_analysis
```

## Requirements

This experiment requires a Mixture-of-Experts model like:
- `mistralai/Mixtral-8x7B-v0.1` (~90GB)

The model must have expert routing infrastructure to analyze.

## Analysis Types

The experiment performs several types of analysis:

### Token Type Patterns
Analyzes which experts are activated for different token types:
- Words, numbers, punctuation
- Content vs function words

### Sequence Patterns
Studies how expert activation changes across sequence positions:
- Beginning, middle, end tokens
- Local vs global context experts

### Semantic Patterns
Examines expert specialization for semantic relationships:
- Synonyms, antonyms, hypernyms
- Associations, analogies
- Arithmetic operations

## Configuration

See `config.yaml` for parameters:
- `model`: MoE model to analyze
- `analysis_types`: Which analyses to run
- `semantic_prompts`: Test prompts for semantic analysis
- `top_k_experts`: Number of top experts to track

## Architecture

```
experiments/expert_analysis/
├── experiment.py      # ExperimentBase implementation
├── config.yaml        # Experiment configuration
├── analyses/          # Analysis output files
├── checkpoints/       # (unused for this experiment)
├── data/              # (unused for this experiment)
└── results/           # Experiment results
```

## Expected Results

The analysis should reveal expert specialization patterns, showing that different experts in MoE models handle different types of linguistic and semantic processing.
