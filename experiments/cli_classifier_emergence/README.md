# CLI Classifier Emergence Experiment

Dual-reward training for vocabulary-aligned arithmetic classifiers.

## Overview

This experiment demonstrates that training V/O projections with dual-reward (generation + classification) creates vocabulary-aligned classifiers that can be read via logit lens at intermediate layers.

### Key Findings

- **Base models have hidden-space classifiers** (detectable via linear probe)
- **Base models do NOT have vocab-aligned classifiers** (0% via logit lens)
- **V/O training creates vocab-aligned classifiers** (measurable via logit lens)

## Running the Experiment

```bash
# Run via framework
lazarus experiment run cli_classifier_emergence

# View results
lazarus experiment status cli_classifier_emergence
```

## Configuration

See `config.yaml` for configurable parameters:
- `model`: Base model to train (default: TinyLlama-1.1B)
- `training.max_steps`: Training iterations
- `classifier.layer_pct`: Layer depth for classifier (0.55 = 55%)
- `classifier.weight`: Weight of classification vs generation loss
- `classifier.targets`: Operation → token mapping

## How It Works

1. **Data Generation**: Creates arithmetic problems labeled by operation (add, subtract, multiply, divide)
2. **Dual-Reward Training**: Applies LoRA to V/O projections with combined loss:
   - Classification loss at intermediate layer (55% depth)
   - Answer loss at final layer
3. **Evaluation**: Tests classifier accuracy on held-out prompts

## Results

The trained model should show vocabulary-aligned classifier signals at the intermediate layer, where the probability mass shifts toward operation tokens (`+`, `-`, `*`, `/` or `add`, `multiply`, etc.) based on the input.

## Architecture

```
experiments/cli_classifier_emergence/
├── experiment.py      # ExperimentBase implementation
├── config.yaml        # Experiment configuration
├── data/              # Generated training data
├── checkpoints/       # Saved model checkpoints
├── results/           # Experiment results
└── archive/           # Historical scripts and writeups
```

## Related Work

This experiment is inspired by observations of vocabulary-aligned classifiers in GPT-OSS at Layer 13 (~54% depth), which show operation type predictions ("multiply", "add") via logit lens with 50-80% probability.
