# Classifier Emergence Experiment

Task classifier emergence through SFT training - logit lens analysis at intermediate layers.

## Overview

This experiment investigates whether task classifiers emerge at intermediate layers through targeted SFT training on verifiable arithmetic tasks. It tests the hypothesis that L13-style task classifiers (as observed in GPT-OSS) can be induced in base models through targeted training.

## Running the Experiment

```bash
# Run via framework
lazarus experiment run classifier_emergence

# View results
lazarus experiment status classifier_emergence
```

## Pipeline

1. Generate arithmetic training data
2. Run baseline logit lens analysis (no classifiers expected)
3. Train model with LoRA using mlx-lm
4. Re-run logit lens at checkpoints to detect emerging classifiers
5. Measure classifier strength vs training steps

## Configuration

See `config.yaml` for parameters:
- `model`: Base model to analyze (default: Llama-3.2-1B)
- `training.checkpoint_steps`: Steps at which to analyze classifiers
- `parameters.lora`: LoRA configuration
- `parameters.task_vocabulary`: Words to look for at intermediate layers

## How It Works

The experiment uses **logit lens** to project intermediate layer hidden states to vocabulary space and checks for task-related tokens (like "multiply", "add", answer digits) appearing with high probability before the final layer.

A classifier is considered "emerged" when task vocabulary tokens appear with >10% probability at some intermediate layer, rather than just at the output layer.

## Architecture

```
experiments/classifier_emergence/
├── experiment.py      # ExperimentBase implementation
├── config.yaml        # Experiment configuration
├── data/              # Generated arithmetic data
├── checkpoints/       # Training checkpoints
└── results/           # Analysis results
```

## Expected Results

After sufficient training, the model should show increasing probability mass on task-related vocabulary tokens at intermediate layers, indicating the emergence of vocabulary-aligned classifiers.
