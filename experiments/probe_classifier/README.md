# Probe Classifier Experiment

Tests whether task information is encoded at intermediate layers using linear probes.

## Key Question

**Can a simple linear probe extract task labels from hidden states?**

This is critical for the virtual expert architecture:
- If YES → We can route using learned projections (no vocabulary alignment needed)
- If NO → We need vocabulary-aligned classifiers or different approach

## How It Works

```
1. Extract hidden states at each layer for arithmetic prompts
2. Train a linear probe: hidden_state → task_label
3. Measure classification accuracy

Linear Probe:
  task_logits = W @ hidden_state + b
  where W is [num_tasks, hidden_dim]
```

## Expected Results

| Layer | Accuracy | Interpretation |
|-------|----------|----------------|
| L4 (25%) | ~33% | Random (no task info yet) |
| L8 (50%) | ~60-80% | Task info emerging |
| L12 (75%) | ~90%+ | Strong task encoding |
| L15 (95%) | ~95%+ | Task fully encoded |

## Run

```bash
lazarus experiment run probe_classifier
```

## Implications

If probe accuracy is >90% at intermediate layers:
- **Routing is viable** without vocabulary classifiers
- Virtual experts can use learned routing matrices
- No need for logit lens approach

If probe accuracy is low (<70%):
- Task info only emerges at final layers
- Vocabulary alignment may be necessary
- Different routing approach needed
