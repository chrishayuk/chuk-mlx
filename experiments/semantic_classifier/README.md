# Semantic Classifier Experiment

Tests whether explicit classifiers improve accuracy when parsing natural language arithmetic.

## Key Difference from classifier_emergence

| Experiment | Input | Classification Required? |
|------------|-------|-------------------------|
| classifier_emergence | `7 * 8 =` | No (operator visible) |
| **semantic_classifier** | `seven times eight` | **Yes** (must parse) |

## Hypothesis

Dual-reward training (explicit classifier at L8) should outperform SFT on semantic input because the model must actually classify the operation, not just read a symbol.

## Run

```bash
lazarus experiment run semantic_classifier
```

## Expected Results

If dual-reward wins:
- Explicit classifiers ARE useful when parsing is required
- L8 is the right place for operation classification

If SFT still wins:
- Classifiers emerge implicitly even with semantic input
- Explicit training isn't needed
