# Two-Stage Classifier Training

Tests whether we can add classifiers WITHOUT destroying computation.

## The Problem

Previous experiments showed:
- **SFT**: Good computation, weak classifiers
- **Dual-reward (70/30)**: Strong classifiers, broken computation

## The Solution

Two-stage training:

```
Stage 1: SFT (500 steps)
  → Build computation circuits
  → Target: 100% symbolic accuracy

Stage 2: Light dual-reward (200 steps)
  → classifier_weight: 0.2 (not 0.7!)
  → answer_weight: 0.8
  → Target: Add classifiers, preserve computation
```

## Expected Outcome

| Metric | After Stage 1 | After Stage 2 |
|--------|---------------|---------------|
| Symbolic accuracy | ~100% | ~100% (preserved) |
| Classifier prob | ~5% | ~30%+ (added) |

## Run

```bash
lazarus experiment run two_stage_classifier
```
