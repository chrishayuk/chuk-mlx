# Injection Point & Layer Compression Findings

## Summary

We investigated whether transformer layers in FunctionGemma-270M can be:
1. **Skipped** - Run the model but skip certain layers
2. **Replaced** - Replace early layers with a smaller "student" network
3. **Injected** - Transform embeddings directly and inject into middle layers

**Key Finding: None of these approaches work.**

## Experiments Conducted

### 1. Layer Skipping Analysis (`injection_upper_bound.py`)

**Question**: What happens if we simply skip layers?

**Results**:
| Layers Skipped | Similarity to Full | Top-1 Accuracy |
|----------------|-------------------|----------------|
| L1 only        | 0.154            | 0.0%           |
| L1-3           | 0.190            | 0.0%           |
| L6-8           | 0.734            | 5.0%           |
| L9-11          | 0.632            | 0.0%           |
| L14-16         | 0.708            | 0.0%           |

**Conclusion**: Skipping even ONE layer causes catastrophic accuracy loss (0-5%).

### 2. Untrained Injection (`find_injection_point.py`)

**Question**: Can we transform embeddings and inject at different layers?

**Results**:
| Injection Point | Similarity | Accuracy |
|-----------------|------------|----------|
| L0 (+ 2 attn)   | 0.284      | 50%      |
| L6 (+ 1 attn)   | 0.009      | 50%      |
| L12 (direct)    | 0.006      | 50%      |

**Conclusion**: Untrained transforms can't match layer expectations. 50% = random guessing.

### 3. Trained Distillation (`normalized_distillation.py`)

**Question**: Can we train a small network to match early layer outputs?

**Training details**:
- Target: Match output of layers 0-N using a student with fewer blocks
- Loss: Normalized cosine similarity + scale matching
- Training converged well (loss 4.7 → 0.03)

**Results**:
| Config          | Training Loss | Cosine to Full | Top-1 Accuracy |
|-----------------|---------------|----------------|----------------|
| L0-3 → 1 block  | 0.026         | 0.309          | 0.0%           |
| L0-5 → 2 blocks | 0.029         | 0.437          | 0.0%           |
| L0-8 → 3 blocks | 0.027         | 0.423          | 0.0%           |

**Conclusion**: Even with well-trained distillation (low loss), the remaining layers can't recover original behavior.

## Key Observations

### 1. Massive Norm Growth Through Layers
```
Layer Input Norms (averaged):
  L0 input:  26 (scaled embeddings)
  L3 input:  1,887 (72x increase)
  L6 input:  9,666 (372x increase)
  L9 input:  18,225 (700x increase)
  L17 input: 33,803 (1,300x increase)
```

### 2. Each Layer is Essential
The injection mechanism works perfectly when using REAL layers (100% accuracy).
But any deviation - skipping, replacing, or approximating - causes complete failure.

### 3. The "Train Deep, Deploy Shallow" Hypothesis is Falsified
We hypothesized that:
- Training enriches embeddings
- At inference, layers become redundant
- A small "context encoder" could replace early layers

**Evidence against this**:
- Skipping any layer = 0% accuracy
- Distilled student with perfect training loss = 0% accuracy
- Cosine similarity to full output remains low (0.3-0.4)

## Interpretation

Each transformer layer in this model performs **non-redundant computation** that cannot be:

1. **Skipped**: Every layer's output is required by subsequent layers
2. **Approximated**: Even matching the direction of activations isn't enough
3. **Compressed**: The transformation each layer performs is unique

This suggests that for FunctionGemma-270M:
- Layers have **sequential dependencies** - each builds on the previous
- The model uses its **full depth** for every forward pass
- There's no "phase transition" where early processing becomes redundant

## Implications for Collapsed Inference

The original goal was to speed up inference by skipping layers. Our findings indicate:

1. **Layer pruning won't work** for this architecture/task
2. **Knowledge distillation** to smaller networks requires different approaches:
   - End-to-end distillation (match final outputs, not intermediate)
   - Larger student networks
   - Different training objectives

3. **The embedding analysis** showing rich semantic structure in raw embeddings remains valid - the embeddings ARE informative. But the layers don't just "pass through" this information; they perform essential transformations.

## Files Created

- `find_injection_point.py` - Test untrained injection at different layers
- `trained_injection.py` - Train transforms to match layer inputs
- `injection_upper_bound.py` - Verify mechanism & test layer skipping
- `layer_distillation.py` - Distill early layers to student network
- `normalized_distillation.py` - Improved distillation with normalized loss

## Next Steps (if continuing this research)

1. **End-to-end distillation**: Train student to match final output, not intermediate
2. **Larger students**: Maybe 10+ blocks needed to approximate 18 layers
3. **Task-specific probing**: Skip layers but add task-specific classifier
4. **Different architectures**: Test on other models (Llama, etc.) to see if this is Gemma-specific
