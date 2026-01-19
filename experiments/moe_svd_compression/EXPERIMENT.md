# MoE Type Detection: Native vs Upcycled Classification

## Overview

This experiment identifies whether an MoE model is **Native-MoE** (trained as MoE from scratch) or **Upcycled-MoE** (dense→MoE conversion, potentially compressible via SVD).

## Key Finding (Updated January 2026)

| Model | Type | Gate Rank | Cosine Sim | Compressible |
|-------|------|-----------|------------|--------------|
| GPT-OSS-20B | Native-MoE | 1408 / 2880 (48.9%) | 0.017 | No |
| OLMoE-1B-7B | Native-MoE | 758 / 1024 (74.0%) | ~0.00 | No |

Both GPT-OSS and OLMoE show characteristics of native MoE training:
- High gate rank (48.9% and 74% respectively)
- Near-zero cosine similarity between experts (orthogonal)
- Neither is compressible via SVD overlay

## CLI Commands

```bash
# Analyze a single model
lazarus introspect moe-expert moe-type-analyze -m openai/gpt-oss-20b

# Analyze specific layer
lazarus introspect moe-expert moe-type-analyze -m allenai/OLMoE-1B-7B-0924 --layer 0

# Compare two models side-by-side
lazarus introspect moe-expert moe-type-compare \
    -m openai/gpt-oss-20b \
    -c allenai/OLMoE-1B-7B-0924

# Save results to JSON
lazarus introspect moe-expert moe-type-analyze -m openai/gpt-oss-20b -o results.json

# Compute overlay representation (base + low-rank deltas)
lazarus introspect moe-expert moe-overlay-compute -m openai/gpt-oss-20b

# Verify reconstruction accuracy
lazarus introspect moe-expert moe-overlay-verify -m openai/gpt-oss-20b

# Estimate storage savings for full model
lazarus introspect moe-expert moe-overlay-estimate -m openai/gpt-oss-20b

# Custom ranks (override auto-selection)
lazarus introspect moe-expert moe-overlay-compute -m openai/gpt-oss-20b \
    --gate-rank 2 --up-rank 128 --down-rank 64
```

## The Hypothesis

**Upcycled-MoE** experts (if they exist) could be represented as:

```
expert_i = base + delta_i

Where:
  base = mean of all experts (full rank)
  delta_i = low-rank matrix (low rank captures 95% variance)
```

**Native-MoE** experts are orthogonal - no shared base exists.

**Current Status:** Both GPT-OSS and OLMoE appear to be Native-MoE. We have not yet found a confirmed Upcycled-MoE model to test this hypothesis.

## Detection Method

The service computes two key metrics:

1. **Gate Rank Ratio**: `effective_rank_95 / max_rank`
   - Upcycled-MoE: < 5% (experts share same gate)
   - Native-MoE: > 50% (diverse gates)

2. **Cosine Similarity**: Mean pairwise similarity between experts
   - Upcycled-MoE: > 0.25 (clustered around base)
   - Native-MoE: < 0.10 (orthogonal)

**Observed Results:**
- GPT-OSS: 48.9% gate rank, 0.017 similarity → Native-MoE
- OLMoE: 74.0% gate rank, ~0 similarity → Native-MoE

## Compression Pipeline

Once a model is detected as Upcycled-MoE (if one is found), you can compute and verify the overlay representation:

```bash
# 1. Detect type (is it compressible?)
lazarus introspect moe-expert moe-type-analyze -m <model>

# 2. Compute overlay (what ranks are needed?)
lazarus introspect moe-expert moe-overlay-compute -m <model>

# 3. Verify reconstruction (is quality preserved?)
lazarus introspect moe-expert moe-overlay-verify -m <model>

# 4. Estimate savings (how much memory saved?)
lazarus introspect moe-expert moe-overlay-estimate -m <model>
```

**Note:** Neither GPT-OSS nor OLMoE are compressible via this method.

## Why This Matters

### Compression Implications

| Model Type | SVD Overlay | Reason |
|------------|-------------|--------|
| Upcycled-MoE | **Potentially compressible** | Experts = base + low-rank delta |
| Native-MoE | **Not applicable** | Experts are orthogonal, no shared base |

### Current Results

Both tested models (GPT-OSS and OLMoE) show Native-MoE characteristics and are NOT compressible via SVD overlay.

## Training Origin Detection

| Training Path | Expert Structure | Detection |
|---------------|------------------|-----------|
| Dense → MoE (Upcycled) | Clustered (low-rank delta) | gate_rank < 5%, sim > 0.3 |
| Native MoE | Orthogonal (full-rank) | gate_rank > 50%, sim < 0.1 |

**Actual Results:**
- GPT-OSS: gate_rank = 48.9%, sim = 0.017 → Native MoE
- OLMoE: gate_rank = 74.0%, sim ≈ 0 → Native MoE

## Example Output

### Single Model Analysis

```
$ lazarus introspect moe-expert moe-type-analyze -m openai/gpt-oss-20b

======================================================================
MOE TYPE ANALYSIS
======================================================================
Model:  openai/gpt-oss-20b
Layer:  0
Type:   PRETRAINED

Evidence:
  Gate Rank:         1408 / 2880 ( 48.9%)
  Cosine Similarity: 0.017

Compression:
  Compressible:      No
======================================================================
```

### Model Comparison

```
$ lazarus introspect moe-expert moe-type-compare -m openai/gpt-oss-20b -c allenai/OLMoE-1B-7B-0924

======================================================================
MOE TYPE COMPARISON
======================================================================
+-----------------------+----------------+----------------+
| Metric                | gpt-oss-20b    | OLMoE-1B-7B-09 |
+-----------------------+----------------+----------------+
| Type                  | PRETRAINED     | PRETRAINED     |
| Confidence            | 73%            | 100%           |
| Gate Rank             | 1408/2880      |  758/1024      |
| Gate Rank %           |          48.9% |          74.0% |
| Cosine Similarity     |          0.017 |         -0.000 |
| Compressible          | No             | No             |
+-----------------------+----------------+----------------+
```

## Performance Notes

- **OLMoE**: ~30 seconds (64 experts, 1024×2048 matrices)
- **GPT-OSS**: ~6 minutes (samples 8/32 experts, 2880×2880 matrices)

SVD on large matrices is expensive. The service samples a subset of experts for efficiency while maintaining accuracy.

## Open Questions

1. **Mixtral**: Native or Upcycled?
2. **DeepSeek-V3**: Native or Upcycled? If Upcycled, compression may be possible
3. **Fine-tuning effect**: Does fine-tuning make Native-MoE more clustered?
4. **Are there any Upcycled-MoE models?**: We have not yet found a model that shows the upcycled signature (low gate rank, high cosine similarity)

## CLI vs Experiment Code

The CLI covers all essential functionality:

| Feature | CLI | Experiment |
|---------|-----|------------|
| Type detection | ✓ `moe-type-analyze` | ✓ |
| Model comparison | ✓ `moe-type-compare` | - |
| Overlay computation | ✓ `moe-overlay-compute` | ✓ |
| Reconstruction verification | ✓ `moe-overlay-verify` | ✓ |
| Storage estimation | ✓ `moe-overlay-estimate` | ✓ |
| Multi-layer batch analysis | - | ✓ |
| Singular value visualization | - | ✓ |
| Multiple variance thresholds | - | ✓ (90%, 95%, 99%) |

**For most use cases, use the CLI.** The experiment code remains for:
- Visualization of singular value decay curves
- Research requiring multiple variance thresholds
- Batch analysis across many layers

## References

- See `RESULTS.md` for detailed experimental results
- Framework: `src/chuk_lazarus/introspection/moe/moe_type.py`, `moe_compression.py`
