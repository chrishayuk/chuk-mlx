# MoE Type Detection: Pseudo vs Native Classification

## Overview

This experiment identifies whether an MoE model is **Pseudo-MoE** (dense→MoE conversion, compressible) or **Native-MoE** (trained natively, not compressible via SVD).

## Key Finding

| Model | Type | Gate Rank | Cosine Sim | Compressible |
|-------|------|-----------|------------|--------------|
| GPT-OSS-20B | Pseudo-MoE | 1 / 2880 (0%) | 0.42 | Yes (8x) |
| OLMoE-1B-7B | Native-MoE | 755 / 1024 (74%) | 0.00 | No |

**The smoking gun:** Gate rank = 1 means all 32 GPT-OSS "experts" use the exact same gating function. There's no mixture - it's one expert with 32 output modes.

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

**Pseudo-MoE** experts can be represented as:

```
expert_i = base + delta_i

Where:
  base = mean of all experts (full rank)
  delta_i = low-rank matrix (rank ~134 captures 95% variance)
```

**Native-MoE** experts are orthogonal - no shared base exists.

## Detection Method

The service computes two key metrics:

1. **Gate Rank Ratio**: `effective_rank_95 / max_rank`
   - Pseudo-MoE: < 5% (experts share same gate)
   - Native-MoE: > 50% (diverse gates)

2. **Cosine Similarity**: Mean pairwise similarity between experts
   - Pseudo-MoE: > 0.25 (clustered around base)
   - Native-MoE: < 0.10 (orthogonal)

## Compression Pipeline

Once a model is detected as Pseudo-MoE, you can compute and verify the overlay representation:

```bash
# 1. Detect type (is it compressible?)
lazarus introspect moe-expert moe-type-analyze -m openai/gpt-oss-20b

# 2. Compute overlay (what ranks are needed?)
lazarus introspect moe-expert moe-overlay-compute -m openai/gpt-oss-20b

# 3. Verify reconstruction (is quality preserved?)
lazarus introspect moe-expert moe-overlay-verify -m openai/gpt-oss-20b

# 4. Estimate savings (how much memory saved?)
lazarus introspect moe-expert moe-overlay-estimate -m openai/gpt-oss-20b
```

## Why This Matters

### Compression Implications

| Model Type | SVD Overlay | Reason |
|------------|-------------|--------|
| Pseudo-MoE | **8x possible** | Experts = base + low-rank delta |
| Native-MoE | **Not applicable** | Experts are orthogonal, no shared base |

### Memory Impact (GPT-OSS Example)

```
STANDARD REPRESENTATION:
  32 experts x 3 projections x (2880 x 2880) x 2 bytes
  = 38.4 GB for 24 layers

OVERLAY REPRESENTATION:
  1 base + 32 low-rank deltas
  = 8.1 GB for 24 layers

SAVINGS: 4.7x memory reduction
```

## Training Origin Detection

| Training Path | Expert Structure | Detection |
|---------------|------------------|-----------|
| Dense → MoE | Clustered (low-rank delta) | gate_rank ≈ 1, sim > 0.3 |
| Native MoE | Orthogonal (full-rank) | gate_rank high, sim ≈ 0 |

## Example Output

### Single Model Analysis

```
$ lazarus introspect moe-expert moe-type-analyze -m openai/gpt-oss-20b

======================================================================
MOE TYPE ANALYSIS
======================================================================
Model:  openai/gpt-oss-20b
Layer:  0
Type:   PSEUDO-MOE

Evidence:
  Gate Rank:            1 / 2880 (  0.0%)
  Up Rank:            337 / 2880 ( 11.7%)
  Down Rank:          206 / 2880 (  7.2%)
  Cosine Similarity: 0.418 (+/- 0.163)

Compression:
  Compressible:      Yes
  Estimated Ratio:   7.9x
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
| Type                  | PSEUDO         | NATIVE         |
| Gate Rank             |    1/2880      |  755/1024      |
| Gate Rank %           |           0.0% |          73.7% |
| Cosine Similarity     |          0.418 |          0.000 |
| Compressible          | Yes (7.9x)     | No             |
+-----------------------+----------------+----------------+
```

## Performance Notes

- **OLMoE**: ~30 seconds (64 experts, 1024×2048 matrices)
- **GPT-OSS**: ~6 minutes (samples 8/32 experts, 2880×2880 matrices)

SVD on large matrices is expensive. The service samples a subset of experts for efficiency while maintaining accuracy.

## Open Questions

1. **Mixtral**: Pseudo or Native?
2. **DeepSeek-V3**: If Pseudo-MoE, 671B could fit on a MacBook
3. **Fine-tuning effect**: Does fine-tuning make Native-MoE more clustered?

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
