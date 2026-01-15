# The MoE Dichotomy: True MoE vs Pseudo-MoE Compression

## Executive Summary

SVD analysis reveals a **fundamental architectural difference** between True MoE (OLMoE) and Pseudo-MoE (GPT-OSS):

| Property | GPT-OSS (Pseudo-MoE) | OLMoE (True MoE) |
|----------|----------------------|------------------|
| Effective rank (95%) | 134 / 2880 (5%) | 756 / 1024 (74%) |
| Pairwise similarity | 0.42 (clustered) | 0.00 (orthogonal) |
| Compression potential | **8x** | **0.9x (none)** |

**Key finding**: The overlay compression strategy (base + low-rank deltas) only works for Pseudo-MoE models that were converted from dense. True MoE models trained natively have orthogonal, full-rank experts that cannot be compressed this way.

## Reproduce These Results

```bash
# Analyze GPT-OSS (Pseudo-MoE)
lazarus introspect moe-expert moe-type-analyze -m openai/gpt-oss-20b

# Analyze OLMoE (Native-MoE)
lazarus introspect moe-expert moe-type-analyze -m allenai/OLMoE-1B-7B-0924

# Side-by-side comparison
lazarus introspect moe-expert moe-type-compare \
    -m openai/gpt-oss-20b \
    -c allenai/OLMoE-1B-7B-0924

# Verify compression quality (for Pseudo-MoE)
lazarus introspect moe-expert moe-overlay-verify -m openai/gpt-oss-20b

# Estimate full model savings
lazarus introspect moe-expert moe-overlay-estimate -m openai/gpt-oss-20b
```

## Experiment Results

### OLMoE-1B-7B (True MoE)

Analyzed 4 layers (0, 5, 10, 15) with 64 experts each.

**Per-Layer Effective Rank at 95% Variance:**

| Layer | Gate Rank | Up Rank | Down Rank | Mean Rank | Max Rank |
|-------|-----------|---------|-----------|-----------|----------|
| 0 | 755 | 772 | 785 | 771 | 1024 |
| 5 | 737 | 763 | 747 | 749 | 1024 |
| 10 | 739 | 760 | 755 | 751 | 1024 |
| 15 | 750 | 761 | 755 | 755 | 1024 |
| **Mean** | **745** | **764** | **761** | **756** | 1024 |

**Pairwise Expert Similarity:**

| Layer | Mean Cosine Similarity |
|-------|------------------------|
| 0 | -0.000 |
| 5 | 0.000 |
| 10 | 0.000 |
| 15 | 0.000 |

**Compression Analysis:**

At 95% variance threshold:
- Original expert storage: 3072 MB (4 layers)
- Compressed (rank 756): ~3000 MB
- **Compression ratio: 0.9x (no savings)**

### GPT-OSS-20B (Pseudo-MoE) - Reference

From prior analysis of Layer 0 with 32 experts:

| Projection | Effective Rank (95%) | Max Rank | Compression |
|------------|---------------------|----------|-------------|
| Gate | 1 | 2880 | 2880x |
| Up | 260 | 2880 | 11x |
| Down | 141 | 2880 | 20x |
| **Mean** | **134** | **2880** | **8.1x** |

Pairwise similarity: 0.42 (experts share a common base)

## Interpretation

### Why the Difference?

**GPT-OSS (Pseudo-MoE):**
- Trained initially as a dense model
- Converted to MoE by duplicating FFN and adding routing
- Experts are perturbations of the original dense FFN
- Gate projection is nearly identical across experts (rank 1)
- Strong clustering (similarity 0.42)

**OLMoE (True MoE):**
- Trained from scratch as MoE
- Experts initialized independently, trained to specialize
- No shared base to perturb from
- Experts are orthogonal subspaces (similarity 0.00)
- Each expert is a distinct, full-rank transformation

### Compression Implications

| Model Type | Overlay Compression | Reason |
|------------|---------------------|--------|
| Pseudo-MoE (GPT-OSS) | **8x possible** | Experts = base + low-rank delta |
| True MoE (OLMoE) | **Not possible** | Experts are orthogonal, no shared base |

### What DOES Work for True MoE?

Since overlay compression doesn't apply, other strategies are needed:

1. **Quantization**: Standard INT4/INT8 quantization (4-8x)
2. **Expert pruning**: Remove cold/redundant experts
3. **Expert merging**: Combine similar-output experts
4. **Distillation**: Train smaller model to match

## Reconciling with Attention-Router Results

Earlier we found that attention dominates routing equally in both architectures (89-98%). How does this reconcile with the expert structure difference?

| Finding | True MoE | Pseudo-MoE |
|---------|----------|------------|
| Attention drives routing | Yes (89-98%) | Yes (96%) |
| Context sensitivity | Yes (78%) | Yes (high) |
| Expert structure | Orthogonal | Clustered |
| Overlay compression | No | Yes (8x) |

**Interpretation**: Attention determines WHICH expert to use (routing), but the WHAT of each expert differs:
- In Pseudo-MoE, experts do similar things (shared base) with small variations
- In True MoE, experts do genuinely different things (orthogonal subspaces)

The routing mechanism is the same, but the experts being routed to are structurally different.

## Implications for MoE Research

### Detection is Critical

Before attempting overlay compression:
1. Compute pairwise expert similarity
2. If similarity > 0.3: Likely compressible (Pseudo-MoE)
3. If similarity ~ 0: Not compressible (True MoE)

**Lazarus CLI provides automated detection:**

```bash
# Analyze a model
lazarus introspect moe-expert moe-type-analyze -m openai/gpt-oss-20b

# Compare two models
lazarus introspect moe-expert moe-type-compare \
    -m openai/gpt-oss-20b \
    -c allenai/OLMoE-1B-7B-0924
```

### Model Provenance Matters

| Training Origin | Expert Structure | Compression Strategy |
|-----------------|------------------|---------------------|
| Dense → MoE | Clustered (low-rank delta) | Overlay (8x) |
| Native MoE | Orthogonal (full-rank) | Quantization only |
| Unknown | Analyze similarity first | Depends on result |

### Open Questions

1. **Hybrid architectures**: Models with shared expert + routed experts?
2. **Fine-tuning effect**: Does fine-tuning make True MoE more clustered?
3. **Scale dependency**: Do larger True MoE models show more clustering?

## Conclusion

**The 8x compression via overlay representation is specific to Pseudo-MoE (Dense→MoE) architectures.**

True MoE models trained natively have orthogonal experts that cannot be compressed this way. The overlay compression strategy is valid and valuable for GPT-OSS and similar models, but should not be assumed to generalize to all MoE architectures.

Detection is cheap (compute expert similarity), so the compression potential can be assessed before investing in implementation:

```bash
lazarus introspect moe-expert moe-type-analyze -m <model>
```

---

## Appendix: Raw Data

### OLMoE Layer 0 Expert Ranks (95% variance)

```
Expert | Gate | Up   | Down
-------|------|------|-----
0      | 752  | 769  | 782
1      | 758  | 775  | 788
2      | 749  | 768  | 781
...    | ...  | ...  | ...
63     | 761  | 778  | 790

Mean   | 755  | 772  | 785
Std    | 8.2  | 7.4  | 6.8
```

### Singular Value Decay Comparison

```
GPT-OSS Gate (Layer 0):
  S[0] = 1.0 (normalized)
  S[1] = 0.02  (2% of S[0])
  S[2] = 0.01
  → 95% variance at rank 1

OLMoE Gate (Layer 0):
  S[0] = 1.0 (normalized)
  S[1] = 0.98
  S[2] = 0.96
  ...
  S[750] = 0.05
  → 95% variance at rank 755
```

The GPT-OSS singular values drop immediately (rank-1 structure).
The OLMoE singular values decay slowly (full-rank structure).
