# MoE Type Classification: Native vs Upcycled

## Overview

This experiment classifies MoE models based on their weight structure to determine training history:

- **Native MoE**: Trained as MoE from scratch - orthogonal experts, high gate rank
- **Upcycled MoE**: Converted from dense model - clustered experts, low gate rank, potentially compressible

## Results

| Model | Type | Gate Rank | Cosine Sim | Confidence |
|-------|------|-----------|------------|------------|
| GPT-OSS-20B | Native | 48.9% | 0.017 | 73% |
| OLMoE-1B-7B | Native | 74.0% | ~0.00 | 100% |

Both tested models show native MoE characteristics.

## Detection Method

Two key metrics distinguish native from upcycled MoE:

1. **Gate Rank Ratio**: `effective_rank_95 / max_rank`
   - Upcycled: < 5% (experts share same gate)
   - Native: > 50% (diverse gates)

2. **Cosine Similarity**: Mean pairwise similarity between experts
   - Upcycled: > 0.25 (clustered around base)
   - Native: < 0.10 (orthogonal)

## CLI Commands

```bash
# Analyze a single model
lazarus introspect moe-expert moe-type-analyze -m openai/gpt-oss-20b

# Compare two models
lazarus introspect moe-expert moe-type-compare \
    -m openai/gpt-oss-20b \
    -c allenai/OLMoE-1B-7B-0924
```

## Implications

### For Native MoE (confirmed for GPT-OSS and OLMoE)
- Not compressible via SVD overlay
- Use standard quantization (INT4/INT8)
- Consider expert pruning or tiered storage

### For Upcycled MoE (if found)
- Potentially compressible via SVD overlay
- Store as: base + low-rank deltas per expert
- Could achieve 5-10x compression

## Models to Test

| Model | Experts | Status |
|-------|---------|--------|
| OLMoE-1B-7B | 64 | ✓ Native |
| GPT-OSS-20B | 32 | ✓ Native |
| Mixtral-8x7B | 8 | Pending |
| DeepSeek-V3 | 256 | Pending |
| Qwen-MoE | 64 | Pending |

## Open Questions

1. Are there any upcycled MoE models in the wild?
2. Does fine-tuning affect expert structure?
3. Can native MoE be post-hoc converted to overlay representation?
