# MoE Type Analysis Results

## Summary

Both tested MoE models show characteristics of **native MoE training** (trained as MoE from scratch):

| Property | GPT-OSS-20B | OLMoE-1B-7B |
|----------|-------------|-------------|
| Type | PRETRAINED | PRETRAINED |
| Confidence | 73% | 100% |
| Gate Rank | 1408/2880 (48.9%) | 758/1024 (74.0%) |
| Cosine Similarity | 0.017 | ~0.000 |
| Compressible via SVD | No | No |

## Reproduce These Results

```bash
lazarus introspect moe-expert moe-type-compare \
    -m openai/gpt-oss-20b \
    -c allenai/OLMoE-1B-7B-0924
```

## OLMoE-1B-7B (Native MoE)

- **64 experts**, 8 active per token
- Gate rank: 758/1024 (74.0%)
- Cosine similarity: ~0.000 (orthogonal experts)
- Classification: PRETRAINED (100% confidence)

The high gate rank and near-zero cosine similarity indicate experts are orthogonal subspaces, consistent with native MoE training.

## GPT-OSS-20B (Native MoE)

- **32 experts**, top-k routing
- Gate rank: 1408/2880 (48.9%)
- Cosine similarity: 0.017 (near-orthogonal experts)
- Classification: PRETRAINED (73% confidence)

Despite different architecture and scale, GPT-OSS also shows native MoE characteristics:
- Moderate-to-high gate rank (48.9%)
- Near-zero cosine similarity (0.017)

## Implications

### Compression

Neither model is compressible via the SVD overlay method. The overlay method requires:
- Low gate rank (< 5%) indicating shared gating
- High cosine similarity (> 0.25) indicating clustered experts

Both models fail these criteria.

### Alternative Compression Strategies

For native MoE models, alternative compression strategies include:
1. **Quantization**: Standard INT4/INT8 quantization (4-8x)
2. **Expert pruning**: Remove cold/redundant experts
3. **Expert merging**: Combine similar-output experts
4. **Distillation**: Train smaller model to match

## Detection Thresholds

The classification uses these thresholds:

| Metric | Upcycled-MoE | Native-MoE |
|--------|--------------|------------|
| Gate Rank Ratio | < 5% | > 50% |
| Cosine Similarity | > 0.25 | < 0.10 |

## Models to Test

| Model | Experts | Status |
|-------|---------|--------|
| OLMoE-1B-7B | 64 | ✓ Native MoE |
| GPT-OSS-20B | 32 | ✓ Native MoE |
| Mixtral-8x7B | 8 | Pending |
| DeepSeek-V3 | 256 | Pending |
| Qwen-MoE | 64 | Pending |
| DBRX | 16 | Pending |

## Conclusion

The tested MoE models (GPT-OSS and OLMoE) both show native MoE characteristics. We have not yet identified an "upcycled" model (dense→MoE conversion) that would be compressible via SVD overlay.
