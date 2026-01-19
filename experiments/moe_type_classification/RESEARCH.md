# MoE Training History Detection from Weight Structure

## Hypothesis

MoE models can be classified into two types based on training history:

**Path A: Native MoE Training**
```
Random init → Train as MoE from start → Load balancing loss

Result:
- Orthogonal experts (cosine ≈ 0)
- High gate rank (50-80% of max)
- Diverse gates
- Cannot compress via SVD overlay
```

**Path B: Dense → MoE Conversion (Upcycled)**
```
Dense pretrain → Split MLP into N copies → Fine-tune as experts

Result (hypothesized):
- Clustered experts (cosine > 0.3)
- Low rank deltas (< 20% of max)
- Nearly identical gates (rank < 5%)
- Compressible via SVD overlay
```

## Results

| Model | Gate Rank | Cosine Sim | Classification |
|-------|-----------|------------|----------------|
| GPT-OSS-20B | 48.9% | 0.017 | Native MoE |
| OLMoE-1B-7B | 74.0% | ~0.00 | Native MoE |

Both tested models show native MoE characteristics. We have not yet found a model that shows the upcycled signature.

## Detection Metrics

| Metric | Native MoE | Upcycled MoE |
|--------|------------|--------------|
| Gate Rank Ratio | > 50% | < 5% |
| Cosine Similarity | < 0.10 | > 0.25 |
| Compressibility | ~1x | 5-10x |

## Method

For each MoE model:

1. **Extract expert weights** (gate, up, down projections)
2. **Compute mean expert** (the implicit "base")
3. **Compute delta = expert - base** for each expert
4. **SVD analysis**: Effective rank at 95% variance
5. **Similarity analysis**: Pairwise cosine similarity

## Models to Classify

| Model | Experts | Status |
|-------|---------|--------|
| OLMoE-1B-7B | 64 | ✓ Native |
| GPT-OSS-20B | 32 | ✓ Native |
| Mixtral-8x7B | 8 | Pending |
| DeepSeek-V3 | 256 | Pending |
| Qwen-MoE | 64 | Pending |
| DBRX | 16 | Pending |

## Implications

### For Inference
| Type | Optimization Strategy |
|------|----------------------|
| Native MoE | Tiered storage, quantization, expert pruning |
| Upcycled MoE | SVD overlay compression (if found) |

### For Training
Native MoE may have parameter efficiency concerns:
- Trains many orthogonal experts
- Some experts rarely activated
- Large memory footprint

If upcycled models exist, they could offer:
- Shared base captures common computation
- Deltas specialize for routing
- Better parameter efficiency

## Running the Analysis

```bash
# Analyze single model
lazarus introspect moe-expert moe-type-analyze -m <model>

# Compare models
lazarus introspect moe-expert moe-type-compare -m <model1> -c <model2>
```

## Open Questions

1. **Do upcycled MoE models exist?** We haven't found one yet.
2. **Layer depth effects**: Do deeper layers show different characteristics?
3. **Expert count effects**: Does having more experts affect structure?
4. **Post-hoc conversion**: Can native MoE be converted to overlay format?
