# Detecting MoE Training History from Weight Structure

## The Discovery

Analysis of two MoE models revealed fundamentally different expert structures:

| Model | Effective Rank | Cosine Similarity | Gate Rank | Compressible |
|-------|----------------|-------------------|-----------|--------------|
| OLMoE-1B-7B | 74% of max | 0.00 | High | No |
| GPT-OSS-20B | 6.7% of max | 0.42 | **0.03%** | 6× yes |

This suggests two distinct training histories produce two distinct architectures.

## The Hypothesis

**Path A: Native MoE**
```
Random init → Train as MoE from start → Load balancing loss

Dynamics:
- Experts pushed apart to maximize coverage
- Each expert develops independently
- No shared structure emerges

Result:
- Orthogonal experts (cosine ≈ 0)
- Full rank (70-80% of max)
- Diverse gates
- Cannot compress
```

**Path B: Dense → MoE Conversion**
```
Dense pretrain → Split MLP into N copies → Fine-tune as experts

Dynamics:
- Experts inherit shared weights from dense model
- Fine-tuning adds small perturbations
- Base structure preserved

Result:
- Clustered experts (cosine ≈ 0.3-0.5)
- Low rank deltas (5-15% of max)
- Nearly identical gates (rank ≈ 1)
- 6×+ compressible
```

## The Smoking Gun: Gate Projection

The gate projection in GPT-OSS has effective rank of **1** across ALL 24 layers.

```
GPT-OSS Gate Projection Rank by Layer:
Layer 0:  1.0
Layer 6:  1.0
Layer 12: 1.0
Layer 18: 1.0
Layer 23: 1.0
```

This means all 32 experts use essentially the same gating function with a 1-dimensional perturbation. That's not "32 independent experts"—it's one expert with 32 output modes.

This only happens if experts were initialized from a shared source.

## Full GPT-OSS Results

```
 Layer |     Gate |       Up |     Down |      Avg
     0 |      1.0 |    260.0 |    140.7 |    133.9
     6 |      1.0 |      1.0 |    533.1 |    178.4
    12 |      1.0 |      1.0 |    768.0 |    256.7
    18 |      1.0 |      1.0 |    436.8 |    146.3
    23 |      1.0 |      3.9 |    764.2 |    256.4
  Mean |      1.0 |     53.4 |    528.6 |    194.3

Compression: 6.0× with overlay architecture
```

Key observations:
- Gate: RANK 1 everywhere (all experts share same gate)
- Up: Mostly rank 1 (same up projection)
- Down: Variable (140-768) - this is where experts actually differ

GPT-OSS is essentially: **1 shared (gate + up) + 32 different down projections**

## Detection Method

**Inputs:** MoE model weights (no training details needed)

**Metrics to compute:**

1. **Effective rank at 95% variance**
   - Per projection (gate, up, down)
   - Averaged across experts
   - Normalized by max possible rank

2. **Pairwise cosine similarity**
   - Between all expert pairs
   - Mean and distribution

3. **Gate projection rank**
   - Specifically the gating/routing projection
   - Rank 1 = strong evidence of Dense→MoE

**Classification:**

| Metric | Native MoE (Type A) | Dense→MoE (Type B) |
|--------|--------------------|--------------------|
| Effective rank | >50% | <20% |
| Cosine similarity | <0.1 | >0.3 |
| Gate rank | >100 | ≈1 |
| Compressibility | ~1× | 5-10× |

## Models to Classify

| Model | Experts | Expected | Confirmed |
|-------|---------|----------|-----------|
| OLMoE-1B-7B | 64 | Type A | ✓ Orthogonal |
| GPT-OSS-20B | 32 | Type B | ✓ Pseudo-MoE |
| Mixtral-8x7B | 8 | Type A? | Pending |
| Mixtral-8x22B | 8 | Type A? | Pending |
| DeepSeek-V2 | 160 | ? | Pending |
| DeepSeek-V3 | 256 | ? | **Priority** |
| Qwen-MoE | 64 | ? | Pending |
| DBRX | 16 | ? | Pending |
| Grok-1 | 8 | ? | Pending |
| JetMoE | 8 | ? | Pending |

**Priority:** DeepSeek-V3 (determines Mac inference strategy)

## Experiment Procedure

### Phase 1: Collect Weight Statistics

For each model, for each MoE layer:

```python
1. Extract all expert weights
2. Compute mean expert (potential "base")
3. Compute delta_i = expert_i - mean for each expert
4. SVD each delta → effective rank at 95%
5. Cosine similarity between all expert pairs
6. Special attention to gate projection rank
```

### Phase 2: Classify Models

Plot each model on:
- X-axis: Mean cosine similarity
- Y-axis: Effective rank (normalized)

Expected clusters:
```
        ↑ Effective Rank
        │
    70% │  ● OLMoE        (Type A: Native MoE)
        │  ● Mixtral?
        │
    50% │ ─ ─ ─ ─ ─ ─ ─ ─ ─ Decision boundary
        │
    20% │              ● GPT-OSS    (Type B: Dense→MoE)
        │              ● ???
        │
        └──────────────────────────────→ Cosine Similarity
           0.0       0.3       0.5
```

### Phase 3: Validate with Training Details

Where available, check model documentation for:
- "Upcycling" mentions
- "Dense initialization"
- "MoE from scratch"

Confirm classification matches known training history.

## Implications

### For Inference

| Type | Optimization Strategy |
|------|----------------------|
| Type A (Native MoE) | Tiered storage, workhorse concentration |
| Type B (Dense→MoE) | Overlay compression (6×), nearly dense inference |

### For Training

Type A wastes capacity:
- Trains 64 orthogonal experts
- Uses 3-8 "workhorses"
- 56% of parameters rarely activated

Type B is more efficient:
- Shared base captures common computation
- Deltas specialize for routing
- All parameters contribute

**Recommendation:** Dense→MoE should be preferred. Get the best of both worlds—dense pretraining quality + MoE routing flexibility.

### For Architecture

This reframes what "experts" means:

| Architecture | What Experts Are |
|--------------|-----------------|
| Native MoE | Independent transformations fighting for tokens |
| Dense→MoE | Shared transformation with routing-dependent adjustments |

Dense→MoE is closer to the "subspace gating" insight—one MLP, dynamically adjusted by task.

## Specific Question: DeepSeek-V3

**If Type A (Native MoE):**
- 671B cannot compress
- Need tiered inference
- Hot/warm/cold with prefetch
- Still runs on Mac, but complex

**If Type B (Dense→MoE):**
- 671B → ~110GB with 6× compression
- Fits entirely in 128GB unified memory
- Nearly dense inference
- Much simpler

**How to find out:**
1. Download DeepSeek-V3 weights (or shard)
2. Run SVD analysis on 2-3 layers
3. Check gate projection rank
4. If gate rank ≈ 1 → Type B → easy path

## Broader Research Questions

1. **Does expert count affect type?**
   - Fewer experts (8) more likely to stay clustered?
   - Many experts (64+) more likely to diverge?

2. **Does layer depth affect structure?**
   - Early layers more clustered (shared preprocessing)?
   - Late layers more diverse (specialized outputs)?

3. **Can you convert Type A → Type B?**
   - Distill orthogonal experts into base + deltas?
   - Lose some capacity, gain compression?

4. **Should new MoE models train Dense→MoE?**
   - Better parameter efficiency
   - Compressible by construction
   - Maintains routing benefits

## The Real Architecture of GPT-OSS

What OpenAI calls "32 experts" is actually:

```
┌─────────────────────────────────────────────────────────────────┐
│  GPT-OSS "MoE" Layer - What It Really Is                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input ──→ [SHARED GATE] ──→ routing weights                    │
│       │                                                         │
│       └──→ [SHARED UP PROJ] ──→ intermediate                    │
│                    │                                            │
│                    ▼                                            │
│            ┌──────────────────────────────────┐                 │
│            │  SELECT ONE OF 32 DOWN PROJS     │                 │
│            │  (only part that actually varies) │                 │
│            └──────────────────────────────────┘                 │
│                    │                                            │
│                    ▼                                            │
│               Output                                            │
│                                                                 │
│  This is NOT 32 experts.                                        │
│  This is 1 expert with 32 output modes.                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Deliverables

1. **Classification tool** (done)
   - `experiments/expert_svd_analysis/analyze_expert_deltas.py`
   - Input: MoE model path
   - Output: Type A or Type B, with metrics

2. **Model taxonomy** (in progress)
   - This document
   - Table of all major MoE models
   - Training history (known or inferred)

3. **Paper/blog post** (future)
   - "Two Architectures of MoE"
   - Training implications
   - Inference implications

## Why This Matters

The MoE field has been treating all MoE models as the same architecture. They're not.

Knowing the type determines:
- Which inference optimizations work
- How much compression is possible
- Whether overlay/tiering/pruning applies

This is **model archaeology**—understanding what training produced, directly from the weights.

## Running the Analysis

```bash
# Test on OLMoE (Type A - orthogonal)
python experiments/expert_svd_analysis/analyze_expert_deltas.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --layers 0,5,10,15

# Test on GPT-OSS (Type B - pseudo-MoE)
python experiments/expert_svd_analysis/analyze_expert_deltas.py \
    --model openai/gpt-oss-20b \
    --layers 0,6,12,18,23
```

## References

- Original discovery: January 2025, SVD analysis comparing OLMoE vs GPT-OSS
- Gate rank-1 observation: GPT-OSS full layer sweep
- Workhorse concentration: Expert utilization profiling experiments
