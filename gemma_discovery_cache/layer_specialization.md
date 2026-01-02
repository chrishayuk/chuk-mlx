# Gemma Layer Specialization Analysis

## Summary

Gemma-3-4B has clear **phase-based layer specialization**, different from GPT-OSS's expert-based MoE routing.

## Layer Phases

### Phase 1: Context Building (L0-L3)
- **Attention-heavy** (MLP/Attn ratio < 1.5)
- Building initial token relationships
- Collecting relevant context
- Attention norm: 700-1500, MLP norm: 200-1000

### Phase 2: Representation Building (L4-L10)
- **Transition zone** (ratio 0.6x to 5x)
- MLP gradually takes over
- Building task-specific representations
- Classification circuits active here

### Phase 3: Computation (L11-L23) ⭐
- **MLP explodes** (ratio 10x to 875x!)
- **KEY LAYER 17**: ratio 386x - major computation
- **KEY LAYER 23**: largest MLP norm (246k!) - peak computation
- Arithmetic, factual recall, reasoning happens here
- Attention nearly dormant

### Phase 4: Output Projection (L24-L33)
- **Extreme MLP dominance** (ratio 280x to 4370x)
- **KEY LAYER 29**: ratio 4370x - massive output transformation
- Projecting internal representations to vocabulary space

## Key Layers

| Layer | MLP/Attn Ratio | Role |
|-------|----------------|------|
| L17 | 386x | Major computation |
| L23 | 876x | Peak computation (largest MLP) |
| L29 | 4371x | Critical output transformation |

## Comparison with GPT-OSS

| Aspect | Gemma-3-4B | GPT-OSS-20B |
|--------|------------|-------------|
| Architecture | Dense MLP | Sparse MoE (32 experts) |
| Specialization | Phase-based | Expert-based |
| Computation location | L11-L23 (middle/late) | Expert routing |
| Peak computation | L17, L23 | Varies by expert |
| Attention role | Context only (early) | Context + routing |
| MLP role | All computation | Per-expert compute |

**Key Insight**: Both models do computation in the same relative position (middle-to-late layers), but Gemma uses DENSE layers with extreme MLP ratios while GPT-OSS routes to SPARSE experts.

## Implications for Interpretability

1. **Probing Target Layers**:
   - Task classification: L4-L10 (representation phase)
   - Arithmetic computation: L17-L23 (computation phase)
   - Output features: L29-L33 (output phase)

2. **Ablation Targets**:
   - L17: Key computation layer (386x MLP ratio)
   - L23: Peak MLP contribution (246k norm)
   - L29: Critical output layer (4370x ratio)

3. **Activation Steering**:
   - Inject directions in L11-L15 to influence computation
   - The computation phase (L16-L23) is where to steer

4. **Why Logit Lens Fails**:
   - Gemma's intermediate representations don't decode cleanly
   - The phase structure means each layer has a different "job"
   - Need to use PROBES not direct vocabulary projection

5. **Lookup Table Location**:
   - The multiplication table structure is in L17-L23
   - Same-row/column similarities are encoded in computation phase
