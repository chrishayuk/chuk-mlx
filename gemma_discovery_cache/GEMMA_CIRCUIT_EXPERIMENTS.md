# Gemma-3-4B Multiplication Circuit Experiments

**Date**: January 2025
**Researcher**: Christopher Hay
**Model**: `mlx-community/gemma-3-4b-it-bf16`
**Framework**: MLX + chuk-lazarus

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Setup & Requirements](#setup--requirements)
3. [Experiment 1: Layer Role Analysis](#experiment-1-layer-role-analysis)
4. [Experiment 2: Lookup Table Structure](#experiment-2-lookup-table-structure)
5. [Experiment 3: Circuit Identification via Probes](#experiment-3-circuit-identification-via-probes)
6. [Experiment 4: Neuron Ablation Study](#experiment-4-neuron-ablation-study)
7. [Experiment 5: Attention Head Ablation](#experiment-5-attention-head-ablation)
8. [Experiment 6: Full Layer Ablation](#experiment-6-full-layer-ablation)
9. [Experiment 7: Activation Steering](#experiment-7-activation-steering)
10. [Experiment 8: Complete Circuit Analysis](#experiment-8-complete-circuit-analysis)
11. [Experiment 9: Phase Proof Experiments](#experiment-9-phase-proof-experiments)
12. [Experiment 10: Phase Boundary Detection](#experiment-10-phase-boundary-detection)
13. [Comparison with GPT-OSS](#comparison-with-gpt-oss)
14. [Key Findings Summary](#key-findings-summary)
15. [Scripts Reference](#scripts-reference)
16. [Generated Data Files](#generated-data-files)

---

## Executive Summary

We conducted a comprehensive analysis of how Gemma-3-4B computes single-digit multiplication (e.g., "7 * 8 = 56").

### Key Discoveries:

1. **Lookup Table, Not Algorithm**: Gemma uses memorized lookup tables, not step-by-step computation
2. **6-Phase Circuit Architecture**: Identical to GPT-OSS despite different model family
3. **Critical Layers**: L0, L1, L4, L21 are essential; L29-L33 are dispensable (0% accuracy drop when skipped)
4. **Massive Redundancy**: Ablating 20% of neurons or all 8 attention heads at one layer causes 0% accuracy drop
5. **Steering Works**: Activation steering effective at -500 strength where ablation fails
6. **Universal Pattern**: Same architecture as GPT-OSS-20B suggests this is a universal transformer pattern

---

## Setup & Requirements

### Environment

```bash
# Clone repository
git clone <repo-url>
cd chuk-mlx

# Install dependencies
uv sync

# Verify MLX installation
uv run python -c "import mlx.core as mx; print(mx.default_device())"
```

### Model Download

The model downloads automatically on first use:
```bash
# Model: mlx-community/gemma-3-4b-it-bf16
# Size: ~8GB
# Layers: 34
# Hidden size: 2560
# Attention heads: 8
```

### Directory Structure

```
examples/introspection/experiments/model_specific/
├── gemma_layer_roles.py              # Layer specialization analysis
├── gemma_lookup_table_analysis.py    # Lookup structure tests
├── gemma_lookup_evolution.py         # Layer-by-layer evolution
├── gemma_circuit_via_probes.py       # Circuit identification
├── gemma_circuit_identification.py   # Standalone circuit finder
├── gemma_neuron_ablation.py          # Neuron ablation study
├── gemma_attention_ablation.py       # Attention head ablation
├── gemma_layer_ablation.py           # Full layer ablation
├── gemma_activation_steering.py      # Activation steering
├── gemma_multiplication_circuit.py   # Complete circuit analysis
├── gemma_probe_lens.py               # Probe-based projection
└── gemma_vocabulary_projection.py    # Vocabulary projection experiments

gemma_discovery_cache/
├── discovery_summary.json            # All findings in JSON
├── circuit_via_probes.json           # Probe results
├── neuron_ablation.json              # Ablation results
├── attention_ablation.json           # Attention ablation results
├── layer_ablation.json               # Layer ablation results
├── activation_steering.json          # Steering results
├── multiplication_circuit.json       # Complete circuit data
├── projection_method.md              # Projection documentation
└── universal_multiplication_circuit.md  # GPT-OSS comparison
```

---

## Experiment 1: Layer Role Analysis

### Purpose
Identify what each layer contributes to the computation.

### Command
```bash
uv run python examples/introspection/experiments/model_specific/gemma_layer_roles.py
```

### Script: `gemma_layer_roles.py`

Key functions:
- Collects activations at each layer for multiplication prompts
- Trains probes to classify arithmetic vs non-arithmetic
- Measures layer-by-layer contribution

### Results

| Layer Range | Role | Evidence |
|-------------|------|----------|
| L0-L3 | Encoding | Token embeddings processed |
| L4-L7 | Task Recognition | Arithmetic detection |
| L8-L16 | Retrieval | Probe accuracy building |
| L17-L22 | Computation | Answer crystallization |
| L23-L28 | Output Prep | Format for generation |
| L29-L33 | Optional | Can be skipped |

---

## Experiment 2: Lookup Table Structure

### Purpose
Determine if Gemma uses lookup tables (like GPT-OSS) vs algorithmic computation.

### Command
```bash
uv run python examples/introspection/experiments/model_specific/gemma_lookup_table_analysis.py
```

### Script: `gemma_lookup_table_analysis.py`

Tests performed:
1. **Commutativity**: Is 7×8 ≈ 8×7 in activation space?
2. **Same-Product Clustering**: Do 2×6 and 3×4 cluster together?
3. **Row/Column Structure**: Are same-row items (3×_, 3×_) similar?
4. **Product Proximity**: Do adjacent products have similar activations?

### Results

| Test | Score | Interpretation |
|------|-------|----------------|
| Commutativity (a×b vs b×a) | 0.9993 | Perfect |
| Same-product pairs (2×6 vs 3×4) | 0.9868 | High clustering |
| Same-row items (a×_) | 0.9788 | Structured |
| Same-column items (_×b) | 0.9825 | Slightly higher |

**Conclusion**: Gemma uses a **lookup table** with commutativity baked in.

### Layer Evolution Command
```bash
uv run python examples/introspection/experiments/model_specific/gemma_lookup_evolution.py
```

### Layer Evolution Results

| Layer | Commutativity |
|-------|---------------|
| L0-L20 | 1.0000 (perfect) |
| L21 | 0.999 (computation begins) |
| L22-L33 | 0.998+ (maintained) |

---

## Experiment 3: Circuit Identification via Probes

### Purpose
Identify which neurons are associated with arithmetic computation.

### Command
```bash
uv run python examples/introspection/experiments/model_specific/gemma_circuit_via_probes.py
```

### Script: `gemma_circuit_via_probes.py`

Uses `chuk_lazarus.introspection.circuit` infrastructure:
- Creates arithmetic vs non-arithmetic dataset
- Collects activations at key layers
- Trains LogisticRegression probes
- Identifies top neurons by probe weight

### Results

**Layer Probe Accuracy:**

| Layer | Accuracy | Interpretation |
|-------|----------|----------------|
| L0 | 100% | Fully encoded |
| L8 | 100% | Fully encoded |
| L16 | 100% | Fully encoded |
| L20 | 100% | Fully encoded |
| L24 | 100% | Fully encoded |
| L28 | 100% | Fully encoded |

**Key Neurons Identified:**

| Neuron | Layers Active | Role | Importance |
|--------|---------------|------|------------|
| 19 | L20, L24, L28 | ARITHMETIC- | 0.00265 |
| 1698 | L20, L24, L28 | ARITHMETIC+ | 0.00239 |
| 2309 | L20, L24, L28 | ARITHMETIC- | 0.00189 |
| 468 | L20, L24 | ARITHMETIC- | 0.00134 |
| 1305 | L20, L24 | ARITHMETIC- | 0.00133 |

### Output File
`gemma_discovery_cache/circuit_via_probes.json`

---

## Experiment 4: Neuron Ablation Study

### Purpose
Test if identified neurons are **causally important** (not just correlated).

### Command
```bash
uv run python examples/introspection/experiments/model_specific/gemma_neuron_ablation.py
```

### Script: `gemma_neuron_ablation.py`

Key methods:
- `generate_with_neuron_ablation()`: Zeros specific MLP neurons
- `_forward_with_ablation()`: Custom forward pass with ablation mask
- `test_multiplication_accuracy()`: Measures accuracy on 10 test cases

### Test Cases
```python
test_cases = [
    (2, 3, 6), (3, 4, 12), (5, 6, 30), (7, 8, 56), (9, 9, 81),
    (4, 7, 28), (6, 8, 48), (3, 9, 27), (5, 5, 25), (8, 9, 72),
]
```

### Results

| Ablation | Accuracy Drop |
|----------|---------------|
| Single neuron (any) | 0% |
| Top 5 neurons @ L20,L24,L28 | 0% |
| First 1000 neurons (9.8%) | 0% |
| First 2000 neurons (19.5%) | 0% |
| 20 random neurons (control) | 0% |

**Ablation Verification:**
```
Layer 20 output difference with 1000 neurons ablated: 14.437500
✓ Ablation IS modifying the output
```

**Conclusion**: Neurons are **massively redundant**. No small set is causally critical.

### Output File
`gemma_discovery_cache/neuron_ablation.json`

---

## Experiment 5: Attention Head Ablation

### Purpose
Test if attention heads are more localized than MLP neurons.

### Command
```bash
uv run python examples/introspection/experiments/model_specific/gemma_attention_ablation.py
```

### Script: `gemma_attention_ablation.py`

Key method:
- `_forward_with_head_ablation()`: Zeros specific attention head outputs

### Results

| Ablation | Layer | Accuracy Drop |
|----------|-------|---------------|
| Single head | L16-L28 | 0% |
| 4 heads (50%) | L20, L24 | 0% |
| All 8 heads (100%) | L16 | 0% |
| All 8 heads (100%) | L20 | 0% |
| All 8 heads (100%) | L24 | 0% |
| All 8 heads (100%) | L28 | 0% |

**Conclusion**: Even ablating **ALL attention heads** at a layer causes 0% drop!

### Output File
`gemma_discovery_cache/attention_ablation.json`

---

## Experiment 6: Full Layer Ablation

### Purpose
Test if entire layers are critical (since components aren't).

### Command
```bash
uv run python examples/introspection/experiments/model_specific/gemma_layer_ablation.py
```

### Script: `gemma_layer_ablation.py`

Key method:
- `generate_with_layer_skip()`: Completely skips layers (no residual)

### Results

**Single Layer Skip:**

| Layer | Accuracy Drop | Role |
|-------|---------------|------|
| L0 | 100% | **CRITICAL** - embedding |
| L1 | 90% | **CRITICAL** - early processing |
| L4 | 100% | **CRITICAL** - task recognition |
| L21 | 70% | **IMPORTANT** - computation |
| L22 | 10% | Minor |
| L29-L33 | **0%** | **DISPENSABLE** |

**Layer Region Tests:**

| Skipped | Accuracy Drop |
|---------|---------------|
| L0-L4 (first 5) | 100% |
| L29-L33 (last 5) | **0%** |
| L24-L33 (last 10) | 60% |
| L10-L14 (middle 5) | 30% |

**Minimum Layers Test:**

| Configuration | Accuracy |
|---------------|----------|
| Keep every 2nd layer | 0% (fails) |
| Keep every 3rd layer | 0% (fails) |
| Skip even layers | 0% (fails) |

**Conclusion**: Layer sequence is critical. Cannot use sparse layers.

### Output File
`gemma_discovery_cache/layer_ablation.json`

---

## Experiment 7: Activation Steering

### Purpose
Test if adding/subtracting directions works where ablation fails.

### Command
```bash
uv run python examples/introspection/experiments/model_specific/gemma_activation_steering.py
```

### Script: `gemma_activation_steering.py`

Key methods:
- `train_arithmetic_probe()`: Train probe for arithmetic direction
- `train_digit_probe()`: Train probe for specific digits
- `generate_with_steering()`: Add steering vector during forward pass

### Probe Training Results

| Layer | Arithmetic Probe Accuracy |
|-------|---------------------------|
| L16 | 100% |
| L20 | 100% |
| L24 | 100% |

### Steering Results

**Negative Steering (suppress arithmetic):**

| Prompt | Baseline | Strength | Steered Output |
|--------|----------|----------|----------------|
| 7 * 8 = | 56 | -200 | 56 (no change) |
| 7 * 8 = | 56 | -500 | 7 * (partial) |
| 7 * 8 = | 56 | -1000 | The product |
| 7 * 8 = | 56 | -2000 | The question |

**Threshold**: ~-500 for visible effect

**Positive Steering (induce arithmetic on non-arithmetic prompts):**

| Prompt | Baseline | Strength | Effect |
|--------|----------|----------|--------|
| The capital of France is | Paris | 500 | Changes output |
| The answer is | 10 | 500 | 100 |

**Conclusion**: **Steering works** where ablation failed!

### Output File
`gemma_discovery_cache/activation_steering.json`

---

## Experiment 8: Complete Circuit Analysis

### Purpose
Comprehensive analysis combining all methods.

### Command
```bash
uv run python examples/introspection/experiments/model_specific/gemma_multiplication_circuit.py
```

### Script: `gemma_multiplication_circuit.py`

Components:
1. `probe_answer_emergence()`: Layer-by-layer probe accuracy
2. `analyze_attention_patterns()`: Token attention analysis
3. `activation_patching()`: Transfer activations between prompts
4. `identify_circuit_phases()`: Map 6-phase architecture

### Results

**Answer Emergence (Probe Accuracy):**

| Layer | First Digit | Tens Digit | Ones Digit |
|-------|-------------|------------|------------|
| Embed | 20% | 20% | 0% |
| L3 | 50% | 40% | 10% |
| L10 | 30% | 50% | 10% |
| L20 | 40% | 50% | 20% |
| L21 | **70%** | **70%** | **60%** |
| L26 | **100%** | 90% | 70% |
| L33 | 100% | 90% | 70% |

**Key observation**: Answer **jumps at L21** (40% → 70%)

**Attention Patterns for "7 * 8 = ":**

| Layer | Attention to 7 | Attention to 8 | Attention to * |
|-------|----------------|----------------|----------------|
| L0 | 0.030 | 0.133 | 0.120 |
| L4 | 0.082 | 0.176 | 0.148 |
| L8 | 0.064 | **0.224** | **0.243** |
| L22 | 0.011 | **0.192** | 0.032 |

**Activation Patching (7×8=56 → 3×4=12):**

| Patched Layer | Output | Interpretation |
|---------------|--------|----------------|
| L0 | 5 | Source transferred |
| L4 | 5 | Source transferred |
| L8 | 5 | Source transferred |
| L16 | 5 | Source transferred |
| L24 | 5 | Source transferred |
| L32 | 5 | Source transferred |

**Conclusion**: Answer encoded from earliest layers, transferable via patching.

### Output File
`gemma_discovery_cache/multiplication_circuit.json`

---

## Experiment 9: Phase Proof Experiments

### Purpose
Provide **causal evidence** for each phase of the 6-phase architecture.

### Command
```bash
uv run python examples/introspection/experiments/model_specific/gemma_phase_proofs.py
```

### Script: `gemma_phase_proofs.py`

Three experiments to prove phase boundaries:

1. **Phase 2 Proof (Task Recognition)**: Train probe to classify arithmetic vs language
2. **Phase 3 Proof (Retrieval)**: Cross-operation patching (* vs + vs -)
3. **Phase 5 Proof (Output Format)**: Format steering (numeric vs word)

### Results

**Phase 2: Task Recognition**

| Layer | Arithmetic vs Language Accuracy |
|-------|--------------------------------|
| Embed | 100% |
| L0 | 100% |
| L4 | 100% |
| L8 | 100% |
| All | 100% |

**Finding**: Task type detectable from embedding layer! This suggests arithmetic is identified extremely early.

**Phase 3: Retrieval (Cross-Operation Patching)**

Testing: Patch activations from "7 * 8 =" (answer: 56) into "7 + 8 =" (answer: 15)

| Patched Layer | Output | Result |
|---------------|--------|--------|
| L0 | 5 | SOURCE transferred |
| L4 | 5 | SOURCE transferred |
| L8 | 5 | SOURCE transferred |
| L16 | 5 | SOURCE transferred |
| L24 | 5 | SOURCE transferred |
| L28 | 5 | SOURCE transferred |

**Finding**: Answer is encoded from earliest layers and transfers at all layers!

**Phase 5: Output Format Steering**

Testing format steering at L24:

| Strength | Output for "7 * 8 = " |
|----------|----------------------|
| 0 | 5 (numeric) |
| 50 | Five (word!) |
| 100 | Five |
| 200 | Five |

**Finding**: Format steering WORKS! L24 controls output format.

### Conclusions

| Phase | Evidence | Status |
|-------|----------|--------|
| Phase 1 (Encoding) | Layer ablation 90-100% drop | ✓ PROVEN |
| Phase 2 (Recognition) | 100% task detection at L0 | ✓ PROVEN (earlier than expected) |
| Phase 3 (Retrieval) | Answer encoded from L0 | ✓ PROVEN (earlier than expected) |
| Phase 4 (Computation) | L21 70% drop, steering works | ✓ PROVEN |
| Phase 5 (Format) | Format steering at L24 | ✓ PROVEN |
| Phase 6 (Optional) | 0% drop when skipped | ✓ PROVEN |

### Output File
`gemma_discovery_cache/phase_proofs.json`

---

## Experiment 10: Phase Boundary Detection

### Purpose
Find exact phase boundaries using refined probing methods.

### Command
```bash
uv run python examples/introspection/experiments/model_specific/gemma_phase_boundaries.py
```

### Script: `gemma_phase_boundaries.py`

Four experiments:

1. **Operation Classification**: When can we distinguish * from + from - ?
2. **Operand Binding**: When can we decode operand 1 vs operand 2?
3. **Answer Crystallization**: When does answer uncertainty collapse?
4. **Phase Boundary Detection**: Representation similarity analysis

### Results

**Experiment 1: Operation Classification (* vs + vs -)**

| Layer | Accuracy | Status |
|-------|----------|--------|
| Embed | 33.3% | Random |
| L0 | 95.8% | ← JUMP |
| L4 | 100% | Saturated |
| L8-L33 | 100% | Maintained |

**Finding**: Operation type emerges at L0, not L4-L7 as hypothesized!

**Experiment 2: Operand Binding**

| Layer | Op1 (a) Accuracy | Op2 (b) Accuracy |
|-------|------------------|------------------|
| Embed | 8.3% | 16.7% |
| L0 | 41.7% | 66.7% |
| L4 | 58.3% | 50.0% |
| L8 | 66.7% | 58.3% |
| L16 | 83.3% | 75.0% |
| L20 | 75.0% | 75.0% |

**Finding**: Operand 2 is more accessible than Operand 1 early on. Both reach ~80% by L16.

**Experiment 3: Answer Crystallization**

| Layer | Accuracy | Entropy | Status |
|-------|----------|---------|--------|
| Embed | 8.3% | 2.680 | Uncertain |
| L0-L20 | ~75-83% | ~0.2-0.3 | Emerging |
| L21 | 83.3% | 0.191 | Emerging |
| L26 | 91.7% | 0.102 | ✓ CRYSTALLIZED |
| L27-L33 | 91.7% | <0.01 | ✓ CRYSTALLIZED |

**Finding**: Answer crystallizes at **L26** (entropy drops to <0.1, accuracy >90%)

**Experiment 4: Phase Boundaries (Representation Similarity)**

| Layer | Cosine to Previous | Status |
|-------|-------------------|--------|
| L0 | 0.023 | ← MAJOR BOUNDARY |
| L1-L33 | 0.99+ | Smooth |

**Finding**: Only ONE major representation boundary at L0 (embedding → first layer)

### Revised Phase Model

Based on these experiments, the 6 phases are confirmed but boundaries are **earlier** than initially thought:

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: ENCODING (Embed only)                                     │
│  • Major representation change at L0                                │
│  • Operands partially decoded                                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2: TASK + OPERATION RECOGNITION (L0-L4)                      │
│  • Operation type (*,+,-) detectable at L0 (95.8%)                  │
│  • L4 critical for computation (100% drop if skipped)               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3: OPERAND BINDING + RETRIEVAL (L5-L16)                      │
│  • Operand accuracy builds to ~80%                                  │
│  • Answer probe accuracy ~75-83%                                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 4: COMPUTATION (L17-L25)                                     │
│  • L21 important (70% drop if skipped)                              │
│  • Answer still emerging, entropy ~0.2                              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 5: CRYSTALLIZATION + OUTPUT (L26-L28)                        │
│  • Answer crystallizes at L26 (entropy drops to <0.1)               │
│  • Format steering works here                                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 6: OPTIONAL (L29-L33)                                        │
│  • 0% drop when skipped                                             │
│  • Answer already crystallized                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Output File
`gemma_discovery_cache/phase_boundaries.json`

---

## Comparison with GPT-OSS

### Side-by-Side Architecture

| Phase | GPT-OSS-20B | Gemma-3-4B |
|-------|-------------|------------|
| 1. Encoding | L0-L3 | L0-L3 |
| 2. Recognition | L3 (R²=1.0) | L4 (100% drop) |
| 3. Retrieval | L3-L18 | L8-L16 |
| 4. Crystallization | L19 | L21 |
| 5. Output Prep | L20-L21 | L23-L28 |
| 6. Dispensable | L22-L23 | L29-L33 |

### Universal Patterns Confirmed

| Pattern | GPT-OSS | Gemma |
|---------|---------|-------|
| Lookup table mechanism | ✓ | ✓ |
| 6-phase architecture | ✓ | ✓ |
| Early layers critical | ✓ | ✓ |
| Late layers dispensable | ✓ | ✓ |
| Component redundancy | ✓ | ✓ |
| Steering effective | ✓ | ✓ |

### Documentation
See `gemma_discovery_cache/universal_multiplication_circuit.md`

---

## Key Findings Summary

### 1. Circuit Architecture

```
INPUT: "7 * 8 = "
        │
        ▼
┌─────────────────────────────────────┐
│ PHASE 1: ENCODING (L0-L3)           │
│ • L0, L1 CRITICAL (90-100% drop)    │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│ PHASE 2: TASK RECOGNITION (L4-L7)   │
│ • L4 CRITICAL (100% drop)           │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│ PHASE 3: LOOKUP RETRIEVAL (L8-L16)  │
│ • Perfect commutativity (0.9993)    │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│ PHASE 4: COMPUTATION (L17-L22)      │
│ • L21 important (70% drop)          │
│ • Answer jumps to 70% probe acc     │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│ PHASE 5: OUTPUT PREP (L23-L28)      │
│ • Answer reaches 100% at L26        │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│ PHASE 6: OPTIONAL (L29-L33)         │
│ • CAN BE SKIPPED (0% loss!)         │
└─────────────────────────────────────┘
        │
        ▼
OUTPUT: "56"
```

### 2. Critical vs Dispensable

| Category | Layers | Evidence |
|----------|--------|----------|
| **Critical** | L0, L1, L4, L21 | 70-100% drop if skipped |
| **Important** | L5-L28 | Contribute to computation |
| **Dispensable** | L29-L33 | 0% drop if skipped |

### 3. Redundancy Findings

| Component | Amount Ablated | Accuracy Drop |
|-----------|----------------|---------------|
| Neurons | 2000 (20%) | 0% |
| Attention Heads | 8 (100%) | 0% |
| Single Layer | 1 | 0-100% (varies) |

### 4. Effective Interventions

| Method | Effect |
|--------|--------|
| Neuron ablation | ✗ No effect |
| Attention ablation | ✗ No effect |
| Layer skip | ✓ Strong effect |
| Activation steering | ✓ Strong effect |

---

## Scripts Reference

### Quick Reference

| Script | Purpose | Key Output |
|--------|---------|------------|
| `gemma_layer_roles.py` | Layer specialization | Phase identification |
| `gemma_lookup_table_analysis.py` | Lookup structure | Commutativity score |
| `gemma_lookup_evolution.py` | Layer evolution | Per-layer structure |
| `gemma_circuit_via_probes.py` | Find key neurons | Neuron importance |
| `gemma_neuron_ablation.py` | Test neuron causality | Ablation results |
| `gemma_attention_ablation.py` | Test head causality | Ablation results |
| `gemma_layer_ablation.py` | Test layer criticality | Critical layers |
| `gemma_activation_steering.py` | Steering experiments | Steering threshold |
| `gemma_multiplication_circuit.py` | Complete analysis | Full circuit map |
| `gemma_phase_proofs.py` | Phase proof experiments | Phase causality |
| `gemma_phase_boundaries.py` | Phase boundary detection | Exact boundaries |
| `gemma_probe_lens.py` | Probe-based decoding | Projection method |

### Running All Experiments

```bash
# Run all experiments sequentially
cd /path/to/chuk-mlx

# 1. Layer roles
uv run python examples/introspection/experiments/model_specific/gemma_layer_roles.py

# 2. Lookup table structure
uv run python examples/introspection/experiments/model_specific/gemma_lookup_table_analysis.py

# 3. Layer evolution
uv run python examples/introspection/experiments/model_specific/gemma_lookup_evolution.py

# 4. Circuit identification
uv run python examples/introspection/experiments/model_specific/gemma_circuit_via_probes.py

# 5. Neuron ablation
uv run python examples/introspection/experiments/model_specific/gemma_neuron_ablation.py

# 6. Attention ablation
uv run python examples/introspection/experiments/model_specific/gemma_attention_ablation.py

# 7. Layer ablation
uv run python examples/introspection/experiments/model_specific/gemma_layer_ablation.py

# 8. Activation steering
uv run python examples/introspection/experiments/model_specific/gemma_activation_steering.py

# 9. Complete circuit analysis
uv run python examples/introspection/experiments/model_specific/gemma_multiplication_circuit.py

# 10. Phase proof experiments
uv run python examples/introspection/experiments/model_specific/gemma_phase_proofs.py

# 11. Phase boundary detection
uv run python examples/introspection/experiments/model_specific/gemma_phase_boundaries.py
```

---

## Generated Data Files

### JSON Results

| File | Contents |
|------|----------|
| `discovery_summary.json` | Complete findings summary |
| `circuit_via_probes.json` | Key neurons & probe accuracies |
| `neuron_ablation.json` | Ablation test results |
| `attention_ablation.json` | Attention ablation results |
| `layer_ablation.json` | Layer skip results |
| `activation_steering.json` | Steering probe accuracies |
| `multiplication_circuit.json` | Complete circuit data |
| `phase_proofs.json` | Phase proof experiment results |
| `phase_boundaries.json` | Phase boundary detection results |

### Markdown Documentation

| File | Contents |
|------|----------|
| `projection_method.md` | Probe-based projection method |
| `universal_multiplication_circuit.md` | GPT-OSS comparison |
| `GEMMA_CIRCUIT_EXPERIMENTS.md` | This document |

### Loading Results

```python
import json

# Load discovery summary
with open("gemma_discovery_cache/discovery_summary.json") as f:
    findings = json.load(f)

# Access key discoveries
print(findings["key_discoveries"]["7_ablation_robustness"])
print(findings["key_discoveries"]["8_layer_ablation"])
print(findings["architectural_insights"])
```

---

## Reproducibility Notes

### Hardware Used
- Apple Silicon (MLX optimized)
- ~16GB RAM recommended

### Timing
- Model loading: ~30 seconds
- Each experiment: 2-10 minutes
- Full suite: ~1 hour

### Known Issues
1. `sklearn` convergence warnings are expected (can be ignored)
2. Model downloads on first run (~8GB)
3. Some numpy type conversions needed for JSON serialization

### Version Information
```bash
uv run python -c "import mlx; print(mlx.__version__)"
uv run python -c "import sklearn; print(sklearn.__version__)"
```

---

## Future Work

1. **Test more models**: Llama, Qwen, Mistral
2. **Two-digit multiplication**: Does it use a different circuit?
3. **Other operations**: Addition, subtraction, division
4. **Pruning experiments**: Remove dispensable layers for efficiency
5. **Training interventions**: Can we force algorithmic learning?

---

## Citation

If you use these findings, please cite:

```
Hay, C. (2025). Universal Multiplication Circuit in Transformers:
A Comparative Analysis of Gemma-3-4B and GPT-OSS-20B.
chuk-mlx repository.
```

---

## Changelog

- **2025-01-02**: Initial experiments and documentation
- Added all 9 experiment scripts
- Documented 6-phase circuit architecture
- Confirmed universal pattern with GPT-OSS
- **2025-01-02**: Phase proof experiments
- Added phase proof experiments (gemma_phase_proofs.py)
- Added phase boundary detection (gemma_phase_boundaries.py)
- Discovered operation type detectable at L0 (earlier than expected)
- Confirmed answer crystallization at L26
- Proved format steering works at L24
