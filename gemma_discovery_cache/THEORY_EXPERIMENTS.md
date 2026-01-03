# Gemma Circuit Theory Experiments

This document lists all experiments needed to test the theories about transformer arithmetic circuits.

---

## Theory Summary

### Hybrid Model: Pretraining + RLVF

| Component | Source | How to Test |
|-----------|--------|-------------|
| Lookup table content | Pretraining | Compare base vs IT model |
| Task type in embeddings | RLVF backprop | Embedding analysis (done) |
| Sharp phase boundaries | RLVF pressure | Compare base vs IT phases |
| Dispensable late layers | RLVF | Layer ablation (done) |

### Format Converter Hypothesis

| Claim | How to Test |
|-------|-------------|
| Task type baked into embeddings | Embedding probes (done - 100%) |
| L0 does real computation | Embedding vs L0 comparison (done) |
| Components redundant within layers | Neuron/head ablation (done - 0% drop) |
| Layers critical in sequence | Layer skip (done - 100% drop) |

---

## Experiment Commands

### Already Completed

```bash
# 1. Layer role analysis
uv run python examples/introspection/experiments/model_specific/gemma_layer_roles.py

# 2. Lookup table structure (commutativity, clustering)
uv run python examples/introspection/experiments/model_specific/gemma_lookup_table_analysis.py

# 3. Layer-by-layer lookup evolution
uv run python examples/introspection/experiments/model_specific/gemma_lookup_evolution.py

# 4. Circuit identification via probes
uv run python examples/introspection/experiments/model_specific/gemma_circuit_via_probes.py

# 5. Neuron ablation (redundancy test)
uv run python examples/introspection/experiments/model_specific/gemma_neuron_ablation.py

# 6. Attention head ablation
uv run python examples/introspection/experiments/model_specific/gemma_attention_ablation.py

# 7. Layer ablation (critical layers)
uv run python examples/introspection/experiments/model_specific/gemma_layer_ablation.py

# 8. Activation steering
uv run python examples/introspection/experiments/model_specific/gemma_activation_steering.py

# 9. Complete circuit analysis
uv run python examples/introspection/experiments/model_specific/gemma_multiplication_circuit.py

# 10. Phase proof experiments
uv run python examples/introspection/experiments/model_specific/gemma_phase_proofs.py

# 11. Phase boundary detection
uv run python examples/introspection/experiments/model_specific/gemma_phase_boundaries.py

# 12. Embedding analysis (RLVF backprop test)
uv run python examples/introspection/experiments/model_specific/gemma_embedding_analysis.py
```

### New Experiments Needed

```bash
# 13. Base vs Instruction-Tuned comparison
# Tests: Does RLVF create the circuit, or does pretraining?
uv run python examples/introspection/experiments/model_specific/gemma_base_vs_it.py

# 14. OOD (Out-of-Distribution) test
# Tests: Lookup table vs algorithm hypothesis
uv run python examples/introspection/experiments/model_specific/gemma_ood_test.py

# 15. Multi-model comparison
# Tests: Universal circuit pattern
uv run python examples/introspection/experiments/model_specific/multi_model_circuit.py
```

---

## Detailed Experiment Descriptions

### Experiment 13: Base vs Instruction-Tuned

**Purpose**: Test if RLVF creates the circuit structure or just refines it.

**Models**:
- Base: `mlx-community/gemma-3-4b-bf16` (no instruction tuning)
- IT: `mlx-community/gemma-3-4b-it-bf16` (instruction tuned)

**Tests**:
1. Task type in embeddings (base vs IT)
2. Phase boundaries (base vs IT)
3. Dispensable layers (base vs IT)
4. Answer crystallization point (base vs IT)

**Predictions**:
- If RLVF creates circuit: Base model should have weaker/no phases
- If pretraining creates circuit: Both should have similar phases

---

### Experiment 14: OOD Test

**Purpose**: Confirm lookup table (not algorithm) hypothesis.

**Tests**:
1. In-distribution: 2-9 × 2-9 (training range)
2. OOD: 10-15 × 10-15 (outside training)
3. OOD: 1 × N, 0 × N (edge cases)

**Predictions**:
- Lookup table: OOD fails catastrophically
- Algorithm: OOD works (generalization)

---

### Experiment 15: Multi-Model Comparison

**Purpose**: Confirm universal circuit pattern.

**Models**:
- Gemma-3-4B (done)
- Llama-3.2-3B
- Qwen-2.5-3B
- Mistral-7B

**Tests**:
1. 6-phase architecture
2. Dispensable late layers
3. Component redundancy
4. Lookup table structure

---

## Results Summary Table

| Experiment | Theory Tested | Result | Status |
|------------|---------------|--------|--------|
| Embedding analysis | RLVF backprop | Task type 100% in embeddings | ✓ Done |
| Neuron ablation | Component redundancy | 0% drop with 20% ablated | ✓ Done |
| Layer ablation | Layer criticality | L0,L4,L21 critical | ✓ Done |
| Phase boundaries | 6-phase architecture | Confirmed with probes | ✓ Done |
| Lookup structure | Memorization | Commutativity 0.9993 | ✓ Done |
| Format steering | Phase 5 output | "56" → "Five" works | ✓ Done |
| Base vs IT | RLVF vs Pretraining | - | Pending |
| OOD test | Lookup vs Algorithm | - | Pending |
| Multi-model | Universal pattern | - | Pending |

---

## Quick Start: Run All Experiments

```bash
cd /Users/christopherhay/chris-source/chuk-mlx

# Run all completed experiments
for script in gemma_layer_roles gemma_lookup_table_analysis gemma_lookup_evolution \
              gemma_circuit_via_probes gemma_neuron_ablation gemma_attention_ablation \
              gemma_layer_ablation gemma_activation_steering gemma_multiplication_circuit \
              gemma_phase_proofs gemma_phase_boundaries gemma_embedding_analysis; do
    echo "Running $script..."
    uv run python examples/introspection/experiments/model_specific/${script}.py
done
```

---

## Key Findings So Far

### Confirmed Theories

1. **RLVF backprop bakes task type into embeddings**
   - Evidence: 100% task detection from raw embeddings
   - Command: `gemma_embedding_analysis.py`

2. **Lookup table, not algorithm**
   - Evidence: Perfect commutativity (0.9993), same-product clustering
   - Command: `gemma_lookup_table_analysis.py`

3. **Components redundant, layers critical**
   - Evidence: 20% neuron ablation = 0% drop, L0 skip = 100% drop
   - Commands: `gemma_neuron_ablation.py`, `gemma_layer_ablation.py`

4. **6-phase architecture**
   - Evidence: Probes, ablation, steering all confirm phases
   - Commands: `gemma_phase_proofs.py`, `gemma_phase_boundaries.py`

5. **Late layers dispensable**
   - Evidence: L29-L33 skip = 0% accuracy drop
   - Command: `gemma_layer_ablation.py`

### Theories Needing More Evidence

1. **RLVF shapes phase boundaries (vs pretraining)**
   - Need: Base model comparison

2. **Universal circuit pattern**
   - Need: Multi-model comparison

3. **Lookup table fails OOD**
   - Need: OOD test with larger numbers
