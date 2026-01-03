# Gemma Experiments Using Lazarus CLI

The `lazarus introspect` CLI can run most of our experiments directly. Here's the mapping:

---

## Quick Reference

```bash
MODEL="mlx-community/gemma-3-4b-it-bf16"
```

---

## Experiment 1: Arithmetic Emergence

**What it tests**: When does the answer emerge across layers?

```bash
# Full arithmetic study
lazarus introspect arithmetic -m $MODEL --output arith_results.json

# Quick mode
lazarus introspect arithmetic -m $MODEL --quick

# Easy problems only (1-digit)
lazarus introspect arithmetic -m $MODEL --easy-only
```

---

## Experiment 2: Layer Ablation

**What it tests**: Which layers are critical?

```bash
# Ablate MLP at each layer
lazarus introspect ablate -m $MODEL \
    --prompt "7 * 8 = " \
    --component mlp \
    --criterion "56" \
    --verbose

# Ablate attention at each layer
lazarus introspect ablate -m $MODEL \
    --prompt "7 * 8 = " \
    --component attention \
    --criterion "56" \
    --verbose

# Ablate specific layers together
lazarus introspect ablate -m $MODEL \
    --prompt "7 * 8 = " \
    --layers "29,30,31,32,33" \
    --multi \
    --criterion "56"

# Test multiple prompts
lazarus introspect ablate -m $MODEL \
    --prompts "7*8=|3*4=|9*9=|5*6=" \
    --component mlp \
    --criterion "correct" \
    --output ablation_results.json
```

---

## Experiment 3: Task Classification Probe

**What it tests**: When can the model distinguish arithmetic from language?

```bash
# Train probe: arithmetic vs language
lazarus introspect probe -m $MODEL \
    --class-a "2*3=|7*8=|4+5=|9-2=" \
    --label-a "arithmetic" \
    --class-b "The cat sat|Hello world|Paris is" \
    --label-b "language" \
    --save-direction arithmetic_direction.npz

# Train probe: multiplication vs addition
lazarus introspect probe -m $MODEL \
    --class-a "2*3=|7*8=|4*5=|9*2=" \
    --label-a "multiplication" \
    --class-b "2+3=|7+8=|4+5=|9+2=" \
    --label-b "addition" \
    --save-direction operation_direction.npz
```

---

## Experiment 4: Direction Orthogonality

**What it tests**: Are arithmetic, operation, and format directions orthogonal?

```bash
# First extract directions
lazarus introspect probe -m $MODEL \
    --class-a "2*3=|7*8=" --label-a "math" \
    --class-b "The cat|Hello" --label-b "text" \
    --save-direction task_direction.npz

lazarus introspect probe -m $MODEL \
    --class-a "2*3=|7*8=" --label-a "mult" \
    --class-b "2+3=|7+8=" --label-b "add" \
    --save-direction op_direction.npz

# Compare orthogonality
lazarus introspect directions task_direction.npz op_direction.npz
```

---

## Experiment 5: Activation Steering

**What it tests**: Can we steer arithmetic behavior?

```bash
# Extract arithmetic direction
lazarus introspect steer -m $MODEL \
    --extract \
    --positive "7*8=56" \
    --negative "The cat sat on the mat" \
    --output arith_steer.npz

# Apply steering (suppress arithmetic)
lazarus introspect steer -m $MODEL \
    -p "7 * 8 = " \
    --direction arith_steer.npz \
    --compare "-500,-200,-100,0,100,200,500"

# Steer specific neuron
lazarus introspect steer -m $MODEL \
    -p "7 * 8 = " \
    --neuron 19 \
    --layer 20 \
    --compare "-100,-50,0,50,100"
```

---

## Experiment 6: Circuit Capture

**What it tests**: Can we capture and transfer computation?

```bash
# Capture multiplication circuit
lazarus introspect circuit capture -m $MODEL \
    --prompts "7*8=56|3*4=12|9*2=18" \
    --layer 21 \
    --save mult_circuit.npz

# View captured circuit
lazarus introspect circuit view mult_circuit.npz

# Test circuit transfer
lazarus introspect circuit test -m $MODEL \
    --circuit mult_circuit.npz \
    --test-prompts "5*6=|4*7="

# Compare multiplication vs addition circuits
lazarus introspect circuit capture -m $MODEL \
    --prompts "7+8=15|3+4=7|9+2=11" \
    --layer 21 \
    --save add_circuit.npz

lazarus introspect circuit compare mult_circuit.npz add_circuit.npz
```

---

## Experiment 7: Neuron Analysis

**What it tests**: Which neurons activate for arithmetic?

```bash
# Analyze neurons across prompts
lazarus introspect neurons -m $MODEL \
    --prompts "7*8=|3*4=|The cat|Hello world" \
    --layer 20 \
    --top-k 20 \
    --output neuron_analysis.json
```

---

## Experiment 8: Layer Representation Similarity

**What it tests**: How does representation change across layers?

```bash
# Compare layer representations
lazarus introspect layer -m $MODEL \
    --prompt "7 * 8 = " \
    --method cosine \
    --output layer_similarity.json
```

---

## Experiment 9: Activation Clustering

**What it tests**: Do arithmetic prompts cluster together?

```bash
# Cluster activations
lazarus introspect cluster -m $MODEL \
    --prompts "2*3=|7*8=|4*5=|The cat|Hello|Paris" \
    --layer 20 \
    --output cluster_viz.png
```

---

## Full Pipeline Example

```bash
MODEL="mlx-community/gemma-3-4b-it-bf16"
OUT="gemma_cli_results"
mkdir -p $OUT

# 1. Arithmetic emergence
lazarus introspect arithmetic -m $MODEL -o $OUT/arithmetic.json

# 2. Layer ablation
lazarus introspect ablate -m $MODEL \
    --prompts "7*8=|3*4=|9*9=" \
    --component mlp \
    -o $OUT/ablation_mlp.json

lazarus introspect ablate -m $MODEL \
    --prompts "7*8=|3*4=|9*9=" \
    --component attention \
    -o $OUT/ablation_attn.json

# 3. Probes
lazarus introspect probe -m $MODEL \
    --class-a "2*3=|7*8=|4*5=|9*2=" \
    --class-b "The cat|Hello world|Paris is|I went" \
    --label-a arithmetic --label-b language \
    --save-direction $OUT/task_dir.npz

lazarus introspect probe -m $MODEL \
    --class-a "2*3=|7*8=|4*5=" \
    --class-b "2+3=|7+8=|4+5=" \
    --label-a mult --label-b add \
    --save-direction $OUT/op_dir.npz

# 4. Orthogonality
lazarus introspect directions $OUT/task_dir.npz $OUT/op_dir.npz

# 5. Steering
lazarus introspect steer -m $MODEL \
    -p "7 * 8 = " \
    --direction $OUT/task_dir.npz \
    --compare "-500,-200,0,200,500" \
    -o $OUT/steering.json

# 6. Circuit capture
lazarus introspect circuit capture -m $MODEL \
    --prompts "7*8=56|3*4=12|9*2=18|6*5=30" \
    --layer 21 \
    --save $OUT/mult_circuit.npz

echo "Results saved to $OUT/"
```

---

## Experiment 10: Operand Direction Extraction

**What it tests**: Does the model use compositional (separate A/B subspaces) or holistic encoding?

```bash
# Analyze operand encoding structure
lazarus introspect operand-directions -m $MODEL \
    --digits 2,3,4,5,6,7,8,9 \
    --operation "*" \
    --layers 8,16,20,24 \
    --output operand_dirs.npz

# Quick check at default layers
lazarus introspect operand-directions -m $MODEL
```

**Key output**:
- A_i vs A_j: If low (<0.5), distinct operand directions (compositional)
- A_i vs B_j: If low (<0.3), orthogonal subspaces
- A_i vs B_i: If high (>0.8), digit identity dominates position

---

## Experiment 11: Embedding Analysis (RLVF Backprop Test)

**What it tests**: Is task type baked into embeddings (RLVF hypothesis)?

```bash
# Full embedding analysis
lazarus introspect embedding -m $MODEL \
    --output embedding_analysis.json

# Test specific operation
lazarus introspect embedding -m $MODEL --operation mult

# Analyze specific layers
lazarus introspect embedding -m $MODEL --layers 0,1,2,4
```

**Key output**:
- Task type from embeddings: If 100%, RLVF backprop confirmed
- Answer R² from embeddings: Should be low (computation required)

---

## Experiment 12: Commutativity Test

**What it tests**: Lookup table (memorization) vs algorithm hypothesis

```bash
# Test all commutative pairs
lazarus introspect commutativity -m $MODEL

# Test specific pairs
lazarus introspect commutativity -m $MODEL \
    --pairs "2*3,3*2|7*8,8*7|4*5,5*4" \
    --layer 20 \
    --output commutativity.json
```

**Key output**:
- Mean similarity >0.999: Strong evidence for lookup table
- Mean similarity <0.9: Model may use different algorithms for A*B vs B*A

---

## Experiment 13: Activation Patching

**What it tests**: Which layers encode computation vs operands?

```bash
# Patch multiplication into addition
lazarus introspect patch -m $MODEL \
    --source "7*8=" --target "7+8="

# Patch at specific layer
lazarus introspect patch -m $MODEL \
    --source "7*8=" --target "7+8=" \
    --layer 20

# Sweep all layers
lazarus introspect patch -m $MODEL \
    --source "7*8=" --target "7+8=" \
    --output patch_results.json
```

**Key output**:
- "TRANSFERRED!" at layer N: Answer production happens at layer N
- "no change": That layer doesn't encode the answer

---

## CLI Coverage Table

| Experiment | Custom Script | CLI Command |
|------------|---------------|-------------|
| Arithmetic emergence | `gemma_multiplication_circuit.py` | `introspect arithmetic` ✓ |
| Layer ablation | `gemma_layer_ablation.py` | `introspect ablate` ✓ |
| Task classification probe | `gemma_circuit_via_probes.py` | `introspect probe` ✓ |
| Direction orthogonality | - | `introspect directions` ✓ |
| Activation steering | `gemma_activation_steering.py` | `introspect steer` ✓ |
| Circuit capture | - | `introspect circuit capture` ✓ |
| Neuron analysis | `gemma_neuron_ablation.py` | `introspect neurons` ✓ |
| Layer representation | - | `introspect layer` ✓ |
| Activation clustering | - | `introspect cluster` ✓ |
| Operand A/B extraction | `gemma_orthogonal_extraction.py` | `introspect operand-directions` ✓ |
| Embedding analysis | `gemma_embedding_analysis.py` | `introspect embedding` ✓ |
| Commutativity test | `gemma_lookup_table_analysis.py` | `introspect commutativity` ✓ |
| Cross-operation patching | `gemma_phase_proofs.py` | `introspect patch` ✓ |
| Phase boundary detection | `gemma_phase_boundaries.py` | Partial: `introspect layer` |

---

## Video Script CLI Commands

Complete CLI commands for "Inside Gemma's Calculator" video:

```bash
MODEL="mlx-community/gemma-3-4b-it-bf16"

# ============================================================
# Section 1: Layer-by-Layer Emergence (6-Phase Architecture)
# ============================================================
# Shows when "56" emerges as top prediction across layers
lazarus introspect analyze -m $MODEL -p "7 * 8 = " --all-layers --raw

# ============================================================
# Section 2: Ablation Surprises
# ============================================================
# Which components are critical?
lazarus introspect ablate -m $MODEL -p "7 * 8 = " --component mlp -v
lazarus introspect ablate -m $MODEL -p "7 * 8 = " --component attention -v
lazarus introspect ablate -m $MODEL -p "7 * 8 = " --layers "29,30,31,32,33" --multi

# ============================================================
# Section 3: Task Recognition Probes
# ============================================================
# When can the model distinguish arithmetic from language?
lazarus introspect probe -m $MODEL \
    --class-a "2*3=|7*8=|4*5=" \
    --class-b "The cat|Hello|Paris" \
    --label-a arithmetic --label-b language

# ============================================================
# Section 4: Embedding Analysis (RLVF Backprop Hypothesis)
# ============================================================
# Is task type baked into embeddings before any computation?
lazarus introspect embedding -m $MODEL

# ============================================================
# Section 5: Operand Encoding (Holistic vs Compositional)
# ============================================================
# Does Gemma use separate A/B subspaces like GPT-OSS?
lazarus introspect operand-directions -m $MODEL

# ============================================================
# Section 6: Commutativity Test (Lookup Table Evidence)
# ============================================================
# Do 2*3 and 3*2 have identical representations?
lazarus introspect commutativity -m $MODEL

# ============================================================
# Section 7: Activation Steering
# ============================================================
# Can we steer the model's arithmetic behavior?
lazarus introspect steer -m $MODEL \
    -p "7 * 8 = " \
    --positive "math=56" --negative "The cat sat" \
    --compare "-500,-200,0,200,500"

# ============================================================
# Section 8: Cross-Operation Patching
# ============================================================
# Can we transfer multiplication computation into addition?
lazarus introspect patch -m $MODEL \
    --source "7*8=" --target "7+8="

# ============================================================
# Section 9: Circuit Capture
# ============================================================
# Capture and test the multiplication circuit
lazarus introspect circuit capture -m $MODEL \
    --prompts "7*8=56|3*4=12" \
    --layer 21 \
    --save circuit.npz

lazarus introspect circuit test -m $MODEL \
    --circuit circuit.npz \
    --test-prompts "5*6=|4*7="
```

---

## Expected Results Summary

| Command | Expected Finding |
|---------|-----------------|
| `analyze` | "56" emerges around layer 21, crystallizes by layer 26 |
| `ablate mlp` | 20% neurons ablated = 0% accuracy drop (redundancy) |
| `ablate attention` | Similar redundancy pattern |
| `ablate layers 29-33` | 0% drop - late layers dispensable |
| `probe` | 100% task classification from layer 0 |
| `embedding` | Task type 93-100% in raw embeddings (RLVF backprop confirmed) |
| `operand-directions` | A vs A: ~1.0 (holistic encoding, not compositional) |
| `commutativity` | Mean similarity 0.999 (lookup table confirmed) |
| `steer` | Can shift output with direction vectors |
| `patch` | No transfer between 7*8→7+8 (holistic = no composition) |
| `circuit capture` | Captures activation patterns for multiplication |
