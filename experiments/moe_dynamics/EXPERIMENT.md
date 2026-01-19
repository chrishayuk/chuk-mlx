# MoE Dynamics: Expert Behavior During Inference

## Overview

This experiment investigates expert behavior dynamics in MoE models through multiple complementary analyses:

1. **Cold Expert Investigation** - What do rarely-activated experts compute?
2. **Cross-Layer Expert Circuits** - Do specific expert combinations form functional units?
3. **Expert Dynamics During Generation** - How does routing change across autoregressive steps?
4. **Expert Interference** - When multiple experts activate, do they interfere?
5. **Expert Merging** - Can similar experts be merged without quality loss?
6. **Attention-Based Routing Prediction** - Can we predict routing from attention alone?
7. **Context-Aware Attention-Routing** - Does attention-routing correlation hold across contexts?
8. **Task-Aware Expert Loading** - Can we predict needed experts from early layers?
9. **Routing Manipulation** - Can we craft inputs that force specific routing?

## Research Questions

### 1. Cold Expert Investigation

**Question**: What do rarely-activated experts (<1% activation) compute? Are they dead weight or handling rare but critical cases?

**Method**:
- Identify cold experts across a diverse prompt corpus
- Construct inputs that specifically activate cold experts
- Measure impact of cold expert ablation on diverse tasks
- Analyze token patterns that trigger cold experts

**Metrics**:
- Activation frequency distribution
- Impact of ablation (perplexity delta)
- Token type patterns for cold expert activation

### 2. Cross-Layer Expert Circuits

**Question**: Do specific expert combinations across layers form functional units?

**Method**:
- Track expert co-occurrence across layers for the same tokens
- Identify stable "circuits" (e.g., L4-E17 → L8-E35 → L12-E3)
- Test if circuits correspond to specific computations (arithmetic, syntax, etc.)

**Expected Patterns**:
```
Arithmetic circuit:  L4-E17 → L8-E35 → L12-E3  (for "127 * 89 =")
Syntax circuit:      L4-E5  → L8-E12 → L12-E8  (for "def foo():")
```

### 3. Expert Dynamics During Generation

**Question**: How does expert selection evolve across autoregressive steps?

**Method**:
- Capture routing trace for each generated token
- Analyze expert "handoffs" between tokens
- Identify generation phase patterns (planning → execution?)
- Compare routing traces for correct vs incorrect outputs

**Analyses**:
- Expert consistency across generation
- Phase transitions in expert usage
- Error correlation with routing anomalies

### 4. Expert Interference

**Question**: When multiple experts activate (top-k), do they compute independently or interfere?

**Method**:
- Compare output of k=1 (single expert) vs k=4 (multiple experts)
- Measure if outputs are linear combinations
- Identify inputs where multi-expert causes degradation
- Test expert pair compatibility

### 5. Expert Merging

**Question**: Can experts with similar activation patterns be merged without quality loss?

**Method**:
- Compute expert similarity (activation correlation, weight similarity)
- Identify merge candidates
- Test merged expert quality on held-out prompts
- Find compression/quality Pareto frontier

### 6. Attention-Based Routing Prediction

**Question**: Can we predict routing from attention patterns alone?

**Background**: Prior work showed attention contributes 89-98% of routing signal. This tests if we can actually predict routing decisions from attention features.

**Method**:
- Extract attention patterns at each layer
- Train simple predictor: attention → expert selection
- Measure prediction accuracy across layers
- Analyze which attention features correlate with expert selection

### 7. Context-Aware Attention-Routing Correlation

**Question**: If attention drives routing context-dependently, do similar attention patterns yield similar routing?

**Background**: Experiment 6 found low prediction accuracy (4.3%) but strong individual correlations. This tests the context-dependency hypothesis more directly.

**Method**:
- Place the same token in N different contexts
- For each context: capture attention pattern and routing decision
- Compute pairwise similarity matrices for attention and routing
- Measure correlation: similar attention → similar routing?

**Hypothesis**: If attention drives routing, contexts with similar attention patterns should have similar routing decisions.

### 8. Task-Aware Expert Loading

**Question**: Can we predict which experts will be needed from early layer activations?

**Background**: L4 probe achieves high task accuracy. Can this predict expert needs for prefetching?

**Method**:
- Build predictor: L4 hidden state → expert usage for later layers
- Measure prediction accuracy
- Test selective expert loading based on prediction
- Measure memory/latency tradeoffs

### 9. Routing Manipulation

**Question**: Can we craft inputs that force specific routing patterns?

**Method**:
- Given target expert sequence, optimize input tokens
- Test adversarial inputs that cause misrouting
- Measure robustness of routing to perturbations

## Running the Experiment

### Via CLI (Recommended)

Each analysis is available as a standalone CLI command:

```bash
# Cold Expert Analysis
lazarus introspect moe-expert cold-experts -m openai/gpt-oss-20b

# Cross-Layer Expert Circuits
lazarus introspect moe-expert expert-circuits -m openai/gpt-oss-20b

# Generation Dynamics
lazarus introspect moe-expert generation-dynamics -m openai/gpt-oss-20b
lazarus introspect moe-expert generation-dynamics -m openai/gpt-oss-20b -p "def fibonacci(n):"

# Expert Interference
lazarus introspect moe-expert expert-interference -m openai/gpt-oss-20b

# Expert Merging
lazarus introspect moe-expert expert-merging -m openai/gpt-oss-20b -t 0.8

# Attention-Based Routing Prediction
lazarus introspect moe-expert attention-prediction -m openai/gpt-oss-20b

# Context-Aware Attention-Routing Correlation
lazarus introspect moe-expert context-attention-routing -m openai/gpt-oss-20b
lazarus introspect moe-expert context-attention-routing -m openai/gpt-oss-20b -t "127"

# Task-Aware Expert Prediction
lazarus introspect moe-expert task-prediction -m openai/gpt-oss-20b -p 4

# Routing Manipulation (resource intensive)
lazarus introspect moe-expert routing-manipulation -m openai/gpt-oss-20b
```

### Via Python Script

```bash
# Run all analyses
python experiments/moe_dynamics/experiment.py

# Run specific analysis
python experiments/moe_dynamics/experiment.py --analysis cold_experts
python experiments/moe_dynamics/experiment.py --analysis circuits
python experiments/moe_dynamics/experiment.py --analysis generation
python experiments/moe_dynamics/experiment.py --analysis interference
python experiments/moe_dynamics/experiment.py --analysis merging
python experiments/moe_dynamics/experiment.py --analysis attention_routing
python experiments/moe_dynamics/experiment.py --analysis context_attention_routing
python experiments/moe_dynamics/experiment.py --analysis task_aware
python experiments/moe_dynamics/experiment.py --analysis manipulation
```

## Results Summary

See [RESULTS.md](./RESULTS.md) for detailed findings. Key outcomes:

| Analysis | Expected | Actual | Verdict |
|----------|----------|--------|---------|
| Cold Experts | Some dead, some rare cases | 12.9% cold, 50 prunable | Pruning opportunity |
| Circuits | Functional circuits exist | 15 pipelines, 87.5% consistency | Strong structure confirmed |
| Generation Dynamics | Phase patterns exist | 45.9% consistency, phases detected | Dynamic routing confirmed |
| Interference | Non-linear interaction | k=4 essential, k=1 breaks output | Cooperation confirmed |
| Merging | 50%+ mergeable | 0% mergeable | Native MoE (orthogonal) |
| Attention Routing | >90% accuracy | 4.3% accuracy, but 0.906 correlation | Complex non-linear mapping |
| Context-Attention-Routing | Strong correlation | 0.32 at L12, weak overall | Context-dependent relationship |
| Task-Aware | High prediction | 94% accuracy, 53% prefetch | Prefetch viable |
| Manipulation | Routing manipulable | Not completed | Future work |

### Key Insights

1. **Native MoE Architecture**: GPT-OSS has orthogonal experts (0% mergeable), confirming it was trained from scratch as MoE, not upcycled from dense.

2. **Attention→Routing Paradox Resolved**: Attention provides 89-98% of routing signal (via hidden state), but the mapping is non-linear. Simple prediction fails (4.3%), but strong correlations exist (0.906).

3. **Middle Layers are Special**: L8-L17 show maximum differentiation, highest stability, and best predictability. This is where context-framing decisions are made.

4. **Multi-Expert Cooperation**: Reducing k=4 to k=1 breaks the model. Experts work cooperatively, not independently.

## Configuration

See `config.yaml` for parameters including:
- Model selection
- Analysis types to run
- Prompt datasets
- Thresholds and hyperparameters
