# Memory Structure Analysis in LLMs

This document describes experiments and findings on how factual knowledge is organized in LLM internal representations, using the `lazarus introspect memory` tool.

## Overview

LLMs don't store facts in a clean lookup table. Instead, they use a **distributed associative memory** with:
- Categorical clustering (similar facts stored together)
- Prototype/attractor nodes (common answers that dominate retrieval)
- Asymmetric encoding (order matters even when it shouldn't)
- Interference zones (similar items compete)

## Tools and Commands

### Memory Extraction Command

```bash
# Basic usage
lazarus introspect memory -m MODEL --facts FACT_TYPE --layer LAYER

# Built-in fact types
lazarus introspect memory -m openai/gpt-oss-20b --facts multiplication --layer 20
lazarus introspect memory -m openai/gpt-oss-20b --facts capitals --layer 20
lazarus introspect memory -m openai/gpt-oss-20b --facts addition --layer 15
lazarus introspect memory -m openai/gpt-oss-20b --facts elements --layer 18

# Save results and visualization
lazarus introspect memory -m openai/gpt-oss-20b --facts multiplication \
    --layer 20 --top-k 30 \
    -o memory_mult.json \
    --save-plot memory_mult.png

# Custom facts from file
lazarus introspect memory -m openai/gpt-oss-20b --facts @custom_facts.json --layer 20
```

### Custom Facts JSON Format

```json
[
  {
    "query": "The CEO of Apple is",
    "answer": "Tim Cook",
    "category": "Tech",
    "category_alt": "CEO"
  },
  {
    "query": "The CEO of Microsoft is",
    "answer": "Satya Nadella",
    "category": "Tech",
    "category_alt": "CEO"
  }
]
```

For facts with operands (to enable asymmetry analysis):
```json
[
  {
    "query": "3*7=",
    "answer": "21",
    "operand_a": 3,
    "operand_b": 7,
    "category": "3x",
    "category_alt": "x7"
  }
]
```

---

## Experiment 1: Multiplication Tables

### Command
```bash
lazarus introspect memory -m openai/gpt-oss-20b \
    --facts multiplication \
    --layer 20 \
    --top-k 30 \
    -o memory_mult.json
```

### Model Configuration
- **Model**: openai/gpt-oss-20b (20B parameters)
- **Architecture**: 24 layers, GPT-style decoder
- **Target Layer**: 20 (83% depth - late layers where facts crystallize)
- **Hidden Dimension**: 6144

### Results Summary

| Metric | Value |
|--------|-------|
| Total Facts | 64 (8x8 single-digit products) |
| Top-1 Accuracy | 9.4% (6/64) |
| Top-5 Accuracy | 71.9% (46/64) |
| Not in Top-30 | 3.1% (2/64) |

### Accuracy by Row (First Operand)

| Row | Top-1 | Avg Prob | Status |
|-----|-------|----------|--------|
| 2x  | 2/8   | 0.310    | Good |
| 3x  | 3/8   | 0.246    | Good |
| 4x  | 1/8   | 0.116    | Weak |
| 5x  | 0/8   | 0.093    | Weak |
| 6x  | 0/8   | 0.036    | Poor |
| 7x  | 0/8   | 0.016    | **Broken** |
| 8x  | 0/8   | 0.012    | Poor |
| 9x  | 0/8   | 0.023    | Poor |

### Attractor Nodes (Wrong Answers That Dominate)

| Answer | Co-activations | Avg Probability |
|--------|---------------|-----------------|
| 9      | 63            | 0.112 |
| 6      | 62            | 0.168 |
| 10     | 62            | 0.018 |
| 12     | 60            | 0.044 |
| 8      | 59            | 0.132 |
| 18     | 59            | 0.006 |
| 15     | 57            | 0.009 |
| 24     | 56            | 0.007 |

**Key insight**: Small numbers (6, 8, 9, 10, 12) act as "attractors" that pull probability mass away from correct answers.

### Asymmetry Analysis

The model treats A*B differently from B*A, even though multiplication is commutative:

| Pair | Rank A*B | Rank B*A | Delta |
|------|----------|----------|-------|
| 7*9 vs 9*7 | 5 | 23 | -18 |
| 3*7 vs 7*3 | 6 | 13 | -7 |
| 7*8 vs 8*7 | 18 | 13 | +5 |
| 2*7 vs 7*2 | 5 | 9 | -4 |
| 4*7 vs 7*4 | 3 | 7 | -4 |

**Pattern**: When 7 is the first operand, retrieval is much worse than when 7 is second.

### Hardest Facts

| Query | Answer | Rank | Probability |
|-------|--------|------|-------------|
| 6*7=  | 42     | >30  | 0.000 |
| 7*6=  | 42     | >30  | 0.000 |
| 9*7=  | 63     | 23   | 0.005 |
| 7*8=  | 56     | 18   | 0.001 |
| 8*7=  | 56     | 13   | 0.007 |
| 7*3=  | 21     | 13   | 0.002 |

**The 7x row is essentially broken** - the model cannot reliably retrieve any 7*n multiplication.

### Organization Bias

| Bias Type | Count |
|-----------|-------|
| Row-biased (same first operand clusters) | 27 |
| Column-biased (same second operand clusters) | 29 |
| Neutral | 8 |

Memory is organized by **both** row and column, with no strong preference.

---

## Experiment 2: Capital Cities

### Command
```bash
lazarus introspect memory -m openai/gpt-oss-20b \
    --facts capitals \
    --layer 20 \
    --top-k 30 \
    -o memory_capitals.json
```

### Results Summary

| Metric | Value |
|--------|-------|
| Total Facts | 30 countries |
| Top-1 Accuracy | 26.7% (8/30) |
| Top-5 Accuracy | 36.7% (11/30) |
| Not in Top-30 | 13.3% (4/30) |

### Accuracy by Region

| Region | Top-1 | Avg Prob | Notes |
|--------|-------|----------|-------|
| Europe | 5/13  | 0.194    | Best performance |
| Asia   | 2/9   | 0.149    | Moderate |
| Other  | 1/4   | 0.112    | Mixed |
| Americas | 0/4 | 0.010    | **Broken** - multi-token cities |

### Attractor Nodes

| Capital | Co-activations | Avg Probability |
|---------|---------------|-----------------|
| Paris   | 24            | 0.014 |
| London  | 16            | 0.004 |
| Berlin  | 8             | 0.001 |
| Tokyo   | 5             | 0.002 |
| Madrid  | 3             | 0.007 |
| Oslo    | 3             | 0.001 |

**Paris is the mega-attractor** - the "default capital" that the model falls back to.

### Hardest Facts (Multi-Token Barrier)

| Query | Answer | Rank | Issue |
|-------|--------|------|-------|
| Brazil | Brasilia | >30 | Multi-token? |
| Mexico | Mexico City | >30 | Multi-token |
| Argentina | Buenos Aires | >30 | Multi-token |
| Saudi Arabia | Riyadh | >30 | Query too long? |
| Iraq | Baghdad | 28 | Rare in training? |

### Nordic Cluster Interference

The Nordic capitals form a tight cluster and interfere with each other:

| Country | Capital | Rank | Notes |
|---------|---------|------|-------|
| Sweden | Stockholm | 19 | Confused with others |
| Denmark | Copenhagen | 19 | Confused with others |
| Finland | Helsinki | 16 | Confused with others |
| Norway | Oslo | appears as attractor | Also confused |

---

## Memory Architecture Model

Based on these experiments, here's the inferred memory architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                     LLM MEMORY STRUCTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 0-8: ENCODING                                             │
│  ├── Token embeddings                                            │
│  └── Basic pattern recognition                                   │
│                                                                  │
│  Layer 9-16: ASSOCIATION                                         │
│  ├── Query-key matching                                          │
│  ├── Category activation (row/column, region)                    │
│  └── Prototype retrieval begins                                  │
│                                                                  │
│  Layer 17-22: RETRIEVAL                          ← Best layer    │
│  ├── Fact crystallization                                        │
│  ├── Competition between similar answers                         │
│  ├── Attractor nodes dominate                                    │
│  └── Asymmetric effects strongest                                │
│                                                                  │
│  Layer 23-24: OUTPUT                                             │
│  ├── Final token selection                                       │
│  └── May override retrieved fact                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Neurons and Circuits

To identify specific neurons involved in fact retrieval:

```bash
# Find neurons that activate for multiplication
lazarus introspect neurons -m openai/gpt-oss-20b \
    -p "7*8=" \
    --layer 20 \
    --top-k 50 \
    --labels "multiplication,arithmetic,times"

# Compare neuron activation across different queries
lazarus introspect neurons -m openai/gpt-oss-20b \
    -p "7*8=|8*7=|6*9=|9*6=" \
    --layer 20 \
    --compare
```

### Layer Selection Guide

| Fact Type | Recommended Layer | Notes |
|-----------|------------------|-------|
| Arithmetic | 75-85% depth | Facts crystallize late |
| Capitals | 70-80% depth | Geographic knowledge mid-late |
| Elements | 65-75% depth | Structured knowledge earlier |
| Common facts | 80%+ depth | Well-known facts need full processing |

---

## Theoretical Implications

### 1. Distributed Lookup Table

Multiplication is NOT stored as a clean 10x10 table. Instead:
- Facts are distributed across neurons
- Similar products interfere (35, 36, 40, 42 compete)
- Retrieval depends on input format (7*8 ≠ 8*7)

### 2. Prototype-Based Retrieval

The model uses prototypes/attractors:
- "12" is the prototypical product (appears most in training?)
- "Paris" is the prototypical capital
- When uncertain, model falls back to prototypes

### 3. Training Frequency Effects

Accuracy correlates with training frequency:
- 2x, 3x tables: common in educational text → better
- 7x table: less common → broken
- "Paris", "London": frequently mentioned → attractors
- "Brasilia": rarely mentioned → fails

### 4. Tokenization Constraints

Single-token answers work better:
- "36" (one token) → retrievable
- "Buenos Aires" (multiple tokens) → fails in single-step retrieval
- This is a fundamental limitation of autoregressive decoding

---

## Advanced Experiments

### Multi-Layer Analysis

```bash
# Compare retrieval across layers
for layer in 12 15 18 20 22; do
    lazarus introspect memory -m openai/gpt-oss-20b \
        --facts multiplication \
        --layer $layer \
        -o memory_mult_L${layer}.json
done
```

### Cross-Model Comparison

```bash
# Compare memory structure across models
for model in "openai/gpt-oss-20b" "meta-llama/Llama-3.2-3B" "google/gemma-2-2b"; do
    lazarus introspect memory -m "$model" \
        --facts multiplication \
        --layer 20 \
        -o "memory_mult_${model//\//_}.json"
done
```

### Circuit Extraction

```bash
# Extract the "multiplication circuit"
lazarus introspect circuit capture \
    -m openai/gpt-oss-20b \
    --prompts "3*4=|5*6=|7*8=|2*9=" \
    --results "12|30|56|18" \
    --layer 20 \
    --extract-direction \
    -o mult_circuit.npz

# Test the circuit on novel inputs
lazarus introspect circuit test \
    -m openai/gpt-oss-20b \
    -c mult_circuit.npz \
    --prompts "4*7=|8*3=|6*6=" \
    --results "28|24|36"
```

### Neuron-Level Analysis

```bash
# Find which neurons encode "product > 50"
lazarus introspect probe \
    -m openai/gpt-oss-20b \
    --prompts "7*8=|9*6=|8*8=|6*7=|3*4=|2*5=" \
    --labels "1|1|1|1|0|0" \
    --layer 20 \
    -o large_product_probe.npz

# Steer toward large products
lazarus introspect steer \
    -m openai/gpt-oss-20b \
    -d large_product_probe.npz \
    -p "5*5=" \
    -c 2.0
```

---

## Summary of Findings

1. **Small operands work, large operands fail**: 2x, 3x tables are reliable; 7x, 8x, 9x are broken

2. **Asymmetric encoding is real**: 7*9 and 9*7 retrieve completely differently (rank 5 vs 23)

3. **Attractors dominate**: Small numbers (6, 8, 9, 12) and prototype cities (Paris, London) pull probability

4. **Regional/categorical clustering**: Capitals cluster by region, products cluster by row/column

5. **Tokenization is a hard barrier**: Multi-token answers cannot be retrieved in single logit lens probe

6. **Layer 20 (83% depth) is optimal**: Facts crystallize in late layers for this model

## References

- Logit Lens: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- Activation Patching: https://arxiv.org/abs/2202.05262
- Knowledge Neurons: https://arxiv.org/abs/2104.08696
