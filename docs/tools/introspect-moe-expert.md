# MoE Expert Analysis

Analyze expert routing patterns in Mixture-of-Experts (MoE) models using the semantic trigram methodology.

## Overview

MoE models route tokens to different expert networks. The common assumption is that experts specialize by **domain** (math expert, code expert) or **token type** (number expert, keyword expert). Our analysis shows this is incorrect.

**Key Finding:** Experts specialize by **semantic trigram patterns** - the relationship between previous, current, and next token types.

## Quick Start

```bash
# Interactive exploration (best for demos)
lazarus introspect moe-expert explore -m openai/gpt-oss-20b

# Demonstrate that domain experts don't exist
lazarus introspect moe-expert domain-test -m openai/gpt-oss-20b

# Demonstrate that single token routing is context-dependent
lazarus introspect moe-expert token-routing -m openai/gpt-oss-20b --token 127

# Full semantic trigram analysis
lazarus introspect moe-expert full-taxonomy -m openai/gpt-oss-20b
```

## The Semantic Trigram Methodology

### Why Domain Classification Fails

**Hypothesis:** "There's a math expert that handles arithmetic"

**Test:** Run 4 prompts each from math, code, language, and reasoning domains.

**Result:** The same experts handle ALL domains. E12 handles math(23), language(3), code(2), reasoning(1).

```bash
lazarus introspect moe-expert domain-test -m openai/gpt-oss-20b
```

### Why Single Token Classification Fails

**Hypothesis:** "Token '127' always routes to the same expert"

**Test:** Run the same token in different contexts:
- `"127"` (solo)
- `"111 127"` (after number)
- `"= 127"` (after operator)
- `"The value is 127."` (in sentence)

**Result:** Token "127" routes to 12 DIFFERENT experts depending on context.

```bash
lazarus introspect moe-expert token-routing -m openai/gpt-oss-20b --token 127
```

### Semantic Trigram Patterns

The key insight: experts specialize by **trigram pattern** - the semantic types of (previous, current, next) tokens.

#### Token Semantic Types

| Type | Description | Examples |
|------|-------------|----------|
| NUM | Numbers | `2`, `127`, `3.14` |
| OP | Operators | `+`, `-`, `*`, `=` |
| KW | Keywords | `def`, `class`, `if` |
| NOUN | Nouns | `cat`, `dog`, `king` |
| VERB | Verbs | `run`, `walk`, `think` |
| ADJ | Adjectives | `big`, `small`, `happy` |
| FUNC | Function words | `the`, `is`, `to`, `of` |
| AS | Analogy marker | `as` |
| TO | Preposition "to" | `to` |
| SYN | Synonym marker | `means`, `equals` |
| ANT | Antonym marker | `versus`, `opposite` |
| WS | Whitespace | spaces, newlines |
| PN | Punctuation | `.`, `,`, `!` |
| ^ | Sequence start | (first position) |
| $ | Sequence end | (last position) |

#### Pattern Categories

| Category | Patterns | Example Prompt |
|----------|----------|----------------|
| arithmetic | `NUM→OP`, `OP→WS→NUM` | `"2 + 3 = 5"` |
| code | `^→KW`, `KW→VAR→BR` | `"def hello():"` |
| analogy | `→AS→`, `FUNC→TO→NOUN` | `"king is to queen as..."` |
| synonym | `→SYN→`, `ADJ→SYN` | `"happy means joyful"` |
| antonym | `→ANT→`, `ADJ→ANT` | `"big versus small"` |
| comparison | `→THAN→`, `ADJ→THAN` | `"bigger than"` |
| causation | `→CAUSE→`, `VERB→CAUSE` | `"because she won"` |
| conditional | `→COND→`, `^→COND` | `"if it rains"` |
| question | `^→QW`, `QW→VERB` | `"what is..."` |

```bash
# Analyze specific categories
lazarus introspect moe-expert full-taxonomy -m openai/gpt-oss-20b --categories arithmetic,analogy,code
```

## Interactive Explorer

The `explore` command provides a real-time REPL for investigating expert routing:

```bash
lazarus introspect moe-expert explore -m openai/gpt-oss-20b
```

### Commands

| Command | Action |
|---------|--------|
| `[prompt]` | Analyze a new prompt |
| `l N` | Switch to layer N |
| `c "prompt"` | Compare with another prompt |
| `a` | Show all layers for current prompt |
| `d N` | Deep dive on position N |
| `q` | Quit |

### Example Session

```
[L11]> King is to queen as man is to woman

TOKENIZATION
----------------------------------------------------------------------
Pos  Token           Type     Trigram
----------------------------------------------------------------------
0    King            NOUN     ^→NOUN→FUNC
1    is              FUNC     NOUN→FUNC→TO
2    to              TO       FUNC→TO→NOUN
3    queen           NOUN     TO→NOUN→AS
4    as              AS       NOUN→AS→NOUN
5    man             NOUN     AS→NOUN→FUNC
6    is              FUNC     NOUN→FUNC→TO
7    to              TO       FUNC→TO→NOUN
8    woman           NOUN     TO→NOUN→$

EXPERT ROUTING (Layer 11)
----------------------------------------------------------------------
Pos  Token        Trigram                Top-4 Experts
----------------------------------------------------------------------
0    King         ^→NOUN→FUNC            E10 E15 E5 E25
2    to           FUNC→TO→NOUN           E30 E0 E2 E18  ← Analogy marker
4    as           NOUN→AS→NOUN           E25 E2 E30 E18 ← Analogy pivot

PATTERN SUMMARY
----------------------------------------------------------------------
  Pos 4 "as" (NOUN→AS→NOUN): analogy pivot -> E25
```

### Comparing Prompts

```
[L11 | "King is to queen..."]> c "2 + 3 = 5"

COMPARISON
======================================================================
"King is to queen as man is to woman" vs "2 + 3 = 5"

EXPERT OVERLAP
----------------------------------------------------------------------
  Shared experts: [2, 5, 10, 12, 15, 18, 25]
  Only in analogy: [0, 7, 9, 21, 30]
  Only in arithmetic: [3, 6, 26, 28]
  Overlap: 44%
```

**Key insight:** Different semantic patterns route to different experts. E30 handles analogy markers (`FUNC→TO→NOUN`) while E28 handles arithmetic operators (`NUM→OP→WS`).

## Layer Evolution

Expert routing changes across layers:

- **Early layers (L0-L7):** Structural patterns (punctuation, position)
- **Middle layers (L8-L15):** Semantic patterns (analogy, arithmetic)
- **Late layers (L16-L23):** Reasoning patterns (causation, comparison)

```bash
# Show all layers for a prompt
lazarus introspect moe-expert explore -m openai/gpt-oss-20b
# Then type: a
```

## Key Findings

### 1. No Domain Experts

The same experts handle math, code, language, AND reasoning. Domain is not what determines routing.

### 2. Context-Dependent Routing

The same token ("127") routes to different experts depending on its context. Token identity alone doesn't determine routing.

### 3. Semantic Trigram Specialization

Experts specialize by trigram pattern:
- `FUNC→TO→NOUN` (analogy marker) → E30
- `NUM→OP→WS` (arithmetic operator) → E28
- `^→NOUN→FUNC` (sequence start) → E10

### 4. 96% Router Signal from Attention

The router's decision comes primarily from attention-processed context, not raw token embeddings. This explains why context matters so much.

## Command Reference

### explore

Interactive REPL for real-time analysis.

```bash
lazarus introspect moe-expert explore -m MODEL [--layer N]
```

### domain-test

Demonstrate that domain experts don't exist.

```bash
lazarus introspect moe-expert domain-test -m MODEL [--layer N]
```

### token-routing

Demonstrate that single token routing is context-dependent.

```bash
lazarus introspect moe-expert token-routing -m MODEL --token TOKEN [--layer N]
```

### full-taxonomy

Full semantic trigram pattern analysis.

```bash
lazarus introspect moe-expert full-taxonomy -m MODEL [--categories CAT1,CAT2] [--verbose]
```

Categories: `arithmetic`, `code`, `synonym`, `antonym`, `analogy`, `hypernym`, `comparison`, `causation`, `conditional`, `question`, `negation`, `temporal`, `quantification`, `context_switch`, `position`, `coordination`

## Video Demo Workflow

For a presentation showing the methodology:

```bash
# 1. "Common assumption: domain experts exist"
lazarus introspect moe-expert domain-test -m openai/gpt-oss-20b
# Result: Same experts handle ALL domains

# 2. "Maybe tokens have stable routing?"
lazarus introspect moe-expert token-routing -m openai/gpt-oss-20b --token 127
# Result: Same token routes to 12 different experts

# 3. "The breakthrough: semantic trigrams"
lazarus introspect moe-expert full-taxonomy -m openai/gpt-oss-20b --categories arithmetic,analogy
# Result: Clear pattern specialization

# 4. "Interactive exploration"
lazarus introspect moe-expert explore -m openai/gpt-oss-20b
# Type: King is to queen as man is to woman
# Compare: c "2 + 3 = 5"
```

## MoE Type Detection & Compression

Determine whether an MoE model is compressible using SVD overlay representation.

### Understanding MoE Types

MoE models fall into two categories based on how they were trained:

| Type | Description | Compression |
|------|-------------|-------------|
| **Pseudo-MoE** | Dense model converted to MoE (upcycling). Experts share a common base with low-rank deltas. | ✓ Compressible via SVD overlay (6-10x) |
| **Native-MoE** | Trained natively as MoE from scratch. Experts are orthogonal (independent). | ✗ Not compressible via SVD (use quantization) |

### Quick Start

```bash
# Analyze a single model
lazarus introspect moe-expert moe-type-analyze -m openai/gpt-oss-20b

# Compare two models side-by-side
lazarus introspect moe-expert moe-type-compare -m openai/gpt-oss-20b -c allenai/OLMoE-1B-7B-0924

# Show orthogonality visualization with heatmap and direction diagram
lazarus introspect moe-expert moe-type-analyze -m openai/gpt-oss-20b --visualize
```

### moe-type-analyze

Detect whether a model is pseudo-MoE (compressible) or native-MoE (not compressible).

```bash
lazarus introspect moe-expert moe-type-analyze -m MODEL [--layer N] [--visualize] [-o output.json]
```

**Options:**
- `--layer N`: Analyze specific layer (default: first MoE layer)
- `--visualize`: Show expert orthogonality heatmap and direction diagram
- `-o FILE`: Save JSON results to file

**Example Output:**

```
======================================================================
MOE TYPE ANALYSIS
======================================================================
Model:  openai/gpt-oss-20b
Layer:  0
Type:   PSEUDO-MOE

Evidence:
  Gate Rank:            1 / 2880 (  0.0%)
  Up Rank:            337 / 2880 ( 11.7%)
  Down Rank:          206 / 2880 (  7.2%)
  Cosine Similarity: 0.418 (+/- 0.163)

Compression:
  Compressible:      Yes
  Estimated Ratio:   7.9x
======================================================================
```

### moe-type-compare

Compare MoE types between two models side-by-side.

```bash
lazarus introspect moe-expert moe-type-compare -m MODEL1 -c MODEL2
```

**Example Output:**

```
======================================================================
MOE TYPE COMPARISON
======================================================================
+-----------------------+----------------+----------------+
| Metric                | gpt-oss-20b    | OLMoE-1B-7B-09 |
+-----------------------+----------------+----------------+
| Type                  | PSEUDO         | NATIVE         |
| Gate Rank             |    1/2880      |  755/1024      |
| Gate Rank %           |          0.0%  |         73.7%  |
| Cosine Similarity     |          0.418 |          0.018 |
| Compressible          | Yes (7.9x)     | No             |
+-----------------------+----------------+----------------+
```

### Orthogonality Visualization (--visualize)

The `--visualize` flag adds two data-driven diagrams:

**1. Direction Diagram** - Shows how expert weight vectors are oriented in 2D space:
- Orthogonal experts → arrows point in different directions
- Clustered experts → arrows cluster together

```
Expert Direction Diagram (2D MDS projection):

┌───────────────────────────────────────────────────────────┐
│                  ↙E7        │         ↘E0                 │
│        ↙E4        ·         │        ·          ↘E3       │
│ ──────←E1──────────────────·┼·───────────────────→E6───── │
│                       ↑E5   │    ↑E2                      │
└───────────────────────────────────────────────────────────┘

  Arrows point in different directions → Experts are ORTHOGONAL
```

**2. Similarity Heatmap** - Shows pairwise cosine similarity between all experts:

```
Expert Similarity Heatmap (cosine similarity):

      0  1  2  3  4  5  6  7
  0  ■  ·  ·  ·  ·  ·  ·  ·  avg:0.02
  1  ·  ■  ·  ·  ·  ·  ·  ·  avg:0.02
  2  ·  ·  ■  ·  ·  ·  ·  ·  avg:0.02
  ...

Legend: · (<0.05)  ░ (0.05-0.15)  ▒ (0.15-0.3)  ▓ (0.3-0.5)  █ (>0.5)  ■ (self)
```

### Key Metrics Explained

| Metric | Pseudo-MoE | Native-MoE |
|--------|------------|------------|
| **Gate Rank Ratio** | < 5% (shared gate) | > 50% (diverse gates) |
| **Cosine Similarity** | > 0.25 (clustered) | < 0.10 (orthogonal) |
| **Interpretation** | Expert = Base + delta | Expert ⟂ Expert |

### Compression Pipeline

If a model is pseudo-MoE (compressible), use the overlay commands:

```bash
# Step 1: Confirm model is pseudo-MoE
lazarus introspect moe-expert moe-type-analyze -m openai/gpt-oss-20b

# Step 2: Compute SVD overlay representation
lazarus introspect moe-expert moe-overlay-compute -m openai/gpt-oss-20b

# Step 3: Verify reconstruction accuracy
lazarus introspect moe-expert moe-overlay-verify -m openai/gpt-oss-20b

# Step 4: Estimate full model storage savings
lazarus introspect moe-expert moe-overlay-estimate -m openai/gpt-oss-20b
```

### Video Demo Workflow (MoE Type Detection)

```bash
# 1. "Are these models compressible?"
lazarus introspect moe-expert moe-type-compare -m openai/gpt-oss-20b -c allenai/OLMoE-1B-7B-0924

# 2. "Show me the expert structure"
lazarus introspect moe-expert moe-type-analyze -m openai/gpt-oss-20b --visualize

# 3. "Pseudo-MoE: experts are clustered around a shared base"
# → High similarity, low gate rank, arrows cluster together

# 4. "Native-MoE: experts are orthogonal"
lazarus introspect moe-expert moe-type-analyze -m allenai/OLMoE-1B-7B-0924 --visualize
# → Low similarity, high gate rank, arrows spread in all directions
```

## See Also

- [introspection.md](../introspection.md) - Main introspection documentation
- [expert-compression.md](../expert-compression.md) - Expert compression analysis
