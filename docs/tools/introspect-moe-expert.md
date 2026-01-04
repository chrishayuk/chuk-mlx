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

## See Also

- [introspection.md](../introspection.md) - Main introspection documentation
- [expert-compression.md](../expert-compression.md) - Expert compression analysis
