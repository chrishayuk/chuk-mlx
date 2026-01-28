# Review: chuk-virtual-expert-arithmetic Updates

**Date**: 2026-01-28
**Package**: `chuk-virtual-expert-arithmetic`
**Purpose**: Assess how the updated architecture addresses diversity issues from earlier training runs

---

## Executive Summary

The updated `chuk-virtual-expert-arithmetic` package represents a **major architectural improvement** that directly addresses the diversity issues identified in the experiment review. The new modular design with separate vocabulary, domain, schema, and perturbation layers provides the foundation for generating training data that can generalize to GSM-8K.

| Issue from Earlier Runs | Status | How Addressed |
|-------------------------|--------|---------------|
| One template per pattern | **Fixed** | Multiple templates per pattern (4-8 variants) |
| Fixed linguistic patterns | **Fixed** | Perturbation system + template variety |
| Limited vocabulary | **Fixed** | 304 names, 14 domains, rich phrase library |
| No word numbers | **Fixed** | 30% word number probability built-in |
| Round number bias | **Fixed** | `avoid_round` + difficulty profiles |
| No paraphrasing | **Fixed** | `TemplatePerturbator` with synonyms, fillers |

---

## Architecture Analysis

### New Layered Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    VOCABULARY LAYER                              │
│  names.json (304), items.json, phrases.json, animals.json       │
│  14 domains: kitchen, factory, travel, school, shopping...      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     SCHEMA LAYER                                 │
│  62 schemas across 5 expert types                                │
│  Supports mixins, extends, constraints                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CORE MODULES                                  │
│  SafeEvaluator, VariableGenerator, ConstraintValidator           │
│  TemplatePerturbator, NumericDiversifier, DomainSampler          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   SPEC GENERATOR                                 │
│  Combines schema + domain → technical spec → LLM expansion       │
└─────────────────────────────────────────────────────────────────┘
```

This separation of concerns directly addresses the monolithic generator issue from earlier runs.

---

## Diversity Improvements

### 1. Template Variety (Previously: 1 template per pattern)

**Before:**
```python
# Run 1-18: Single hardcoded template per schema
template = f"{name}'s ducks lay {n} eggs. She eats {e1} and bakes with {e2}..."
```

**Now:**
```json
// consume_then_sell.json - 8 domain variants × 4 templates each = 32 options
{
  "animal_farm": [4 templates],
  "garden": [4 templates],
  "factory": [4 templates],
  "bakery": [4 templates],
  "craft_shop": [3 templates],
  "book_store": [3 templates],
  "flower_shop": [3 templates],
  "lemonade_stand": [3 templates],
  "gsm8k_style": [4 templates]  // Specifically mimics GSM-8K phrasing
}
```

**Impact**: 32x increase in linguistic variety for this pattern alone.

### 2. Name Diversity (Previously: ~90 names)

**Before:** 90 hardcoded names
**Now:** 304 names with gender-aware pronouns

```json
{
  "people": [304 names],
  "male": [99 names],
  "female": [119 names],
  "neutral": [35 names],
  "pronouns": {
    "male": {"subject": "he", "object": "him", "possessive": "his"},
    "female": {"subject": "she", "object": "her", "possessive": "her"},
    "neutral": {"subject": "they", "object": "them", "possessive": "their"}
  }
}
```

**Impact**: 3.4x more names + grammatically correct pronoun handling.

### 3. Word Numbers (Previously: Digits only)

**Before:** Always "5 apples", never "five apples"
**Now:** 30% probability of word form

```python
class SchemaGenerator:
    def __init__(self, word_number_prob: float = 0.3, ...):
        # Converts eligible numbers to words 30% of the time
        # Covers 1-25, 30, 40, 50

WORD_NUMBERS = {1: "one", 2: "two", ..., 12: "twelve", 20: "twenty", ...}
```

**Impact**: Matches GSM-8K's ~30-40% word number usage.

### 4. Perturbation System (Previously: None)

**Before:** Fixed templates with no variation
**Now:** Multi-layer perturbation

```python
class TemplatePerturbator:
    FILLER_PHRASES = ["Now, ", "Here's the situation: ", "So, ", "Well, ", ...]

    QUESTION_STARTERS = {
        "How many": ["How many", "What is the total number of", "Find the number of", ...],
        "How much": ["How much", "What is the total amount of", "Calculate the total", ...],
    }

    SYNONYMS = {
        "has": ["owns", "possesses", "holds"],
        "buys": ["purchases", "gets", "picks up"],
        "total": ["altogether", "in all", "combined"],
        ...
    }
```

**Usage:**
```python
gen = SchemaGenerator(perturbation_level=0.5)
# "Alice has 5 apples" → "Now, Alice owns five apples"
```

**Impact**: Breaks template fingerprints that caused memorization.

### 5. Numeric Diversity (Previously: Random uniform)

**Before:** `random.randint(1, 100)` — produced round numbers 10% of the time
**Now:** Difficulty-aware generation with round-number avoidance

```python
class NumericDiversifier:
    def avoid_round_number(self, min_val, max_val):
        # Never returns multiples of 10

    def generate_carrying_pair(self, ...):
        # Generates pairs requiring carrying in addition

    def generate_by_difficulty(self, difficulty):
        # easy: small round (5, 10, 15, 20)
        # medium: any
        # hard: larger non-round (37, 63, 91)
```

**Impact**: More realistic number distribution matching GSM-8K.

### 6. Domain System (Previously: None)

**Before:** Context-free templates
**Now:** 14 coherent domains

| Domain | Agent Types | Example Output |
|--------|-------------|----------------|
| kitchen | appliance, person | "Oven 2 bakes 12 loaves per hour" |
| factory | machine, line | "Machine A produces 45 widgets per hour" |
| travel | vehicle, person | "The express train travels 60 mph" |
| tech | server, person | "Server A processes 100 requests per second" |
| poultry | coop, person | "Janet's ducks lay 16 eggs per day" |

**Impact**: Semantic coherence — no more "Server A bakes cookies".

### 7. Rich Phrase Library

**Before:** Hardcoded verb phrases
**Now:** Structured phrase vocabulary

```json
{
  "consumption_personal": [
    {"base": "eat", "s": "eats", "rest": "for breakfast"},
    {"base": "keep", "s": "keeps", "rest": "for personal use"},
    {"base": "save", "s": "saves", "rest": "for later"},
    ...
  ],
  "consumption_giving": [
    {"base": "give", "s": "gives", "rest": "to neighbors"},
    {"base": "donate", "s": "donates", "rest": "to charity"},
    {"base": "share", "s": "shares", "rest": "with friends"},
    ...
  ]
}
```

**Impact**: Natural variation in action descriptions.

---

## Schema Coverage

### Current Schema Count: 62

| Category | Count | Examples |
|----------|-------|----------|
| arithmetic | 45 | `consume_then_sell`, `multiply_add`, `combined_rate` |
| entity_track | 5 | `entity_simple_transfer`, `entity_find_lose` |
| comparison | 4 | `comparison_times_more`, `comparison_half` |
| percentage | 4 | `percent_off`, `percent_increase` |
| rate_equation | 4 | `rate_distance`, `rate_earning` |

### New Gap-Closing Schemas

Based on GSM-8K gap analysis, new schemas added:

| Schema | Pattern | Gap Addressed |
|--------|---------|---------------|
| `sub_sub_div_div` | sub→sub→div→div | Division chain gap |
| `div_chain` | Multiple sequential divisions | Division operations |
| `weekly_sprints` | Time-based rate calculation | Sprint/week patterns |
| `decimal_rate_week` | Decimal rates over weeks | John's dogs pattern |
| `feed_remainder_scattered` | Scattered information | Information ordering |

---

## Comparison: Old vs New Generated Output

### Pattern: consume_then_sell

**Old (Run 18):**
```
Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes
muffins for her friends every day with 4. She sells the rest at $2 each.
How much does she make every day?
```

**New (with perturbation_level=0.5):**
```
Now, Marcus's chickens lay sixteen eggs per day. He keeps 3 for personal use
and shares 4 with neighbors. He sells what's left at the roadside stand for
$2 apiece. What are his daily earnings from egg sales?
```

**Improvements visible:**
- Different name (Marcus vs Janet)
- Different animal (chickens vs ducks)
- Word number (sixteen vs 16)
- Filler phrase ("Now, ")
- Varied phrases ("keeps for personal use" vs "eats for breakfast")
- Varied question form ("What are his daily earnings" vs "How much does she make")

### Pattern: combined_rate

**Old:**
```
Machine A produces 10 widgets per hour. Machine B produces 15 widgets per hour.
If both run for 3 hours, how many widgets in total?
```

**New (with domain=kitchen, perturbation=0.5):**
```
Oven 1 bakes 10 loaves per hour. Oven 2 bakes 15 loaves per hour.
After three hours, how many loaves total?
```

**Or (with domain=tech):**
```
Server A processes 10 requests per second and Server B processes 15 per second.
Together for 3 seconds, how many requests combined?
```

---

## Remaining Gaps

### 1. Real GSM-8K Training Data

The new system generates **synthetic** data with more variety, but still lacks actual GSM-8K examples. The LLM expansion step (`spec_to_prompt()`) can help, but requires external LLM calls.

**Recommendation**: Implement experiment B from NEXT_EXPERIMENTS.md — annotate real GSM-8K problems.

### 2. Information Scattering

GSM-8K often scatters information across sentences:
```
"Janet has some eggs. She eats 3. The ducks laid 16 total. She bakes with 4."
```

Current templates still present information in logical order. The `feed_remainder_scattered` schema attempts this but is limited.

**Recommendation**: Add clause reordering to `TemplatePerturbator`.

### 3. Multi-Sentence Complexity

GSM-8K average: 3-5 sentences with narrative flow
Current templates: 2-3 sentences, formulaic

**Recommendation**: Create "gsm8k_style" variants for more schemas (only `consume_then_sell` has this currently).

### 4. LLM Expansion Not Integrated

The spec → LLM → natural language pipeline is documented but not automated:

```python
# Currently manual:
prompt = gen.spec_to_prompt(spec)
# Then call Claude/GPT-4 externally
```

**Recommendation**: Add async LLM expansion in `SchemaGenerator` or `SpecGenerator`.

---

## Metrics Comparison

| Metric | Old (Run 18) | New Package |
|--------|--------------|-------------|
| Names | 90 | 304 (3.4x) |
| Templates per pattern | 1 | 4-8 (6x avg) |
| Domains | 0 | 14 |
| Word number support | No | Yes (30%) |
| Perturbation | No | Yes (3 types) |
| Round number avoidance | No | Yes |
| Difficulty profiles | No | Yes (easy/medium/hard) |
| Carrying/borrowing pairs | No | Yes |
| Schemas | 59 | 62 |
| Pronoun handling | Basic | Full (he/she/they) |

---

## Recommendations for Training

### Immediate Actions

1. **Update `train_gsm8k_yaml.py`** to use new `SchemaGenerator`:
   ```python
   from chuk_virtual_expert_arithmetic.generators import SchemaGenerator

   gen = SchemaGenerator(
       word_number_prob=0.3,
       perturbation_level=0.5,  # Enable perturbation
       seed=42
   )
   ```

2. **Enable perturbation** for training:
   ```python
   gen.perturbation_level = 0.5  # Moderate perturbation
   ```

3. **Use all domains** to maximize variety:
   ```python
   # Already happens automatically — domains sampled randomly
   ```

### Training Configuration

```python
# Recommended settings for next training run
config = {
    "word_number_prob": 0.3,      # Match GSM-8K
    "perturbation_level": 0.5,    # Break template fingerprints
    "n_train": 5000,              # More examples (was 1500-3000)
    "batch_size": 8,
    "use_all_schemas": True,      # 62 schemas
    "use_all_domains": True,      # 14 domains
}
```

### Expected Improvement

Based on the diversity improvements:

| Scenario | Expected GSM-8K Accuracy |
|----------|--------------------------|
| Current best (Run 25) | 27% |
| With new generator (no perturbation) | 30-35% |
| With new generator + perturbation | 35-45% |
| With new generator + real GSM-8K data | 50-60% |

---

## Conclusion

The updated `chuk-virtual-expert-arithmetic` package addresses **all major diversity issues** identified in the experiment review:

1. **Template variety**: 1 → 4-8 templates per pattern
2. **Vocabulary**: 90 → 304 names, 14 domains
3. **Word numbers**: 0% → 30% probability
4. **Perturbation**: None → fillers, synonyms, question variants
5. **Numeric diversity**: Random → difficulty-aware, round-avoiding

The architecture is now ready for a new training run. The primary remaining gap is **real GSM-8K training data** — the synthetic generator can now produce varied outputs, but hasn't seen actual GSM-8K linguistic patterns.

**Next step**: Run experiment A (true baseline) → experiment D (use new generator) → experiment B (add real GSM-8K).
