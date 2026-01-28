# Generator Output vs GSM-8K: Side-by-Side Comparison

**Date**: 2026-01-28
**Purpose**: Compare new generator outputs against actual GSM-8K problems to identify remaining gaps

---

## Pattern 1: consume_then_sell (Janet's Ducks)

### GSM-8K Original
> Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

### Generator Output (Clean)
> Bella's garden produces 16 beans every day during the growing season. She keeps 6 for her family and donates two to the food bank. She sells the rest for $2 each at a roadside stand. How much does Bella earn daily?

### Generator Output (Perturbed)
> Interestingly, Julie's garden produces 29 cucumbers every day during the growing season. She keeps 5 for her family and donates 3 to the food bank. She sells the rest for $2 each at a roadside stand. How much does Julie earn daily?

### Analysis

| Feature | GSM-8K | Generator | Match |
|---------|--------|-----------|-------|
| Name | Janet | Bella/Julie | ✅ Different names |
| Word numbers | "three", "four" | "two", "29" | ⚠️ Partial (has word numbers but mixed) |
| Narrative context | "bakes muffins for friends" | "donates to food bank" | ✅ Varied context |
| Question phrasing | "How much in dollars does she make every day at the farmers' market?" | "How much does Bella earn daily?" | ⚠️ Generator shorter |
| Sentence count | 4 sentences | 4 sentences | ✅ Similar |
| Filler phrases | None | "Interestingly," | ✅ Added variation |

**Gaps**:
- GSM-8K has more elaborate question ("How much in dollars...at the farmers' market")
- GSM-8K repeats location context in question
- Generator question is more formulaic

---

## Pattern 2: material_half (Robe Fiber)

### GSM-8K Original
> A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?

### Generator Output (Clean)
> To build a pen, you need 6 units of batches and half as much groups. Find the total amount of material needed?

### Generator Output (Perturbed)
> Consider this: to build a pen, you need 6 units of batches and half as much groups. Find the total amount of material needed?

### Analysis

| Feature | GSM-8K | Generator | Match |
|---------|--------|-----------|-------|
| Domain | Textiles (robe, fiber) | Construction (pen, batches) | ⚠️ Domain exists but weird |
| Specificity | "blue fiber", "white fiber" | "batches", "groups" | ❌ GSM-8K more concrete |
| Question style | "How many bolts in total does it take?" | "Find the total amount of material needed?" | ⚠️ Different style |
| Length | 1 sentence + question | 1 sentence + question | ✅ Same |

**Gaps**:
- Generator uses abstract nouns ("batches", "groups") instead of concrete materials
- GSM-8K uses color modifiers for clarity ("blue fiber", "white fiber")
- Domain coherence issue: "batches" and "groups" don't make sense together

---

## Pattern 3: weekly_sprints (James Sprints)

### GSM-8K Original
> James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?

### Generator Output (decimal_rate_week)
> Lonnie exercises eleven times a day to stay in shape for the upcoming marathon. Each session lasts 2.5 hours and includes both cardio and strength training. How many hours does he exercise in 7 days?

### Analysis

| Feature | GSM-8K | Generator | Match |
|---------|--------|-----------|-------|
| Structure | rate × frequency × time | rate × frequency × time | ✅ Same |
| Name | James | Lonnie | ✅ Different |
| Word numbers | "3 sprints 3 times" | "eleven times" | ✅ Has word numbers |
| Narrative | "decides to run" | "to stay in shape for the upcoming marathon" | ✅ Generator has more context |
| Question | "How many total meters does he run a week?" | "How many hours does he exercise in 7 days?" | ✅ Similar |

**Gaps**:
- Generator has MORE context than GSM-8K (unusual)
- GSM-8K is more direct, generator adds filler ("includes both cardio and strength training")

---

## Pattern 4: twice_relationship (Toulouse Sheep)

### GSM-8K Original
> Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?

### Generator Output (twice_as_much)
> Samantha collected twice as many boxes as Diego did. Diego collected twelve boxes. How many did Samantha collect?

### Generator Output (Perturbed)
> As it turns out, Samantha collected twice as many boxes as Diego did. Diego collected twelve boxes. How many did Samantha collect?

### Analysis

| Feature | GSM-8K | Generator | Match |
|---------|--------|-----------|-------|
| Complexity | 3 entities, 2 relationships | 2 entities, 1 relationship | ❌ GSM-8K more complex |
| Chaining | Yes (2→4→8 multiplier chain) | No (single comparison) | ❌ Missing |
| Question | "How many together if X has Y?" | "How many did X collect?" | ❌ Different structure |
| Word numbers | "20 sheep" | "twelve boxes" | ✅ Word number used |

**Gaps**:
- GSM-8K chains multiple multiplicative relationships
- GSM-8K asks for aggregate of all entities
- Generator handles only single comparison
- Need multi-entity comparison schema

---

## Pattern 5: scattered_information (Wendi's Chickens)

### GSM-8K Original
> Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?

### Generator Output (None - no direct equivalent)

### Analysis

| Feature | GSM-8K | Generator | Match |
|---------|--------|-----------|-------|
| Sentence count | 5 sentences | N/A | ❌ No equivalent |
| Information scattering | Key info (20 chickens) at END | N/A | ❌ Not implemented |
| Detail level | "seeds, mealworms and vegetables" | N/A | ❌ No decorative detail |
| Multi-step reasoning | Yes (rate × count - given) | N/A | ❌ Missing schema |

**Gaps**:
- No schema for "scattered information" pattern
- No schema for "find missing component" (total - given = remainder)
- GSM-8K buries key information (flock size) at end of problem

---

## Pattern 6: percentage_chain (Josh House Flip)

### GSM-8K Original
> Josh decides to try flipping a house. He buys a house for $80000 and then puts in $50000 in repairs. This increased the value of the house by 150%. How much profit did he make?

### Generator Output (percent_increase - closest)
> The price of a $45 jacket increased by 25%. What is the new price?

### Analysis

| Feature | GSM-8K | Generator | Match |
|---------|--------|-----------|-------|
| Multi-step | Yes (cost + repairs, then %) | No (single %) | ❌ Missing chain |
| Narrative | "decides to try flipping" | None | ❌ No narrative |
| Final question | "How much profit?" | "What is the new price?" | ❌ Different ask |
| Domain | Real estate | Generic | ⚠️ Less specific |

**Gaps**:
- No "profit calculation" schema (revenue - cost)
- No percentage applied to ORIGINAL value then compared to TOTAL cost
- Generator handles simple percentage, not percentage chains

---

## Pattern 7: conditional_rate (Kylar's Glasses)

### GSM-8K Original
> Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?

### Generator Output (None - no equivalent)

### Analysis

**Gaps**:
- No "alternating price" schema
- No conditional pricing ("every second item")
- This requires: (n/2 × full_price) + (n/2 × discounted_price)

---

## Pattern 8: decimal_rate (John's Dogs)

### GSM-8K Original
> John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?

### Generator Output (decimal_rate_week)
> Lonnie exercises eleven times a day to stay in shape for the upcoming marathon. Each session lasts 2.5 hours and includes both cardio and strength training. How many hours does he exercise in 7 days?

### Analysis

| Feature | GSM-8K | Generator | Match |
|---------|--------|-----------|-------|
| Decimal | ".5 hours" | "2.5 hours" | ✅ Both have decimals |
| Structure | count × rate × days | count × rate × days | ✅ Same |
| Phrasing | "takes care of their business" | "includes both cardio and strength training" | ✅ Both have detail |
| Question | "How many hours a week" | "How many hours...in 7 days" | ✅ Equivalent |

**Match**: This pattern is well-covered.

---

## Summary: Coverage Analysis

### Well-Covered Patterns ✅

| Pattern | GSM-8K Example | Generator Schema |
|---------|----------------|------------------|
| consume_then_sell | Janet's ducks | `consume_then_sell` (8 variants) |
| decimal_rate | John's dogs | `decimal_rate_week` |
| simple_twice | Gail's fish tanks | `twice_relationship` |
| material_half | Robe fiber | `material_half` |

### Partially Covered ⚠️

| Pattern | GSM-8K Example | Issue |
|---------|----------------|-------|
| multiply_add | Party supplies | Generator works but less natural phrasing |
| combined_rate | Machines/workers | Domain coherence issues ("train creates emails") |

### Not Covered ❌

| Pattern | GSM-8K Example | Missing Schema |
|---------|----------------|----------------|
| Multi-entity chain | Toulouse sheep (3 entities) | `chained_comparison` |
| Scattered info | Wendi's chickens | `scattered_info` or perturbation |
| Percentage + profit | Josh house flip | `percent_profit_chain` |
| Conditional pricing | Kylar's glasses | `alternating_price` |
| Restart/recovery | Carla's download | `interrupted_process` |

---

## Key Gaps Identified

### 1. Domain Coherence Issues

The generator sometimes produces semantically incoherent outputs:

```
❌ "the express train creates 14 emails per week"
❌ "you need 6 units of batches and half as much groups"
```

**Root cause**: Domain sampler not enforcing item-verb compatibility in all cases.

### 2. Question Elaboration

GSM-8K questions are often more elaborate:

```
GSM-8K: "How much in dollars does she make every day at the farmers' market?"
Generator: "How much does Bella earn daily?"
```

**Root cause**: Generator question templates are shorter and more formulaic.

### 3. Multi-Entity Chains

GSM-8K frequently chains 3+ entities:
- Toulouse → Charleston → Seattle (2 multiplicative relationships)
- Generator only handles 2 entities

**Root cause**: Missing `chained_comparison` schema.

### 4. Information Scattering

GSM-8K buries key information in later sentences:
- Wendi problem: "if the size of Wendi's flock is 20 chickens" (at end)
- Generator presents information in logical order

**Root cause**: Perturbator doesn't reorder clauses.

### 5. Decorative Detail

GSM-8K adds irrelevant but realistic detail:
- "containing seeds, mealworms and vegetables to help keep them healthy"
- "for his new apartment"

**Root cause**: Generator templates don't include decorative clauses.

---

## Recommendations

### Immediate Fixes

1. **Fix domain coherence** — Validate item-verb-agent compatibility
2. **Elaborate questions** — Add longer question variants to templates
3. **Add clause reordering** — Move key info to end of problem

### New Schemas Needed

| Schema | Pattern | Priority |
|--------|---------|----------|
| `chained_comparison` | A = 2×B, B = 4×C, find total | High |
| `percent_profit_chain` | cost + repairs, % increase, profit | High |
| `alternating_price` | every Nth item discounted | Medium |
| `interrupted_process` | partial work, restart, complete | Low |
| `scattered_remainder` | total - given1 - given2 = ? | High |

### Template Improvements

1. Add "gsm8k_style" variants to ALL schemas (not just `consume_then_sell`)
2. Add decorative clauses: "to help keep them healthy", "for the upcoming event"
3. Vary question length: short vs elaborate

---

## Metrics Summary

| Metric | Generator | GSM-8K | Gap |
|--------|-----------|--------|-----|
| Avg sentence count | 2-3 | 3-5 | Need more |
| Word number usage | 30% | 30-40% | ✅ Close |
| Question length | 8-12 words | 12-20 words | Need longer |
| Multi-entity problems | 2 max | 3+ common | Need chains |
| Information order | Logical | Scattered | Need reordering |
| Decorative detail | Minimal | Frequent | Need more |
| Domain coherence | Sometimes wrong | Always coherent | Need fixing |
