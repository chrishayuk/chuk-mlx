# Experiment Review: GSM-8K YAML Trace Training

**Date**: 2026-01-27
**Reviewer**: Claude
**Status**: Critical analysis of 25-run experiment series

---

## 1. Core Thesis

The experiment tests whether small LLMs (1-3B parameters) can achieve high accuracy on math reasoning by **separating structure from computation**:

- Model outputs symbolic YAML traces (structure extraction)
- External solver executes traces deterministically (computation)

**Verdict**: The thesis is sound and well-motivated. Transformers struggle with arithmetic but excel at pattern matching. Outsourcing computation to deterministic solvers is a valid approach.

---

## 2. Methodology Strengths

### Well-designed reward shaping

```
1.0: Correct answer
0.7: Valid trace, wrong answer
0.5: Invalid trace structure
0.3: Wrong expert
0.0: Parse failure
```

This graduated reward provides learning signal at every stage. The model first learns to produce valid YAML, then correct routing, then valid structure, then correct wiring.

### Anti-short-circuit constraint

Preventing the model from regurgitating extracted values forces it to actually wire computation graphs. This is a smart design decision that addressed a real failure mode (Run 3: rate_equation regression from 98% to 86%).

### Systematic ablations

25 runs testing:
- 4 different models (TinyLlama, SmolLM2, Llama-3.2-1B, Llama-3.2-3B)
- Layer unfreezing (6, 8, 16 layers)
- Variable naming strategies (abstract, semantic, hybrid)
- Composition patterns (2-expert, 3-expert chains)

Good experimental rigor with clear documentation of what changed between runs.

### Pattern documentation

The 17 training patterns documented in `PATTERNS.md` are valuable for future work. Each pattern includes:
- Problem description
- Implementation details
- Evidence from specific runs
- Clear principles

---

## 3. Critical Issues

### Issue 1: Evaluation Methodology Flaw

The 10-sample probe showed **90% accuracy**, but 100-sample showed **~2%**. This is a **45x discrepancy**.

**Problem**: The 10 sample problems were hardcoded in `gsm8k_loader.py` (lines 197-248) and likely influenced training pattern design. This creates data leakage:

- Training patterns were designed to match these specific 10 problems
- "Evaluation" was really testing pattern coverage, not generalization

**Evidence**: The sample problems exactly match schemas:

| Sample Problem | Matching Schema |
|----------------|-----------------|
| Janet's ducks | `consume_then_sell` |
| Robe fiber | `material_half` |
| House flipping | `cost_increase_profit` |
| James sprints | `interleaved_mul_mul` |
| John's dogs | `decimal_rate_week` |

This explains why 10-sample showed 90% while 100-sample showed 2%.

### Issue 2: Synthetic-Only Training

All training data comes from `TraceGenerator.generate_balanced()`. No actual GSM-8K problems in training.

**Problem**: Synthetic templates have:
- Fixed linguistic patterns (one template per pattern)
- Limited vocabulary (~90 names, ~50 items)
- Consistent structure

GSM-8K has:
- 1319 unique problem formulations
- Natural language variation
- Information scattered throughout prose
- Ambiguous phrasing

**Result**: 100% parse rate (learned format), ~2% accuracy (didn't learn reasoning).

The model memorized the mapping from template → trace, not from problem semantics → computation.

### Issue 3: Expert Routing vs. Problem Understanding

The model learns to route based on surface patterns, not problem semantics.

```
# Failure mode from RESULTS.md:
Question: "Josh buys a house for $80,000..."
Model: expert=entity_track (because "house" is an entity)
Trace: includes PercentIncreaseStep (because problem mentions "increase")
Error: Cross-expert contamination
```

The routing decision and operation selection are coupled to surface features, not mathematical structure.

### Issue 4: Token Limit Discovery Was Late

Run 17 revealed `max_tokens=250` was truncating multi-expert traces. This basic issue wasn't caught for 17 runs.

| Pattern | Tokens Required | Status with 250 limit |
|---------|-----------------|----------------------|
| 2-expert simple | 159-167 | OK |
| 2-expert complex | 217-293 | Truncated |
| 3-expert | 264-307 | Truncated |

**Suggestion**: Add trace length verification in data generation pipeline. Fail fast on truncation.

### Issue 5: Chat Template Incompatibility

SmolLM2-Instruct failed badly (65% SFT vs TinyLlama's 100%) due to chat template mismatch.

```python
# Current format (TinyLlama-specific):
"<|system|>\n{SYSTEM_PROMPT}</s>\n<|user|>\n{question}</s>\n<|assistant|>\n```yaml\n"
```

This format is baked into the approach. Testing with other models requires template adaptation.

---

## 4. Results Interpretation

### What the experiment proved

1. **Small models CAN learn structured output formats** — 100% parse rate achieved
2. **Anti-short-circuit constraints work** — Eliminated regurgitation failure mode
3. **Structural consistency matters** — All patterns in an expert need same shape
4. **Hybrid variable naming works** — Semantic inits + fixed scaffolding
5. **Full fine-tuning causes catastrophic forgetting** — 17% → 7% with all layers unfrozen
6. **Instruction-tuning effects vary by model** — Llama-Instruct helps, SmolLM2-Instruct hurts

### What the experiment did NOT prove

1. **That this approach scales to real math reasoning** — 27% best result is far from SOTA
2. **That 59 schemas generalize to 1319 problems** — Pattern memorization, not understanding
3. **That the model understands math** — It matches templates, doesn't reason

### The "best result" in context

The 27% accuracy (Llama-3.2-3B-Instruct) should be contextualized:

| Approach | GSM-8K Accuracy |
|----------|-----------------|
| This experiment (best) | 27% |
| GPT-4 (zero-shot CoT) | ~92% |
| Fine-tuned 7B models | 50-70% |
| Random baseline | ~0% |

The result is better than random but far from competitive.

---

## 5. Missing Components

### Not included in experiment

1. **No training on actual GSM-8K examples** — Only synthetic data used
2. **No linguistic diversity** — One template per pattern, fixed phrasing
3. **No natural language chain-of-thought** — Pure symbolic traces only
4. **No detailed error analysis on 100-sample** — Only 10-sample analyzed
5. **No comparison to baselines** — How does vanilla fine-tuning perform?
6. **No retrieval augmentation** — No similar-problem lookup

### Data that would help

1. Per-problem failure analysis on 100+ samples
2. Attention visualization on correct vs incorrect traces
3. Comparison of model outputs vs ground truth traces
4. Linguistic coverage analysis (GSM-8K vocab vs training vocab)

---

## 6. Recommendations

### Short-term fixes

| Fix | Effort | Expected Impact |
|-----|--------|-----------------|
| Include GSM-8K training split with trace annotations | Medium | High |
| Expand evaluation to 100+ random samples | Low | High (better signal) |
| Add linguistic augmentation (paraphrase templates) | Medium | Medium |
| Test without chat template (raw completion) | Low | Unknown |
| Add trace length validation in generator | Low | Prevents truncation bugs |

### Architectural changes

1. **Hybrid approach** — Model generates natural language CoT first, then structured trace
   - Leverage pre-trained reasoning capabilities
   - Use trace as verification, not sole output

2. **Iterative refinement** — Let model revise traces based on execution feedback
   - If trace fails, show error and let model correct
   - Multi-turn generation

3. **Retrieval-augmented** — Find similar solved problems to guide trace generation
   - Use embedding similarity to find relevant examples
   - Include in prompt as few-shot demonstrations

4. **Curriculum learning** — Start with simple patterns, gradually increase complexity
   - Currently all patterns trained simultaneously
   - May help with composition learning

### Evaluation improvements

1. **Stratified sampling** — Ensure evaluation covers all difficulty levels and problem types
2. **Error taxonomy** — Classify failures systematically (value extraction, operation selection, etc.)
3. **Baseline comparison** — Compare to vanilla SFT on GSM-8K with same compute budget
4. **Separate validation set** — Don't use same 10 problems for development and evaluation

---

## 7. Summary Assessment

| Aspect | Assessment | Details |
|--------|------------|---------|
| **Thesis** | Sound | Separation of structure/compute is valid |
| **Implementation** | Good | Clean code, proper reward shaping, modular design |
| **Experimentation** | Good | Systematic ablations, clear run tracking |
| **Evaluation** | Flawed | 10-sample probe was misleading; need larger random samples |
| **Results** | Modest | 27% best vs 2% baseline, but far from SOTA |
| **Documentation** | Excellent | Thorough patterns/results docs, good organization |
| **Generalization** | Poor | Format mastery ≠ reasoning capability |

---

## 8. Conclusion

The experiment successfully demonstrates that small models can learn structured output formats with high fidelity (100% parse rate). The training patterns, anti-short-circuit constraints, and hybrid variable naming are valuable contributions.

However, the experiment does not demonstrate math reasoning capability. The 100-sample evaluation reveals that pattern memorization doesn't generalize to novel problems. The gap between 90% (10-sample) and 2% (100-sample) accuracy exposes a fundamental evaluation flaw.

**Key insight**: The model learned to match templates, not to reason about math. The 59 training schemas cover 93% of GSM-8K problem *structures*, but the model can't map novel linguistic expressions to those structures.

**Path forward**: The approach needs:
1. Training on real GSM-8K examples (not just synthetic)
2. Diverse linguistic patterns (not one template per pattern)
3. Larger evaluation samples (not 10 hardcoded problems)
4. Possibly a hybrid approach combining natural language CoT with structured verification

The infrastructure is solid. The missing piece is data diversity.
