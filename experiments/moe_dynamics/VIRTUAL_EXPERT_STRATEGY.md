# Virtual Expert Compression Strategy

**Date**: 2026-01-19
**Status**: Architecture Designed, Ready for Implementation

---

## Executive Summary

We propose flipping the MoE compression strategy from "prune by frequency" to "prune by externalizability":

| Approach | Result |
|----------|--------|
| **OLD**: Prune cold experts | Lost fluency, kept bad math |
| **NEW**: Prune externalizable experts | Keep fluency, externalize math to tools |

---

## 1. The Problem with Frequency-Based Pruning

Our initial 71% expert reduction achieved:
- Memory: 40GB → 21GB (49% reduction)
- Speed: 5.1 → 77.8 tok/s (15x faster)
- **But**: Perplexity +109% (30.66 vs 14.63)

The model retained computation experts (which do math badly via lookup tables) but lost fluency experts (which are the core LLM strength).

---

## 2. Expert Capability Analysis

We probed 4,608 experts across 12 task categories:

### Externalizable Tasks (can be replaced by tools)
| Category | Tool | Experts Found |
|----------|------|---------------|
| Arithmetic | calculator | ~500 |
| Symbolic Math | sympy | ~300 |
| DateTime | system clock | ~300 |
| Current Data | web APIs | ~300 |
| Code Execution | interpreter | ~400 |
| Unit Conversion | pint | ~250 |

**Total externalizable: 1,323 experts**

### Core LLM Tasks (must keep)
| Category | Why Keep |
|----------|----------|
| Language Fluency | Core LLM strength |
| Style/Tone | Can't externalize |
| Reasoning | Requires context |
| World Knowledge | Static, fast |
| Code Understanding | != execution |
| Creative | Emergent property |

**Total fluency experts: 1,657 experts**

---

## 3. The New Strategy

```
Input → Embed → [L2 Detector] → Route Decision
                    ↓                    ↓
              Normal Path          Virtual Expert
                    ↓                    ↓
           Fluency Experts       External Tool
                    ↓                    ↓
                Output    ←←←←    Inject Result
```

### Step 1: Identify Externalizable Experts
Experts that primarily activate for:
- Arithmetic operations
- DateTime queries
- Current data requests
- Code execution
- Unit conversions

### Step 2: Remove Externalizable Experts
Instead of keeping bad math experts, remove them entirely.

### Step 3: Keep Fluency Experts
Even if "cold" by frequency, these are essential for:
- Natural language generation
- Style and tone
- Reasoning chains
- Creative outputs

### Step 4: Add Virtual Expert Detection
At layer 2 (early), detect task type and route:
- Math queries → Calculator (sympy/eval)
- Time queries → System clock
- Current data → Web APIs
- Code execution → Sandboxed interpreter
- Unit conversion → pint library

---

## 4. Virtual Experts Implemented

### Calculator
```python
Input:  "127 * 89 = "
Output: "11303"  # 100% accurate, not lookup
```

### DateTime
```python
Input:  "What day is today?"
Output: "Monday, January 19, 2026"  # Always correct
```

### Interpreter
```python
Input:  "Run this: sorted([3,1,4,1,5,9])"
Output: "[1, 1, 3, 4, 5, 9]"  # Actual execution
```

### Unit Converter
```python
Input:  "Convert 100 meters to feet"
Output: "100.0 meter = 328.08 feet"  # Exact
```

---

## 5. Expected Outcomes

| Metric | Frequency Pruning | Virtual Expert Strategy |
|--------|-------------------|------------------------|
| Experts | 1,344 (29%) | ~1,657 (36%) |
| Perplexity | 30.66 (+109%) | **~15** (target) |
| Math Accuracy | ~95% (lookup) | **100%** (exact) |
| Speed | 77.8 tok/s | ~60 tok/s (slightly slower) |
| Memory | 21GB | ~25GB |
| Current Data | Hallucination | **Accurate** (APIs) |

---

## 6. Implementation Plan

### Phase 1: Build Capability-Aware Lite Model
```bash
# Use probing results to select experts
python build_capability_aware_lite.py \
    --keep-fluency \
    --remove-externalizable \
    --output ./gpt-oss-120b-ve
```

### Phase 2: Integrate Virtual Expert Router
```python
from virtual_expert_architecture import VirtualExpertRouter

router = VirtualExpertRouter(model, tokenizer)
response = router.generate("127 * 89 = ")
# Routes to calculator → "11303"
```

### Phase 3: Validate Quality
```bash
# Compare perplexity
python evaluate_perplexity_120b.py \
    --model ./gpt-oss-120b-ve \
    --compare
```

### Phase 4: Fine-tune (if needed)
If perplexity still high, distill fluency from original model.

---

## 7. Files Created

```
experiments/moe_dynamics/
├── probe_expert_capabilities.py    # Expert capability probing
├── virtual_expert_architecture.py  # Virtual expert router
├── results/capability_profile_120b.json  # Probing results
└── VIRTUAL_EXPERT_STRATEGY.md      # This document
```

---

## 8. Key Insight

> **LLMs should do what LLMs do best: language.**
>
> Math, time, current data, and code execution are better handled by specialized tools that give exact answers.

The compression question isn't "which experts are cold?" but rather "which capabilities can be externalized?"

---

## 9. Next Steps

1. **Build capability-aware lite model** - Remove externalizable, keep fluency
2. **Integrate virtual router** - Early-layer detection and routing
3. **Evaluate perplexity** - Should be much better than frequency pruning
4. **Add more virtual experts** - Web search, RAG, database queries
5. **Benchmark** - Compare math accuracy, fluency, speed

---

## Appendix: Virtual Expert Demo Output

```
✓ [calculator] 127 * 89 =
  → 11303

✓ [calculator] What is 15% of 200?
  → 30.0

✓ [datetime] What day is today?
  → Monday, January 19, 2026

✓ [interpreter] Run this: sorted([3,1,4,1,5,9])
  → [1, 1, 3, 4, 5, 9]

✓ [unit_converter] Convert 100 meters to feet
  → 100.0 meter = 328.08 feet

✓ [LLM] Once upon a time, in a land far away,
  → (uses language model - fluency preserved)

✓ [LLM] The capital of France is
  → (uses language model - knowledge preserved)
```
