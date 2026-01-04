# Introspection & MoE Differentiation Roadmap

## Vision

Position Lazarus as **"The only framework that understands your experts"** - the definitive tool for LLM interpretability and MoE analysis on Apple Silicon.

## Current Strengths (Moat)

### Introspection Tier 1 (Rare & Hard to Replicate)
| Capability | Location | Why It's Unique |
|------------|----------|-----------------|
| Commutativity Analysis | `patcher.py` | Reveals lookup tables vs algorithms |
| Format Sensitivity Detection | `layer_analysis.py` | Research-grade tokenizer debugging |
| Multi-Family Ablation | `ablation/study.py` | Single interface for 7+ architectures |
| Steering + Logit Lens | `steering/core.py` | Few projects combine these |
| Criterion-Based Causality | `ablation/` | Arbitrary predicates, not just output diff |

### MoE Tier 1 (Unique Capabilities)
| Capability | Location | Why It's Unique |
|------------|----------|-----------------|
| Expert Taxonomy Generation | `moe/identification.py` | Semantic understanding, not just stats |
| Token-Level Router Transparency | `moe/expert_router.py` | Most tools show sequence-level only |
| Co-activation Pair Analysis | `moe/router.py` | Reveals collaboration patterns |
| Compression Planning | `moe/compression.py` | Actionable size/quality estimates |
| 22+ CLI Handlers | `cli/commands/introspect/moe_expert/` | Unmatched UX |

---

## Phase 1: Complete What's Started

### 1.1 Implement `generate_with_topk_sync()`
**File:** `src/chuk_lazarus/introspection/moe/expert_router.py:558-561`

Currently a placeholder that just calls normal generation. Need to:
- Modify router to use custom k value during generation
- Support both softmax and sigmoid routers
- Handle GPT-OSS batched vs standard MoE architectures

```python
# Current (placeholder):
def _generate_with_topk_sync(self, prompt: str, k: int, max_tokens: int) -> str:
    return self._generate_normal_sync(prompt, max_tokens)

# Target: Actually modify k during generation
```

### 1.2 Add Activation Overlap to Compression
**File:** `src/chuk_lazarus/introspection/moe/compression.py:102`

Currently hardcoded to 0.0:
```python
activation_overlap=0.0,  # Requires activation data
```

Need to:
- Capture expert activations across a dataset
- Compute Jaccard similarity of activation patterns
- Weight merge candidates by both weight AND activation overlap

### 1.3 Implement Expert Vocabulary Contribution
**File:** `src/chuk_lazarus/introspection/moe/logit_lens.py`

Add per-expert logit contributions to understand vocabulary specialization:
- Which tokens each expert "prefers" to predict
- Expert-specific vocabulary statistics
- Token-to-expert preference mapping

---

## Phase 2: Visualization & Shareability

### 2.1 Circuit Graph Export
**New file:** `src/chuk_lazarus/introspection/circuit/export.py`

Export discovered circuits as:
- DOT format (Graphviz)
- JSON graph format
- HTML interactive visualization
- Mermaid diagrams

### 2.2 MoE Routing Heatmaps
**New file:** `src/chuk_lazarus/introspection/moe/visualization.py`

Create visualization utilities:
- Token × Expert activation heatmaps
- Layer-wise routing flow diagrams
- Expert utilization bar charts
- Matplotlib + optional Plotly backends

### 2.3 Jupyter Widget Support
**New file:** `src/chuk_lazarus/introspection/widgets.py`

Interactive widgets for notebooks:
- Expert selector with live generation
- Layer slider with routing display
- Activation steering controls

---

## Phase 3: Advanced Causal Analysis

### 3.1 Cross-Layer Expert Tracking
**New file:** `src/chuk_lazarus/introspection/moe/tracking.py`

Track expert evolution through model depth:
- Match experts across layers by specialization
- Identify "math pipeline" (experts that handle math across layers)
- Visualize expert role evolution
- Compute cross-layer expert alignment scores

### 3.2 Counterfactual Interventions
**Extend:** `src/chuk_lazarus/introspection/patcher.py`

"What if" experiments:
- "What if expert X was suppressed?"
- "What if this neuron fired differently?"
- Intervention effect propagation tracking
- Causal graph construction from interventions

### 3.3 Automated Circuit Discovery
**New file:** `src/chuk_lazarus/introspection/circuit/discovery.py`

End-to-end pipeline:
- Input: Task dataset (e.g., arithmetic problems)
- Output: Discovered circuits with confidence scores
- Automatic ablation sweeps
- Direction extraction and validation

---

## Phase 4: Research-Grade Features

### 4.1 Expert Distillation
**New file:** `src/chuk_lazarus/training/distillation/expert_distill.py`

Compress MoE → dense via specialization:
- Identify specialist experts
- Train dense model to mimic expert behavior
- Quality-aware compression
- Benchmark distillation quality

### 4.2 Routing-Aware Fine-Tuning
**Extend:** `src/chuk_lazarus/training/trainers/`

Expert-aware training:
- Freeze generalist experts, train specialists
- Route-specific LoRA adapters
- Expert load balancing loss terms
- Specialization-preserving regularization

### 4.3 Expert Transplantation
**New file:** `src/chuk_lazarus/introspection/moe/transplant.py`

Transfer experts between models:
- Expert similarity across models
- Weight alignment and transplantation
- Quality validation after transplant
- Cross-model expert comparison

---

## Implementation Priority

| Phase | Task | Effort | Impact | Priority |
|-------|------|--------|--------|----------|
| 1.1 | `generate_with_topk_sync()` | Medium | High | P0 |
| 1.2 | Activation overlap | Medium | High | P0 |
| 1.3 | Expert vocabulary | Low | Medium | P1 |
| 2.1 | Circuit graph export | Low | High | P1 |
| 2.2 | Routing heatmaps | Medium | High | P1 |
| 2.3 | Jupyter widgets | Medium | Medium | P2 |
| 3.1 | Cross-layer tracking | High | High | P1 |
| 3.2 | Counterfactuals | High | High | P2 |
| 3.3 | Auto circuit discovery | Very High | Very High | P2 |
| 4.1 | Expert distillation | Very High | Medium | P3 |
| 4.2 | Routing-aware training | High | Medium | P3 |
| 4.3 | Expert transplant | High | Low | P3 |

---

## Success Metrics

### Phase 1 Complete When:
- [x] `generate_with_topk` produces different output for k=1 vs k=4
- [x] Compression candidates include activation overlap > 0
- [x] Expert vocabulary maps show top tokens per expert

### Phase 2 Complete When:
- [x] Circuits exportable to DOT/JSON/HTML
- [x] Routing heatmaps render in CLI and notebooks
- [ ] Jupyter widgets functional

### Phase 3 Complete When:
- [x] Cross-layer expert pipelines identified
- [x] Counterfactual experiments documented
- [ ] Auto-discovery finds known circuits (e.g., induction heads)

### Phase 4 Complete When:
- [ ] MoE → dense distillation workflow documented
- [ ] Routing-aware LoRA training example
- [ ] Expert transplant between Llama variants

---

## Competitive Positioning After Roadmap

| Competitor | Current Gap | After Roadmap |
|------------|-------------|---------------|
| TransformerLens | No MoE, sync-only | Full MoE + async + visualization |
| nnsight | Basic MoE | Expert taxonomy + compression + transplant |
| Baukit | No production models | Multi-arch + Pydantic + CLI |
| HF Transformers | No introspection | Deep understanding + causal discovery |

---

## Getting Started

```bash
# Run tests for new features
pytest tests/introspection/moe/ -v

# Example: Test top-k variation
lazarus introspect moe-expert topk --model openai/gpt-oss-20b --k 1 --prompt "127 * 89 ="

# Example: Generate routing heatmap
lazarus introspect moe-expert heatmap --model openai/gpt-oss-20b --prompt "def fib(n):" --output heatmap.png
```
