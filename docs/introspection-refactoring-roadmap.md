# Introspection Layer Refactoring Roadmap

## Current State Analysis

### Files Over 500 Lines (Need Breaking Up)
| File | Lines | Status |
|------|-------|--------|
| moe.py | 3,288 | **CRITICAL** - 8 concerns in one file |
| circuit/dataset.py | 774 | Should split data models from loaders |
| external_memory.py | 723 | Could extract models |
| circuit/probes.py | 693 | Acceptable for now |
| hooks.py | 600 | Slightly over, MoE should compose not duplicate |
| analyzer/core.py | 580 | Acceptable |
| circuit/geometry.py | 573 | Acceptable |
| circuit/collector.py | 562 | Acceptable |
| circuit/cli.py | 553 | Acceptable |
| layer_analysis.py | 548 | Acceptable |

### Well-Structured Subpackages (Use as Templates)
```
ablation/           # 5 files, clean separation
├── __init__.py     # 28 lines - exports
├── config.py       # 81 lines - config classes
├── models.py       # 26 lines - Pydantic models
├── adapter.py      # 256 lines - model interface
└── study.py        # 476 lines - core logic

steering/           # 5 files
├── __init__.py     # exports
├── config.py       # config + enums
├── core.py         # 494 lines - main class
├── hook.py         # hook implementation
└── legacy.py       # deprecated

analyzer/           # 4 files
├── __init__.py     # exports
├── config.py       # config
├── models.py       # 213 lines - Pydantic models
└── core.py         # 580 lines - async analyzer
```

### Design Principles

1. **No file over 500 lines** - split if larger
2. **Composition over duplication** - MoEHooks wraps ModelHooks
3. **JSON for data** - prompts, keywords, categories
4. **Pydantic native** - BaseModel with ConfigDict(frozen=True)
5. **Use existing enums** - from enums.py
6. **Async-native** - follow analyzer/ patterns

---

## Phase 1: MoE Subpackage (Priority 1)

Break `moe.py` (3,288 lines) into:

```
moe/
├── __init__.py           # ~30 lines - exports
├── enums.py              # ~50 lines - MoE-specific enums
├── config.py             # ~80 lines - MoECaptureConfig
├── models.py             # ~150 lines - Pydantic models
├── detector.py           # ~100 lines - architecture detection
├── hooks.py              # ~200 lines - MoEHooks (composes ModelHooks)
├── router.py             # ~150 lines - router analysis
├── analysis.py           # ~150 lines - analyze_moe_model, print_moe_analysis
├── datasets/
│   ├── __init__.py       # ~30 lines
│   ├── loader.py         # ~100 lines - load from JSON
│   ├── prompts.json      # data file
│   └── categories.json   # data file
│
├── ablation/             # Subpackage for expert ablation
│   ├── __init__.py       # ~20 lines
│   ├── models.py         # ~50 lines - ExpertAblationResult
│   └── study.py          # ~250 lines - MoEAblation
│
├── logit_lens/           # Subpackage for MoE logit lens
│   ├── __init__.py       # ~20 lines
│   ├── models.py         # ~50 lines - ExpertContribution, MoELayerPrediction
│   └── lens.py           # ~180 lines - MoELogitLens
│
├── identification/       # Subpackage for expert identification
│   ├── __init__.py       # ~20 lines
│   ├── models.py         # ~80 lines - ExpertIdentity, ExpertIdentificationResult
│   ├── categorizer.py    # ~200 lines - token categorization
│   └── identifier.py     # ~350 lines - ExpertIdentifier
│
└── compression/          # Subpackage for expert compression
    ├── __init__.py       # ~20 lines
    ├── models.py         # ~100 lines - CompressionPlan, etc.
    ├── estimator.py      # ~150 lines - size estimation
    └── compressor.py     # ~300 lines - ExpertCompressor
```

**Total: ~2,400 lines across 20+ files (avg ~120 lines each)**

### Integration Points

1. **moe/hooks.py composes hooks.ModelHooks**
   ```python
   class MoEHooks:
       def __init__(self, model):
           self._hooks = ModelHooks(model)  # Compose, don't inherit
   ```

2. **moe/ablation/ reuses ablation/ patterns**
   ```python
   from ..ablation import AblationStudy, ModelAdapter
   ```

3. **moe/logit_lens/ shares with logit_lens.py**
   ```python
   from ..logit_lens import LogitLens, TokenEvolution
   ```

4. **moe/enums.py extends enums.py**
   ```python
   # Add MoE-specific enums to introspection/enums.py
   class ExpertCategory(str, Enum): ...
   class MoEArchitecture(str, Enum): ...
   ```

---

## Phase 2: Fix Other Large Files

### 2.1 external_memory.py (723 lines)
Split into:
```
external_memory/
├── __init__.py
├── models.py       # MemoryAnalysisResult, AttractorNode, etc.
├── analyzer.py     # Core analysis logic
└── storage.py      # Memory storage/retrieval
```

### 2.2 circuit/dataset.py (774 lines)
Split into:
```
circuit/
├── dataset/
│   ├── __init__.py
│   ├── models.py   # LabeledPrompt, etc.
│   ├── loader.py   # Load from files
│   └── builder.py  # Build datasets programmatically
```

### 2.3 hooks.py (600 lines)
Consider extracting:
- `hooks/config.py` - CaptureConfig, LayerSelection
- `hooks/state.py` - CapturedState
- `hooks/core.py` - ModelHooks class

---

## Phase 3: Standardize Models Location

Currently models are scattered:
- `analyzer/models.py` (213 lines)
- `ablation/models.py` (26 lines)
- `models/facts.py` (228 lines)
- `models/arithmetic.py` (219 lines)
- `circuit/` has models inline

### Proposed Structure
```
introspection/
├── models/                 # Centralized models
│   ├── __init__.py         # Re-exports all
│   ├── base.py             # Shared base classes
│   ├── arithmetic.py       # Arithmetic models (existing)
│   ├── facts.py            # Fact models (existing)
│   ├── analysis.py         # Analysis result models
│   ├── patching.py         # Patching/intervention models
│   ├── probing.py          # Probing models
│   ├── uncertainty.py      # Uncertainty models
│   └── memory.py           # Memory analysis models
│
├── enums.py                # All enums (existing, extend)
```

---

## Phase 4: CLI Refactoring

The CLI command `moe_expert.py` (1,723 lines) should:
1. **Use moe/ subpackage** - don't duplicate logic
2. **Be thin** - just argument parsing and output formatting
3. **Compose modules** - call into moe/ for actual work

```python
# Instead of ExpertRouter class in CLI:
from ...introspection.moe import MoEHooks, detect_moe_architecture
from ...introspection.moe.identification import ExpertIdentifier
from ...introspection.moe.ablation import MoEAblation
```

---

## Implementation Order

### Week 1: Foundation
1. Create `moe/datasets/` with JSON files and loader ✓ (started)
2. Create `moe/enums.py` and add to `enums.py`
3. Create `moe/config.py`
4. Create `moe/models.py`

### Week 2: Core MoE
5. Create `moe/detector.py`
6. Create `moe/hooks.py` (compose ModelHooks)
7. Create `moe/router.py`
8. Create `moe/analysis.py`

### Week 3: MoE Features
9. Create `moe/ablation/` subpackage
10. Create `moe/logit_lens/` subpackage
11. Create `moe/identification/` subpackage
12. Create `moe/compression/` subpackage

### Week 4: Integration
13. Create `moe/__init__.py` with full exports
14. Update main `introspection/__init__.py`
15. Refactor CLI to use new modules
16. Deprecate old `moe.py`

### Future: Other Files
17. Split `external_memory.py`
18. Split `circuit/dataset.py`
19. Consider `hooks/` subpackage
20. Centralize models

---

## Backward Compatibility

```python
# Old moe.py becomes a re-export shim:
"""DEPRECATED: Use introspection.moe subpackage instead."""
import warnings
warnings.warn(
    "moe.py is deprecated. Use introspection.moe instead.",
    DeprecationWarning,
    stacklevel=2,
)
from .moe import *
```

Keep for 2 releases, then remove.

---

## Success Criteria

- [ ] No file over 500 lines
- [ ] MoEHooks composes ModelHooks (no code duplication)
- [ ] All prompts/keywords in JSON files
- [ ] All models are Pydantic BaseModel
- [ ] MoE enums added to enums.py
- [ ] CLI is thin, uses moe/ modules
- [ ] All existing tests pass
- [ ] Backward compatible imports work
