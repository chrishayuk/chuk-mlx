# MoE Introspection Refactoring Roadmap

## Problem Statement

`moe.py` is a 3,288-line monolith containing 8 distinct concerns:
1. Architecture detection
2. Hook capture
3. Router analysis
4. Ablation
5. Logit lens
6. Expert identification
7. Expert compression
8. Hardcoded prompts/keywords

Additionally:
- Duplicates patterns from `hooks.py`, `logit_lens.py`, `ablation/`
- Uses `@dataclass` instead of Pydantic `BaseModel`
- Hardcoded magic strings for prompts and token categories
- Not async-native

## Constraints

- **No file over 500 lines** (strict limit for maintainability)
- **Compose existing modules** - don't duplicate hooks.py, logit_lens.py, ablation/
- **Use existing enums** from enums.py where applicable
- **Use existing models** from models.py where applicable
- **Async-native** - follow analyzer/ patterns

## Target Architecture

```
introspection/
├── moe/                          # NEW subpackage
│   ├── __init__.py               # Clean exports
│   ├── config.py                 # MoECaptureConfig, etc.
│   ├── models.py                 # Pydantic models (frozen, validated)
│   ├── detector.py               # Architecture detection
│   ├── hooks.py                  # MoEHooks (composes ModelHooks)
│   ├── router.py                 # Router analysis utilities
│   ├── ablation.py               # MoEAblation (reuses ablation/ patterns)
│   ├── logit_lens.py             # MoELogitLens (shares with main)
│   ├── identifier.py             # ExpertIdentifier
│   ├── compression.py            # ExpertCompressor
│   ├── utils.py                  # Token categorization, etc.
│   └── datasets/
│       ├── __init__.py
│       ├── prompts.json          # Test prompts by category
│       ├── categories.json       # Token category keywords
│       └── loader.py             # Load and validate datasets
│
├── hooks.py                      # Unchanged (MoEHooks composes this)
├── logit_lens.py                 # Unchanged (MoELogitLens shares patterns)
├── ablation/                     # Unchanged (MoEAblation reuses patterns)
└── moe.py                        # DEPRECATED - re-exports from moe/
```

## Design Principles

1. **Composition over inheritance** - MoEHooks wraps ModelHooks
2. **Data-driven config** - JSON for prompts, keywords, categories
3. **Pydantic-native** - BaseModel with ConfigDict(frozen=True)
4. **No magic strings** - All literals in JSON config
5. **Async-ready** - Design for future async support
6. **Backward compatible** - Old imports still work during transition

## Implementation Phases

### Phase 1: Foundation (Priority 1)

#### 1.1 Create JSON Datasets
```
moe/datasets/
├── prompts.json        # Categorized test prompts
├── categories.json     # Token category keywords
└── loader.py           # Pydantic models + loaders
```

**prompts.json structure:**
```json
{
  "version": "1.0",
  "categories": {
    "code": {
      "python": ["def fibonacci(n):", ...],
      "javascript": ["const x = () => {", ...],
      "rust": ["fn main() {", ...],
      "sql": ["SELECT * FROM", ...]
    },
    "math": {
      "arithmetic": ["127 * 89 = ", ...],
      "algebra": ["Solve for x: 2x + 5 = 13", ...]
    },
    "structure": {
      "punctuation": ["Hello, how are you?", ...],
      "proper_nouns": ["Barack Obama was", ...]
    }
  }
}
```

#### 1.2 Create config.py + models.py
Extract from moe.py:
- `MoECaptureConfig` → config.py
- All dataclasses → models.py as Pydantic BaseModel

```python
# models.py
class ExpertIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    expert_idx: int = Field(ge=0)
    layer_idx: int = Field(ge=0)
    primary_category: ExpertCategory
    confidence: float = Field(ge=0, le=1)
```

#### 1.3 Create detector.py
Extract from moe.py lines 51-125:
- `MoEArchitecture` enum
- `detect_moe_architecture()`
- `get_moe_layer_info()`

### Phase 2: Core Modules (Priority 2)

#### 2.1 Create hooks.py with Composition
```python
class MoEHooks:
    """MoE-aware hooks that compose ModelHooks."""

    def __init__(self, model: nn.Module):
        self._hooks = ModelHooks(model)  # Delegate
        self.architecture = detect_moe_architecture(model)

    def configure(self, config: MoECaptureConfig) -> Self:
        # Convert to CaptureConfig and delegate
        self._hooks.configure(self._to_capture_config(config))
        return self
```

#### 2.2 Create router.py
Pure analysis functions:
- `analyze_router_entropy()`
- `compute_expert_utilization()`
- `print_router_summary()`

#### 2.3 Create utils.py
Token categorization extracted from ExpertIdentifier:
- `categorize_token()`
- `detect_semantic_clusters()`
- Load keywords from categories.json

### Phase 3: Heavy Modules (Priority 3)

#### 3.1 Create identifier.py
The 642-line ExpertIdentifier class, refactored to:
- Use datasets/prompts.json
- Use utils.py for token categorization
- Return Pydantic models

#### 3.2 Create compression.py
The 500-line ExpertCompressor, standalone.

#### 3.3 Create ablation.py
Reuse patterns from `ablation/study.py`:
```python
class MoEAblation:
    def __init__(self, model, tokenizer):
        self._study = AblationStudy(ModelAdapter(...))
        self._hooks = MoEHooks(model)
```

#### 3.4 Create logit_lens.py
Share patterns with main `logit_lens.py`:
```python
class MoELogitLens:
    def __init__(self, model, tokenizer):
        self._lens = LogitLens(...)  # Compose
        self._hooks = MoEHooks(model)
```

### Phase 4: Integration (Priority 4)

#### 4.1 Update moe.py to re-export
```python
# moe.py - DEPRECATED
"""Use introspection.moe subpackage instead."""
from .moe import *  # Re-export everything
```

#### 4.2 Refactor CLI commands
Update `cli/commands/introspect/moe_expert.py` to use new modules.

#### 4.3 Update __init__.py exports
Maintain backward compatibility.

## File Size Comparison

| Before | After |
|--------|-------|
| moe.py: 3,288 lines | moe/: ~2,700 lines total |

Breakdown:
- config.py: ~80 lines
- models.py: ~150 lines
- detector.py: ~100 lines
- hooks.py: ~250 lines (reduced via composition)
- router.py: ~200 lines
- utils.py: ~300 lines
- identifier.py: ~600 lines
- compression.py: ~400 lines
- ablation.py: ~200 lines
- logit_lens.py: ~200 lines
- datasets/: ~200 lines code + JSON data

## Key Patterns to Follow

### From ablation/ subpackage:
```
ablation/
├── config.py    # Config classes only
├── models.py    # Pydantic models only
├── adapter.py   # Model interface
├── study.py     # Core logic
└── __init__.py  # Clean exports
```

### From circuit/ subpackage:
- JSON datasets in `probe_datasets/`
- Separate dataset.py for loading

### From analyzer/ subpackage:
- Async-native design
- Clear config → models → core separation

## Migration Path

1. **Phase 1**: Create moe/ alongside moe.py (non-breaking)
2. **Phase 2**: moe.py re-exports from moe/ (non-breaking)
3. **Phase 3**: Deprecation warnings (1-2 releases)
4. **Phase 4**: Remove moe.py

## Success Criteria

- [ ] No magic strings in Python code
- [ ] All models are Pydantic BaseModel with frozen=True
- [ ] MoEHooks composes ModelHooks (no duplication)
- [ ] MoEAblation reuses ablation/ patterns
- [ ] MoELogitLens shares with logit_lens.py
- [ ] Prompts loaded from JSON
- [ ] Token categories loaded from JSON
- [ ] CLI works with new structure
- [ ] All existing tests pass
- [ ] Backward compatible imports work
