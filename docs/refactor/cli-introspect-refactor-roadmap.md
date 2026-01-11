# CLI Introspect Refactoring Roadmap

## Executive Summary

The `src/chuk_lazarus/cli/commands/introspect/` directory contains overly large, flat CLI files that violate separation of concerns. The worst offender is `moe_expert.py` at **4,928 lines** with **zero test coverage**.

### Key Principles for Refactor
1. **Async-native**: All I/O operations use async/await
2. **Pydantic-native**: All data structures use Pydantic models
3. **No magic strings**: Enums and constants for all categorical values
4. **CLI is thin**: Only argument parsing, validation, and output formatting
5. **Framework has logic**: Business logic lives in `introspection/` package
6. **No dictionary goop**: Typed models instead of raw dicts
7. **Tests mirror structure**: `tests/` mirrors `src/` with 90%+ coverage per file

---

## Current State Analysis

### File Sizes in `cli/commands/introspect/`

| File | Lines | Size | Tests | Coverage |
|------|-------|------|-------|----------|
| `moe_expert.py` | 4,928 | 185KB | **NONE** | **0%** |
| `circuit.py` | 1,276 | 44KB | Yes | ~70% |
| `neurons.py` | 981 | 35KB | Yes | ~60% |
| `analyze.py` | 877 | 38KB | Yes | ~75% |
| `memory.py` | 807 | 29KB | Yes | ~65% |
| `probing.py` | 614 | 22KB | Yes | ~55% |
| `embedding.py` | 558 | 20KB | Yes | ~70% |
| `ablation.py` | 448 | 16KB | Yes | ~65% |
| `clustering.py` | 319 | 11KB | Yes | ~60% |
| `generation.py` | 273 | 9KB | Yes | ~70% |
| `steering.py` | 251 | 9KB | Yes | ~65% |
| `arithmetic.py` | 221 | 9KB | Yes | ~60% |
| `patching.py` | 183 | 6KB | Yes | ~70% |
| `layer.py` | 152 | 5KB | Yes | ~75% |
| `virtual_expert.py` | 414 | 13KB | Yes | ~50% |

### Critical Issues in `moe_expert.py`

1. **21 action handlers** in if/elif chain (lines 41-88)
2. **Magic strings**: "chat", "compare", "ablate", "topk", etc.
3. **Untyped dicts**: `_get_moe_info()` returns `dict` not Pydantic model
4. **Sync-only**: No async/await despite I/O-heavy operations
5. **1,000+ line class**: `ExpertRouter` (lines 140-1161)
6. **Hardcoded test data**: Benchmark problems at lines 1340-1344
7. **Print-based output**: No structured output format
8. **Zero tests**: 4,928 lines of completely untested code

---

## Target Architecture

### Directory Structure After Refactor

```
src/chuk_lazarus/
├── introspection/
│   └── moe/
│       ├── __init__.py           # Public API exports
│       ├── enums.py              # MoEAction, ExpertCategory, etc. (exists, expand)
│       ├── models.py             # Pydantic models (exists, expand)
│       ├── config.py             # Configuration models (exists)
│       ├── detector.py           # MoE detection (exists)
│       ├── hooks.py              # Routing hooks (exists)
│       ├── router.py             # Router analysis (exists, expand)
│       ├── ablation.py           # Expert ablation (exists)
│       ├── expert_router.py      # NEW: ExpertRouter class from CLI
│       ├── generation.py         # NEW: Generation with routing control
│       ├── analysis.py           # NEW: Expert analysis functions
│       ├── tokenizer_analysis.py # NEW: Token-expert mapping
│       ├── entropy.py            # NEW: Routing entropy analysis
│       ├── taxonomy.py           # NEW: Expert taxonomy/patterns
│       └── output.py             # NEW: Structured output formatters

├── cli/
│   └── commands/
│       └── introspect/
│           ├── moe_expert/                  # NEW: Split into submodule
│           │   ├── __init__.py              # Exports main dispatcher
│           │   ├── enums.py                 # MoEAction enum
│           │   ├── dispatcher.py            # Action dispatch table
│           │   ├── handlers/
│           │   │   ├── __init__.py
│           │   │   ├── chat.py              # chat action
│           │   │   ├── compare.py           # compare action
│           │   │   ├── ablate.py            # ablate action
│           │   │   ├── topk.py              # topk action
│           │   │   ├── collaboration.py     # collab action
│           │   │   ├── pairs.py             # pairs action
│           │   │   ├── interactive.py       # interactive action
│           │   │   ├── weights.py           # weights action
│           │   │   ├── tokenizer.py         # tokenizer action
│           │   │   ├── control_tokens.py    # control-tokens action
│           │   │   ├── trace.py             # trace action
│           │   │   ├── entropy.py           # entropy action
│           │   │   ├── divergence.py        # divergence action
│           │   │   ├── role.py              # role action
│           │   │   ├── context_test.py      # context-test action
│           │   │   ├── vocab_map.py         # vocab-map action
│           │   │   ├── router_probe.py      # router-probe action
│           │   │   ├── pattern_discovery.py # pattern-discovery action
│           │   │   ├── taxonomy.py          # full-taxonomy action
│           │   │   └── layer_sweep.py       # layer-sweep action
│           │   └── formatters.py            # Output formatting utilities
│           └── moe_expert.py                # DEPRECATED: Thin wrapper

tests/
├── introspection/
│   └── moe/
│       ├── test_expert_router.py    # Tests for ExpertRouter class
│       ├── test_generation.py       # Tests for generation functions
│       ├── test_analysis.py         # Tests for analysis functions
│       ├── test_tokenizer_analysis.py
│       ├── test_entropy.py
│       └── test_taxonomy.py
└── cli/
    └── commands/
        └── introspect/
            └── moe_expert/
                ├── conftest.py              # Shared fixtures
                ├── test_dispatcher.py       # Dispatcher tests
                └── handlers/
                    ├── test_chat.py
                    ├── test_compare.py
                    ├── test_ablate.py
                    ├── test_topk.py
                    ├── test_collaboration.py
                    ├── test_pairs.py
                    ├── test_interactive.py
                    ├── test_weights.py
                    ├── test_tokenizer.py
                    ├── test_control_tokens.py
                    ├── test_trace.py
                    ├── test_entropy.py
                    ├── test_divergence.py
                    ├── test_role.py
                    ├── test_context_test.py
                    ├── test_vocab_map.py
                    ├── test_router_probe.py
                    ├── test_pattern_discovery.py
                    ├── test_taxonomy.py
                    └── test_layer_sweep.py
```

---

## Phase 0: Externalize Hardcoded Data to JSON

**Principle**: All prompts, test data, taxonomies, benchmarks, and examples must be in JSON files, not hardcoded in Python.

### Current Violations

The codebase has hardcoded data in multiple files:

| File | Location | Data Type | Lines |
|------|----------|-----------|-------|
| `moe_expert.py` | `_ablate_expert()` | Benchmark problems | 1340-1344 |
| `moe_expert.py` | `_test_expert_pairs()` | Benchmark problems | 1681-1685 |
| `moe_expert.py` | `_test_context_independence()` | Test prompts | 3703-3708 |
| `moe_expert.py` | `_discover_expert_patterns()` | Test prompts dict | 4195-4260 |
| `probing.py` | `_detect_uncertainty()` | Working/broken prompts | 255-268 |
| `virtual_expert.py` | `introspect_virtual_expert()` | Default problems | 254-261 |

### Target: JSON Dataset Files

Create a structured dataset directory:

```
src/chuk_lazarus/introspection/
└── datasets/
    ├── __init__.py              # Dataset loaders
    ├── benchmarks/
    │   ├── arithmetic.json      # Math benchmark problems
    │   ├── multiplication.json  # Multiplication-specific
    │   └── schema.json          # JSON schema for validation
    ├── probing/
    │   ├── uncertainty.json     # Working/broken prompts for uncertainty detection
    │   └── schema.json
    ├── moe/
    │   ├── prompts.json         # (exists) - Category prompts
    │   ├── context_tests.json   # Context independence tests
    │   ├── pattern_discovery.json # Pattern discovery test prompts
    │   ├── taxonomies/
    │   │   ├── expert_roles.json
    │   │   └── token_categories.json
    │   └── schema.json
    └── common/
        ├── format_sensitivity.json
        └── schema.json
```

### Example JSON Structures

#### `benchmarks/arithmetic.json`
```json
{
  "$schema": "./schema.json",
  "version": "1.0.0",
  "description": "Arithmetic benchmark problems for expert ablation testing",
  "problems": {
    "simple": [
      {"prompt": "2 + 2 = ", "answer": 4, "operation": "addition"},
      {"prompt": "5 * 5 = ", "answer": 25, "operation": "multiplication"},
      {"prompt": "10 - 3 = ", "answer": 7, "operation": "subtraction"}
    ],
    "medium": [
      {"prompt": "23 * 17 = ", "answer": 391, "operation": "multiplication"},
      {"prompt": "456 + 789 = ", "answer": 1245, "operation": "addition"}
    ],
    "hard": [
      {"prompt": "127 * 89 = ", "answer": 11303, "operation": "multiplication"},
      {"prompt": "999 * 888 = ", "answer": 887112, "operation": "multiplication"},
      {"prompt": "1234 + 5678 = ", "answer": 6912, "operation": "addition"}
    ]
  }
}
```

#### `probing/uncertainty.json`
```json
{
  "$schema": "./schema.json",
  "version": "1.0.0",
  "description": "Calibration prompts for uncertainty detection",
  "working_prompts": {
    "description": "Prompts that should trigger compute pathway",
    "prompts": [
      "100 - 37 = ",
      "50 + 25 = ",
      "10 * 10 = ",
      "200 - 50 = ",
      "25 * 4 = "
    ]
  },
  "broken_prompts": {
    "description": "Prompts that should trigger refusal/uncertainty",
    "prompts": [
      "100 - 37 =",
      "50 + 25 =",
      "10 * 10 =",
      "200 - 50 =",
      "25 * 4 ="
    ]
  }
}
```

#### `moe/pattern_discovery.json`
```json
{
  "$schema": "./schema.json",
  "version": "1.0.0",
  "description": "Test prompts for expert pattern discovery",
  "categories": {
    "num_seq": {
      "description": "Pure number sequences",
      "prompts": [
        "1", "42", "127", "999", "3.14",
        "1 2", "42 127", "100 200", "1 2 3", "10 20 30 40",
        "1 + 2", "42 * 3", "100 - 50", "10 / 2"
      ]
    },
    "word_seq": {
      "description": "Pure word sequences",
      "prompts": [
        "the", "Hello", "world", "Python",
        "the cat", "Hello world", "a b c",
        "The quick brown fox"
      ]
    },
    "code_patterns": {
      "description": "Code-like patterns",
      "prompts": [
        "def ", "class ", "import ", "return ",
        "def foo():", "class Bar:", "import numpy"
      ]
    },
    "punctuation": {
      "description": "Punctuation-heavy patterns",
      "prompts": [
        ".", ",", "!", "?",
        "...", "!?", "\"Hello\"", "'world'"
      ]
    }
  }
}
```

#### `moe/context_tests.json`
```json
{
  "$schema": "./schema.json",
  "version": "1.0.0",
  "description": "Context independence test prompts",
  "tests": [
    {"prompt": "111 127", "context_type": "number"},
    {"prompt": "222 127", "context_type": "number"},
    {"prompt": "abc 127", "context_type": "word"},
    {"prompt": "xyz 127", "context_type": "word"}
  ],
  "target_token": "127"
}
```

### Dataset Loader Pattern

```python
# src/chuk_lazarus/introspection/datasets/__init__.py

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class DatasetLoader:
    """Load and cache JSON datasets with Pydantic validation."""

    _base_path: Path = Path(__file__).parent

    @classmethod
    @lru_cache(maxsize=32)
    def load_json(cls, relative_path: str) -> dict:
        """Load raw JSON data with caching."""
        path = cls._base_path / relative_path
        with open(path) as f:
            return json.load(f)

    @classmethod
    def load_model(cls, relative_path: str, model_class: type[T]) -> T:
        """Load JSON and validate with Pydantic model."""
        data = cls.load_json(relative_path)
        return model_class.model_validate(data)


# Convenience functions
def get_arithmetic_benchmarks() -> ArithmeticBenchmark:
    """Load arithmetic benchmark problems."""
    return DatasetLoader.load_model("benchmarks/arithmetic.json", ArithmeticBenchmark)


def get_uncertainty_prompts() -> UncertaintyPrompts:
    """Load uncertainty detection calibration prompts."""
    return DatasetLoader.load_model("probing/uncertainty.json", UncertaintyPrompts)


def get_pattern_discovery_prompts() -> PatternDiscoveryPrompts:
    """Load pattern discovery test prompts."""
    return DatasetLoader.load_model("moe/pattern_discovery.json", PatternDiscoveryPrompts)
```

### Pydantic Models for Datasets

```python
# src/chuk_lazarus/introspection/datasets/models.py

from pydantic import BaseModel, ConfigDict, Field


class ArithmeticProblem(BaseModel):
    """A single arithmetic problem."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    answer: int
    operation: str


class ArithmeticBenchmark(BaseModel):
    """Full arithmetic benchmark dataset."""

    model_config = ConfigDict(frozen=True)

    version: str
    description: str
    problems: dict[str, tuple[ArithmeticProblem, ...]]

    def get_all_problems(self) -> list[ArithmeticProblem]:
        """Get all problems flattened."""
        result = []
        for difficulty_problems in self.problems.values():
            result.extend(difficulty_problems)
        return result

    def get_by_difficulty(self, difficulty: str) -> tuple[ArithmeticProblem, ...]:
        """Get problems by difficulty level."""
        return self.problems.get(difficulty, ())


class UncertaintyPrompts(BaseModel):
    """Calibration prompts for uncertainty detection."""

    model_config = ConfigDict(frozen=True)

    version: str
    description: str
    working_prompts: dict[str, list[str]]
    broken_prompts: dict[str, list[str]]

    @property
    def working(self) -> list[str]:
        return self.working_prompts.get("prompts", [])

    @property
    def broken(self) -> list[str]:
        return self.broken_prompts.get("prompts", [])


class PatternCategory(BaseModel):
    """A category of test prompts for pattern discovery."""

    model_config = ConfigDict(frozen=True)

    description: str
    prompts: tuple[str, ...] = Field(default_factory=tuple)


class PatternDiscoveryPrompts(BaseModel):
    """Test prompts for expert pattern discovery."""

    model_config = ConfigDict(frozen=True)

    version: str
    description: str
    categories: dict[str, PatternCategory]

    def get_category(self, name: str) -> PatternCategory | None:
        return self.categories.get(name)

    def get_all_prompts(self) -> list[tuple[str, str]]:
        """Get all (category_name, prompt) tuples."""
        result = []
        for cat_name, cat in self.categories.items():
            for prompt in cat.prompts:
                result.append((cat_name, prompt))
        return result
```

### Migration Steps

1. **Create dataset directory structure**
2. **Create JSON schema files for validation**
3. **Extract hardcoded data to JSON files**:
   - `moe_expert.py` lines 1340-1344 → `benchmarks/arithmetic.json`
   - `moe_expert.py` lines 4195-4260 → `moe/pattern_discovery.json`
   - `moe_expert.py` lines 3703-3708 → `moe/context_tests.json`
   - `probing.py` lines 255-268 → `probing/uncertainty.json`
4. **Create Pydantic models for each dataset**
5. **Create loader functions**
6. **Update CLI files to use loaders instead of hardcoded data**
7. **Add tests for dataset loading and validation**

### Benefits

1. **Separation of concerns**: Data separate from logic
2. **Easier updates**: Modify JSON without touching Python
3. **Validation**: Pydantic models catch data errors
4. **Discoverability**: All test data in one place
5. **Extensibility**: Easy to add new benchmarks
6. **Testing**: Mock datasets for unit tests
7. **Documentation**: JSON files self-document expected formats

---

## Phase 1: Foundation (Framework Layer)

### 1.1 Expand `introspection/moe/enums.py`

Add `MoEAction` enum for all 21 actions:

```python
# src/chuk_lazarus/introspection/moe/enums.py

class MoEAction(str, Enum):
    """Available MoE expert actions."""

    ANALYZE = "analyze"
    CHAT = "chat"
    COMPARE = "compare"
    ABLATE = "ablate"
    TOPK = "topk"
    COLLABORATION = "collab"
    PAIRS = "pairs"
    INTERACTIVE = "interactive"
    WEIGHTS = "weights"
    TOKENIZER = "tokenizer"
    CONTROL_TOKENS = "control-tokens"
    TRACE = "trace"
    ENTROPY = "entropy"
    DIVERGENCE = "divergence"
    ROLE = "role"
    CONTEXT_TEST = "context-test"
    VOCAB_MAP = "vocab-map"
    ROUTER_PROBE = "router-probe"
    PATTERN_DISCOVERY = "pattern-discovery"
    FULL_TAXONOMY = "full-taxonomy"
    LAYER_SWEEP = "layer-sweep"
```

### 1.2 Expand `introspection/moe/models.py`

Add missing Pydantic models:

```python
# Add to src/chuk_lazarus/introspection/moe/models.py

class MoEModelInfo(BaseModel):
    """Complete MoE model information."""

    model_config = ConfigDict(frozen=True)

    moe_layers: tuple[int, ...] = Field(default_factory=tuple)
    num_experts: int = Field(ge=0)
    num_experts_per_tok: int = Field(ge=0)
    total_layers: int = Field(ge=1)
    architecture: MoEArchitecture = MoEArchitecture.GENERIC
    has_shared_expert: bool = False


class GenerationStats(BaseModel):
    """Statistics from expert-controlled generation."""

    model_config = ConfigDict(frozen=True)

    expert_idx: int = Field(ge=0)
    tokens_generated: int = Field(ge=0)
    layers_modified: int = Field(ge=0)
    moe_type: str
    prompt_tokens: int = Field(ge=0)


class ExpertChatResult(BaseModel):
    """Result from chatting with a specific expert."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    response: str
    expert_idx: int = Field(ge=0)
    stats: GenerationStats


class ExpertComparisonResult(BaseModel):
    """Result from comparing multiple experts."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    expert_results: tuple[ExpertChatResult, ...] = Field(default_factory=tuple)


class TopKVariationResult(BaseModel):
    """Result from varying top-k experts."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    k_value: int = Field(ge=1)
    response: str
    active_experts: tuple[int, ...] = Field(default_factory=tuple)


class RouterWeightCapture(BaseModel):
    """Captured router weights for a single position."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    position_idx: int = Field(ge=0)
    expert_indices: tuple[int, ...] = Field(default_factory=tuple)
    weights: tuple[float, ...] = Field(default_factory=tuple)


class LayerRoutingAnalysis(BaseModel):
    """Routing analysis for a single layer."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    entropy: RouterEntropy
    utilization: ExpertUtilization
    coactivation: CoactivationAnalysis


class ExpertPattern(BaseModel):
    """Discovered pattern for an expert."""

    model_config = ConfigDict(frozen=True)

    expert_idx: int = Field(ge=0)
    layer_idx: int = Field(ge=0)
    pattern_type: str
    trigger_tokens: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(ge=0, le=1)
    sample_activations: int = Field(ge=0)


class ExpertTaxonomy(BaseModel):
    """Complete taxonomy of all experts."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    num_layers: int = Field(ge=1)
    num_experts: int = Field(ge=1)
    expert_identities: tuple[ExpertIdentity, ...] = Field(default_factory=tuple)
    patterns: tuple[ExpertPattern, ...] = Field(default_factory=tuple)
    layer_analyses: tuple[LayerRoutingAnalysis, ...] = Field(default_factory=tuple)
```

### 1.3 Create `introspection/moe/expert_router.py`

Extract `ExpertRouter` class from CLI:

```python
# src/chuk_lazarus/introspection/moe/expert_router.py

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import mlx.core as mx

from .config import MoECaptureConfig
from .detector import get_moe_layer_info, is_moe_model
from .enums import MoEArchitecture
from .models import (
    GenerationStats,
    MoEModelInfo,
    RouterWeightCapture,
)

if TYPE_CHECKING:
    from ....models_v2.core.protocols import ModelProtocol


class ExpertRouter:
    """Async-native utility for manipulating expert routing.

    Example:
        >>> async with ExpertRouter.from_pretrained("openai/gpt-oss-20b") as router:
        ...     result = await router.generate_with_forced_expert(
        ...         prompt="127 * 89 = ",
        ...         expert_idx=6,
        ...         max_tokens=20,
        ...     )
        ...     print(result.response)
    """

    def __init__(
        self,
        model: ModelProtocol,
        tokenizer,
        model_info: MoEModelInfo,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._info = model_info

        if not self._info.moe_layers:
            raise ValueError("Model has no MoE layers")

    @classmethod
    async def from_pretrained(cls, model_id: str) -> ExpertRouter:
        """Load model and create router."""
        # Async model loading
        ...

    async def __aenter__(self) -> ExpertRouter:
        return self

    async def __aexit__(self, *args) -> None:
        # Cleanup if needed
        pass

    @property
    def info(self) -> MoEModelInfo:
        """Get MoE model information."""
        return self._info

    async def generate_with_forced_expert(
        self,
        prompt: str,
        expert_idx: int,
        *,
        max_tokens: int = 100,
        layers: list[int] | None = None,
        temperature: float = 0.0,
    ) -> tuple[str, GenerationStats]:
        """Generate with routing forced to a specific expert."""
        ...

    async def generate_with_ablated_expert(
        self,
        prompt: str,
        expert_idx: int,
        *,
        max_tokens: int = 100,
        layers: list[int] | None = None,
    ) -> tuple[str, GenerationStats]:
        """Generate with a specific expert ablated (removed from routing)."""
        ...

    async def generate_with_topk(
        self,
        prompt: str,
        k: int,
        *,
        max_tokens: int = 100,
    ) -> tuple[str, GenerationStats]:
        """Generate with custom top-k expert selection."""
        ...

    async def capture_router_weights(
        self,
        prompt: str,
        *,
        layers: list[int] | None = None,
    ) -> list[RouterWeightCapture]:
        """Capture router weights for each token position."""
        ...

    async def analyze_coactivation(
        self,
        prompts: list[str],
        *,
        layer_idx: int | None = None,
    ) -> CoactivationAnalysis:
        """Analyze expert co-activation patterns across prompts."""
        ...
```

### 1.4 Create Config Models

```python
# src/chuk_lazarus/introspection/moe/config.py (expand existing)

class ExpertChatConfig(BaseModel):
    """Configuration for expert chat."""

    model_config = ConfigDict(frozen=True)

    expert_idx: int = Field(ge=0)
    max_tokens: int = Field(default=100, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    layers: tuple[int, ...] | None = None
    apply_chat_template: bool = True


class ExpertCompareConfig(BaseModel):
    """Configuration for expert comparison."""

    model_config = ConfigDict(frozen=True)

    expert_indices: tuple[int, ...] = Field(min_length=2)
    max_tokens: int = Field(default=100, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class TaxonomyConfig(BaseModel):
    """Configuration for full expert taxonomy."""

    model_config = ConfigDict(frozen=True)

    sample_prompts_per_category: int = Field(default=10, ge=1)
    layers: tuple[int, ...] | None = None
    include_patterns: bool = True
    include_entropy: bool = True
```

---

## Phase 2: CLI Refactoring

### 2.1 Create Action Enum

```python
# src/chuk_lazarus/cli/commands/introspect/moe_expert/enums.py

from enum import Enum


class MoEAction(str, Enum):
    """CLI actions for moe-expert command."""

    ANALYZE = "analyze"
    CHAT = "chat"
    COMPARE = "compare"
    ABLATE = "ablate"
    TOPK = "topk"
    COLLABORATION = "collab"
    PAIRS = "pairs"
    INTERACTIVE = "interactive"
    WEIGHTS = "weights"
    TOKENIZER = "tokenizer"
    CONTROL_TOKENS = "control-tokens"
    TRACE = "trace"
    ENTROPY = "entropy"
    DIVERGENCE = "divergence"
    ROLE = "role"
    CONTEXT_TEST = "context-test"
    VOCAB_MAP = "vocab-map"
    ROUTER_PROBE = "router-probe"
    PATTERN_DISCOVERY = "pattern-discovery"
    FULL_TAXONOMY = "full-taxonomy"
    LAYER_SWEEP = "layer-sweep"

    @property
    def handler_module(self) -> str:
        """Get the handler module name."""
        return self.value.replace("-", "_")
```

### 2.2 Create Dispatcher with Dispatch Table

```python
# src/chuk_lazarus/cli/commands/introspect/moe_expert/dispatcher.py

from __future__ import annotations

import asyncio
from argparse import Namespace
from typing import Callable

from .enums import MoEAction
from .handlers import (
    handle_chat,
    handle_compare,
    handle_ablate,
    handle_topk,
    handle_collaboration,
    handle_pairs,
    handle_interactive,
    handle_weights,
    handle_tokenizer,
    handle_control_tokens,
    handle_trace,
    handle_entropy,
    handle_divergence,
    handle_role,
    handle_context_test,
    handle_vocab_map,
    handle_router_probe,
    handle_pattern_discovery,
    handle_full_taxonomy,
    handle_layer_sweep,
    handle_analyze,
)

# Dispatch table: action -> handler
_HANDLERS: dict[MoEAction, Callable[[Namespace], None]] = {
    MoEAction.ANALYZE: handle_analyze,
    MoEAction.CHAT: handle_chat,
    MoEAction.COMPARE: handle_compare,
    MoEAction.ABLATE: handle_ablate,
    MoEAction.TOPK: handle_topk,
    MoEAction.COLLABORATION: handle_collaboration,
    MoEAction.PAIRS: handle_pairs,
    MoEAction.INTERACTIVE: handle_interactive,
    MoEAction.WEIGHTS: handle_weights,
    MoEAction.TOKENIZER: handle_tokenizer,
    MoEAction.CONTROL_TOKENS: handle_control_tokens,
    MoEAction.TRACE: handle_trace,
    MoEAction.ENTROPY: handle_entropy,
    MoEAction.DIVERGENCE: handle_divergence,
    MoEAction.ROLE: handle_role,
    MoEAction.CONTEXT_TEST: handle_context_test,
    MoEAction.VOCAB_MAP: handle_vocab_map,
    MoEAction.ROUTER_PROBE: handle_router_probe,
    MoEAction.PATTERN_DISCOVERY: handle_pattern_discovery,
    MoEAction.FULL_TAXONOMY: handle_full_taxonomy,
    MoEAction.LAYER_SWEEP: handle_layer_sweep,
}


def dispatch(args: Namespace) -> None:
    """Dispatch to appropriate handler based on action."""
    action_str = getattr(args, "action", "chat")

    try:
        action = MoEAction(action_str)
    except ValueError:
        print(f"Unknown action: {action_str}")
        print(f"Available actions: {', '.join(a.value for a in MoEAction)}")
        return

    handler = _HANDLERS.get(action)
    if handler is None:
        print(f"Handler not implemented for action: {action.value}")
        return

    handler(args)
```

### 2.3 Handler Pattern (Example: Chat)

```python
# src/chuk_lazarus/cli/commands/introspect/moe_expert/handlers/chat.py

from __future__ import annotations

import asyncio
from argparse import Namespace

from .....introspection.moe import ExpertRouter
from .....introspection.moe.config import ExpertChatConfig
from ..formatters import format_chat_result


def handle_chat(args: Namespace) -> None:
    """Handle the 'chat' action - chat with a specific expert."""
    asyncio.run(_async_chat(args))


async def _async_chat(args: Namespace) -> None:
    """Async implementation of chat handler."""
    # Validate arguments
    if not hasattr(args, "expert") or args.expert is None:
        print("Error: --expert/-e is required for chat action")
        return

    if not hasattr(args, "prompt") or args.prompt is None:
        print("Error: --prompt/-p is required for chat action")
        return

    # Build typed config from args
    config = ExpertChatConfig(
        expert_idx=args.expert,
        max_tokens=getattr(args, "max_tokens", 100),
        temperature=getattr(args, "temperature", 0.0),
        apply_chat_template=not getattr(args, "raw", False),
    )

    # Delegate to framework
    async with ExpertRouter.from_pretrained(args.model) as router:
        result = await router.chat_with_expert(
            prompt=args.prompt,
            config=config,
        )

    # Format and print output
    output = format_chat_result(result, verbose=getattr(args, "verbose", False))
    print(output)
```

### 2.4 Output Formatters

```python
# src/chuk_lazarus/cli/commands/introspect/moe_expert/formatters.py

from __future__ import annotations

from .....introspection.moe.models import (
    ExpertChatResult,
    ExpertComparisonResult,
    ExpertTaxonomy,
    RouterWeightCapture,
)


def format_header(title: str, width: int = 70) -> str:
    """Format a section header."""
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def format_chat_result(result: ExpertChatResult, *, verbose: bool = False) -> str:
    """Format chat result for display."""
    lines = [
        format_header(f"CHAT WITH EXPERT {result.expert_idx}"),
        f"Prompt: {result.prompt}",
        "",
        "Response:",
        result.response,
    ]

    if verbose:
        lines.extend([
            "",
            "Statistics:",
            f"  Tokens generated: {result.stats.tokens_generated}",
            f"  Layers modified: {result.stats.layers_modified}",
            f"  MoE type: {result.stats.moe_type}",
        ])

    return "\n".join(lines)


def format_comparison_result(result: ExpertComparisonResult, *, verbose: bool = False) -> str:
    """Format comparison result for display."""
    lines = [format_header("EXPERT COMPARISON")]
    lines.append(f"Prompt: {result.prompt}")
    lines.append("")

    for expert_result in result.expert_results:
        lines.append(f"--- Expert {expert_result.expert_idx} ---")
        lines.append(expert_result.response)
        lines.append("")

    return "\n".join(lines)


def format_taxonomy(taxonomy: ExpertTaxonomy, *, verbose: bool = False) -> str:
    """Format full taxonomy for display."""
    # ... structured output formatting
    pass
```

---

## Phase 3: Test Implementation

### 3.1 Test File Structure

Tests must mirror source structure with 90%+ coverage per file:

```
tests/
├── introspection/
│   └── moe/
│       ├── conftest.py                  # Shared fixtures
│       ├── test_enums.py                # 100% coverage
│       ├── test_models.py               # 100% coverage
│       ├── test_config.py               # 100% coverage
│       ├── test_detector.py             # 90%+ coverage
│       ├── test_hooks.py                # 90%+ coverage
│       ├── test_router.py               # 90%+ coverage
│       ├── test_expert_router.py        # 90%+ coverage (NEW)
│       ├── test_generation.py           # 90%+ coverage (NEW)
│       ├── test_analysis.py             # 90%+ coverage (NEW)
│       ├── test_tokenizer_analysis.py   # 90%+ coverage (NEW)
│       ├── test_entropy.py              # 90%+ coverage (NEW)
│       └── test_taxonomy.py             # 90%+ coverage (NEW)
│
└── cli/
    └── commands/
        └── introspect/
            └── moe_expert/
                ├── conftest.py              # Shared CLI fixtures
                ├── test_dispatcher.py       # 90%+ coverage
                ├── test_formatters.py       # 90%+ coverage
                └── handlers/
                    ├── conftest.py          # Handler fixtures
                    ├── test_chat.py         # 90%+ coverage
                    ├── test_compare.py      # 90%+ coverage
                    ├── test_ablate.py       # 90%+ coverage
                    ├── test_topk.py         # 90%+ coverage
                    ├── test_collaboration.py
                    ├── test_pairs.py
                    ├── test_interactive.py
                    ├── test_weights.py
                    ├── test_tokenizer.py
                    ├── test_control_tokens.py
                    ├── test_trace.py
                    ├── test_entropy.py
                    ├── test_divergence.py
                    ├── test_role.py
                    ├── test_context_test.py
                    ├── test_vocab_map.py
                    ├── test_router_probe.py
                    ├── test_pattern_discovery.py
                    ├── test_taxonomy.py
                    └── test_layer_sweep.py
```

### 3.2 Test Fixtures (conftest.py)

```python
# tests/introspection/moe/conftest.py

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    MoEModelInfo,
    GenerationStats,
    ExpertChatResult,
)


@pytest.fixture
def mock_moe_model_info() -> MoEModelInfo:
    """Standard MoE model info for testing."""
    return MoEModelInfo(
        moe_layers=(0, 1, 2, 3, 4, 5, 6, 7),
        num_experts=32,
        num_experts_per_tok=4,
        total_layers=8,
        architecture=MoEArchitecture.GPT_OSS,
        has_shared_expert=False,
    )


@pytest.fixture
def mock_generation_stats() -> GenerationStats:
    """Standard generation stats for testing."""
    return GenerationStats(
        expert_idx=6,
        tokens_generated=20,
        layers_modified=8,
        moe_type="gpt_oss",
        prompt_tokens=10,
    )


@pytest.fixture
def mock_expert_router(mock_moe_model_info, mock_generation_stats):
    """Mock ExpertRouter for testing."""
    with patch("chuk_lazarus.introspection.moe.ExpertRouter") as mock_cls:
        mock_router = AsyncMock()
        mock_router.info = mock_moe_model_info
        mock_router.generate_with_forced_expert = AsyncMock(
            return_value=("Test output", mock_generation_stats)
        )
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)
        mock_cls.from_pretrained = AsyncMock(return_value=mock_router)
        yield mock_cls


@pytest.fixture
def mock_model():
    """Mock MLX model for testing."""
    mock = MagicMock()
    mock.model.layers = [MagicMock() for _ in range(8)]
    for i, layer in enumerate(mock.model.layers):
        layer.mlp.router.num_experts = 32
        layer.mlp.router.num_experts_per_tok = 4
    return mock


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    mock = MagicMock()
    mock.encode.return_value = [1, 2, 3, 4, 5]
    mock.decode.return_value = "decoded text"
    mock.chat_template = None
    return mock
```

### 3.3 Example Test File

```python
# tests/cli/commands/introspect/moe_expert/handlers/test_chat.py

import pytest
from argparse import Namespace
from unittest.mock import AsyncMock, patch

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.chat import (
    handle_chat,
    _async_chat,
)
from chuk_lazarus.introspection.moe.models import ExpertChatResult, GenerationStats


class TestHandleChat:
    """Tests for chat handler."""

    @pytest.fixture
    def chat_args(self) -> Namespace:
        """Standard args for chat command."""
        return Namespace(
            model="test-model",
            expert=6,
            prompt="127 * 89 = ",
            max_tokens=100,
            temperature=0.0,
            raw=False,
            verbose=False,
        )

    def test_chat_basic(self, chat_args, mock_expert_router, capsys):
        """Test basic chat with expert."""
        handle_chat(chat_args)

        captured = capsys.readouterr()
        assert "CHAT WITH EXPERT 6" in captured.out
        assert "Test output" in captured.out

    def test_chat_missing_expert(self, chat_args, capsys):
        """Test error when expert not specified."""
        chat_args.expert = None

        handle_chat(chat_args)

        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "--expert" in captured.out

    def test_chat_missing_prompt(self, chat_args, capsys):
        """Test error when prompt not specified."""
        chat_args.prompt = None

        handle_chat(chat_args)

        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "--prompt" in captured.out

    def test_chat_verbose_output(self, chat_args, mock_expert_router, capsys):
        """Test verbose output includes stats."""
        chat_args.verbose = True

        handle_chat(chat_args)

        captured = capsys.readouterr()
        assert "Statistics:" in captured.out
        assert "Tokens generated:" in captured.out

    def test_chat_raw_mode(self, chat_args, mock_expert_router):
        """Test raw mode skips chat template."""
        chat_args.raw = True

        handle_chat(chat_args)

        # Verify config was created with apply_chat_template=False
        mock_router = mock_expert_router.from_pretrained.return_value
        call_kwargs = mock_router.chat_with_expert.call_args.kwargs
        assert call_kwargs["config"].apply_chat_template is False

    def test_chat_custom_temperature(self, chat_args, mock_expert_router):
        """Test custom temperature setting."""
        chat_args.temperature = 0.7

        handle_chat(chat_args)

        mock_router = mock_expert_router.from_pretrained.return_value
        call_kwargs = mock_router.chat_with_expert.call_args.kwargs
        assert call_kwargs["config"].temperature == 0.7

    @pytest.mark.asyncio
    async def test_async_chat_directly(self, chat_args, mock_expert_router):
        """Test async implementation directly."""
        await _async_chat(chat_args)

        mock_expert_router.from_pretrained.assert_called_once_with("test-model")


class TestChatWithExpertIntegration:
    """Integration tests for chat functionality."""

    @pytest.fixture
    def mock_full_router(self, mock_moe_model_info):
        """Full mock of ExpertRouter with all methods."""
        with patch("chuk_lazarus.introspection.moe.ExpertRouter") as mock_cls:
            mock_router = AsyncMock()
            mock_router.info = mock_moe_model_info

            # Mock chat result
            result = ExpertChatResult(
                prompt="127 * 89 = ",
                response="11303",
                expert_idx=6,
                stats=GenerationStats(
                    expert_idx=6,
                    tokens_generated=5,
                    layers_modified=8,
                    moe_type="gpt_oss",
                    prompt_tokens=8,
                ),
            )
            mock_router.chat_with_expert = AsyncMock(return_value=result)
            mock_router.__aenter__ = AsyncMock(return_value=mock_router)
            mock_router.__aexit__ = AsyncMock(return_value=None)
            mock_cls.from_pretrained = AsyncMock(return_value=mock_router)
            yield mock_cls

    def test_chat_formats_correctly(self, mock_full_router, capsys):
        """Test output is formatted correctly."""
        args = Namespace(
            model="test-model",
            expert=6,
            prompt="127 * 89 = ",
            max_tokens=100,
            temperature=0.0,
            raw=False,
            verbose=True,
        )

        handle_chat(args)

        captured = capsys.readouterr()
        assert "CHAT WITH EXPERT 6" in captured.out
        assert "127 * 89 =" in captured.out
        assert "11303" in captured.out
        assert "Tokens generated: 5" in captured.out
```

---

## Phase 4: Migration Strategy

### 4.1 Step-by-Step Migration

1. **Create framework layer first** (no breaking changes)
   - Add new models to `introspection/moe/models.py`
   - Add `MoEAction` enum to `introspection/moe/enums.py`
   - Create `introspection/moe/expert_router.py` with async API
   - Write tests for all new framework code

2. **Create CLI submodule structure**
   - Create `cli/commands/introspect/moe_expert/` directory
   - Add `enums.py`, `dispatcher.py`, `formatters.py`
   - Create `handlers/` directory with handler files

3. **Migrate handlers one at a time**
   - Start with simplest: `chat`, `compare`
   - Move to complex: `full-taxonomy`, `layer-sweep`
   - Each handler:
     - Extract to own file
     - Convert to async pattern
     - Add tests before moving on

4. **Update main.py registration**
   - Change import from `moe_expert.introspect_moe_expert` to new dispatcher
   - Update action choices to use enum values

5. **Deprecate old file**
   - Keep `moe_expert.py` as thin wrapper during transition
   - Add deprecation warning
   - Remove after all handlers migrated

### 4.2 Coverage Requirements

Each file MUST have:
- Corresponding test file in mirrored location
- 90%+ line coverage
- Tests for:
  - Happy path
  - Error conditions
  - Edge cases
  - All public functions/methods

Coverage verification:
```bash
pytest tests/introspection/moe/ --cov=src/chuk_lazarus/introspection/moe --cov-report=term-missing --cov-fail-under=90

pytest tests/cli/commands/introspect/moe_expert/ --cov=src/chuk_lazarus/cli/commands/introspect/moe_expert --cov-report=term-missing --cov-fail-under=90
```

---

## Phase 5: Other CLI Files

Apply same patterns to other large CLI files:

| File | Lines | Target Lines | Strategy |
|------|-------|--------------|----------|
| `circuit.py` | 1,276 | ~300 | Extract to `introspection/circuit/` |
| `neurons.py` | 981 | ~250 | Extract to `introspection/neurons/` |
| `analyze.py` | 877 | ~200 | Already good pattern, minor cleanup |
| `memory.py` | 807 | ~200 | Extract to `introspection/memory/` |
| `probing.py` | 614 | ~150 | Extract to `introspection/probing/` |

---

## Implementation Checklist

### Phase 0: Externalize Data
- [ ] Create `introspection/datasets/` directory structure
- [ ] Create `benchmarks/arithmetic.json` with all math problems
- [ ] Create `moe/pattern_discovery.json` with pattern test prompts
- [ ] Create `moe/context_tests.json` with context independence tests
- [ ] Create `probing/uncertainty.json` with calibration prompts
- [ ] Create Pydantic models in `datasets/models.py`
- [ ] Create `DatasetLoader` class and convenience functions
- [ ] Update `moe_expert.py` to use dataset loaders (7 locations)
- [ ] Update `probing.py` to use dataset loaders
- [ ] Update `virtual_expert.py` to use dataset loaders
- [ ] Add tests for all dataset models and loaders (90%+ coverage)
- [ ] Create JSON schemas for validation

### Phase 1: Foundation
- [ ] Expand `introspection/moe/enums.py` with `MoEAction`
- [ ] Add all Pydantic models to `introspection/moe/models.py`
- [ ] Create `introspection/moe/expert_router.py` with async API
- [ ] Create `introspection/moe/generation.py`
- [ ] Create `introspection/moe/analysis.py`
- [ ] Create `introspection/moe/output.py` for formatters
- [ ] Write tests for all new framework code (90%+ coverage)

### Phase 2: CLI Refactoring
- [ ] Create `cli/commands/introspect/moe_expert/` submodule
- [ ] Create `enums.py` with `MoEAction`
- [ ] Create `dispatcher.py` with dispatch table
- [ ] Create `formatters.py`
- [ ] Create handler files (21 total):
  - [ ] `handlers/chat.py`
  - [ ] `handlers/compare.py`
  - [ ] `handlers/ablate.py`
  - [ ] `handlers/topk.py`
  - [ ] `handlers/collaboration.py`
  - [ ] `handlers/pairs.py`
  - [ ] `handlers/interactive.py`
  - [ ] `handlers/weights.py`
  - [ ] `handlers/tokenizer.py`
  - [ ] `handlers/control_tokens.py`
  - [ ] `handlers/trace.py`
  - [ ] `handlers/entropy.py`
  - [ ] `handlers/divergence.py`
  - [ ] `handlers/role.py`
  - [ ] `handlers/context_test.py`
  - [ ] `handlers/vocab_map.py`
  - [ ] `handlers/router_probe.py`
  - [ ] `handlers/pattern_discovery.py`
  - [ ] `handlers/taxonomy.py`
  - [ ] `handlers/layer_sweep.py`
  - [ ] `handlers/analyze.py`

### Phase 3: Tests
- [ ] Create test fixtures in `tests/introspection/moe/conftest.py`
- [ ] Create test fixtures in `tests/cli/commands/introspect/moe_expert/conftest.py`
- [ ] Write tests for each framework module (90%+ coverage each)
- [ ] Write tests for each CLI handler (90%+ coverage each)
- [ ] Write tests for dispatcher
- [ ] Write tests for formatters

### Phase 4: Migration
- [ ] Update `main.py` to use new dispatcher
- [ ] Add deprecation warning to old `moe_expert.py`
- [ ] Remove old `moe_expert.py` after verification
- [ ] Update documentation

### Phase 5: Other Files
- [ ] Audit and refactor `circuit.py`
- [ ] Audit and refactor `neurons.py`
- [ ] Audit and refactor `memory.py`
- [ ] Audit and refactor `probing.py`
- [ ] Ensure 90%+ coverage for all CLI files

---

## Success Metrics

1. **File Size**: No CLI file > 500 lines
2. **Coverage**: All files have 90%+ test coverage
3. **Type Safety**: No `dict` return types, all Pydantic models
4. **Async**: All I/O operations use async/await
5. **No Magic Strings**: All categorical values use enums
6. **Separation**: CLI files only do parsing/formatting, logic in framework
7. **Tests Mirror Structure**: Every `src/x/y.py` has `tests/x/test_y.py`
