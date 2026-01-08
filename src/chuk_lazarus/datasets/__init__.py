"""Dataset loading utilities for chuk-lazarus.

This module provides shared datasets used across the framework:
- Facts (multiplication, addition, capitals, elements)
- Calibration prompts for uncertainty detection
- Benchmark problems for evaluation
- Test categories for expert analysis

These datasets are used by CLI commands, training, and evaluation -
they are NOT specific to introspection.

Example:
    >>> from chuk_lazarus.datasets import load_facts, FactType
    >>> facts = load_facts(FactType.MULTIPLICATION)
    >>> for fact in facts[:5]:
    ...     print(f"{fact['query']} -> {fact['answer']}")
"""

from __future__ import annotations

from .facts import (
    FactType,
    load_facts,
    load_multiplication_facts,
    load_addition_facts,
    load_capital_facts,
    load_element_facts,
)
from .calibration import (
    CalibrationPrompts,
    load_calibration_prompts,
)
from .benchmarks import (
    load_expert_benchmark,
    load_expert_test_categories,
)

__all__ = [
    # Fact loading
    "FactType",
    "load_facts",
    "load_multiplication_facts",
    "load_addition_facts",
    "load_capital_facts",
    "load_element_facts",
    # Calibration
    "CalibrationPrompts",
    "load_calibration_prompts",
    # Benchmarks
    "load_expert_benchmark",
    "load_expert_test_categories",
]
