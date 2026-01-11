"""Dataset loading utilities for introspection.

Provides cached loading of JSON datasets with Pydantic validation.

Example:
    >>> from chuk_lazarus.introspection.datasets import get_arithmetic_benchmarks
    >>> benchmarks = get_arithmetic_benchmarks()
    >>> for problem in benchmarks.get_by_difficulty("hard"):
    ...     print(f"{problem.prompt} -> {problem.answer}")
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .models import (
    ArithmeticBenchmark,
    ArithmeticProblem,
    ContextTest,
    ContextTestDataset,
    LayerExpectation,
    LayerSweepCategory,
    LayerSweepDataset,
    LayerSweepSubcategory,
    PatternCategory,
    PatternDiscoveryDataset,
    UncertaintyDataset,
    UncertaintyPromptsSection,
)

T = TypeVar("T", bound=BaseModel)

# Base path for datasets
_BASE_PATH = Path(__file__).parent


class DatasetLoader:
    """Load and cache JSON datasets with Pydantic validation."""

    @classmethod
    @lru_cache(maxsize=32)
    def load_json(cls, relative_path: str) -> dict:
        """Load raw JSON data with caching.

        Args:
            relative_path: Path relative to the datasets directory.

        Returns:
            The parsed JSON data as a dictionary.

        Raises:
            FileNotFoundError: If the dataset file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        path = _BASE_PATH / relative_path
        with open(path) as f:
            return json.load(f)

    @classmethod
    def load_model(cls, relative_path: str, model_class: type[T]) -> T:
        """Load JSON and validate with Pydantic model.

        Args:
            relative_path: Path relative to the datasets directory.
            model_class: The Pydantic model class to validate against.

        Returns:
            A validated Pydantic model instance.

        Raises:
            FileNotFoundError: If the dataset file doesn't exist.
            pydantic.ValidationError: If the data doesn't match the model.
        """
        data = cls.load_json(relative_path)
        return model_class.model_validate(data)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the JSON loading cache."""
        cls.load_json.cache_clear()


# =============================================================================
# Convenience Functions
# =============================================================================


def get_arithmetic_benchmarks() -> ArithmeticBenchmark:
    """Load arithmetic benchmark problems.

    Returns:
        ArithmeticBenchmark with problems organized by difficulty (simple, medium, hard).

    Example:
        >>> benchmarks = get_arithmetic_benchmarks()
        >>> hard_problems = benchmarks.get_by_difficulty("hard")
        >>> for p in hard_problems:
        ...     print(f"{p.prompt} = {p.answer}")
    """
    return DatasetLoader.load_model("benchmarks/arithmetic.json", ArithmeticBenchmark)


def get_uncertainty_prompts() -> UncertaintyDataset:
    """Load uncertainty detection calibration prompts.

    Returns:
        UncertaintyDataset with working and broken prompt sets.

    Example:
        >>> dataset = get_uncertainty_prompts()
        >>> print(dataset.working)  # Prompts that should compute
        >>> print(dataset.broken)   # Prompts that may refuse
    """
    return DatasetLoader.load_model("probing/uncertainty.json", UncertaintyDataset)


def get_context_tests() -> ContextTestDataset:
    """Load context independence test prompts.

    Returns:
        ContextTestDataset with test cases for context independence analysis.

    Example:
        >>> tests = get_context_tests()
        >>> for test in tests.tests:
        ...     print(f"{test.prompt} ({test.context_type})")
    """
    return DatasetLoader.load_model("moe/context_tests.json", ContextTestDataset)


def get_pattern_discovery_prompts() -> PatternDiscoveryDataset:
    """Load pattern discovery test prompts.

    Returns:
        PatternDiscoveryDataset with categorized prompts for pattern analysis.

    Example:
        >>> patterns = get_pattern_discovery_prompts()
        >>> for cat_name in patterns.get_category_names():
        ...     cat = patterns.get_category(cat_name)
        ...     print(f"{cat_name}: {len(cat.prompts)} prompts")
    """
    return DatasetLoader.load_model("moe/pattern_discovery.json", PatternDiscoveryDataset)


def get_layer_sweep_tests() -> LayerSweepDataset:
    """Load layer sweep test prompts.

    Returns:
        LayerSweepDataset with categorized prompts for layer sweep analysis.

    Example:
        >>> tests = get_layer_sweep_tests()
        >>> for cat_name in tests.get_category_names():
        ...     cat = tests.get_category(cat_name)
        ...     print(f"{cat_name}: {len(cat.get_all_prompts())} prompts")
    """
    return DatasetLoader.load_model("moe/layer_sweep_tests.json", LayerSweepDataset)


__all__ = [
    # Loader
    "DatasetLoader",
    # Convenience functions
    "get_arithmetic_benchmarks",
    "get_uncertainty_prompts",
    "get_context_tests",
    "get_pattern_discovery_prompts",
    "get_layer_sweep_tests",
    # Models
    "ArithmeticBenchmark",
    "ArithmeticProblem",
    "UncertaintyDataset",
    "UncertaintyPromptsSection",
    "ContextTestDataset",
    "ContextTest",
    "PatternDiscoveryDataset",
    "PatternCategory",
    "LayerSweepDataset",
    "LayerSweepCategory",
    "LayerSweepSubcategory",
    "LayerExpectation",
]
