"""
Prompt datasets for MoE expert analysis.

Loads categorized prompts from JSON for analyzing expert specialization patterns.
Categories are designed to reveal:
- Domain-level specialization (math vs code vs facts)
- Language-level specialization (Python vs Rust)
- Token-level specialization (punctuation, proper nouns)

Based on ST-MoE paper findings that experts often specialize by token type
rather than semantic domain.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field


class PromptCategory(str, Enum):
    """Categories for prompt classification."""

    # Code by language
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    RUST = "rust"
    SQL = "sql"
    GO = "go"
    TYPESCRIPT = "typescript"

    # Math by type
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    STATISTICS = "statistics"
    CALCULUS = "calculus"

    # Facts by domain
    GEOGRAPHY = "geography"
    HISTORY = "history"
    SCIENCE = "science"
    POP_CULTURE = "pop_culture"
    TECHNOLOGY = "technology"

    # Language structure
    PUNCTUATION = "punctuation"
    PROPER_NOUNS = "proper_nouns"
    PRONOUNS = "pronouns"
    PREPOSITIONS = "prepositions"
    ARTICLES = "articles"
    CONJUNCTIONS = "conjunctions"

    # Creative
    POETRY = "poetry"
    STORYTELLING = "storytelling"
    DIALOGUE = "dialogue"

    # Reasoning
    LOGIC = "logic"
    ANALOGIES = "analogies"
    CAUSATION = "causation"


class PromptCategoryGroup(str, Enum):
    """Higher-level groupings of categories."""

    CODE = "code"
    MATH = "math"
    FACTS = "facts"
    STRUCTURE = "structure"
    CREATIVE = "creative"
    REASONING = "reasoning"


# Mapping of groups to their categories
CATEGORY_GROUPS: dict[PromptCategoryGroup, list[PromptCategory]] = {
    PromptCategoryGroup.CODE: [
        PromptCategory.PYTHON,
        PromptCategory.JAVASCRIPT,
        PromptCategory.RUST,
        PromptCategory.SQL,
        PromptCategory.GO,
        PromptCategory.TYPESCRIPT,
    ],
    PromptCategoryGroup.MATH: [
        PromptCategory.ARITHMETIC,
        PromptCategory.ALGEBRA,
        PromptCategory.STATISTICS,
        PromptCategory.CALCULUS,
    ],
    PromptCategoryGroup.FACTS: [
        PromptCategory.GEOGRAPHY,
        PromptCategory.HISTORY,
        PromptCategory.SCIENCE,
        PromptCategory.POP_CULTURE,
        PromptCategory.TECHNOLOGY,
    ],
    PromptCategoryGroup.STRUCTURE: [
        PromptCategory.PUNCTUATION,
        PromptCategory.PROPER_NOUNS,
        PromptCategory.PRONOUNS,
        PromptCategory.PREPOSITIONS,
        PromptCategory.ARTICLES,
        PromptCategory.CONJUNCTIONS,
    ],
    PromptCategoryGroup.CREATIVE: [
        PromptCategory.POETRY,
        PromptCategory.STORYTELLING,
        PromptCategory.DIALOGUE,
    ],
    PromptCategoryGroup.REASONING: [
        PromptCategory.LOGIC,
        PromptCategory.ANALOGIES,
        PromptCategory.CAUSATION,
    ],
}


class CategoryPrompts(BaseModel):
    """Prompts for a specific category."""

    model_config = ConfigDict(frozen=True)

    category: PromptCategory
    group: PromptCategoryGroup
    prompts: tuple[str, ...] = Field(default_factory=tuple)
    description: str = ""


class PromptDataset(BaseModel):
    """Full prompt dataset loaded from JSON."""

    model_config = ConfigDict(frozen=True)

    version: str
    description: str
    categories: dict[PromptCategoryGroup, dict[PromptCategory, tuple[str, ...]]]


# Cache for loaded data
_cached_data: dict | None = None


def _get_dataset_path() -> Path:
    """Get path to prompts.json."""
    return Path(__file__).parent / "prompts.json"


def _load_data() -> dict:
    """Load and cache prompt data from JSON."""
    global _cached_data
    if _cached_data is None:
        with open(_get_dataset_path()) as f:
            _cached_data = json.load(f)
    return _cached_data


def _get_group_for_category(category: PromptCategory) -> PromptCategoryGroup:
    """Get the group a category belongs to."""
    for group, categories in CATEGORY_GROUPS.items():
        if category in categories:
            return group
    raise ValueError(f"Category {category} not found in any group")


def get_category_prompts(category: PromptCategory) -> CategoryPrompts:
    """Get prompts for a specific category."""
    data = _load_data()
    group = _get_group_for_category(category)

    # Navigate JSON structure: categories -> group -> category
    group_data = data.get("categories", {}).get(group.value, {})
    prompts = group_data.get(category.value, [])

    return CategoryPrompts(
        category=category,
        group=group,
        prompts=tuple(prompts),
    )


def get_all_prompts() -> dict[PromptCategory, CategoryPrompts]:
    """Get all category prompts."""
    return {cat: get_category_prompts(cat) for cat in PromptCategory}


def get_prompts_by_group(group: PromptCategoryGroup) -> list[CategoryPrompts]:
    """Get all prompts for a category group."""
    categories = CATEGORY_GROUPS.get(group, [])
    return [get_category_prompts(cat) for cat in categories]


def get_prompts_flat(
    categories: Sequence[PromptCategory] | None = None,
) -> list[tuple[PromptCategory, str]]:
    """Get flattened list of (category, prompt) tuples."""
    if categories is None:
        categories = list(PromptCategory)

    result = []
    for cat in categories:
        cat_prompts = get_category_prompts(cat)
        for prompt in cat_prompts.prompts:
            result.append((cat, prompt))
    return result


def get_prompts_for_group_flat(group: PromptCategoryGroup) -> list[str]:
    """Get all prompts for a group as a flat list."""
    prompts = []
    for cat_prompts in get_prompts_by_group(group):
        prompts.extend(cat_prompts.prompts)
    return prompts


def get_grouped_prompts() -> dict[str, list[str]]:
    """Get prompts organized by category name (for CLI use).

    Returns:
        Dict mapping category name (uppercase) -> list of prompts
    """
    result = {}
    for cat in PromptCategory:
        cat_prompts = get_category_prompts(cat)
        if cat_prompts.prompts:
            # Use uppercase category name to match CLI expectations
            result[cat.name.upper()] = list(cat_prompts.prompts)
    return result
