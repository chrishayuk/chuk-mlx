"""MoE analysis datasets.

Provides categorized prompts and token categories for expert analysis.
Data is loaded from JSON files for easy customization.
"""

from .prompts import (
    CATEGORY_GROUPS,
    CategoryPrompts,
    PromptCategory,
    PromptCategoryGroup,
    PromptDataset,
    get_all_prompts,
    get_category_prompts,
    get_grouped_prompts,
    get_prompts_by_group,
    get_prompts_flat,
    get_prompts_for_group_flat,
)

__all__ = [
    # Enums
    "PromptCategory",
    "PromptCategoryGroup",
    # Models
    "CategoryPrompts",
    "PromptDataset",
    # Constants
    "CATEGORY_GROUPS",
    # Functions
    "get_category_prompts",
    "get_all_prompts",
    "get_grouped_prompts",
    "get_prompts_by_group",
    "get_prompts_flat",
    "get_prompts_for_group_flat",
]
