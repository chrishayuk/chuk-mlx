"""Expert identification and specialization analysis.

Provides tools for identifying what each expert in an MoE model
specializes in, based on:
- Token activation patterns
- Category-specific routing
- Semantic clustering
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from pydantic import BaseModel, ConfigDict, Field

from .config import MoECaptureConfig
from .datasets import PromptCategory, get_category_prompts
from .enums import ExpertCategory, ExpertRole
from .models import ExpertIdentity

if TYPE_CHECKING:
    from .hooks import MoEHooks


class CategoryActivation(BaseModel):
    """Expert activation for a prompt category."""

    model_config = ConfigDict(frozen=True)

    category: PromptCategory
    expert_idx: int = Field(ge=0)
    layer_idx: int = Field(ge=0)
    activation_count: int = Field(ge=0)
    activation_rate: float = Field(ge=0, le=1)
    avg_weight: float = Field(ge=0, le=1)


class ExpertProfile(BaseModel):
    """Complete profile of an expert's behavior."""

    model_config = ConfigDict(frozen=True)

    expert_idx: int = Field(ge=0)
    layer_idx: int = Field(ge=0)
    total_activations: int = Field(ge=0)
    category_breakdown: tuple[CategoryActivation, ...] = Field(default_factory=tuple)
    primary_category: ExpertCategory
    role: ExpertRole
    confidence: float = Field(ge=0, le=1)


def identify_expert(
    hooks: MoEHooks,
    layer_idx: int,
    expert_idx: int,
    tokenizer: Any,
    prompts_per_category: int = 5,
) -> ExpertIdentity:
    """
    Identify what an expert specializes in.

    Args:
        hooks: MoEHooks with model reference
        layer_idx: Layer containing the expert
        expert_idx: Expert to identify
        tokenizer: Tokenizer for encoding prompts
        prompts_per_category: Number of prompts to test per category

    Returns:
        ExpertIdentity with specialization info
    """
    category_counts: dict[PromptCategory, int] = defaultdict(int)
    category_totals: dict[PromptCategory, int] = defaultdict(int)
    total_activations = 0

    for category in PromptCategory:
        prompts = get_category_prompts(category)
        if not prompts:
            continue

        for prompt in prompts.prompts[:prompts_per_category]:
            input_ids = mx.array(tokenizer.encode(prompt))[None, :]

            hooks.configure(
                MoECaptureConfig(
                    layers=[layer_idx],
                    capture_selected_experts=True,
                )
            )
            hooks.forward(input_ids)

            selected = hooks.moe_state.selected_experts.get(layer_idx)
            if selected is not None:
                flat = selected.reshape(-1).tolist()
                count = flat.count(expert_idx)
                category_counts[category] += count
                category_totals[category] += len(flat)
                total_activations += count

    # Determine primary category
    if not category_counts:
        return ExpertIdentity(
            expert_idx=expert_idx,
            layer_idx=layer_idx,
            primary_category=ExpertCategory.UNKNOWN,
            role=ExpertRole.RARE,
            confidence=0.0,
            activation_rate=0.0,
        )

    # Map PromptCategory to ExpertCategory
    category_mapping = {
        PromptCategory.PYTHON: ExpertCategory.CODE,
        PromptCategory.JAVASCRIPT: ExpertCategory.CODE,
        PromptCategory.RUST: ExpertCategory.CODE,
        PromptCategory.SQL: ExpertCategory.CODE,
        PromptCategory.GO: ExpertCategory.CODE,
        PromptCategory.TYPESCRIPT: ExpertCategory.CODE,
        PromptCategory.ARITHMETIC: ExpertCategory.MATH,
        PromptCategory.ALGEBRA: ExpertCategory.MATH,
        PromptCategory.CALCULUS: ExpertCategory.MATH,
        PromptCategory.STATISTICS: ExpertCategory.MATH,
        PromptCategory.GEOMETRY: ExpertCategory.MATH,
        PromptCategory.LOGIC: ExpertCategory.REASONING,
        PromptCategory.SCIENCE: ExpertCategory.KNOWLEDGE,
        PromptCategory.HISTORY: ExpertCategory.KNOWLEDGE,
        PromptCategory.GEOGRAPHY: ExpertCategory.KNOWLEDGE,
        PromptCategory.CULTURE: ExpertCategory.KNOWLEDGE,
        PromptCategory.TECH: ExpertCategory.KNOWLEDGE,
        PromptCategory.LISTS: ExpertCategory.STRUCTURE,
        PromptCategory.TABLES: ExpertCategory.STRUCTURE,
        PromptCategory.JSON: ExpertCategory.STRUCTURE,
        PromptCategory.MARKDOWN: ExpertCategory.STRUCTURE,
        PromptCategory.STORY: ExpertCategory.LANGUAGE,
        PromptCategory.POETRY: ExpertCategory.LANGUAGE,
        PromptCategory.DIALOGUE: ExpertCategory.LANGUAGE,
        PromptCategory.DESCRIPTION: ExpertCategory.LANGUAGE,
        PromptCategory.ANALYSIS: ExpertCategory.REASONING,
        PromptCategory.COMPARISON: ExpertCategory.REASONING,
        PromptCategory.CAUSATION: ExpertCategory.REASONING,
        PromptCategory.PLANNING: ExpertCategory.REASONING,
    }

    # Aggregate by ExpertCategory
    expert_category_scores: dict[ExpertCategory, float] = defaultdict(float)
    for prompt_cat, count in category_counts.items():
        expert_cat = category_mapping.get(prompt_cat, ExpertCategory.UNKNOWN)
        total = category_totals[prompt_cat]
        if total > 0:
            expert_category_scores[expert_cat] += count / total

    # Find primary category
    if not expert_category_scores:
        primary = ExpertCategory.UNKNOWN
        confidence = 0.0
    else:
        primary = max(expert_category_scores, key=expert_category_scores.get)
        total_score = sum(expert_category_scores.values())
        confidence = expert_category_scores[primary] / total_score if total_score > 0 else 0.0

    # Determine role
    total_possible = sum(category_totals.values())
    activation_rate = total_activations / total_possible if total_possible > 0 else 0.0

    if activation_rate < 0.01:
        role = ExpertRole.RARE
    elif confidence > 0.7:
        role = ExpertRole.SPECIALIST
    elif len([c for c, s in expert_category_scores.items() if s > 0.1]) >= 3:
        role = ExpertRole.GENERALIST
    else:
        role = ExpertRole.GENERALIST

    # Get secondary categories
    sorted_cats = sorted(
        expert_category_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    secondary = tuple(c for c, _ in sorted_cats[1:4] if c != primary)

    return ExpertIdentity(
        expert_idx=expert_idx,
        layer_idx=layer_idx,
        primary_category=primary,
        secondary_categories=secondary,
        role=role,
        confidence=confidence,
        activation_rate=activation_rate,
    )


def identify_all_experts(
    hooks: MoEHooks,
    layer_idx: int,
    tokenizer: Any,
    prompts_per_category: int = 3,
) -> list[ExpertIdentity]:
    """
    Identify all experts in a layer.

    Args:
        hooks: MoEHooks with model reference
        layer_idx: Layer to analyze
        tokenizer: Tokenizer
        prompts_per_category: Prompts per category

    Returns:
        List of ExpertIdentity for all experts
    """
    info = hooks.get_layer_info(layer_idx)
    if info is None:
        return []

    identities = []
    for expert_idx in range(info.num_experts):
        identity = identify_expert(hooks, layer_idx, expert_idx, tokenizer, prompts_per_category)
        identities.append(identity)

    return identities


def find_specialists(
    identities: list[ExpertIdentity],
    category: ExpertCategory | None = None,
) -> list[ExpertIdentity]:
    """
    Find specialist experts.

    Args:
        identities: List of expert identities
        category: Optional category to filter by

    Returns:
        List of specialist experts
    """
    specialists = [i for i in identities if i.role == ExpertRole.SPECIALIST]

    if category is not None:
        specialists = [i for i in specialists if i.primary_category == category]

    return sorted(specialists, key=lambda x: x.confidence, reverse=True)


def find_generalists(
    identities: list[ExpertIdentity],
) -> list[ExpertIdentity]:
    """
    Find generalist experts.

    Args:
        identities: List of expert identities

    Returns:
        List of generalist experts
    """
    return [i for i in identities if i.role == ExpertRole.GENERALIST]


def cluster_experts_by_specialization(
    identities: list[ExpertIdentity],
) -> dict[ExpertCategory, list[ExpertIdentity]]:
    """
    Cluster experts by their primary specialization.

    Args:
        identities: List of expert identities

    Returns:
        Dict mapping category -> list of experts
    """
    clusters: dict[ExpertCategory, list[ExpertIdentity]] = defaultdict(list)

    for identity in identities:
        clusters[identity.primary_category].append(identity)

    # Sort within each cluster by confidence
    for cat in clusters:
        clusters[cat].sort(key=lambda x: x.confidence, reverse=True)

    return dict(clusters)


def print_expert_summary(identities: list[ExpertIdentity]) -> None:
    """Print a summary of expert identities."""
    if not identities:
        print("No experts identified")
        return

    layer = identities[0].layer_idx
    print(f"\nExpert Identity Summary (Layer {layer})")
    print("=" * 60)

    # Group by role
    by_role: dict[ExpertRole, list[ExpertIdentity]] = defaultdict(list)
    for i in identities:
        by_role[i.role].append(i)

    for role in [ExpertRole.SPECIALIST, ExpertRole.GENERALIST, ExpertRole.RARE]:
        experts = by_role.get(role, [])
        if not experts:
            continue

        print(f"\n{role.value.upper()}S ({len(experts)}):")
        for e in sorted(experts, key=lambda x: x.confidence, reverse=True):
            secondary = ", ".join(c.value for c in e.secondary_categories[:2])
            print(
                f"  Expert {e.expert_idx:2d}: {e.primary_category.value:12s} "
                f"(conf={e.confidence:.2f}, rate={e.activation_rate:.3f}) "
                f"[{secondary}]"
            )
