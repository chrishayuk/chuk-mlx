"""Expert compression and merging utilities.

Provides tools for analyzing which experts can be merged or pruned
to reduce model size while preserving quality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from .models import CompressionPlan

if TYPE_CHECKING:
    from .hooks import MoEHooks


class ExpertSimilarity(BaseModel):
    """Similarity between two experts."""

    model_config = ConfigDict(frozen=True)

    expert_a: int = Field(ge=0)
    expert_b: int = Field(ge=0)
    layer_idx: int = Field(ge=0)
    weight_cosine_similarity: float = Field(ge=-1, le=1)
    activation_overlap: float = Field(ge=0, le=1)
    merge_candidate: bool = False


class CompressionAnalysis(BaseModel):
    """Analysis of compression opportunities."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    num_experts: int = Field(ge=1)
    merge_candidates: tuple[tuple[int, int], ...] = Field(default_factory=tuple)
    prune_candidates: tuple[int, ...] = Field(default_factory=tuple)
    estimated_size_reduction: float = Field(ge=0, le=1)
    estimated_quality_loss: float = Field(ge=0)


def compute_expert_similarity(
    model: nn.Module,
    layer_idx: int,
    expert_a: int,
    expert_b: int,
) -> ExpertSimilarity:
    """
    Compute similarity between two experts.

    Args:
        model: The model
        layer_idx: Layer containing experts
        expert_a: First expert index
        expert_b: Second expert index

    Returns:
        ExpertSimilarity with metrics
    """
    layers = _get_model_layers(model)
    if layer_idx >= len(layers):
        raise ValueError(f"Layer {layer_idx} out of range")

    layer = layers[layer_idx]
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise ValueError(f"Layer {layer_idx} has no MLP")

    experts = getattr(mlp, "experts", None)
    if experts is None or not isinstance(experts, list):
        raise ValueError(f"Layer {layer_idx} has no experts list")

    if expert_a >= len(experts) or expert_b >= len(experts):
        raise ValueError("Expert index out of range")

    # Get weight matrices
    exp_a = experts[expert_a]
    exp_b = experts[expert_b]

    # Compute cosine similarity of down projection weights
    if hasattr(exp_a, "down_proj") and hasattr(exp_b, "down_proj"):
        w_a = exp_a.down_proj.weight.reshape(-1)
        w_b = exp_b.down_proj.weight.reshape(-1)

        dot = mx.sum(w_a * w_b)
        norm_a = mx.linalg.norm(w_a)
        norm_b = mx.linalg.norm(w_b)
        cosine_sim = float(dot / (norm_a * norm_b + 1e-10))
    else:
        cosine_sim = 0.0

    return ExpertSimilarity(
        expert_a=expert_a,
        expert_b=expert_b,
        layer_idx=layer_idx,
        weight_cosine_similarity=cosine_sim,
        activation_overlap=0.0,  # Requires activation data
        merge_candidate=cosine_sim > 0.8,
    )


def compute_similarity_matrix(
    model: nn.Module,
    layer_idx: int,
) -> list[ExpertSimilarity]:
    """
    Compute pairwise similarity between all experts.

    Args:
        model: The model
        layer_idx: Layer to analyze

    Returns:
        List of ExpertSimilarity for all pairs
    """
    layers = _get_model_layers(model)
    if layer_idx >= len(layers):
        return []

    layer = layers[layer_idx]
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return []

    experts = getattr(mlp, "experts", None)
    if experts is None or not isinstance(experts, list):
        return []

    similarities = []
    for i in range(len(experts)):
        for j in range(i + 1, len(experts)):
            sim = compute_expert_similarity(model, layer_idx, i, j)
            similarities.append(sim)

    return similarities


def find_merge_candidates(
    similarities: list[ExpertSimilarity],
    threshold: float = 0.8,
) -> list[tuple[int, int]]:
    """
    Find expert pairs that are good merge candidates.

    Args:
        similarities: List of ExpertSimilarity
        threshold: Cosine similarity threshold

    Returns:
        List of (expert_a, expert_b) tuples
    """
    candidates = []
    for sim in similarities:
        if sim.weight_cosine_similarity >= threshold:
            candidates.append((sim.expert_a, sim.expert_b))

    return sorted(candidates, key=lambda x: x[0])


def find_prune_candidates(
    hooks: MoEHooks,
    layer_idx: int,
    threshold: float = 0.01,
) -> list[int]:
    """
    Find experts that rarely activate and can be pruned.

    Args:
        hooks: MoEHooks with captured state
        layer_idx: Layer to analyze
        threshold: Activation rate threshold

    Returns:
        List of expert indices that rarely activate
    """
    utilization = hooks.get_expert_utilization(layer_idx)
    if utilization is None:
        return []

    prune_candidates = []
    for idx, freq in enumerate(utilization.expert_frequencies):
        if freq < threshold:
            prune_candidates.append(idx)

    return prune_candidates


def create_compression_plan(
    hooks: MoEHooks,
    layer_idx: int,
    target_experts: int | None = None,
    merge_threshold: float = 0.8,
    prune_threshold: float = 0.01,
) -> CompressionPlan:
    """
    Create a plan for compressing experts in a layer.

    Args:
        hooks: MoEHooks with model reference
        layer_idx: Layer to compress
        target_experts: Target number of experts (None = auto)
        merge_threshold: Similarity threshold for merging
        prune_threshold: Activation threshold for pruning

    Returns:
        CompressionPlan with merge groups and estimates
    """
    info = hooks.get_layer_info(layer_idx)
    if info is None:
        return CompressionPlan(
            source_num_experts=0,
            target_num_experts=0,
            merge_groups=(),
            estimated_quality_loss=1.0,
            estimated_size_reduction=0.0,
        )

    num_experts = info.num_experts

    # Find similar experts
    similarities = compute_similarity_matrix(hooks.model, layer_idx)
    merge_pairs = find_merge_candidates(similarities, merge_threshold)

    # Find prunable experts
    prune_list = find_prune_candidates(hooks, layer_idx, prune_threshold)

    # Build merge groups (greedy clustering)
    merged: set[int] = set()
    merge_groups: list[tuple[int, ...]] = []

    for a, b in merge_pairs:
        if a not in merged and b not in merged:
            # Check if we can extend an existing group
            extended = False
            for i, group in enumerate(merge_groups):
                if a in group or b in group:
                    merge_groups[i] = tuple(set(group) | {a, b})
                    merged.add(a)
                    merged.add(b)
                    extended = True
                    break

            if not extended:
                merge_groups.append((a, b))
                merged.add(a)
                merged.add(b)

    # Add unmerged experts as singleton groups
    for exp_idx in range(num_experts):
        if exp_idx not in merged and exp_idx not in prune_list:
            merge_groups.append((exp_idx,))

    # Calculate target
    if target_experts is None:
        target_experts = len(merge_groups)

    # Estimate quality loss (heuristic)
    total_merged = sum(len(g) - 1 for g in merge_groups if len(g) > 1)
    total_pruned = len(prune_list)
    quality_loss = (total_merged * 0.05 + total_pruned * 0.1) / num_experts

    # Estimate size reduction
    experts_removed = num_experts - len(merge_groups)
    size_reduction = experts_removed / num_experts

    return CompressionPlan(
        source_num_experts=num_experts,
        target_num_experts=len(merge_groups),
        merge_groups=tuple(merge_groups),
        estimated_quality_loss=min(1.0, quality_loss),
        estimated_size_reduction=size_reduction,
    )


def analyze_compression_opportunities(
    hooks: MoEHooks,
    merge_threshold: float = 0.8,
    prune_threshold: float = 0.01,
) -> list[CompressionAnalysis]:
    """
    Analyze compression opportunities across all MoE layers.

    Args:
        hooks: MoEHooks with model reference
        merge_threshold: Similarity threshold for merging
        prune_threshold: Activation threshold for pruning

    Returns:
        List of CompressionAnalysis, one per layer
    """
    analyses = []

    for layer_idx in hooks.moe_layers:
        info = hooks.get_layer_info(layer_idx)
        if info is None:
            continue

        plan = create_compression_plan(hooks, layer_idx, None, merge_threshold, prune_threshold)

        merge_pairs = [(g[0], g[1]) for g in plan.merge_groups if len(g) >= 2]
        prune_list = find_prune_candidates(hooks, layer_idx, prune_threshold)

        analyses.append(
            CompressionAnalysis(
                layer_idx=layer_idx,
                num_experts=info.num_experts,
                merge_candidates=tuple(merge_pairs),
                prune_candidates=tuple(prune_list),
                estimated_size_reduction=plan.estimated_size_reduction,
                estimated_quality_loss=plan.estimated_quality_loss,
            )
        )

    return analyses


def print_compression_summary(analyses: list[CompressionAnalysis]) -> None:
    """Print compression analysis summary."""
    if not analyses:
        print("No compression analysis available")
        return

    print("\nCompression Opportunity Summary")
    print("=" * 60)

    total_merge = 0
    total_prune = 0

    for a in analyses:
        merge_count = len(a.merge_candidates)
        prune_count = len(a.prune_candidates)
        total_merge += merge_count
        total_prune += prune_count

        print(
            f"Layer {a.layer_idx:2d}: {a.num_experts} experts | "
            f"merge={merge_count} prune={prune_count} | "
            f"size-{a.estimated_size_reduction:.0%} quality-{a.estimated_quality_loss:.1%}"
        )

    print("-" * 60)
    print(f"Total: {total_merge} merge candidates, {total_prune} prune candidates")


def _get_model_layers(model: nn.Module) -> list[nn.Module]:
    """Get transformer layers from model."""
    for attr in ["model", "transformer", "decoder"]:
        submodel = getattr(model, attr, None)
        if submodel is not None:
            layers = getattr(submodel, "layers", None)
            if layers is not None:
                return list(layers)
    return list(getattr(model, "layers", []))
