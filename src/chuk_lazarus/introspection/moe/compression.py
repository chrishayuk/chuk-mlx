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


class ExpertActivationStats(BaseModel):
    """Activation statistics for an expert across a dataset."""

    model_config = ConfigDict(frozen=True)

    expert_idx: int = Field(ge=0)
    layer_idx: int = Field(ge=0)
    activation_count: int = Field(ge=0)
    token_positions: tuple[int, ...] = Field(default_factory=tuple)
    total_samples: int = Field(ge=0)

    @property
    def activation_rate(self) -> float:
        """Compute activation rate as fraction of total samples."""
        if self.total_samples == 0:
            return 0.0
        return self.activation_count / self.total_samples


class ActivationOverlapResult(BaseModel):
    """Result of computing activation overlap between two experts."""

    model_config = ConfigDict(frozen=True)

    expert_a: int = Field(ge=0)
    expert_b: int = Field(ge=0)
    layer_idx: int = Field(ge=0)
    jaccard_similarity: float = Field(ge=0, le=1)
    overlap_count: int = Field(ge=0, description="Number of samples where both activate")
    union_count: int = Field(ge=0, description="Number of samples where either activates")
    a_only_count: int = Field(ge=0, description="Samples where only A activates")
    b_only_count: int = Field(ge=0, description="Samples where only B activates")


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


# =============================================================================
# Activation Overlap Analysis
# =============================================================================


def collect_expert_activations(
    hooks: MoEHooks,
    prompts: list[str],
    layer_idx: int,
    tokenizer: object,
) -> dict[int, set[int]]:
    """
    Collect which experts activate for each prompt.

    Args:
        hooks: MoEHooks with model reference
        prompts: List of prompts to analyze
        layer_idx: Layer to analyze
        tokenizer: Tokenizer for encoding prompts

    Returns:
        Dict mapping expert_idx -> set of prompt indices where it activated
    """
    info = hooks.get_layer_info(layer_idx)
    if info is None:
        return {}

    num_experts = info.num_experts
    expert_activations: dict[int, set[int]] = {i: set() for i in range(num_experts)}

    for prompt_idx, prompt in enumerate(prompts):
        # Encode prompt
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]

        # Capture router weights for this prompt
        captured = hooks.capture_router_weights(input_ids, [layer_idx])

        if layer_idx in captured:
            for position_data in captured[layer_idx]:
                # position_data contains selected expert indices
                for expert_idx in position_data.get("selected_experts", []):
                    if 0 <= expert_idx < num_experts:
                        expert_activations[expert_idx].add(prompt_idx)

    return expert_activations


def compute_activation_overlap(
    expert_a_activations: set[int],
    expert_b_activations: set[int],
    expert_a: int,
    expert_b: int,
    layer_idx: int,
) -> ActivationOverlapResult:
    """
    Compute Jaccard similarity between two experts' activation patterns.

    Args:
        expert_a_activations: Set of sample indices where expert A activated
        expert_b_activations: Set of sample indices where expert B activated
        expert_a: Index of expert A
        expert_b: Index of expert B
        layer_idx: Layer index

    Returns:
        ActivationOverlapResult with Jaccard similarity and counts
    """
    intersection = expert_a_activations & expert_b_activations
    union = expert_a_activations | expert_b_activations

    overlap_count = len(intersection)
    union_count = len(union)
    a_only = len(expert_a_activations - expert_b_activations)
    b_only = len(expert_b_activations - expert_a_activations)

    jaccard = overlap_count / union_count if union_count > 0 else 0.0

    return ActivationOverlapResult(
        expert_a=expert_a,
        expert_b=expert_b,
        layer_idx=layer_idx,
        jaccard_similarity=jaccard,
        overlap_count=overlap_count,
        union_count=union_count,
        a_only_count=a_only,
        b_only_count=b_only,
    )


def compute_expert_similarity_with_activations(
    model: nn.Module,
    layer_idx: int,
    expert_a: int,
    expert_b: int,
    expert_activations: dict[int, set[int]] | None = None,
) -> ExpertSimilarity:
    """
    Compute similarity between two experts including activation overlap.

    Args:
        model: The model
        layer_idx: Layer containing experts
        expert_a: First expert index
        expert_b: Second expert index
        expert_activations: Optional pre-computed activation sets per expert

    Returns:
        ExpertSimilarity with weight similarity and activation overlap
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

    # Compute weight cosine similarity
    exp_a = experts[expert_a]
    exp_b = experts[expert_b]

    if hasattr(exp_a, "down_proj") and hasattr(exp_b, "down_proj"):
        w_a = exp_a.down_proj.weight.reshape(-1)
        w_b = exp_b.down_proj.weight.reshape(-1)

        dot = mx.sum(w_a * w_b)
        norm_a = mx.linalg.norm(w_a)
        norm_b = mx.linalg.norm(w_b)
        cosine_sim = float(dot / (norm_a * norm_b + 1e-10))
    else:
        cosine_sim = 0.0

    # Compute activation overlap if activations provided
    activation_overlap = 0.0
    if expert_activations is not None:
        a_acts = expert_activations.get(expert_a, set())
        b_acts = expert_activations.get(expert_b, set())
        if a_acts or b_acts:
            intersection = len(a_acts & b_acts)
            union = len(a_acts | b_acts)
            activation_overlap = intersection / union if union > 0 else 0.0

    # Merge candidate considers both weight similarity and activation overlap
    # High weight similarity + high activation overlap = strong merge candidate
    combined_score = (cosine_sim + activation_overlap) / 2 if activation_overlap > 0 else cosine_sim
    merge_candidate = combined_score > 0.7

    return ExpertSimilarity(
        expert_a=expert_a,
        expert_b=expert_b,
        layer_idx=layer_idx,
        weight_cosine_similarity=cosine_sim,
        activation_overlap=activation_overlap,
        merge_candidate=merge_candidate,
    )


def compute_similarity_matrix_with_activations(
    model: nn.Module,
    layer_idx: int,
    expert_activations: dict[int, set[int]] | None = None,
) -> list[ExpertSimilarity]:
    """
    Compute pairwise similarity between all experts with activation overlap.

    Args:
        model: The model
        layer_idx: Layer to analyze
        expert_activations: Optional pre-computed activation sets per expert

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
            sim = compute_expert_similarity_with_activations(
                model, layer_idx, i, j, expert_activations
            )
            similarities.append(sim)

    return similarities


def find_merge_candidates_with_activations(
    similarities: list[ExpertSimilarity],
    weight_threshold: float = 0.8,
    activation_threshold: float = 0.5,
    require_both: bool = False,
) -> list[tuple[int, int, float, float]]:
    """
    Find expert pairs that are good merge candidates using both metrics.

    Args:
        similarities: List of ExpertSimilarity
        weight_threshold: Minimum weight cosine similarity
        activation_threshold: Minimum activation overlap
        require_both: If True, both thresholds must be met

    Returns:
        List of (expert_a, expert_b, weight_sim, activation_overlap) tuples
    """
    candidates = []
    for sim in similarities:
        weight_ok = sim.weight_cosine_similarity >= weight_threshold
        activation_ok = sim.activation_overlap >= activation_threshold

        if require_both:
            if weight_ok and activation_ok:
                candidates.append(
                    (
                        sim.expert_a,
                        sim.expert_b,
                        sim.weight_cosine_similarity,
                        sim.activation_overlap,
                    )
                )
        else:
            # Either high weight similarity OR high activation overlap
            if weight_ok or activation_ok:
                candidates.append(
                    (
                        sim.expert_a,
                        sim.expert_b,
                        sim.weight_cosine_similarity,
                        sim.activation_overlap,
                    )
                )

    # Sort by combined score (average of both metrics)
    return sorted(
        candidates,
        key=lambda x: (x[2] + x[3]) / 2,
        reverse=True,
    )


def print_activation_overlap_matrix(
    similarities: list[ExpertSimilarity],
    num_experts: int,
) -> None:
    """Print a matrix showing activation overlap between experts."""
    print("\nActivation Overlap Matrix")
    print("=" * 60)

    # Header row
    header = "     " + " ".join(f"{i:5d}" for i in range(num_experts))
    print(header)
    print("-" * len(header))

    # Build matrix
    matrix: dict[tuple[int, int], float] = {}
    for sim in similarities:
        matrix[(sim.expert_a, sim.expert_b)] = sim.activation_overlap
        matrix[(sim.expert_b, sim.expert_a)] = sim.activation_overlap

    # Print rows
    for i in range(num_experts):
        row = f"{i:3d}: "
        for j in range(num_experts):
            if i == j:
                row += "  1.0 "
            else:
                overlap = matrix.get((i, j), 0.0)
                if overlap > 0.7:
                    row += f" {overlap:.2f}*"
                elif overlap > 0.3:
                    row += f" {overlap:.2f} "
                else:
                    row += f" {overlap:.2f} "
        print(row)
