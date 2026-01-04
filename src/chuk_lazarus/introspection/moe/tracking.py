"""
Cross-layer expert tracking for MoE models.

Tracks how expert specialization evolves through model depth:
- Match experts across layers by specialization
- Identify functional pipelines (e.g., "math expert" through layers)
- Visualize expert role evolution
- Compute cross-layer alignment scores

Example:
    >>> from chuk_lazarus.introspection.moe import MoEHooks
    >>> from chuk_lazarus.introspection.moe.tracking import (
    ...     track_expert_pipeline,
    ...     compute_layer_alignment,
    ...     identify_functional_pipelines,
    ... )
    >>>
    >>> hooks = MoEHooks(model)
    >>> # Find "math pipeline" - experts that handle math across layers
    >>> pipeline = track_expert_pipeline(hooks, category="math", prompts=math_prompts)
    >>> print(pipeline.experts_by_layer)
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .enums import ExpertCategory

if TYPE_CHECKING:
    from .models import LayerRouterWeights


class ExpertPipelineNode(BaseModel):
    """A single expert in a pipeline across layers."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    expert_idx: int = Field(ge=0)
    activation_rate: float = Field(ge=0, le=1)
    category: ExpertCategory | None = None
    confidence: float = Field(ge=0, le=1, default=0.0)


class ExpertPipeline(BaseModel):
    """A chain of experts across layers that serve a similar function."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Pipeline name (e.g., 'Math Pipeline')")
    category: ExpertCategory = Field(description="Primary category")
    nodes: tuple[ExpertPipelineNode, ...] = Field(default_factory=tuple)
    consistency_score: float = Field(
        ge=0, le=1, default=0.0, description="How consistent the pipeline is across layers"
    )
    coverage: float = Field(
        ge=0, le=1, default=0.0, description="Fraction of layers covered by this pipeline"
    )

    @property
    def experts_by_layer(self) -> dict[int, int]:
        """Get expert index for each layer."""
        return {node.layer_idx: node.expert_idx for node in self.nodes}

    @property
    def layers(self) -> list[int]:
        """Get layers in this pipeline."""
        return sorted(n.layer_idx for n in self.nodes)

    def get_expert_at_layer(self, layer_idx: int) -> int | None:
        """Get expert index at a specific layer."""
        for node in self.nodes:
            if node.layer_idx == layer_idx:
                return node.expert_idx
        return None


class LayerAlignmentResult(BaseModel):
    """Result of computing alignment between two layers."""

    model_config = ConfigDict(frozen=True)

    layer_a: int = Field(ge=0)
    layer_b: int = Field(ge=0)
    alignment_score: float = Field(ge=0, le=1)
    matched_pairs: tuple[tuple[int, int], ...] = Field(
        default_factory=tuple, description="Matched (expert_a, expert_b) pairs"
    )
    category_agreement: float = Field(
        ge=0, le=1, default=0.0, description="How often matched experts have same category"
    )


class CrossLayerAnalysis(BaseModel):
    """Complete cross-layer expert analysis."""

    model_config = ConfigDict(frozen=True)

    num_layers: int = Field(ge=0)
    num_experts: int = Field(ge=1)
    pipelines: tuple[ExpertPipeline, ...] = Field(default_factory=tuple)
    layer_alignments: tuple[LayerAlignmentResult, ...] = Field(default_factory=tuple)
    global_consistency: float = Field(ge=0, le=1, default=0.0)

    def get_pipeline_for_category(self, category: ExpertCategory) -> ExpertPipeline | None:
        """Get pipeline for a specific category."""
        for pipeline in self.pipelines:
            if pipeline.category == category:
                return pipeline
        return None


# =============================================================================
# Core Tracking Functions
# =============================================================================


def compute_expert_activation_profile(
    all_layer_weights: list[LayerRouterWeights],
    num_experts: int,
) -> dict[int, np.ndarray]:
    """
    Compute activation profile for each expert across positions.

    Args:
        all_layer_weights: Router weights for all layers
        num_experts: Total number of experts

    Returns:
        Dict mapping layer_idx -> activation matrix [positions × experts]
    """
    profiles: dict[int, np.ndarray] = {}

    for layer_weights in all_layer_weights:
        layer_idx = layer_weights.layer_idx
        num_positions = len(layer_weights.positions)

        matrix = np.zeros((num_positions, num_experts))

        for pos_idx, pos in enumerate(layer_weights.positions):
            for exp_idx, weight in zip(pos.expert_indices, pos.weights):
                if 0 <= exp_idx < num_experts:
                    matrix[pos_idx, exp_idx] = weight

        profiles[layer_idx] = matrix

    return profiles


def compute_layer_alignment(
    profile_a: np.ndarray,
    profile_b: np.ndarray,
    layer_a: int,
    layer_b: int,
) -> LayerAlignmentResult:
    """
    Compute alignment between expert profiles at two layers.

    Uses correlation between activation patterns to find matching experts.

    Args:
        profile_a: Activation matrix for layer A [positions × experts]
        profile_b: Activation matrix for layer B [positions × experts]
        layer_a: Index of layer A
        layer_b: Index of layer B

    Returns:
        LayerAlignmentResult with alignment score and matched pairs
    """
    num_experts = profile_a.shape[1]

    # Compute correlation matrix between experts
    correlation_matrix = np.zeros((num_experts, num_experts))

    for i in range(num_experts):
        for j in range(num_experts):
            act_a = profile_a[:, i]
            act_b = profile_b[:, j]

            # Handle zero variance
            if np.std(act_a) < 1e-10 or np.std(act_b) < 1e-10:
                correlation_matrix[i, j] = 0.0
            else:
                correlation_matrix[i, j] = np.corrcoef(act_a, act_b)[0, 1]

    # Greedy matching based on highest correlation
    matched_pairs: list[tuple[int, int]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()

    # Sort all pairs by correlation (descending)
    all_pairs = []
    for i in range(num_experts):
        for j in range(num_experts):
            all_pairs.append((i, j, correlation_matrix[i, j]))

    all_pairs.sort(key=lambda x: x[2], reverse=True)

    for i, j, corr in all_pairs:
        if i not in used_a and j not in used_b and corr > 0.3:
            matched_pairs.append((i, j))
            used_a.add(i)
            used_b.add(j)

    # Compute overall alignment score
    if matched_pairs:
        avg_corr = np.mean([correlation_matrix[i, j] for i, j in matched_pairs])
    else:
        avg_corr = 0.0

    return LayerAlignmentResult(
        layer_a=layer_a,
        layer_b=layer_b,
        alignment_score=float(max(0, avg_corr)),
        matched_pairs=tuple(matched_pairs),
        category_agreement=0.0,  # Updated later with identity info
    )


def track_expert_across_layers(
    profiles: dict[int, np.ndarray],
    start_layer: int,
    start_expert: int,
    threshold: float = 0.3,
) -> list[ExpertPipelineNode]:
    """
    Track an expert's function across layers by correlation.

    Args:
        profiles: Activation profiles per layer
        start_layer: Starting layer index
        start_expert: Starting expert index
        threshold: Minimum correlation to continue tracking

    Returns:
        List of ExpertPipelineNode representing the tracked path
    """
    if start_layer not in profiles:
        return []

    nodes: list[ExpertPipelineNode] = []
    layers = sorted(profiles.keys())

    # Start with the initial expert
    current_expert = start_expert
    current_layer_idx = layers.index(start_layer)

    # Add starting node
    start_profile = profiles[start_layer][:, start_expert]
    activation_rate = float(np.mean(start_profile > 0.01))

    nodes.append(
        ExpertPipelineNode(
            layer_idx=start_layer,
            expert_idx=start_expert,
            activation_rate=activation_rate,
        )
    )

    # Track forward through layers
    for layer_idx in layers[current_layer_idx + 1 :]:
        prev_layer = nodes[-1].layer_idx
        prev_expert = nodes[-1].expert_idx
        prev_profile = profiles[prev_layer][:, prev_expert]

        # Find best matching expert in current layer
        curr_profile_matrix = profiles[layer_idx]
        num_experts = curr_profile_matrix.shape[1]

        best_expert = -1
        best_corr = -1.0

        for exp_idx in range(num_experts):
            exp_profile = curr_profile_matrix[:, exp_idx]

            if np.std(prev_profile) < 1e-10 or np.std(exp_profile) < 1e-10:
                continue

            corr = np.corrcoef(prev_profile, exp_profile)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_expert = exp_idx

        # Only continue if correlation is above threshold
        if best_corr >= threshold and best_expert >= 0:
            activation_rate = float(np.mean(curr_profile_matrix[:, best_expert] > 0.01))
            nodes.append(
                ExpertPipelineNode(
                    layer_idx=layer_idx,
                    expert_idx=best_expert,
                    activation_rate=activation_rate,
                    confidence=float(best_corr),
                )
            )
        else:
            break  # Pipeline ends here

    return nodes


def identify_functional_pipelines(
    profiles: dict[int, np.ndarray],
    expert_identities: list[dict[str, Any]] | None = None,
    min_coverage: float = 0.5,
    correlation_threshold: float = 0.3,
) -> list[ExpertPipeline]:
    """
    Identify functional pipelines across layers.

    Args:
        profiles: Activation profiles per layer
        expert_identities: Optional expert identity info with category
        min_coverage: Minimum fraction of layers a pipeline must cover
        correlation_threshold: Minimum correlation for tracking

    Returns:
        List of identified pipelines
    """
    if not profiles:
        return []

    layers = sorted(profiles.keys())
    num_layers = len(layers)
    first_layer = layers[0]
    num_experts = profiles[first_layer].shape[1]

    # Track from each expert in the first layer
    pipelines: list[ExpertPipeline] = []
    used_starts: set[int] = set()

    for start_expert in range(num_experts):
        if start_expert in used_starts:
            continue

        nodes = track_expert_across_layers(
            profiles, first_layer, start_expert, correlation_threshold
        )

        coverage = len(nodes) / num_layers

        if coverage >= min_coverage:
            # Determine category from identities if available
            category = ExpertCategory.GENERALIST
            if expert_identities:
                # Look for most common category in pipeline
                categories: dict[ExpertCategory, int] = defaultdict(int)
                for node in nodes:
                    for identity in expert_identities:
                        if (
                            identity.get("layer_idx") == node.layer_idx
                            and identity.get("expert_idx") == node.expert_idx
                        ):
                            cat = identity.get("primary_category")
                            if cat:
                                categories[ExpertCategory(cat)] += 1

                if categories:
                    category = max(categories, key=categories.get)

            # Compute consistency score
            if len(nodes) > 1:
                confidences = [n.confidence for n in nodes[1:] if n.confidence > 0]
                consistency = float(np.mean(confidences)) if confidences else 0.0
            else:
                consistency = 1.0

            pipeline = ExpertPipeline(
                name=f"{category.value.title()} Pipeline (E{start_expert})",
                category=category,
                nodes=tuple(nodes),
                consistency_score=consistency,
                coverage=coverage,
            )
            pipelines.append(pipeline)

            # Mark starting expert as used
            used_starts.add(start_expert)

    # Sort by coverage and consistency
    pipelines.sort(key=lambda p: (p.coverage, p.consistency_score), reverse=True)

    return pipelines


def analyze_cross_layer_routing(
    all_layer_weights: list[LayerRouterWeights],
    num_experts: int,
    expert_identities: list[dict[str, Any]] | None = None,
) -> CrossLayerAnalysis:
    """
    Comprehensive cross-layer routing analysis.

    Args:
        all_layer_weights: Router weights for all layers
        num_experts: Total number of experts
        expert_identities: Optional expert identity info

    Returns:
        CrossLayerAnalysis with pipelines and alignments
    """
    # Compute profiles
    profiles = compute_expert_activation_profile(all_layer_weights, num_experts)

    if not profiles:
        return CrossLayerAnalysis(
            num_layers=0,
            num_experts=num_experts,
            pipelines=(),
            layer_alignments=(),
            global_consistency=0.0,
        )

    layers = sorted(profiles.keys())
    num_layers = len(layers)

    # Compute layer alignments
    alignments: list[LayerAlignmentResult] = []
    for i in range(len(layers) - 1):
        layer_a = layers[i]
        layer_b = layers[i + 1]
        alignment = compute_layer_alignment(
            profiles[layer_a],
            profiles[layer_b],
            layer_a,
            layer_b,
        )
        alignments.append(alignment)

    # Identify pipelines
    pipelines = identify_functional_pipelines(
        profiles,
        expert_identities,
        min_coverage=0.3,
    )

    # Compute global consistency
    if alignments:
        global_consistency = float(np.mean([a.alignment_score for a in alignments]))
    else:
        global_consistency = 0.0

    return CrossLayerAnalysis(
        num_layers=num_layers,
        num_experts=num_experts,
        pipelines=tuple(pipelines),
        layer_alignments=tuple(alignments),
        global_consistency=global_consistency,
    )


# =============================================================================
# Printing and Visualization
# =============================================================================


def print_pipeline_summary(pipelines: list[ExpertPipeline]) -> None:
    """Print summary of identified pipelines."""
    print("\nExpert Pipelines Across Layers")
    print("=" * 60)

    if not pipelines:
        print("No pipelines identified")
        return

    for i, pipeline in enumerate(pipelines):
        print(f"\n{i + 1}. {pipeline.name}")
        print(f"   Category: {pipeline.category.value}")
        print(
            f"   Coverage: {pipeline.coverage:.0%} ({len(pipeline.nodes)}/{len(pipeline.layers)} layers)"
        )
        print(f"   Consistency: {pipeline.consistency_score:.2f}")

        # Show expert path
        path = " → ".join(f"L{n.layer_idx}:E{n.expert_idx}" for n in pipeline.nodes)
        print(f"   Path: {path}")


def print_alignment_matrix(alignments: list[LayerAlignmentResult]) -> None:
    """Print layer alignment scores."""
    print("\nLayer-to-Layer Alignment")
    print("=" * 40)

    for alignment in alignments:
        score = alignment.alignment_score
        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        print(f"L{alignment.layer_a:2d} → L{alignment.layer_b:2d}: {bar} {score:.2f}")

    if alignments:
        avg = np.mean([a.alignment_score for a in alignments])
        print("-" * 40)
        print(f"Average alignment: {avg:.2f}")
