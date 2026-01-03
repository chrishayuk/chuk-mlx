"""Router analysis utilities for MoE models."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from .detector import get_moe_layers
from .models import CoactivationAnalysis, ExpertPair, ExpertUtilization

if TYPE_CHECKING:
    from .hooks import MoEHooks


def analyze_coactivation(
    hooks: "MoEHooks",
    layer_idx: int,
) -> CoactivationAnalysis | None:
    """Analyze which experts frequently co-activate together.

    Args:
        hooks: MoEHooks with captured state
        layer_idx: Layer to analyze

    Returns:
        CoactivationAnalysis or None if no data
    """
    if layer_idx not in hooks.moe_state.selected_experts:
        return None

    info = hooks.get_layer_info(layer_idx)
    if info is None:
        return None

    selected = hooks.moe_state.selected_experts[layer_idx]
    batch_size, seq_len, k = selected.shape

    # Count pair occurrences
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    expert_counts: dict[int, int] = defaultdict(int)
    total = 0

    for b in range(batch_size):
        for s in range(seq_len):
            experts = sorted(selected[b, s].tolist())
            total += 1

            for exp in experts:
                expert_counts[exp] += 1

            for i, exp1 in enumerate(experts):
                for exp2 in experts[i + 1:]:
                    pair_counts[(exp1, exp2)] += 1

    # Build sorted pairs
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    top_pairs = []
    for (exp_a, exp_b), count in sorted_pairs[:20]:
        top_pairs.append(ExpertPair(
            expert_a=exp_a,
            expert_b=exp_b,
            coactivation_count=count,
            coactivation_rate=count / total if total > 0 else 0,
        ))

    # Find specialists (high coactivation with specific partners)
    specialist_pairs = [p for p in top_pairs if p.coactivation_rate > 0.3]

    # Find generalists (appear in many pairs)
    expert_pair_counts: dict[int, int] = defaultdict(int)
    for (exp_a, exp_b), _ in sorted_pairs:
        expert_pair_counts[exp_a] += 1
        expert_pair_counts[exp_b] += 1

    generalists = [
        exp for exp, count in expert_pair_counts.items()
        if count >= info.num_experts // 2
    ]

    return CoactivationAnalysis(
        layer_idx=layer_idx,
        total_activations=total,
        top_pairs=tuple(top_pairs),
        specialist_pairs=tuple(specialist_pairs),
        generalist_experts=tuple(sorted(generalists)),
    )


def compute_routing_diversity(
    hooks: "MoEHooks",
    layer_idx: int,
) -> float:
    """Compute routing diversity score (0=always same experts, 1=uniform).

    Args:
        hooks: MoEHooks with captured state
        layer_idx: Layer to analyze

    Returns:
        Diversity score between 0 and 1
    """
    utilization = hooks.get_expert_utilization(layer_idx)
    if utilization is None:
        return 0.0

    # Use load balance score as diversity measure
    return utilization.load_balance_score


def get_dominant_experts(
    hooks: "MoEHooks",
    layer_idx: int,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Get the most frequently activated experts.

    Args:
        hooks: MoEHooks with captured state
        layer_idx: Layer to analyze
        top_k: Number of experts to return

    Returns:
        List of (expert_idx, activation_rate) tuples
    """
    utilization = hooks.get_expert_utilization(layer_idx)
    if utilization is None:
        return []

    # Sort by frequency
    indexed_freqs = list(enumerate(utilization.expert_frequencies))
    indexed_freqs.sort(key=lambda x: x[1], reverse=True)

    return [(idx, freq) for idx, freq in indexed_freqs[:top_k]]


def get_rare_experts(
    hooks: "MoEHooks",
    layer_idx: int,
    threshold: float = 0.01,
) -> list[int]:
    """Get experts that rarely activate.

    Args:
        hooks: MoEHooks with captured state
        layer_idx: Layer to analyze
        threshold: Activation rate threshold

    Returns:
        List of rarely-activated expert indices
    """
    utilization = hooks.get_expert_utilization(layer_idx)
    if utilization is None:
        return []

    return [
        idx for idx, freq in enumerate(utilization.expert_frequencies)
        if freq < threshold
    ]


def compare_routing(
    hooks_a: "MoEHooks",
    hooks_b: "MoEHooks",
    layer_idx: int,
) -> dict[str, float]:
    """Compare routing patterns between two forward passes.

    Args:
        hooks_a: First MoEHooks
        hooks_b: Second MoEHooks
        layer_idx: Layer to compare

    Returns:
        Dictionary with comparison metrics
    """
    if layer_idx not in hooks_a.moe_state.selected_experts:
        return {}
    if layer_idx not in hooks_b.moe_state.selected_experts:
        return {}

    selected_a = hooks_a.moe_state.selected_experts[layer_idx]
    selected_b = hooks_b.moe_state.selected_experts[layer_idx]

    if selected_a.shape != selected_b.shape:
        return {"shape_mismatch": 1.0}

    # Compute overlap
    batch_size, seq_len, k = selected_a.shape
    total_matches = 0
    total_positions = 0

    for b in range(batch_size):
        for s in range(seq_len):
            set_a = set(selected_a[b, s].tolist())
            set_b = set(selected_b[b, s].tolist())
            overlap = len(set_a & set_b)
            total_matches += overlap
            total_positions += k

    overlap_rate = total_matches / total_positions if total_positions > 0 else 0

    return {
        "overlap_rate": overlap_rate,
        "divergence": 1.0 - overlap_rate,
    }
