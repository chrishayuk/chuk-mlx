"""
Utility functions for model analysis.

This module contains helper functions for computing entropy,
divergence, and other mathematical operations used in analysis.
"""

from __future__ import annotations

import mlx.core as mx


def compute_entropy(probs: mx.array) -> float:
    """
    Compute Shannon entropy of a probability distribution.

    Args:
        probs: Probability distribution array

    Returns:
        Shannon entropy value
    """
    # Avoid log(0) by clipping
    probs_clipped = mx.clip(probs, 1e-10, 1.0)
    entropy = -mx.sum(probs_clipped * mx.log(probs_clipped))
    return float(entropy)


def compute_kl_divergence(p: mx.array, q: mx.array) -> float:
    """
    Compute KL divergence D(P || Q).

    Args:
        p: Source probability distribution
        q: Target probability distribution

    Returns:
        KL divergence value (always >= 0)
    """
    # Clip to avoid log(0) and division by zero
    p_clipped = mx.clip(p, 1e-10, 1.0)
    q_clipped = mx.clip(q, 1e-10, 1.0)
    kl = mx.sum(p_clipped * mx.log(p_clipped / q_clipped))
    return float(mx.maximum(kl, mx.array(0.0)))  # KL should be >= 0


def compute_js_divergence(p: mx.array, q: mx.array) -> float:
    """
    Compute Jensen-Shannon divergence (symmetric, bounded [0, ln(2)]).

    Args:
        p: First probability distribution
        q: Second probability distribution

    Returns:
        JS divergence value
    """
    m = (p + q) / 2
    js = 0.5 * compute_kl_divergence(p, m) + 0.5 * compute_kl_divergence(q, m)
    return js


def get_layers_to_capture(
    num_layers: int,
    layer_strategy: str,
    layer_step: int = 4,
    custom_layers: list[int] | None = None,
) -> list[int]:
    """
    Determine which layers to capture based on strategy.

    Args:
        num_layers: Total number of layers in the model
        layer_strategy: Strategy name ('all', 'evenly_spaced', 'first_last', 'custom')
        layer_step: Step size for evenly spaced capture
        custom_layers: Specific layers when using 'custom' strategy

    Returns:
        Sorted list of layer indices to capture
    """
    if layer_strategy == "all":
        return list(range(num_layers))

    if layer_strategy == "first_last":
        return [0, num_layers - 1]

    if layer_strategy == "custom":
        if custom_layers:
            return sorted(set(custom_layers))
        return [0, num_layers - 1]

    # evenly_spaced
    layers = list(range(0, num_layers, layer_step))
    if (num_layers - 1) not in layers:
        layers.append(num_layers - 1)
    return sorted(set(layers))
