# pruning.py

import mlx.core as mx
import mlx.nn as nn


def find_pruneable_heads_and_indices(
    heads: list[int], n_heads: int, pruned_heads: set[int]
) -> tuple[list[int], mx.array]:
    """
    Find the heads and their indices taking pruning into account.

    Args:
        heads: List of head indices to prune
        n_heads: Total number of heads
        pruned_heads: Set of already pruned heads

    Returns:
        Tuple of (list of heads to prune, index of remaining heads)
    """
    to_prune = set(heads) - pruned_heads
    to_prune = sorted(to_prune)

    # Create index of remaining heads
    index = mx.cumsum(mx.ones((n_heads,), dtype=mx.int32), 0) - 1
    index = mx.array([i for i in index if i not in to_prune])

    return to_prune, index


def prune_linear_layer(layer: nn.Linear, index: mx.array, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer.

    Args:
        layer: Linear layer to prune
        index: Index of heads to keep
        dim: Dimension along which to prune the layer

    Returns:
        Pruned linear layer
    """
    if dim == 0:
        return nn.Linear(index.shape[0], layer.weight.shape[1], bias=layer.bias is not None)
    elif dim == 1:
        return nn.Linear(layer.weight.shape[0], index.shape[0], bias=layer.bias is not None)
    else:
        raise ValueError(f"Invalid dimension: {dim}")


def update_pruned_heads(pruned_heads: set[int], heads_to_prune: list[int]) -> set[int]:
    """
    Update the set of pruned heads.

    Args:
        pruned_heads: Set of already pruned heads
        heads_to_prune: List of new heads to prune

    Returns:
        Updated set of pruned heads
    """
    return pruned_heads.union(heads_to_prune)


def update_head_importance(head_importance: mx.array, pruned_heads: set[int]) -> mx.array:
    """
    Update head importance array after pruning.

    Args:
        head_importance: Array of head importance values
        pruned_heads: Set of pruned head indices

    Returns:
        Updated head importance array
    """
    return mx.array([i for j, i in enumerate(head_importance) if j not in pruned_heads])


def prune_heads(
    heads: list[int],
    n_heads: int,
    pruned_heads: set[int],
    head_importance: mx.array,
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
    o_proj: nn.Linear,
) -> tuple[int, set[int], mx.array, nn.Linear, nn.Linear, nn.Linear, nn.Linear]:
    """
    Prune attention heads. This method removes the specified heads from the attention mechanism.

    Args:
        heads: List of head indices to prune
        n_heads: Current number of attention heads
        pruned_heads: Set of already pruned heads
        head_importance: Array of head importance values
        q_proj, k_proj, v_proj, o_proj: Linear projection layers

    Returns:
        Tuple of (
            updated number of heads,
            updated set of pruned heads,
            updated head importance array,
            pruned q_proj, k_proj, v_proj, o_proj layers
        )
    """
    if len(heads) == 0:
        return n_heads, pruned_heads, head_importance, q_proj, k_proj, v_proj, o_proj

    heads_to_prune, index = find_pruneable_heads_and_indices(heads, n_heads, pruned_heads)

    # Prune linear layers
    q_proj = prune_linear_layer(q_proj, index)
    k_proj = prune_linear_layer(k_proj, index)
    v_proj = prune_linear_layer(v_proj, index)
    o_proj = prune_linear_layer(o_proj, index, dim=1)

    # Update configs
    n_heads = n_heads - len(heads_to_prune)
    pruned_heads = update_pruned_heads(pruned_heads, heads_to_prune)
    head_importance = update_head_importance(head_importance, pruned_heads)

    return n_heads, pruned_heads, head_importance, q_proj, k_proj, v_proj, o_proj
