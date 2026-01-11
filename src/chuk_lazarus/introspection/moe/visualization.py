"""
MoE routing visualization utilities.

Provides tools for visualizing expert routing patterns:
- Token × Expert activation heatmaps
- Layer-wise routing flow diagrams
- Expert utilization bar charts
- Cross-layer routing evolution

Example:
    >>> from chuk_lazarus.introspection.moe import ExpertRouter
    >>> from chuk_lazarus.introspection.moe.visualization import (
    ...     plot_routing_heatmap,
    ...     plot_expert_utilization,
    ...     save_routing_heatmap,
    ... )
    >>>
    >>> router = await ExpertRouter.from_pretrained("model")
    >>> weights = await router.capture_router_weights("Hello world")
    >>> fig = plot_routing_heatmap(weights, layer_idx=0)
    >>> fig.savefig("heatmap.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .models import ExpertUtilization, LayerRouterWeights


# =============================================================================
# Heatmap Data Structures
# =============================================================================


def routing_weights_to_matrix(
    layer_weights: LayerRouterWeights,
    num_experts: int,
) -> tuple[np.ndarray, list[str]]:
    """
    Convert LayerRouterWeights to a 2D matrix for heatmap plotting.

    Args:
        layer_weights: Router weights for a single layer
        num_experts: Total number of experts

    Returns:
        Tuple of (matrix [positions × experts], token_labels)
    """
    positions = layer_weights.positions
    num_positions = len(positions)

    # Initialize matrix with zeros
    matrix = np.zeros((num_positions, num_experts))
    tokens: list[str] = []

    for pos_idx, pos in enumerate(positions):
        tokens.append(pos.token if pos.token else f"[{pos_idx}]")

        # Fill in the weights for selected experts
        for exp_idx, weight in zip(pos.expert_indices, pos.weights):
            if 0 <= exp_idx < num_experts:
                matrix[pos_idx, exp_idx] = weight

    return matrix, tokens


def multi_layer_routing_matrix(
    all_layer_weights: list[LayerRouterWeights],
    num_experts: int,
    aggregation: str = "mean",
) -> np.ndarray:
    """
    Aggregate routing across multiple layers.

    Args:
        all_layer_weights: Router weights for all layers
        num_experts: Total number of experts
        aggregation: How to aggregate ("mean", "max", "sum")

    Returns:
        Matrix [positions × experts] with aggregated weights
    """
    if not all_layer_weights:
        return np.zeros((0, num_experts))

    matrices = []

    for layer_weights in all_layer_weights:
        matrix, _ = routing_weights_to_matrix(layer_weights, num_experts)
        matrices.append(matrix)

    stacked = np.stack(matrices, axis=0)  # [layers, positions, experts]

    if aggregation == "mean":
        return np.mean(stacked, axis=0)
    elif aggregation == "max":
        return np.max(stacked, axis=0)
    elif aggregation == "sum":
        return np.sum(stacked, axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


# =============================================================================
# Matplotlib Plotting
# =============================================================================


def plot_routing_heatmap(
    layer_weights: LayerRouterWeights,
    num_experts: int,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 8),
    cmap: str = "YlOrRd",
    show_values: bool = False,
    ax: Any = None,
) -> Any:
    """
    Plot a token × expert routing heatmap using matplotlib.

    Args:
        layer_weights: Router weights for a single layer
        num_experts: Total number of experts
        title: Plot title (defaults to layer info)
        figsize: Figure size
        cmap: Colormap name
        show_values: Whether to show weight values in cells
        ax: Existing matplotlib axes (creates new figure if None)

    Returns:
        matplotlib Figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from err

    matrix, tokens = routing_weights_to_matrix(layer_weights, num_experts)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Create heatmap
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    # Configure axes
    ax.set_xlabel("Expert Index")
    ax.set_ylabel("Token Position")

    # Set ticks
    ax.set_xticks(range(num_experts))
    ax.set_xticklabels(range(num_experts))

    # Only show token labels if not too many
    if len(tokens) <= 30:
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=8)
    else:
        # Show every 5th token
        tick_positions = list(range(0, len(tokens), 5))
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([tokens[i] for i in tick_positions], fontsize=8)

    # Add colorbar
    fig.colorbar(im, ax=ax, label="Routing Weight")

    # Add values to cells if requested
    if show_values and matrix.shape[0] * matrix.shape[1] < 500:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 0.01:
                    text_color = "white" if matrix[i, j] > 0.5 else "black"
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=6,
                    )

    # Title
    if title is None:
        title = f"Expert Routing - Layer {layer_weights.layer_idx}"
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_multi_layer_heatmap(
    all_layer_weights: list[LayerRouterWeights],
    num_experts: int,
    title: str = "Cross-Layer Expert Routing",
    figsize: tuple[int, int] = (14, 10),
    cmap: str = "YlOrRd",
) -> Any:
    """
    Plot routing heatmaps for multiple layers in a grid.

    Args:
        all_layer_weights: Router weights for all layers
        num_experts: Total number of experts
        title: Overall title
        figsize: Figure size
        cmap: Colormap name

    Returns:
        matplotlib Figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError("matplotlib required") from err

    num_layers = len(all_layer_weights)
    if num_layers == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    # Determine grid layout
    cols = min(4, num_layers)
    rows = (num_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, layer_weights in enumerate(all_layer_weights):
        ax = axes[i]
        matrix, tokens = routing_weights_to_matrix(layer_weights, num_experts)

        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"Layer {layer_weights.layer_idx}", fontsize=10)
        ax.set_xlabel("Expert", fontsize=8)
        ax.set_ylabel("Token", fontsize=8)

    # Hide unused subplots
    for i in range(num_layers, len(axes)):
        axes[i].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.colorbar(im, ax=axes[:num_layers], label="Weight", shrink=0.8)
    plt.tight_layout()

    return fig


def plot_expert_utilization(
    utilization: ExpertUtilization,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    color: str = "#4ECDC4",
    highlight_threshold: float = 0.15,
) -> Any:
    """
    Plot expert utilization as a bar chart.

    Args:
        utilization: Expert utilization statistics
        title: Plot title
        figsize: Figure size
        color: Bar color
        highlight_threshold: Threshold for highlighting over/under-used experts

    Returns:
        matplotlib Figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError("matplotlib required") from err

    fig, ax = plt.subplots(figsize=figsize)

    num_experts = utilization.num_experts
    frequencies = list(utilization.expert_frequencies)
    x = list(range(num_experts))

    # Color bars based on utilization
    uniform = 1.0 / num_experts
    colors = []
    for freq in frequencies:
        if freq > uniform * (1 + highlight_threshold):
            colors.append("#FF6B6B")  # Over-used (red)
        elif freq < uniform * (1 - highlight_threshold):
            colors.append("#95E1D3")  # Under-used (light green)
        else:
            colors.append(color)  # Normal

    ax.bar(x, frequencies, color=colors, edgecolor="black", linewidth=0.5)

    # Add uniform distribution line
    ax.axhline(
        y=uniform,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Uniform ({uniform:.2%})",
    )

    # Configure
    ax.set_xlabel("Expert Index")
    ax.set_ylabel("Activation Frequency")
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend()

    if title is None:
        title = f"Expert Utilization - Layer {utilization.layer_idx} (Balance: {utilization.load_balance_score:.2%})"
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_routing_flow(
    all_layer_weights: list[LayerRouterWeights],
    num_experts: int,
    token_idx: int = -1,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> Any:
    """
    Plot how routing changes across layers for a specific token.

    Args:
        all_layer_weights: Router weights for all layers
        num_experts: Total number of experts
        token_idx: Token position to track (-1 for last)
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError("matplotlib required") from err

    fig, ax = plt.subplots(figsize=figsize)

    layers = []
    expert_weights: dict[int, list[float]] = {i: [] for i in range(num_experts)}

    for layer_weights in all_layer_weights:
        layers.append(layer_weights.layer_idx)

        # Get weights for the specified token
        positions = layer_weights.positions
        if not positions:
            for exp in range(num_experts):
                expert_weights[exp].append(0.0)
            continue

        pos = positions[token_idx] if abs(token_idx) < len(positions) else positions[-1]

        # Initialize all experts to 0 for this layer
        layer_exp_weights = [0.0] * num_experts
        for exp_idx, weight in zip(pos.expert_indices, pos.weights):
            if 0 <= exp_idx < num_experts:
                layer_exp_weights[exp_idx] = weight

        for exp in range(num_experts):
            expert_weights[exp].append(layer_exp_weights[exp])

    # Plot lines for each expert
    cmap = plt.cm.get_cmap("tab20")
    for exp_idx in range(num_experts):
        weights = expert_weights[exp_idx]
        if max(weights) > 0.01:  # Only plot active experts
            ax.plot(
                layers,
                weights,
                marker="o",
                label=f"Expert {exp_idx}",
                color=cmap(exp_idx / num_experts),
                linewidth=2,
                markersize=4,
            )

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Routing Weight")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    if title is None:
        title = f"Routing Flow Across Layers (Token {token_idx})"
    ax.set_title(title)

    plt.tight_layout()
    return fig


# =============================================================================
# ASCII Visualization (for terminal)
# =============================================================================


def routing_heatmap_ascii(
    layer_weights: LayerRouterWeights,
    num_experts: int,
    max_width: int = 80,
) -> str:
    """
    Generate ASCII art heatmap for terminal display.

    Args:
        layer_weights: Router weights for a single layer
        num_experts: Total number of experts
        max_width: Maximum line width

    Returns:
        ASCII heatmap string
    """
    matrix, tokens = routing_weights_to_matrix(layer_weights, num_experts)

    # Characters for intensity
    chars = " ░▒▓█"

    lines = [f"Layer {layer_weights.layer_idx} Routing Heatmap"]
    lines.append("=" * min(num_experts * 3 + 10, max_width))

    # Header
    header = "Token".ljust(8) + "".join(f"{i:3d}" for i in range(num_experts))
    lines.append(header[:max_width])
    lines.append("-" * len(header[:max_width]))

    # Rows
    for pos_idx, (row, token) in enumerate(zip(matrix, tokens)):
        # Truncate token
        token_display = token[:6].ljust(8)
        row_chars = ""
        for weight in row:
            char_idx = int(weight * (len(chars) - 1))
            row_chars += f" {chars[char_idx]} "
        lines.append(f"{token_display}{row_chars}"[:max_width])

    return "\n".join(lines)


def utilization_bar_ascii(
    utilization: ExpertUtilization,
    bar_width: int = 40,
) -> str:
    """
    Generate ASCII bar chart for expert utilization.

    Args:
        utilization: Expert utilization statistics
        bar_width: Width of each bar

    Returns:
        ASCII bar chart string
    """
    lines = [f"Expert Utilization - Layer {utilization.layer_idx}"]
    lines.append(f"Load Balance Score: {utilization.load_balance_score:.2%}")
    lines.append("=" * (bar_width + 15))

    max_freq = max(utilization.expert_frequencies) if utilization.expert_frequencies else 1.0
    uniform = 1.0 / utilization.num_experts

    for exp_idx, freq in enumerate(utilization.expert_frequencies):
        bar_len = int((freq / max_freq) * bar_width)
        bar = "█" * bar_len + "░" * (bar_width - bar_len)

        # Mark if over/under utilized
        marker = " "
        if freq > uniform * 1.2:
            marker = "▲"  # Over-used
        elif freq < uniform * 0.8:
            marker = "▼"  # Under-used

        lines.append(f"E{exp_idx:2d} {bar} {freq:.1%} {marker}")

    lines.append("-" * (bar_width + 15))
    lines.append(f"Uniform: {uniform:.1%}")

    return "\n".join(lines)


# =============================================================================
# File I/O
# =============================================================================


def save_routing_heatmap(
    layer_weights: LayerRouterWeights,
    num_experts: int,
    path: str | Path,
    format: str = "png",
    **kwargs: Any,
) -> None:
    """
    Save routing heatmap to file.

    Args:
        layer_weights: Router weights
        num_experts: Number of experts
        path: Output path
        format: File format (png, pdf, svg)
        **kwargs: Additional arguments for plot_routing_heatmap
    """
    fig = plot_routing_heatmap(layer_weights, num_experts, **kwargs)
    fig.savefig(str(path), format=format, dpi=150, bbox_inches="tight")

    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass


def save_utilization_chart(
    utilization: ExpertUtilization,
    path: str | Path,
    format: str = "png",
    **kwargs: Any,
) -> None:
    """
    Save utilization bar chart to file.

    Args:
        utilization: Expert utilization data
        path: Output path
        format: File format
        **kwargs: Additional arguments for plot_expert_utilization
    """
    fig = plot_expert_utilization(utilization, **kwargs)
    fig.savefig(str(path), format=format, dpi=150, bbox_inches="tight")

    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
