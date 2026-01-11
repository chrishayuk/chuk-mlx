"""Clustering service for CLI commands.

This module provides services for activation clustering analysis using PCA.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import numpy as np


class ClusteringConfig(BaseModel):
    """Configuration for clustering analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    prompts: list[str] = Field(..., description="Prompts to cluster")
    labels: list[str] = Field(..., description="Labels for prompts")
    target_layers: list[int] | None = Field(default=None, description="Target layers")
    layer_depth_ratio: float | None = Field(default=None, description="Layer depth ratio")
    grid_width: int = Field(default=60, description="ASCII grid width")
    grid_height: int = Field(default=20, description="ASCII grid height")
    save_plot: str | None = Field(default=None, description="Path to save plot")


class ClusteringResult(BaseModel):
    """Result of clustering analysis."""

    model_config = ConfigDict(frozen=True)

    layer_results: list[dict[str, Any]] = Field(default_factory=list)
    model_id: str = Field(default="")
    unique_labels: list[str] = Field(default_factory=list)
    prompt_count: int = Field(default=0)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            "CLUSTERING ANALYSIS",
            f"{'=' * 70}",
            f"Model: {self.model_id}",
            f"Prompts: {self.prompt_count}",
            f"Classes: {', '.join(self.unique_labels)}",
        ]

        for layer_result in self.layer_results:
            layer = layer_result["layer"]
            pca_var = layer_result.get("pca_variance", [0, 0])
            lines.extend(
                [
                    "",
                    f"Layer {layer}:",
                    f"  PCA variance: {pca_var[0]:.1%}, {pca_var[1]:.1%}",
                ]
            )

            # Show cluster centers
            if "cluster_stats" in layer_result:
                for label, stats in layer_result["cluster_stats"].items():
                    center = stats.get("center", [0, 0])
                    lines.append(f"  {label}: center=({center[0]:.2f}, {center[1]:.2f})")

            # Show ASCII visualization if available
            if "ascii_grid" in layer_result:
                lines.append("")
                lines.append(layer_result["ascii_grid"])

        return "\n".join(lines)


class ClusteringService:
    """Service for clustering analysis."""

    @classmethod
    async def analyze(cls, config: ClusteringConfig) -> ClusteringResult:
        """Analyze activation clusters using PCA.

        Projects hidden states to 2D to see if different prompt types
        cluster separately.
        """
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from sklearn.decomposition import PCA

        from ...models_v2 import load_model
        from ..accessor import ModelAccessor

        # Load model using framework loader
        load_result = load_model(config.model)
        model = load_result.model
        tokenizer = load_result.tokenizer
        model_config = load_result.config

        num_layers = getattr(model_config, "num_hidden_layers", 32)
        accessor = ModelAccessor(model=model, config=model_config)

        # Determine target layers
        if config.target_layers:
            target_layers = config.target_layers
        elif config.layer_depth_ratio:
            target_layers = [int(num_layers * config.layer_depth_ratio)]
        else:
            target_layers = [int(num_layers * 0.5)]

        def get_hidden_at_layer(prompt: str, layer: int) -> np.ndarray:
            """Get hidden state at specific layer."""
            input_ids = mx.array(tokenizer.encode(prompt))[None, :]
            layers = accessor.layers
            embed = accessor.embed

            h = embed(input_ids)
            scale = accessor.embedding_scale
            if scale:
                h = h * scale

            seq_len = input_ids.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

            for idx, lyr in enumerate(layers):
                try:
                    out = lyr(h, mask=mask)
                except TypeError:
                    out = lyr(h)
                h = (
                    out.hidden_states
                    if hasattr(out, "hidden_states")
                    else (out[0] if isinstance(out, tuple) else out)
                )
                if idx == layer:
                    return np.array(h[0, -1, :].tolist())

            return np.array(h[0, -1, :].tolist())

        # Get unique labels
        unique_labels = list(dict.fromkeys(config.labels))

        # Collect activations for all layers
        activations_by_layer = {layer: [] for layer in target_layers}

        for prompt in config.prompts:
            for target_layer in target_layers:
                h = get_hidden_at_layer(prompt, target_layer)
                activations_by_layer[target_layer].append(h)

        # Create symbols for each label
        symbols = {}
        used_symbols = set()
        fallback_symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        fallback_idx = 0

        for label in unique_labels:
            symbol = label[0].upper()
            if symbol in used_symbols:
                while fallback_idx < len(fallback_symbols):
                    symbol = fallback_symbols[fallback_idx]
                    fallback_idx += 1
                    if symbol not in used_symbols:
                        break
            symbols[label] = symbol
            used_symbols.add(symbol)

        # Process each layer
        layer_results = []

        for target_layer in target_layers:
            X = np.array(activations_by_layer[target_layer])

            pca = PCA(n_components=2)
            projected = pca.fit_transform(X)

            # Compute cluster statistics
            cluster_stats = {}
            for label in unique_labels:
                mask = np.array([lbl == label for lbl in config.labels])
                points = projected[mask]
                center = np.mean(points, axis=0)
                cluster_stats[label] = {
                    "center": center.tolist(),
                    "count": int(mask.sum()),
                }

            # Create ASCII grid
            ascii_grid = cls._create_ascii_grid(
                projected,
                config.labels,
                symbols,
                config.grid_width,
                config.grid_height,
            )

            layer_results.append(
                {
                    "layer": target_layer,
                    "pca_variance": pca.explained_variance_ratio_.tolist(),
                    "cluster_stats": cluster_stats,
                    "ascii_grid": ascii_grid,
                }
            )

        return ClusteringResult(
            layer_results=layer_results,
            model_id=config.model,
            unique_labels=unique_labels,
            prompt_count=len(config.prompts),
        )

    @staticmethod
    def _create_ascii_grid(
        projected: np.ndarray,
        labels: list[str],
        symbols: dict[str, str],
        width: int,
        height: int,
    ) -> str:
        """Create ASCII visualization of PCA projection."""

        # Get bounds
        x_min, x_max = projected[:, 0].min(), projected[:, 0].max()
        y_min, y_max = projected[:, 1].min(), projected[:, 1].max()

        # Add padding
        x_range = x_max - x_min or 1
        y_range = y_max - y_min or 1
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1

        # Create empty grid
        grid = [[" " for _ in range(width)] for _ in range(height)]

        # Plot points
        for i, (point, label) in enumerate(zip(projected, labels)):
            x = int((point[0] - x_min) / (x_max - x_min) * (width - 1))
            y = int((point[1] - y_min) / (y_max - y_min) * (height - 1))
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            grid[height - 1 - y][x] = symbols.get(label, "?")

        # Build string with border
        lines = ["+" + "-" * width + "+"]
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append("+" + "-" * width + "+")

        # Add legend
        legend = "Legend: " + ", ".join(f"{symbol}={label}" for label, symbol in symbols.items())
        lines.append(legend)

        return "\n".join(lines)


__all__ = [
    "ClusteringConfig",
    "ClusteringResult",
    "ClusteringService",
]
