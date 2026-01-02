"""
Attention analysis and extraction utilities.

Provides tools for extracting and analyzing attention patterns,
including aggregation strategies and focus analysis.

Example:
    >>> from chuk_lazarus.introspection import ModelHooks
    >>> from chuk_lazarus.introspection.attention import AttentionAnalyzer
    >>>
    >>> hooks = ModelHooks(model)
    >>> hooks.capture_layers([0, 4, 8], capture_attention=True)
    >>> hooks.forward(input_ids)
    >>>
    >>> analyzer = AttentionAnalyzer(hooks.state, tokenizer)
    >>> focus = analyzer.get_attention_focus(layer=4, position=-1)
    >>> print(focus.top_attended_tokens)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from .hooks import CapturedState


class AggregationStrategy(str, Enum):
    """Strategy for aggregating attention across heads."""

    MEAN = "mean"
    """Average attention across all heads."""

    MAX = "max"
    """Maximum attention across heads (most confident head)."""

    MIN = "min"
    """Minimum attention across heads (consensus)."""

    HEAD = "head"
    """Select a specific head (requires head_idx parameter)."""


@dataclass
class AttentionFocus:
    """Analysis of what tokens a position attends to."""

    query_position: int
    """The position doing the attending."""

    query_token: str | None
    """The token at the query position."""

    layer_idx: int
    """Which layer this analysis is from."""

    attention_weights: mx.array
    """Full attention distribution over keys. Shape: [seq_len]"""

    tokens: list[str]
    """All tokens in the sequence."""

    @property
    def top_attended_positions(self) -> list[tuple[int, float]]:
        """
        Get positions with highest attention, sorted descending.

        Returns:
            List of (position, attention_weight) tuples
        """
        weights = self.attention_weights.tolist()
        indexed = list(enumerate(weights))
        return sorted(indexed, key=lambda x: x[1], reverse=True)

    @property
    def top_attended_tokens(self) -> list[tuple[str, float]]:
        """
        Get tokens with highest attention, sorted descending.

        Returns:
            List of (token, attention_weight) tuples
        """
        top_pos = self.top_attended_positions
        return [(self.tokens[pos], weight) for pos, weight in top_pos]

    def summary(self, top_k: int = 5) -> str:
        """
        Generate a text summary of attention focus.

        Args:
            top_k: Number of top attended tokens to show

        Returns:
            Human-readable summary string
        """
        lines = [f"Layer {self.layer_idx}, position {self.query_position}"]
        if self.query_token:
            lines[0] += f" ('{self.query_token}')"
        lines.append("Top attended tokens:")
        for token, weight in self.top_attended_tokens[:top_k]:
            lines.append(f"  {weight:.3f}: '{token}'")
        return "\n".join(lines)


@dataclass
class AttentionPattern:
    """Full attention pattern for a layer."""

    layer_idx: int
    """Which layer this is from."""

    weights: mx.array
    """Attention weights. Shape: [batch, heads, query_seq, key_seq]"""

    tokens: list[str] | None
    """Tokens in the sequence, if available."""

    @property
    def num_heads(self) -> int:
        """Number of attention heads."""
        return self.weights.shape[1]

    @property
    def seq_len(self) -> int:
        """Sequence length."""
        return self.weights.shape[2]

    def aggregate(
        self,
        strategy: AggregationStrategy = AggregationStrategy.MEAN,
        head_idx: int | None = None,
    ) -> mx.array:
        """
        Aggregate attention weights across heads.

        Args:
            strategy: How to combine heads
            head_idx: Required if strategy is HEAD

        Returns:
            Aggregated attention of shape [batch, query_seq, key_seq]
        """
        if strategy == AggregationStrategy.MEAN:
            return mx.mean(self.weights, axis=1)
        elif strategy == AggregationStrategy.MAX:
            return mx.max(self.weights, axis=1)
        elif strategy == AggregationStrategy.MIN:
            return mx.min(self.weights, axis=1)
        elif strategy == AggregationStrategy.HEAD:
            if head_idx is None:
                raise ValueError("head_idx required for HEAD strategy")
            return self.weights[:, head_idx, :, :]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def get_head(self, head_idx: int) -> mx.array:
        """
        Get attention pattern for a specific head.

        Args:
            head_idx: Head index

        Returns:
            Attention of shape [batch, query_seq, key_seq]
        """
        return self.weights[:, head_idx, :, :]


class AttentionAnalyzer:
    """
    Analyzer for attention patterns captured by ModelHooks.

    Provides high-level analysis methods for understanding
    what the model attends to.
    """

    def __init__(
        self,
        state: CapturedState,
        tokenizer: Any | None = None,
    ):
        """
        Initialize analyzer with captured state.

        Args:
            state: CapturedState from ModelHooks
            tokenizer: Optional tokenizer for decoding tokens
        """
        self.state = state
        self.tokenizer = tokenizer
        self._tokens: list[str] | None = None

    @property
    def tokens(self) -> list[str]:
        """Get decoded tokens from input_ids."""
        if self._tokens is not None:
            return self._tokens

        if self.state.input_ids is None:
            return []

        if self.tokenizer is None:
            # Return string representations of token IDs
            ids = self.state.input_ids
            if ids.ndim > 1:
                ids = ids[0]
            return [f"[{i}]" for i in ids.tolist()]

        # Decode each token individually
        ids = self.state.input_ids
        if ids.ndim > 1:
            ids = ids[0]

        tokens = []
        for token_id in ids.tolist():
            try:
                token = self.tokenizer.decode([token_id])
                tokens.append(token)
            except Exception:
                tokens.append(f"[{token_id}]")

        self._tokens = tokens
        return tokens

    def get_attention_pattern(self, layer_idx: int) -> AttentionPattern | None:
        """
        Get the full attention pattern for a layer.

        Args:
            layer_idx: Layer index

        Returns:
            AttentionPattern or None if not captured
        """
        if layer_idx not in self.state.attention_weights:
            return None

        return AttentionPattern(
            layer_idx=layer_idx,
            weights=self.state.attention_weights[layer_idx],
            tokens=self.tokens,
        )

    def get_attention_focus(
        self,
        layer_idx: int,
        position: int = -1,
        head_idx: int | None = None,
        aggregation: AggregationStrategy = AggregationStrategy.MEAN,
    ) -> AttentionFocus | None:
        """
        Analyze what a specific position attends to.

        Args:
            layer_idx: Which layer to analyze
            position: Query position (-1 for last)
            head_idx: Specific head, or None to aggregate
            aggregation: How to aggregate across heads

        Returns:
            AttentionFocus analysis or None if layer not captured
        """
        pattern = self.get_attention_pattern(layer_idx)
        if pattern is None:
            return None

        # Aggregate across heads
        if head_idx is not None:
            weights = pattern.get_head(head_idx)[0]  # Remove batch dim
        else:
            weights = pattern.aggregate(aggregation)[0]  # Remove batch dim

        # Handle position
        if position < 0:
            position = weights.shape[0] + position

        # Get attention distribution for this position
        attn_dist = weights[position]

        return AttentionFocus(
            query_position=position,
            query_token=self.tokens[position] if self.tokens else None,
            layer_idx=layer_idx,
            attention_weights=attn_dist,
            tokens=self.tokens,
        )

    def find_high_attention_pairs(
        self,
        layer_idx: int,
        threshold: float = 0.1,
        aggregation: AggregationStrategy = AggregationStrategy.MEAN,
    ) -> list[tuple[int, int, float]]:
        """
        Find query-key pairs with high attention.

        Args:
            layer_idx: Which layer to analyze
            threshold: Minimum attention weight
            aggregation: How to aggregate across heads

        Returns:
            List of (query_pos, key_pos, attention_weight) tuples
        """
        pattern = self.get_attention_pattern(layer_idx)
        if pattern is None:
            return []

        weights = pattern.aggregate(aggregation)[0]  # Remove batch dim
        weights_np = weights.tolist()

        pairs = []
        for q in range(len(weights_np)):
            for k in range(len(weights_np[q])):
                if weights_np[q][k] >= threshold:
                    pairs.append((q, k, weights_np[q][k]))

        return sorted(pairs, key=lambda x: x[2], reverse=True)

    def get_attention_entropy(
        self,
        layer_idx: int,
        aggregation: AggregationStrategy = AggregationStrategy.MEAN,
    ) -> mx.array | None:
        """
        Compute attention entropy per query position.

        High entropy = diffuse attention (attending to many tokens).
        Low entropy = focused attention (attending to few tokens).

        Args:
            layer_idx: Which layer to analyze
            aggregation: How to aggregate across heads

        Returns:
            Entropy values of shape [seq_len] or None
        """
        pattern = self.get_attention_pattern(layer_idx)
        if pattern is None:
            return None

        weights = pattern.aggregate(aggregation)[0]  # [seq, seq]

        # Compute entropy: -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        log_weights = mx.log(weights + eps)
        entropy = -mx.sum(weights * log_weights, axis=-1)

        return entropy

    def compare_layers(
        self,
        position: int = -1,
        aggregation: AggregationStrategy = AggregationStrategy.MEAN,
    ) -> dict[int, AttentionFocus]:
        """
        Compare attention focus across all captured layers.

        Args:
            position: Query position to analyze
            aggregation: How to aggregate across heads

        Returns:
            Dict mapping layer_idx -> AttentionFocus
        """
        result = {}
        for layer_idx in self.state.attention_weights:
            focus = self.get_attention_focus(
                layer_idx=layer_idx,
                position=position,
                aggregation=aggregation,
            )
            if focus is not None:
                result[layer_idx] = focus
        return result


def extract_attention_weights(
    model: nn.Module,
    input_ids: mx.array,
    layers: list[int] | None = None,
) -> dict[int, mx.array]:
    """
    Convenience function to extract attention weights from a model.

    This is a simpler interface when you just want attention weights
    without the full ModelHooks infrastructure.

    Note: This requires the model to return attention weights in its
    forward pass. Not all models support this.

    Args:
        model: The model
        input_ids: Input token IDs
        layers: Which layers to capture (None = all)

    Returns:
        Dict mapping layer_idx -> attention weights
    """
    from .hooks import CaptureConfig, ModelHooks

    hooks = ModelHooks(model)
    hooks.configure(
        CaptureConfig(
            layers=layers if layers is not None else "all",
            capture_hidden_states=False,
            capture_attention_weights=True,
        )
    )
    hooks.forward(input_ids)
    return dict(hooks.state.attention_weights)
