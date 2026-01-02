"""
Logit lens implementation for layer-by-layer prediction analysis.

The "logit lens" technique projects intermediate hidden states to
vocabulary logits, revealing how predictions evolve through layers.

Key insight: For tool routing, if the correct tool token appears
early (layers 2-4), a shallow model might suffice. If it only
emerges in final layers, depth is necessary.

Example:
    >>> from chuk_lazarus.introspection import ModelHooks
    >>> from chuk_lazarus.introspection.logit_lens import LogitLens
    >>>
    >>> hooks = ModelHooks(model)
    >>> hooks.capture_layers("all")
    >>> hooks.forward(input_ids)
    >>>
    >>> lens = LogitLens(hooks, tokenizer)
    >>> predictions = lens.get_layer_predictions(top_k=5)
    >>> lens.print_evolution()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from .hooks import ModelHooks


@dataclass
class LayerPrediction:
    """Predictions at a specific layer."""

    layer_idx: int
    """Layer index (0-indexed)."""

    position: int
    """Sequence position these predictions are for."""

    top_tokens: list[str]
    """Top-k predicted tokens."""

    top_probs: list[float]
    """Probabilities for top-k tokens."""

    top_ids: list[int]
    """Token IDs for top-k tokens."""

    target_token: str | None = None
    """Target token we're tracking (if any)."""

    target_rank: int | None = None
    """Rank of target token (1-indexed, None if not in top-k)."""

    target_prob: float | None = None
    """Probability of target token."""

    def __repr__(self) -> str:
        top_str = ", ".join(
            f"'{t}':{p:.3f}" for t, p in zip(self.top_tokens[:3], self.top_probs[:3])
        )
        return f"Layer {self.layer_idx}: [{top_str}]"


@dataclass
class TokenEvolution:
    """Track a specific token's probability across layers."""

    token: str
    """The token being tracked."""

    token_id: int
    """Token ID."""

    layers: list[int]
    """Layer indices."""

    probabilities: list[float]
    """Probability at each layer."""

    ranks: list[int | None]
    """Rank at each layer (1-indexed, None if not in top-k)."""

    @property
    def emergence_layer(self) -> int | None:
        """
        First layer where token appears in top-1.

        Returns:
            Layer index or None if never top-1
        """
        for layer, rank in zip(self.layers, self.ranks):
            if rank == 1:
                return layer
        return None

    @property
    def first_significant_layer(self, threshold: float = 0.1) -> int | None:
        """
        First layer where token probability exceeds threshold.

        Args:
            threshold: Probability threshold

        Returns:
            Layer index or None if never exceeds threshold
        """
        for layer, prob in zip(self.layers, self.probabilities):
            if prob >= threshold:
                return layer
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "token": self.token,
            "token_id": self.token_id,
            "layers": self.layers,
            "probabilities": self.probabilities,
            "ranks": self.ranks,
            "emergence_layer": self.emergence_layer,
        }


class LogitLens:
    """
    Logit lens analysis for understanding prediction evolution.

    Projects hidden states from each layer to vocabulary logits,
    showing how the model's predictions develop through its depth.
    """

    def __init__(
        self,
        hooks: ModelHooks,
        tokenizer: Any | None = None,
    ):
        """
        Initialize logit lens.

        Args:
            hooks: ModelHooks with captured states
            tokenizer: Tokenizer for decoding
        """
        self.hooks = hooks
        self.tokenizer = tokenizer

    def get_layer_predictions(
        self,
        position: int = -1,
        top_k: int = 10,
        normalize: bool = True,
    ) -> list[LayerPrediction]:
        """
        Get top-k predictions at each captured layer.

        Args:
            position: Sequence position (-1 for last)
            top_k: Number of top predictions per layer
            normalize: Whether to apply final norm before projection

        Returns:
            List of LayerPrediction for each captured layer
        """
        predictions = []

        for layer_idx in sorted(self.hooks.state.hidden_states.keys()):
            logits = self.hooks.get_layer_logits(layer_idx, normalize=normalize)
            if logits is None:
                continue

            # Get logits for specific position
            if logits.ndim == 3:
                pos_logits = logits[0, position, :]  # [vocab]
            else:
                pos_logits = logits[position, :]  # [vocab]

            # Get probabilities
            probs = mx.softmax(pos_logits, axis=-1)

            # Get top-k
            # MLX doesn't have topk, so we sort
            sorted_indices = mx.argsort(probs)[::-1][:top_k]
            top_ids = sorted_indices.tolist()
            top_probs = probs[sorted_indices].tolist()

            # Decode tokens
            if self.tokenizer is not None:
                top_tokens = [self.tokenizer.decode([tid]) for tid in top_ids]
            else:
                top_tokens = [f"[{tid}]" for tid in top_ids]

            predictions.append(
                LayerPrediction(
                    layer_idx=layer_idx,
                    position=position if position >= 0 else logits.shape[-2] + position,
                    top_tokens=top_tokens,
                    top_probs=top_probs,
                    top_ids=top_ids,
                )
            )

        return predictions

    def track_token(
        self,
        token: str | int,
        position: int = -1,
        normalize: bool = True,
        top_k_for_rank: int = 100,
    ) -> TokenEvolution:
        """
        Track a specific token's probability evolution across layers.

        Args:
            token: Token string or ID to track
            position: Sequence position
            normalize: Whether to apply final norm
            top_k_for_rank: Consider top-k for ranking

        Returns:
            TokenEvolution showing probability/rank at each layer
        """
        # Resolve token to ID
        if isinstance(token, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer required to track string token")

            # Try to find exact match in vocabulary first
            # This handles cases where "!" could be token ID 29991 vs "▁!" being 1738
            token_id = None
            if hasattr(self.tokenizer, "get_vocab"):
                vocab = self.tokenizer.get_vocab()
                if token in vocab:
                    token_id = vocab[token]

            # Fall back to encoding if not found in vocab
            if token_id is None:
                try:
                    token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                except TypeError:
                    token_ids = self.tokenizer.encode(token)
                if not token_ids:
                    raise ValueError(f"Token '{token}' not in vocabulary")
                if len(token_ids) > 1:
                    # Multi-token string - warn user and track first token only
                    first_token = self.tokenizer.decode([token_ids[0]])
                    import warnings

                    warnings.warn(
                        f"'{token}' is not a single token in this vocabulary. "
                        f"It encodes to {len(token_ids)} tokens. "
                        f"Tracking first token only: '{first_token}' (id={token_ids[0]})",
                        stacklevel=2,
                    )
                token_id = token_ids[0]

            token_str = token
        else:
            token_id = token
            token_str = self.tokenizer.decode([token_id]) if self.tokenizer else f"[{token_id}]"

        layers = []
        probabilities = []
        ranks = []

        for layer_idx in sorted(self.hooks.state.hidden_states.keys()):
            logits = self.hooks.get_layer_logits(layer_idx, normalize=normalize)
            if logits is None:
                continue

            # Get logits for specific position
            if logits.ndim == 3:
                pos_logits = logits[0, position, :]
            else:
                pos_logits = logits[position, :]

            probs = mx.softmax(pos_logits, axis=-1)

            # Get probability of target token
            target_prob = float(probs[token_id])

            # Get rank
            sorted_indices = mx.argsort(probs)[::-1][:top_k_for_rank].tolist()
            if token_id in sorted_indices:
                rank = sorted_indices.index(token_id) + 1
            else:
                rank = None

            layers.append(layer_idx)
            probabilities.append(target_prob)
            ranks.append(rank)

        return TokenEvolution(
            token=token_str,
            token_id=token_id,
            layers=layers,
            probabilities=probabilities,
            ranks=ranks,
        )

    def find_emergence_point(
        self,
        token: str | int,
        position: int = -1,
        threshold: float = 0.5,
    ) -> int | None:
        """
        Find the layer where a token's probability crosses a threshold.

        Useful for determining "when" a routing decision happens.

        Args:
            token: Token to track
            position: Sequence position
            threshold: Probability threshold

        Returns:
            Layer index where threshold is first exceeded, or None
        """
        evolution = self.track_token(token, position)
        for layer, prob in zip(evolution.layers, evolution.probabilities):
            if prob >= threshold:
                return layer
        return None

    def compare_tokens(
        self,
        tokens: list[str | int],
        position: int = -1,
    ) -> dict[str, TokenEvolution]:
        """
        Compare probability evolution of multiple tokens.

        Useful for seeing when one tool token "wins" over others.

        Args:
            tokens: List of tokens to compare
            position: Sequence position

        Returns:
            Dict mapping token string -> TokenEvolution
        """
        return {
            (tok if isinstance(tok, str) else f"[{tok}]"): self.track_token(tok, position)
            for tok in tokens
        }

    def print_evolution(
        self,
        position: int = -1,
        top_k: int = 5,
    ) -> None:
        """
        Print a human-readable evolution table.

        Args:
            position: Sequence position
            top_k: Number of top tokens to show
        """
        predictions = self.get_layer_predictions(position=position, top_k=top_k)

        if not predictions:
            print("No layers captured")
            return

        print(f"\nLogit Lens (position {predictions[0].position})")
        print("=" * 60)

        for pred in predictions:
            tokens_str = " | ".join(f"{t}:{p:.2f}" for t, p in zip(pred.top_tokens, pred.top_probs))
            print(f"Layer {pred.layer_idx:2d}: {tokens_str}")

    def to_dict(
        self,
        position: int = -1,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Export predictions to dictionary for JSON serialization.

        Args:
            position: Sequence position
            top_k: Number of top predictions

        Returns:
            Serializable dictionary
        """
        predictions = self.get_layer_predictions(position=position, top_k=top_k)

        return {
            "position": predictions[0].position if predictions else position,
            "layers": [
                {
                    "layer": p.layer_idx,
                    "top_tokens": p.top_tokens,
                    "top_probs": p.top_probs,
                    "top_ids": p.top_ids,
                }
                for p in predictions
            ],
        }


def run_logit_lens(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    track_token: str | None = None,
    layers: list[int] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Convenience function to run logit lens on a prompt.

    Args:
        model: The model
        tokenizer: Tokenizer
        prompt: Input prompt
        track_token: Optional token to specifically track
        layers: Which layers to capture (None = all)
        top_k: Number of top predictions per layer

    Returns:
        Dictionary with predictions and optional token evolution
    """
    from .hooks import CaptureConfig, ModelHooks

    # Tokenize
    input_ids = mx.array(tokenizer.encode(prompt))[None, :]

    # Setup hooks
    hooks = ModelHooks(model)
    hooks.configure(
        CaptureConfig(
            layers=layers if layers is not None else "all",
            capture_hidden_states=True,
            positions="last",
        )
    )
    hooks.forward(input_ids)

    # Analyze
    lens = LogitLens(hooks, tokenizer)
    result = lens.to_dict(top_k=top_k)

    # Track specific token if requested
    if track_token is not None:
        evolution = lens.track_token(track_token)
        result["tracked_token"] = evolution.to_dict()

    return result
