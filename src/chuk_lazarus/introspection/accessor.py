"""Model accessor for unified access to model components.

Provides a consistent interface to access model layers, embeddings,
and other components regardless of the specific model architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import mlx.core as mx


@runtime_checkable
class HasLayers(Protocol):
    """Protocol for models with layers attribute."""

    layers: Any


@runtime_checkable
class HasModel(Protocol):
    """Protocol for models with nested model attribute."""

    model: Any


@runtime_checkable
class HasEmbedTokens(Protocol):
    """Protocol for models with embed_tokens attribute."""

    embed_tokens: Any


@runtime_checkable
class HasNorm(Protocol):
    """Protocol for models with norm attribute."""

    norm: Any


@runtime_checkable
class HasLMHead(Protocol):
    """Protocol for models with lm_head attribute."""

    lm_head: Any


class ModelAccessor(BaseModel):
    """Unified accessor for model components.

    Handles different model architectures by providing a consistent
    interface to access layers, embeddings, and other components.

    Example:
        >>> accessor = ModelAccessor(model=model, config=config)
        >>> layers = accessor.layers
        >>> embed = accessor.embed_tokens
        >>> for layer in layers:
        ...     output = layer(hidden_states)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any = Field(description="The neural network model")
    config: Any = Field(default=None, description="Optional model configuration")

    @property
    def layers(self) -> list:
        """Get the transformer layers."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        elif hasattr(self.model, "layers"):
            return list(self.model.layers)
        raise AttributeError("Cannot find layers in model")

    @property
    def num_layers(self) -> int:
        """Get the number of layers."""
        return len(self.layers)

    @property
    def embed_tokens(self) -> Any:
        """Get the token embedding layer."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens
        elif hasattr(self.model, "embed_tokens"):
            return self.model.embed_tokens
        raise AttributeError("Cannot find embed_tokens in model")

    @property
    def norm(self) -> Any | None:
        """Get the final normalization layer."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            return self.model.model.norm
        elif hasattr(self.model, "norm"):
            return self.model.norm
        return None

    @property
    def lm_head(self) -> Any | None:
        """Get the language model head."""
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        return None

    @property
    def embedding_scale(self) -> float | None:
        """Get the embedding scale factor if configured."""
        if self.config is not None:
            return getattr(self.config, "embedding_scale", None)
        return None

    @property
    def hidden_size(self) -> int:
        """Get the hidden dimension size."""
        if self.config is not None:
            if hasattr(self.config, "hidden_size"):
                return self.config.hidden_size
            if hasattr(self.config, "d_model"):
                return self.config.d_model
        # Try to infer from embeddings
        embed = self.embed_tokens
        if hasattr(embed, "weight"):
            return embed.weight.shape[-1]
        raise AttributeError("Cannot determine hidden_size")

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        if self.config is not None:
            if hasattr(self.config, "vocab_size"):
                return self.config.vocab_size
        # Try to infer from embeddings
        embed = self.embed_tokens
        if hasattr(embed, "weight"):
            return embed.weight.shape[0]
        raise AttributeError("Cannot determine vocab_size")

    @property
    def has_tied_embeddings(self) -> bool:
        """Check if embeddings are tied with lm_head."""
        lm_head = self.lm_head
        if lm_head is None:
            return True  # Tied by default when no explicit head

        embed = self.embed_tokens
        if hasattr(lm_head, "weight") and hasattr(embed, "weight"):
            # Check if they share the same weight tensor
            try:
                import mlx.core as mx

                return mx.array_equal(lm_head.weight, embed.weight)
            except Exception:
                return False
        return False

    def get_layer(self, idx: int) -> Any:
        """Get a specific layer by index."""
        layers = self.layers
        if idx < 0:
            idx = len(layers) + idx
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"Layer index {idx} out of range [0, {len(layers)})")
        return layers[idx]

    def set_layer(self, idx: int, layer: Any) -> None:
        """Set a specific layer by index."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.model.model.layers[idx] = layer
        elif hasattr(self.model, "layers"):
            self.model.layers[idx] = layer
        else:
            raise AttributeError("Cannot set layer in model")

    def embed(self, input_ids: mx.array) -> mx.array:
        """Embed input tokens with optional scaling."""

        h = self.embed_tokens(input_ids)
        scale = self.embedding_scale
        if scale is not None:
            h = h * scale
        return h

    def apply_norm_and_head(self, hidden_states: mx.array) -> mx.array:
        """Apply final norm and lm_head to get logits."""
        h = hidden_states
        if self.norm is not None:
            h = self.norm(h)

        lm_head = self.lm_head
        if lm_head is not None:
            outputs = lm_head(h)
            # Handle HeadOutput wrapper vs raw logits
            if hasattr(outputs, "logits"):
                return outputs.logits
            return outputs
        else:
            # Tied embeddings
            return h @ self.embed_tokens.weight.T

    def create_causal_mask(self, seq_len: int, dtype: mx.Dtype | None = None) -> mx.array:
        """Create a causal attention mask."""
        import mlx.nn as nn

        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        if dtype is not None:
            mask = mask.astype(dtype)
        return mask


class AsyncModelAccessor(ModelAccessor):
    """Async-compatible model accessor.

    Provides the same interface as ModelAccessor but designed
    for use in async contexts.
    """

    async def forward_through_layers(
        self,
        input_ids: mx.array,
        layers: list[int] | None = None,
        capture_hidden_states: bool = True,
    ) -> dict[int, mx.array]:
        """Run forward pass and capture hidden states at specified layers.

        Args:
            input_ids: Input token IDs
            layers: List of layer indices to capture (None = all)
            capture_hidden_states: Whether to capture hidden states

        Returns:
            Dictionary mapping layer index to hidden states
        """
        import mlx.core as mx

        h = self.embed(input_ids)
        mask = self.create_causal_mask(input_ids.shape[1], h.dtype)

        all_layers = self.layers
        if layers is None:
            layers = list(range(len(all_layers)))

        captured: dict[int, mx.array] = {}

        for idx, layer in enumerate(all_layers):
            try:
                out = layer(h, mask=mask)
            except TypeError:
                out = layer(h)

            # Handle different return types
            if hasattr(out, "hidden_states"):
                h = out.hidden_states
            elif isinstance(out, tuple):
                h = out[0]
            else:
                h = out

            if capture_hidden_states and idx in layers:
                captured[idx] = h

            # Evaluate periodically to avoid memory buildup
            if idx % 4 == 0:
                mx.eval(h)

        return captured
