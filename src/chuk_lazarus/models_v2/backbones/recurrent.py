"""
Recurrent backbone.

RNN-based architecture using LSTM, GRU, or MinGRU.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..blocks.recurrent import RecurrentBlockWrapper, RecurrentWithFFN
from ..components.embeddings import create_token_embedding
from ..components.normalization import RMSNorm
from .base import Backbone, BackboneOutput


class RecurrentBackbone(Backbone):
    """
    Recurrent backbone (LSTM/GRU/MinGRU).

    Consists of:
    - Token embeddings
    - Stack of recurrent blocks (with optional FFN)
    - Final layer normalization

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_layers: Number of recurrent blocks
        rnn_type: Type of RNN ("lstm", "gru", "mingru")
        with_ffn: Include FFN sublayer in each block
        intermediate_size: FFN intermediate dimension
        bidirectional: Use bidirectional RNN
        norm_eps: Normalization epsilon

    Example:
        >>> backbone = RecurrentBackbone(
        ...     vocab_size=32000,
        ...     d_model=768,
        ...     num_layers=12,
        ...     rnn_type="mingru",
        ... )
        >>> input_ids = mx.array([[1, 2, 3, 4, 5]])
        >>> output = backbone(input_ids)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        rnn_type: str = "mingru",
        with_ffn: bool = True,
        intermediate_size: int | None = None,
        bidirectional: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self._vocab_size = vocab_size
        self._hidden_size = d_model
        self._num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        # Token embeddings
        self.embed_tokens = create_token_embedding(
            vocab_size=vocab_size,
            hidden_size=d_model,
        )

        # Recurrent blocks
        if with_ffn:
            self.layers = [
                RecurrentWithFFN(
                    d_model=d_model,
                    rnn_type=rnn_type,
                    num_layers=1,  # Each block is one RNN layer
                    intermediate_size=intermediate_size,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        else:
            self.layers = [
                RecurrentBlockWrapper(
                    d_model=d_model,
                    rnn_type=rnn_type,
                    num_layers=1,
                    bidirectional=bidirectional,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]

        # Final normalization
        self.norm = RMSNorm(d_model, eps=norm_eps)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,  # Not used
        cache: list[Any] | None = None,
        output_hidden_states: bool = False,
    ) -> BackboneOutput:
        """
        Forward pass.

        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            attention_mask: Ignored
            cache: Optional list of hidden states
            output_hidden_states: Return all layer hidden states

        Returns:
            BackboneOutput with hidden states and cache
        """
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Track hidden states if requested
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        # Initialize cache list
        new_cache = []

        # Process through layers
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            output = layer(hidden_states, cache=layer_cache)
            hidden_states = output.hidden_states
            new_cache.append(output.cache)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        return BackboneOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            cache=new_cache,
        )

    def init_cache(
        self,
        batch_size: int,
        max_seq_len: int,  # Not used for RNNs
    ) -> list[Any]:
        """Initialize cache for all layers."""
        return [layer.init_cache(batch_size, max_seq_len) for layer in self.layers]

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        self.embed_tokens = embeddings


def create_recurrent_backbone(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    rnn_type: str = "mingru",
    with_ffn: bool = True,
) -> RecurrentBackbone:
    """Factory function for RecurrentBackbone."""
    return RecurrentBackbone(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        rnn_type=rnn_type,
        with_ffn=with_ffn,
    )
