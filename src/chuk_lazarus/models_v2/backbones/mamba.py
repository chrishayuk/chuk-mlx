"""
Mamba backbone.

Pure Mamba architecture with SSM blocks instead of attention.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..blocks import MambaBlockWrapper
from ..components.embeddings import create_token_embedding
from ..components.normalization import RMSNorm
from ..core.config import ModelConfig, SSMConfig
from .base import Backbone, BackboneOutput


class MambaBackbone(Backbone):
    """
    Mamba backbone (pure SSM).

    Consists of:
    - Token embeddings
    - Stack of Mamba blocks
    - Final layer normalization

    Unlike transformers, Mamba has O(n) complexity and constant
    memory during inference.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_layers: Number of Mamba blocks
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor
        norm_eps: Normalization epsilon

    Example:
        >>> backbone = MambaBackbone(
        ...     vocab_size=32000,
        ...     d_model=768,
        ...     num_layers=24,
        ... )
        >>> input_ids = mx.array([[1, 2, 3, 4, 5]])
        >>> output = backbone(input_ids)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self._vocab_size = vocab_size
        self._hidden_size = d_model
        self._num_layers = num_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        # Token embeddings
        self.embed_tokens = create_token_embedding(
            vocab_size=vocab_size,
            hidden_size=d_model,
        )

        # Mamba blocks
        self.layers = [
            MambaBlockWrapper(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
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
            attention_mask: Ignored (Mamba is inherently causal)
            cache: Optional list of (conv_state, ssm_state) caches
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
        max_seq_len: int,  # Not used for Mamba
    ) -> list[tuple[mx.array, mx.array]]:
        """Initialize cache for all layers."""
        return [layer.init_cache(batch_size, max_seq_len) for layer in self.layers]

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        self.embed_tokens = embeddings

    @classmethod
    def from_config(cls, config: ModelConfig) -> MambaBackbone:
        """Create from ModelConfig."""
        # Get SSM config if available
        ssm_config = config.ssm_config or SSMConfig()

        return cls(
            vocab_size=config.vocab_size,
            d_model=config.hidden_size,
            num_layers=config.num_hidden_layers,
            d_state=ssm_config.state_size,
            d_conv=ssm_config.conv_kernel,
            expand=ssm_config.expand,
            norm_eps=config.rms_norm_eps,
        )


def create_mamba_backbone(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
) -> MambaBackbone:
    """Factory function for MambaBackbone."""
    return MambaBackbone(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )
