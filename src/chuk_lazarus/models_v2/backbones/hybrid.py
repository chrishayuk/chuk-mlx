"""
Hybrid backbone.

Combines different block types (attention, SSM, RNN) in a single backbone.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..blocks import HybridBlock, MambaBlockWrapper, TransformerBlock
from ..blocks.hybrid import AlternatingHybrid
from ..components.embeddings import create_token_embedding
from ..components.normalization import RMSNorm
from ..core.enums import BlockType, HybridCombineMode, HybridMixStrategy
from .base import Backbone, BackboneOutput


class HybridBackbone(Backbone):
    """
    Hybrid backbone combining different block types.

    Supports multiple mixing strategies (see HybridMixStrategy enum):
    - ALTERNATING: Alternate between attention and Mamba layers
    - INTERLEAVED: Every Nth layer is attention, rest are Mamba
    - PARALLEL: Each layer has both attention and Mamba in parallel

    Args:
        vocab_size: Vocabulary size
        hidden_size: Model dimension
        num_layers: Number of blocks
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads
        d_state: Mamba state dimension
        intermediate_size: FFN intermediate dimension
        mix_strategy: How to mix block types
        attention_ratio: Fraction of layers using attention (for interleaved)
        norm_eps: Normalization epsilon

    Example:
        >>> backbone = HybridBackbone(
        ...     vocab_size=32000,
        ...     hidden_size=768,
        ...     num_layers=24,
        ...     num_heads=12,
        ...     mix_strategy=HybridMixStrategy.ALTERNATING,
        ... )
        >>> input_ids = mx.array([[1, 2, 3, 4, 5]])
        >>> output = backbone(input_ids)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        intermediate_size: int | None = None,
        mix_strategy: HybridMixStrategy | str = HybridMixStrategy.ALTERNATING,
        attention_ratio: float = 0.5,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        # Normalize mix_strategy to enum
        self.mix_strategy = (
            HybridMixStrategy(mix_strategy) if isinstance(mix_strategy, str) else mix_strategy
        )

        num_kv_heads = num_kv_heads or num_heads
        intermediate_size = intermediate_size or hidden_size * 4

        # Token embeddings
        self.embed_tokens = create_token_embedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )

        # Create layers based on mix strategy
        self.layers = []
        self.layer_types = []  # Track what type each layer is

        for i in range(num_layers):
            if self.mix_strategy == HybridMixStrategy.ALTERNATING:
                # Alternate: even = attention, odd = Mamba
                block = AlternatingHybrid(
                    d_model=hidden_size,
                    layer_idx=i,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    d_state=d_state,
                    d_conv=d_conv,
                    intermediate_size=intermediate_size,
                    use_attention_first=True,
                    norm_eps=norm_eps,
                )
                self.layer_types.append(BlockType.TRANSFORMER if i % 2 == 0 else BlockType.MAMBA)

            elif self.mix_strategy == HybridMixStrategy.INTERLEAVED:
                # Place attention layers at regular intervals
                attention_interval = (
                    int(1 / attention_ratio) if attention_ratio > 0 else num_layers + 1
                )
                use_attention = i % attention_interval == 0

                if use_attention:
                    block = TransformerBlock(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        intermediate_size=intermediate_size,
                        norm_eps=norm_eps,
                    )
                    self.layer_types.append(BlockType.TRANSFORMER)
                else:
                    block = MambaBlockWrapper(
                        d_model=hidden_size,
                        d_state=d_state,
                        d_conv=d_conv,
                        norm_eps=norm_eps,
                    )
                    self.layer_types.append(BlockType.MAMBA)

            elif self.mix_strategy == HybridMixStrategy.PARALLEL:
                # Each layer has both attention and Mamba in parallel
                block = HybridBlock(
                    d_model=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    d_state=d_state,
                    d_conv=d_conv,
                    intermediate_size=intermediate_size,
                    combine_mode=HybridCombineMode.ADD,
                    norm_eps=norm_eps,
                )
                self.layer_types.append(BlockType.HYBRID)

            else:
                raise ValueError(f"Unknown mix_strategy: {self.mix_strategy}")

            self.layers.append(block)

        # Final normalization
        self.norm = RMSNorm(hidden_size, eps=norm_eps)

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
        attention_mask: mx.array | None = None,
        cache: list[Any] | None = None,
        output_hidden_states: bool = False,
    ) -> BackboneOutput:
        """
        Forward pass.

        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            attention_mask: Optional attention mask (used by attention layers)
            cache: Optional list of per-layer caches
            output_hidden_states: Return all layer hidden states

        Returns:
            BackboneOutput with hidden states and cache
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Create causal mask for attention layers
        if attention_mask is None:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(hidden_states.dtype)
        else:
            mask = attention_mask

        # Track hidden states if requested
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        # Initialize cache list
        new_cache = []

        # Process through layers
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            layer_type = self.layer_types[i]

            # Pass mask only to attention-containing layers
            if layer_type in (BlockType.TRANSFORMER, BlockType.HYBRID):
                output = layer(hidden_states, mask=mask, cache=layer_cache)
            else:
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
        max_seq_len: int,
    ) -> list[Any]:
        """Initialize cache for all layers."""
        return [layer.init_cache(batch_size, max_seq_len) for layer in self.layers]

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        self.embed_tokens = embeddings


def create_hybrid_backbone(
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    mix_strategy: HybridMixStrategy | str = HybridMixStrategy.ALTERNATING,
) -> HybridBackbone:
    """Factory function for HybridBackbone."""
    return HybridBackbone(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        mix_strategy=mix_strategy,
    )
