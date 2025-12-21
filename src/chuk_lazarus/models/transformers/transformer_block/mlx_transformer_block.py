from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models.config import ModelConfig
from chuk_lazarus.models.mlp.mlp_factory import create_mlp
from chuk_lazarus.models.transformers.transformer_block.base_transformer_block import (
    BaseTransformerBlock,
)


class MLXTransformerBlock(BaseTransformerBlock, nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        attention_layer: type[nn.Module],
        norm_layer: Callable[[int, float], nn.Module],
    ):
        nn.Module.__init__(self)  # Initialize MLX nn.Module
        BaseTransformerBlock.__init__(self, config, attention_layer, norm_layer)

        #  set the hidden size
        self.hidden_size = config.hidden_size

        # create the attention layer
        self.self_attn = attention_layer(config)

        # Create MLP layer
        self.mlp = self.create_mlp(config)  # Use the overridden method

        # Ensure `eps` is set to a valid float value, e.g., 1e-6, if not provided
        eps = config.rms_norm_eps if config.rms_norm_eps is not None else 1e-6

        # Create normalization layers
        self.input_layernorm = norm_layer(config.hidden_size, eps=eps)
        self.post_attention_layernorm = norm_layer(config.hidden_size, eps=config.rms_norm_eps)

    def create_mlp(self, config):
        # MLX-specific MLP creation
        return create_mlp(config)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        return self.forward(hidden_states, attention_mask, cache)

    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        # Self-attention
        normed_hidden_states = self.input_layernorm(hidden_states)
        attention_output, cache = self.self_attn(normed_hidden_states, attention_mask, cache)

        # Residual connection for attention output
        hidden_states = hidden_states + attention_output

        # Post-attention normalization and MLP
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed_hidden_states)

        # Residual connection for MLP output
        hidden_states = mx.add(hidden_states, mlp_output)

        # Apply mask if provided
        if attention_mask is not None:
            # Convert to binary mask: 1 for valid tokens, 0 for masked tokens
            binary_attention_mask = mx.where(
                attention_mask < 0, mx.zeros_like(attention_mask), mx.ones_like(attention_mask)
            )

            # Handle different mask shapes:
            # - 2D (batch_size, seq_length): already correct shape
            # - 2D (seq_length, seq_length): take diagonal to get (seq_length,)
            if len(attention_mask.shape) == 2:
                batch_size, seq_len = hidden_states.shape[:2]
                mask_shape = attention_mask.shape
                # If mask is (seq_length, seq_length), take diagonal
                if mask_shape[0] == mask_shape[1] and mask_shape[0] == seq_len:
                    binary_attention_mask = mx.diagonal(binary_attention_mask)
                    binary_attention_mask = mx.expand_dims(binary_attention_mask, axis=0)
                    binary_attention_mask = mx.broadcast_to(
                        binary_attention_mask, (batch_size, seq_len)
                    )

            # Expand to (batch_size, seq_length, 1) for broadcasting with hidden_states
            binary_attention_mask = mx.expand_dims(binary_attention_mask, axis=-1)

            # Apply the mask to hidden_states
            hidden_states = hidden_states * binary_attention_mask

        return hidden_states, cache
