import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, Callable, Type
from core.models.mlp.mlp_factory import create_mlp
from core.models.model_config import ModelConfig
from core.models.transformers.base_transformer_block import BaseTransformerBlock

class MLXTransformerBlock(BaseTransformerBlock, nn.Module):
    def __init__(self, config: ModelConfig, attention_layer: Type[nn.Module], norm_layer: Callable[[int, float], nn.Module]):
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

    def __call__(self, 
                 hidden_states: mx.array, 
                 attention_mask: Optional[mx.array] = None, 
                 cache: Optional[Tuple[mx.array, mx.array]] = None) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        return self.forward(hidden_states, attention_mask, cache)


    def forward(self, 
                hidden_states: mx.array, 
                attention_mask: Optional[mx.array] = None, 
                cache: Optional[Tuple[mx.array, mx.array]] = None) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        
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

        # if attention_mask is not None:
        #     # Check the attention mask values before applying
        #     print(f"Attention mask values before expansion: {attention_mask}")
            
        #     # If the attention mask is 2D, reduce it to seq_length
        #     if len(attention_mask.shape) == 2:
        #         attention_mask = mx.diagonal(attention_mask)  # Shape: (seq_length)

        #     # Expand the mask to include the batch dimension
        #     attention_mask = mx.expand_dims(attention_mask, axis=0)  # Shape: (1, seq_length)
        #     attention_mask = mx.broadcast_to(attention_mask, (hidden_states.shape[0], attention_mask.shape[1]))  # Shape: (batch_size, seq_length)

        #     # Expand along the hidden size dimension to match hidden_states
        #     attention_mask = mx.expand_dims(attention_mask, axis=-1)  # Shape: (batch_size, seq_length, 1)
        #     attention_mask = mx.broadcast_to(attention_mask, hidden_states.shape)  # Shape: (batch_size, seq_length, hidden_size)

        #     # Check the expanded mask
        #     print(f"Expanded mask shape: {attention_mask.shape}")
        #     print(f"Expanded mask values: {attention_mask}")

        #     # Apply the mask to hidden_states
        #     masked_hidden_states = mx.multiply(hidden_states, attention_mask)

        #     # Check the masked hidden states
        #     print(f"Masked hidden states: {masked_hidden_states}")

        #     # Set the masked hidden states back to hidden_states
        #     hidden_states = masked_hidden_states

        # Return the outputs
        return hidden_states, cache






