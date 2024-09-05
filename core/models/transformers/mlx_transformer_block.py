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

        # Create MLP layer
        self.mlp = self.create_mlp(config)  # Use the overridden method

        # Ensure `eps` is set to a valid float value, e.g., 1e-6, if not provided
        eps = config.rms_norm_eps if config.rms_norm_eps is not None else 1e-6
        
        # Create normalization layers
        self.input_layernorm = norm_layer(config.hidden_size, eps=eps)
        self.post_attention_layernorm = norm_layer(config.hidden_size, eps=config.rms_norm_eps)

    def create_mlp(self, config):
        # MLX-specific MLP creation
        return create_mlp(config)  # Assuming this is the function that creates the MLP

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
        
        # Apply the mask again after the residual connection
        if attention_mask is not None:
            # Expand the mask to match the shape of hidden_states
            attention_mask_expanded = mx.expand_dims(attention_mask, axis=-1)
            attention_mask_expanded = mx.broadcast_to(attention_mask_expanded, hidden_states.shape)
            
            # Apply the mask: set values to zero where mask is zero
            hidden_states = mx.multiply(hidden_states, attention_mask_expanded)

        # Post-attention normalization and MLP
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed_hidden_states)
        
        # Residual connection for MLP output
        hidden_states = hidden_states + mlp_output
        
        # Apply the mask again after the MLP residual connection
        if attention_mask is not None:
            hidden_states = mx.multiply(hidden_states, attention_mask_expanded)

        # Return the outputs
        return hidden_states, cache


