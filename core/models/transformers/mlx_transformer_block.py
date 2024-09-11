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

        if attention_mask is not None:
            #print(f"Attention mask values before expansion: {attention_mask}")
            
            # Convert to a binary mask: 1 for valid tokens, 0 for masked tokens
            binary_attention_mask = mx.where(attention_mask < 0, mx.zeros_like(attention_mask), mx.ones_like(attention_mask))

            # Reduce the mask to (batch_size, seq_length)
            if len(attention_mask.shape) == 2:  # Handling a 2D (seq_length, seq_length) mask
                binary_attention_mask = mx.diagonal(binary_attention_mask)  # Shape: (seq_length)

            # Ensure the binary mask is aligned with the hidden states
            binary_attention_mask = mx.expand_dims(binary_attention_mask, axis=0)  # Shape: (1, seq_length)
            binary_attention_mask = mx.broadcast_to(binary_attention_mask, (hidden_states.shape[0], binary_attention_mask.shape[1]))  # Shape: (batch_size, seq_length)

            # Expand along the hidden size dimension to match hidden_states (batch_size, seq_length, hidden_size)
            binary_attention_mask = mx.expand_dims(binary_attention_mask, axis=-1)  # Shape: (batch_size, seq_length, 1)
            binary_attention_mask = mx.broadcast_to(binary_attention_mask, hidden_states.shape)  # Shape: (batch_size, seq_length, hidden_size)

            # Apply the binary mask to hidden_states
            hidden_states = mx.multiply(hidden_states, binary_attention_mask)

            # Print for debugging
            #print(f"Binary attention mask applied: {binary_attention_mask}")
            #print(f"Hidden states after applying binary mask: {hidden_states}")

        return hidden_states, cache







