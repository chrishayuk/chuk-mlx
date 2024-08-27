import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, Callable, Type
from core.models.mlp.mlp_factory import create_mlp
from core.models.model_config import ModelConfig

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, attention_layer: Type[nn.Module], norm_layer: Callable[[int, float], nn.Module]):
        # call the constructor
        super().__init__()

        #  set the hidden size
        self.hidden_size = config.hidden_size

        # create the attention layer
        self.self_attn = attention_layer(config)

        # create the MLP layer
        self.mlp = create_mlp(config)

        # Ensure `eps` is set to a valid float value, e.g., 1e-6, if not provided
        eps = config.rms_norm_eps if config.rms_norm_eps is not None else 1e-6
        
        # Use the provided norm_layer function to create normalization layers
        self.input_layernorm = norm_layer(config.hidden_size, eps=eps)
        self.post_attention_layernorm = norm_layer(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(   
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        # Self-attention
        normed_hidden_states = self.input_layernorm(hidden_states)
        attention_output, cache = self.self_attn(normed_hidden_states, attention_mask, cache)
        hidden_states = mx.add(hidden_states, attention_output)  # Residual connection

        # Execute our MLP, performing normalization on the input
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed_hidden_states)
        hidden_states = mx.add(hidden_states, mlp_output)  # Residual connection

        # Return the updated hidden states
        return hidden_states, cache