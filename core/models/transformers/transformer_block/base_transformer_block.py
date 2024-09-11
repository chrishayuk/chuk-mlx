from typing import Callable, Type

class BaseTransformerBlock:
    def __init__(self, config, attention_layer: Type, norm_layer: Callable[[int, float], Type]):
        # Set the hidden size
        self.hidden_size = config.hidden_size

        # Create the attention layer
        self.self_attn = attention_layer(config)

        # Create the MLP layer
        self.mlp = self.create_mlp(config)  # To be implemented in subclasses

        # Ensure `eps` is set to a valid float value, e.g., 1e-6, if not provided
        eps = config.rms_norm_eps if config.rms_norm_eps is not None else 1e-6

        # Use the provided norm_layer function to create normalization layers
        self.input_layernorm = norm_layer(config.hidden_size, eps=eps)
        self.post_attention_layernorm = norm_layer(config.hidden_size, eps=config.rms_norm_eps)

    def create_mlp(self, config):
        # Placeholder to be implemented by subclasses
        raise NotImplementedError

    def forward(self, hidden_states, attention_mask=None, cache=None):
        # Placeholder for the forward pass
        raise NotImplementedError
