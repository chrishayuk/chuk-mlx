import mlx.core as mx
import mlx.nn as nn
from core.models.architectures.transformer_block import TransformerBlock
from core.models.model_config import ModelConfig

class TransformerBaseModel(nn.Module):
    def __init__(self, config: ModelConfig, attention_layer, norm_layer):
        super().__init__()

        # Define the model parameters
        self.args = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Construct the layers
        self.layers = [
            TransformerBlock(
                config=config,
                attention_layer=attention_layer,
                norm_layer=norm_layer
            ) for _ in range(config.num_hidden_layers)
        ]
        
        # Create the final normalization layer
        self.norm = norm_layer(config.hidden_size, eps=config.rms_norm_eps)
        self._mask_cache = {}

    def embed_inputs(self, inputs):
        # Embed the inputs
        return self.embed_tokens(inputs)

    def scale_embeddings(self, embeddings):
        # No scaling by default
        return embeddings  

    def get_mask(self, seq_length, dtype):
        # Centralized mask caching
        if seq_length not in self._mask_cache:
            self._mask_cache[seq_length] = nn.MultiHeadAttention.create_additive_causal_mask(seq_length).astype(dtype)
        return self._mask_cache[seq_length]

    def __call__(self, inputs: mx.array, cache=None):
        # Embed and scale input tokens
        h = self.embed_inputs(inputs)
        h = self.scale_embeddings(h)

        # Create causal mask for self-attention
        mask = None
        if h.shape[1] > 1:
            mask = self.get_mask(h.shape[1], h.dtype)

        # Initialize cache if not provided
        if cache is None:
            cache = [None] * len(self.layers)

        # Process input through each transformer layer
        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        # Apply final normalization
        return self.norm(h), cache
