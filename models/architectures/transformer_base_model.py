import mlx.core as mx
import mlx.nn as nn
from models.architectures.transformer_block import TransformerBlock
from models.model_config import ModelConfig

class TransformerBaseModel(nn.Module):
    def __init__(self, config: ModelConfig, attention_layer, norm_layer):
        super().__init__()
        self.args = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.layers = [
            TransformerBlock(
                config=config,
                attention_layer=attention_layer,
                norm_layer=norm_layer
            ) for _ in range(config.num_hidden_layers)
        ]
        
        self.norm = norm_layer(config.hidden_size, eps=config.rms_norm_eps)

    def embed_inputs(self, inputs):
        # embed
        return self.embed_tokens(inputs)

    def scale_embeddings(self, embeddings):
        # No scaling by default
        return embeddings  

    def __call__(self, inputs: mx.array, cache=None):
        # Embed and scale input tokens
        h = self.embed_inputs(inputs)
        h = self.scale_embeddings(h)

        # Create causal mask for self-attention
        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        # Initialize cache if not provided
        if cache is None:
            cache = [None] * len(self.layers)

        # Process input through each transformer layer
        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        # Apply final normalization
        return self.norm(h), cache