import mlx.core as mx
import mlx.nn as nn
from models.model_config import ModelConfig
from models.llama.llama_layer import LlamaLayer
  
class LlamaModel(nn.Module):
    def __init__(self, config: ModelConfig):
        # initialize
        super().__init__()

        # we store the config for later
        self.config = config

        # set the vocabulary size
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        # set the embeddings layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # now we create the layers 
        self.layers = [
            # create a llama layer for each hidden layer
            LlamaLayer(config=config) for _ in range(config.num_hidden_layers)
        ]

        # set the normalization layer as using RMS normalization
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.norm(h), cache
