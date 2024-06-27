import mlx.nn as nn
from models.architectures.starcoder2.starcoder2_attention import StarCoder2Attention
from models.architectures.starcoder2.starcoder2_transformer_block import StarCoder2TransformerBlock
from models.model_config import ModelConfig
from models.architectures.transformer_base_model import TransformerBaseModel

class StarCoder2Model(TransformerBaseModel):
    def __init__(self, config: ModelConfig):
        # call the constructor of the base class
        super().__init__(
            config, 
            attention_layer=StarCoder2Attention,
            norm_layer=lambda hidden_size, eps: nn.LayerNorm(hidden_size, eps=eps)
        )
        
        # set the embeddings layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # set the dropout
        # Use a default value if embedding_dropout is not in the config
        embedding_dropout = getattr(config, 'embedding_dropout', 0.1)  # Default to 0.1 if not specified
        self.dropout = nn.Dropout(embedding_dropout)

        # construct the layers
        self.layers = [StarCoder2TransformerBlock(config) for _ in range(config.num_hidden_layers)]

        # create the final normalization layer
        # Use rms_norm_eps if available, otherwise default to a small value
        norm_eps = getattr(config, 'rms_norm_eps', 1e-6)
        self.norm = nn.LayerNorm(config.hidden_size, eps=norm_eps)

    def __call__(self, inputs, cache=None):
        # Embed tokens
        h = self.embed_tokens(inputs)

        # set the drop out
        h = self.dropout(h)

        # Initialize cache if not provided
        if cache is None:
            cache = [None] * len(self.layers)

        # Process input through each transformer layer
        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, cache=cache[e])

        # Apply final normalization
        return self.norm(h), cache