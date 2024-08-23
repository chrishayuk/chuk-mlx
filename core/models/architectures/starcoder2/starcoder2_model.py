import mlx.nn as nn
import mlx.core as mx
from core.models.model_config import ModelConfig
from core.models.architectures.transformer_base_model import TransformerBaseModel
from core.models.architectures.starcoder2.starcoder2_transformer_block import StarCoder2TransformerBlock
from core.models.architectures.starcoder2.starcoder2_attention import StarCoder2Attention

class Starcoder2Model(TransformerBaseModel):
    def __init__(self, args: ModelConfig):
        super().__init__(
            args,
            attention_layer=StarCoder2Attention,
            norm_layer=lambda size, eps: nn.LayerNorm(size, eps=eps)
        )
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [StarCoder2TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.norm_epsilon)

    def __call__(self, inputs: mx.array, cache=None):
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