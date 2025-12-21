import mlx.core as mx
import mlx.nn as nn
from chuk_lazarus.models.config import ModelConfig
from models.architectures.starcoder2.starcoder2_attention import StarCoder2Attention
from models.architectures.starcoder2.starcoder2_mlp import StarCoder2MLP

class StarCoder2TransformerBlock(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads

        self.self_attn = StarCoder2Attention(args)
        self.mlp = StarCoder2MLP(args)
        self.input_layernorm = nn.LayerNorm(args.hidden_size, eps=args.norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(args.hidden_size, eps=args.norm_epsilon)
        self.args = args

    def __call__(self, x: mx.array, mask: mx.array = None, cache=None):
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache