import mlx.core as mx
import mlx.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self):
        # initialize
        super().__init__()

        # parameter   
        self.num_attention_heads = 32
        self.hidden_size = 4096
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache
    
class LlamaModel(nn.Module):
    def __init__(self):
        # initialize
        super().__init__()

        # embedding layer is vocab and hidden_size (dimensions)
        vocab_size = 32000
        hidden_size = 4096
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # we need 32 hidden layers (attention layers)
        num_hidden_layers = 32
        self.layers = [
            TransformerLayer(args=args) for _ in range(num_hidden_layers)
        ]

