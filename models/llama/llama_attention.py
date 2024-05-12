from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
from models.model_config import ModelConfig

class LlamaAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        # initialize
        super().__init__()

        # set the dimensions
        dimensions = config.hidden_size

        # set the number of attention heads
        self.n_heads = n_heads = config.num_attention_heads

        # set the number of Key Value (KV) Heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads

        # head dimensions is total dimensions / number of heads
        dimensions_per_head = config.hidden_size // n_heads

        # set the scaling
        self.scale = dimensions_per_head**-0.5

        # set Q, K, V and O
        self.q_proj = nn.Linear(dimensions, n_heads * dimensions_per_head, bias=False)
        self.k_proj = nn.Linear(dimensions, n_kv_heads * dimensions_per_head, bias=False)
        self.v_proj = nn.Linear(dimensions, n_kv_heads * dimensions_per_head, bias=False)
        self.o_proj = nn.Linear(n_heads * dimensions_per_head, dimensions, bias=False)

        # rope scaling is 1 by default
        rope_scale = 1

        # rope scaling
        if config.rope_scaling is not None:
            # TODO: need to setup model config to handle this
            if config.rope_scaling["type"] == "linear":
                # set the scaling to the factor
                rope_scale = 1 / config.rope_scaling["factor"]

        # TODO: I think this could be cleaner for non RoPE impelmentations
        # setup RoPE
        self.rope = nn.RoPE(
            dimensions_per_head,
            traditional=config.rope_traditional,
            base=config.rope_theta,
            scale=rope_scale,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)