import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
from models.model_config import ModelConfig


class GemmaAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        # call the base constructor
        super().__init__()

        # set the dimensions
        dimensions = config.hidden_size

        # set the attention heads
        self.n_heads = n_heads = config.num_attention_heads

        # set the KV heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads

        # Use head_dim if provided, otherwise calculate dimensions_per_head
        dimensions_per_head = config.head_dim

        # set scale
        self.scale = dimensions_per_head**-0.5

        # set q,k,v,o
        self.q_proj = nn.Linear(dimensions, n_heads * dimensions_per_head, bias=config.attention_bias)
        self.k_proj = nn.Linear(dimensions, n_kv_heads * dimensions_per_head, bias=config.attention_bias)
        self.v_proj = nn.Linear(dimensions, n_kv_heads * dimensions_per_head, bias=config.attention_bias)
        self.o_proj = nn.Linear(n_heads * dimensions_per_head, dimensions, bias=config.attention_bias)

        # Conditional RoPE setup based on max_position_embeddings
        self.rope = nn.RoPE(
            dimensions_per_head,
            traditional=config.rope_traditional,
            base=config.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # calculate queries, keys and values
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # check if we have rope
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