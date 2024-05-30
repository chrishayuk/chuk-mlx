import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
from models.model_config import ModelConfig


class LlamaAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        # call the base constructor
        super().__init__()

        # Validate the configuration values
        if config.hidden_size is None:
            raise ValueError("hidden_size must be provided in the config.")
        if config.num_attention_heads is None:
            raise ValueError("num_attention_heads must be provided in the config.")
        if config.num_key_value_heads is None:
            raise ValueError("num_key_value_heads must be provided in the config.")

        # set the dimensions
        dimensions = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads

        # dimensions per head is hidden size / heads
        # Use head_dim if provided, otherwise calculate dimensions_per_head
        dimensions_per_head = config.head_dim if hasattr(config, 'head_dim') and config.head_dim is not None else config.hidden_size // n_heads
        
        # Add a debug print to trace the values
        #print(f"dimensions: {dimensions}, n_heads: {n_heads}, n_kv_heads: {n_kv_heads}, dimensions_per_head: {dimensions_per_head}")

        if dimensions_per_head is None:
            raise ValueError("dimensions_per_head could not be determined. Check the configuration.")

        # set scale
        self.scale = dimensions_per_head**-0.5

        # set q,k,v,o
        self.q_proj = nn.Linear(dimensions, n_heads * dimensions_per_head, bias=config.attention_bias)
        self.k_proj = nn.Linear(dimensions, n_kv_heads * dimensions_per_head, bias=config.attention_bias)
        self.v_proj = nn.Linear(dimensions, n_kv_heads * dimensions_per_head, bias=config.attention_bias)
        self.o_proj = nn.Linear(n_heads * dimensions_per_head, dimensions, bias=config.attention_bias)

        # Conditional RoPE setup based on max_position_embeddings
        self.rope = None
        if config.max_position_embeddings:
            # set the scale and max position embeddings
            rope_scale = 1.0
            rope_base = config.max_position_embeddings

            # check if we have scaling configured
            if config.rope_scaling:
                if config.rope_scaling["type"] == "linear":
                    rope_scale = 1 / config.rope_scaling["factor"]

            # set rope
            self.rope = nn.RoPE(
                dims=dimensions_per_head,
                traditional=config.rope_traditional,
                base=rope_base,
                scale=rope_scale
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
        if self.rope is not None:
            # check for caching
            if cache is not None:
                key_cache, value_cache = cache
                queries = self.rope(queries, offset=key_cache.shape[2])
                keys = self.rope(keys, offset=key_cache.shape[2])
                keys = mx.concatenate([key_cache, keys], axis=2)
                values = mx.concatenate([value_cache, values], axis=2)
            else:
                # return queries and keys using rope
                queries = self.rope(queries)
                keys = self.rope(keys)
        else:
            if cache is not None:
                key_cache, value_cache = cache
                keys = mx.concatenate([key_cache, keys], axis=2)
                values = mx.concatenate([value_cache, values], axis=2)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output), (keys, values)
