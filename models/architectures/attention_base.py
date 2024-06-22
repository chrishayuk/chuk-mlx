import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
from models.model_config import ModelConfig

class AttentionBase(nn.Module):
    """
    Base class for attention mechanisms used in transformer models.
    This class implements the core functionality of multi-head attention with support for
    grouped-query attention and rotary position embeddings (RoPE).
    """
    def __init__(self, config: ModelConfig):
        # constructor
        super().__init__()
        
        # Set up the dimensions and number of heads
        self.dimensions = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.dimensions_per_head = config.head_dim or (config.hidden_size // config.num_attention_heads)
        
        # Scaling factor for attention scores
        self.scale = self.dimensions_per_head**-0.5
        
        # Projection layers for queries, keys, values, and output
        self.q_proj = nn.Linear(self.dimensions, self.n_heads * self.dimensions_per_head, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.dimensions, self.n_kv_heads * self.dimensions_per_head, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.dimensions, self.n_kv_heads * self.dimensions_per_head, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.dimensions_per_head, self.dimensions, bias=config.attention_bias)
        
        # Set up RoPE
        self.rope = self._setup_rope(config)

    def _setup_rope(self, config):
        """
        Set up the Rotary Position Embedding (RoPE).
        This method implements the Llama-style RoPE setup as a default.
        """
        if config.max_position_embeddings:
            rope_scale = 1.0
            if config.rope_scaling:
                if config.rope_scaling["type"] == "linear":
                    rope_scale = 1 / config.rope_scaling["factor"]
            return nn.RoPE(
                dims=self.dimensions_per_head,
                traditional=config.rope_traditional,
                base=config.max_position_embeddings,
                scale=rope_scale
            )
        return None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """ Compute the attention mechanism. """
        B, L, D = x.shape

        # Project input to queries, keys, and values
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Reshape and transpose for multi-head attention
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Apply RoPE and handle caching
        if self.rope is not None:
            if cache is not None:
                key_cache, value_cache = cache
                queries = self.rope(queries, offset=key_cache.shape[2])
                keys = self.rope(keys, offset=key_cache.shape[2])
                keys = mx.concatenate([key_cache, keys], axis=2)
                values = mx.concatenate([value_cache, values], axis=2)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)
        elif cache is not None:
            key_cache, value_cache = cache
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
            
        # Compute scaled dot-product attention
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        # Reshape output and project back to original dimensions
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        # return the output projection
        return self.o_proj(output), (keys, values)