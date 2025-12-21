import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Tuple
from chuk_lazarus.models.config import ModelConfig

class AttentionBase(nn.Module):
    """
    Base class for attention mechanisms used in transformer models.
    Implements multi-head attention with support for grouped-query attention
    and rotary position embeddings (RoPE).
    """

    def __init__(self, config: ModelConfig):
        # call the parent constructor
        super().__init__()
        
        # Set up dimensions and number of heads
        self.dimensions = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.dimensions_per_head = config.head_dim or (config.hidden_size // config.num_attention_heads)
        
        # Scaling factor for attention scores
        self.scale = self.dimensions_per_head**-0.5
        
        # Initialize projection layers
        self._initialize_projections(config)

        # Set up RoPE
        self.rope = self._setup_rope(config)

        # Compile these functions for better performance
        self._project_inputs = mx.compile(self._project_inputs)

    def _initialize_projections(self, config: ModelConfig):
        """Initialize the projection layers for queries, keys, values, and output."""
        self.q_proj = nn.Linear(self.dimensions, self.n_heads * self.dimensions_per_head, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.dimensions, self.n_kv_heads * self.dimensions_per_head, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.dimensions, self.n_kv_heads * self.dimensions_per_head, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.dimensions_per_head, self.dimensions, bias=config.attention_bias)

    def _setup_rope(self, config: ModelConfig) -> Optional[nn.RoPE]:
        """Set up the Rotary Position Embedding (RoPE)."""
        if config.max_position_embeddings:
            rope_scale = 1.0
            if config.rope_scaling:
                rope_scale = 1 / config.rope_scaling.get("factor", 1.0)

            return nn.RoPE(
                dims=self.dimensions_per_head,
                traditional=config.rope_traditional,
                base=config.max_position_embeddings,
                scale=rope_scale
            )
        return None

    def set_inv_freq(self, inv_freq: any):
        """Set the inverse frequency for RoPE and reset caches."""
        if self.rope:
            self.rope.inv_freq = inv_freq
            self.rope.cos_cached = None
            self.rope.sin_cached = None

    def _apply_rope(self, x: mx.array, offset: int = 0):
        """Apply Rotary Position Embedding to the input tensor."""
        if self.rope is not None:
            return self.rope(x, offset=offset)
        return x

    def extend_position_embeddings(self, new_max_position: int):
        """Extend the maximum sequence length for position embeddings."""
        if self.rope is not None:
            self.rope.base = new_max_position
            print(f"Extended position embeddings to {new_max_position}")
        else:
            print("No RoPE to extend")

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, L, _ = x.shape

        # Layer normalization, if applicable
        if hasattr(self, 'layer_norm'):
            x = self.layer_norm(x)

        # Project input to queries, keys, and values
        q, k, v = self._project_inputs(x)

        # Reshape, apply RoPE, and handle caching
        q, k, v = self._reshape_and_apply_rope(q, k, v, B, L, cache)

        # Compute scaled dot-product attention
        output = self._compute_attention(q, k, v, mask)

        # Reshape output and project back to original dimensions
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Output projection
        output_projection = self.o_proj(output), (k, v)

        # Apply residual dropout if it exists
        if hasattr(self, 'residual_dropout'):
            output_projection = self.residual_dropout(output_projection)

        return output_projection

    def _reshape_and_apply_rope(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        B: int,
        L: int,
        cache: Optional[Tuple[mx.array, mx.array]]
    ) -> Tuple[mx.array, mx.array, mx.array]:
        # Reshape q, k, and v
        q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            offset = key_cache.shape[2]
            q, k = map(lambda x: self._apply_rope(x, offset), (q, k))
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)
        else:
            q, k = map(self._apply_rope, (q, k))

        return q, k, v

    def _project_inputs(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Project the input tensor to queries, keys, and values."""
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def _compute_attention(self, q: mx.array, k: mx.array, v: mx.array, mask: mx.array) -> mx.array:
        """Compute scaled dot-product attention."""
        scale = self.scale
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    def update_rope_params(self, new_params: Dict[str, any]):
        """Update the parameters of the RoPE."""
        if self.rope:
            for key, value in new_params.items():
                setattr(self.rope, key, value)
