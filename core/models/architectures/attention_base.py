# attention_base.py
from functools import lru_cache
import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Tuple
from core.models.model_config import ModelConfig
from core.utils.memory_utils import log_memory_usage

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

        # Set up RoPE and initialize mask cache
        self.rope = self._setup_rope(config)

        # setup the masking cache
        self._mask_cache: Dict[Tuple[int, mx.Dtype], mx.array] = {}

        # Compile these functions for better performance
        self._project_inputs = mx.compile(self._project_inputs)
        #self._reshape_and_apply_rope = mx.compile(self._reshape_and_apply_rope)
        self._compute_attention = mx.compile(self._compute_attention)
        self._get_mask = mx.compile(self._get_mask)


    #    # Conditionally initialize attention dropout
    #     if config.attention_dropout_prob is not None:
    #         self.attention_dropout = nn.Dropout(config.attention_dropout_prob)
    #     else:
    #         self.attention_dropout = None

    #     # Conditionally initialize residual dropout
    #     if config.residual_dropout_prob is not None:
    #         self.residual_dropout = nn.Dropout(config.residual_dropout_prob)
    #     else:
    #         self.residual_dropout = None

    #     # # Conditionally initialize layer normalization
    #     # if config.layer_norm:
    #     #     self.layer_norm = nn.LayerNorm(config.hidden_size)
    #     # else:
    #     #     self.layer_norm = None

    def _initialize_projections(self, config: ModelConfig):
        """Initialize the projection layers for queries, keys, values, and output."""
        self.q_proj = nn.Linear(self.dimensions, self.n_heads * self.dimensions_per_head, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.dimensions, self.n_kv_heads * self.dimensions_per_head, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.dimensions, self.n_kv_heads * self.dimensions_per_head, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.dimensions_per_head, self.dimensions, bias=config.attention_bias)

    
    def _setup_rope(self, config: ModelConfig) -> Optional[nn.RoPE]:
        """Set up the Rotary Position Embedding (RoPE)."""

        # check if we have max_position_embeddings enabled (i.e. RoPE)
        if config.max_position_embeddings:
            # set the rope scale as 1.0
            rope_scale = 1.0

            # check if RoPE scaling is enabled
            if config.rope_scaling:
                # set the rope_scale using the scaling factor
                rope_scale = 1 / config.rope_scaling.get("factor", 1.0)

            # return the RoPE layer
            return nn.RoPE(
                # set the dimensions per head
                dims=self.dimensions_per_head,

                # set whether traditional rope
                traditional=config.rope_traditional,

                # set max_position_embeddings
                base=config.max_position_embeddings,

                # set the scale
                scale=rope_scale
            )
        
        # return none
        return None

    def set_inv_freq(self, inv_freq: any):
        """Set the inverse frequency for RoPE and reset caches."""
        if self.rope:
            self.rope.inv_freq = inv_freq

            # Recalculate cos and sin cache
            self.rope.cos_cached = None
            self.rope.sin_cached = None

    #@lru_cache(maxsize=128)
    def _get_mask(self, seq_length: int, dtype: mx.Dtype) -> mx.array:
        """Retrieve or create a causal mask for the attention mechanism."""
        # The causal mask is used in attention mechanisms, particularly in Transformer models, 
        # to ensure that each position in the sequence can only attend to previous positions (or itself) 
        # and not future positions. 

        # check if we have a mask cached already for this sequence length
        cache_key = (seq_length, dtype)

        if cache_key not in self._mask_cache:
            # create and cache mask if not already present
            self._mask_cache[cache_key] = nn.MultiHeadAttention.create_additive_causal_mask(seq_length).astype(dtype)
        
        # return the mask if not already present
        return self._mask_cache[cache_key]

    def _apply_rope(self, x: mx.array, offset: int = 0):
        """Apply Rotary Position Embedding to the input tensor."""

        # check if rope is configured
        if self.rope is not None:
            # return the rope layer
            return self.rope(x, offset=offset)
        return x

    def extend_position_embeddings(self, new_max_position: int):
        """
        Extend the maximum sequence length for position embeddings.
        Useful for processing longer sequences than the model was originally trained on.
        """
        # check if we have a rope layer
        if self.rope is not None:
            # set the new max position embeddings
            self.rope.base = new_max_position
            print(f"Extended position embeddings to {new_max_position}")
        else:
            # no rope
            print("No RoPE to extend")

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, L, _ = x.shape

        # layer normalization
        if hasattr(self, 'layer_norm'):
            x = self.layer_norm(x)

        # Project input to queries, keys, and values
        #compiled_inputs = mx.compile(self._project_inputs)
        q, k, v = self._project_inputs(x)

        # Reshape, apply RoPE, and handle caching
        q, k, v = self._reshape_and_apply_rope(q, k, v, B, L, cache)

        # Ensure that mask calculation handles batching efficiently
        if mask is None and x.shape[1] > 1:
            mask = self._get_mask(L, x.dtype)
        
        # Compute scaled dot-product attention
        #compiled_attention = mx.compile(self._compute_attention)
        output = self._compute_attention(q, k, v, mask)

        # Reshape output and project back to original dimensions
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Output projection
        output_projection = self.o_proj(output), (k, v)

        # Apply residual dropout if it exists
        if hasattr(self, 'residual_dropout'):
            output = self.residual_dropout(output)

        # Return the projection
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
        
        # reshape q,k
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

        # return q, k, v
        return q, k, v
    
    def _project_inputs(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Project the input tensor to queries, keys, and values."""
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)
    
    def _compute_attention(self, q: mx.array, k: mx.array, v: mx.array, mask: mx.array) -> mx.array:
        # get the scale as local
        scale = self.scale

        # do the dot product attention
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    
    def update_rope_params(self, new_params: Dict[str, any]):
        if self.rope:
            for key, value in new_params.items():
                setattr(self.rope, key, value)

