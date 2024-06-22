import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
from models.model_config import ModelConfig
from .pruning import prune_heads

class AttentionBase(nn.Module):
    """
    Base class for attention mechanisms used in transformer models.
    Implements multi-head attention with support for grouped-query attention
    and rotary position embeddings (RoPE).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Set up dimensions and number of heads
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
        
        # Set up RoPE and initialize mask cache
        self.rope = self._setup_rope(config)
        self._mask_cache = {}

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

    def _get_mask(self, seq_length: int, dtype: mx.Dtype):
        """
        Get or create a causal mask for the attention mechanism.
        Caches masks for efficiency when processing sequences of the same length repeatedly.
        """
        if seq_length not in self._mask_cache:
            self._mask_cache[seq_length] = nn.MultiHeadAttention.create_additive_causal_mask(seq_length).astype(dtype)
        return self._mask_cache[seq_length]

    def _apply_rope(self, x: mx.array, offset: int = 0):
        """
        Apply Rotary Position Embedding to the input tensor.
        """
        if self.rope is not None:
            return self.rope(x, offset=offset)
        return x

    def extend_position_embeddings(self, new_max_position: int):
        """
        Extend the maximum sequence length for position embeddings.
        Useful for processing longer sequences than the model was originally trained on.
        """
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
        """
        Compute the attention mechanism.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, hidden_size)
            mask: Optional attention mask
            cache: Optional key/value cache for attention

        Returns:
            Tuple of output tensor and updated cache
        """
        B, L, _ = x.shape

        # Project input to queries, keys, and values
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Reshape and transpose for multi-head attention
        q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Apply RoPE and handle caching
        if cache is not None:
            key_cache, value_cache = cache
            offset = key_cache.shape[2]
            q = self._apply_rope(q, offset)
            k = self._apply_rope(k, offset)
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)
        else:
            q = self._apply_rope(q)
            k = self._apply_rope(k)

        # Get or create causal mask if needed
        if mask is None and L > 1:
            mask = self._get_mask(L, x.dtype)

        # Compute scaled dot-product attention
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        # Reshape output and project back to original dimensions
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Return the output projection and the updated cache
        return self.o_proj(output), (k, v)