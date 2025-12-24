"""
Attention mechanism components.

Provides:
- MultiHeadAttention: Standard multi-head attention
- GroupedQueryAttention: GQA (Llama 2+, fewer KV heads)
- SlidingWindowAttention: Local attention window (Mistral)
- MultiQueryAttention: MQA (single KV head)

All attention implementations support:
- Rotary Position Embeddings (RoPE)
- KV caching for efficient generation
- Causal masking for autoregressive models
"""

from .base import AttentionBase, create_causal_mask
from .grouped_query import GroupedQueryAttention
from .multi_head import MultiHeadAttention
from .sliding_window import SlidingWindowAttention

__all__ = [
    "AttentionBase",
    "create_causal_mask",
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "SlidingWindowAttention",
]
