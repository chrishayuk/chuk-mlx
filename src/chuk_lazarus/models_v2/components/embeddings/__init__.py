"""
Embedding components.

Provides:
- TokenEmbedding: Standard token embeddings with optional scaling
- RoPE: Rotary Position Embeddings
- ALiBi: Attention with Linear Biases
- LearnedPositionEmbedding: Learned absolute positions
- SinusoidalPositionEmbedding: Fixed sinusoidal positions
"""

from .alibi import ALiBi, compute_alibi_bias
from .learned import LearnedPositionEmbedding
from .rope import RoPE
from .sinusoidal import SinusoidalPositionEmbedding
from .token import TokenEmbedding, create_token_embedding

__all__ = [
    "TokenEmbedding",
    "create_token_embedding",
    "RoPE",
    "ALiBi",
    "compute_alibi_bias",
    "LearnedPositionEmbedding",
    "SinusoidalPositionEmbedding",
]
