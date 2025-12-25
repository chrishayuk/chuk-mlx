"""
Gemma 3 model family.

Supports:
- Gemma 3 270M (FunctionGemma base)
- Gemma 3 1B
- Gemma 3 4B
- Gemma 3 12B
- Gemma 3 27B
- FunctionGemma (function calling variant)

Gemma 3 features:
- Alternating sliding window / global attention
- Query/key pre-attention normalization
- 4 normalization layers per block
- Gated GELU activation
- Large vocabulary (262k tokens)
"""

from .config import GemmaConfig
from .convert import (
    GEMMA_WEIGHT_MAP,
    convert_hf_weights,
    convert_mlx_community_weights,
)
from .model import (
    FunctionGemmaForCausalLM,
    GemmaAttention,
    GemmaBlock,
    GemmaForCausalLM,
    GemmaMLP,
    GemmaModel,
    GemmaRMSNorm,
)

__all__ = [
    # Config
    "GemmaConfig",
    # Model components
    "GemmaRMSNorm",
    "GemmaMLP",
    "GemmaAttention",
    "GemmaBlock",
    "GemmaModel",
    # Full models
    "GemmaForCausalLM",
    "FunctionGemmaForCausalLM",
    # Weight conversion
    "convert_hf_weights",
    "convert_mlx_community_weights",
    "GEMMA_WEIGHT_MAP",
]
