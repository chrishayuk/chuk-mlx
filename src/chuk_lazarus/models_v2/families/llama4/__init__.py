"""
Llama 4 model family.

Supports:
- Llama 4 Scout (17B active / 109B total)
- Llama 4 Maverick (17B active / 400B total)
- Multimodal variants with vision encoder

Key features:
- MoE (Mixture of Experts) with shared expert
- iRoPE (interleaved RoPE and NoPE layers)
- QK normalization
- Native multimodal support

Reference: https://llama.meta.com/llama4/
"""

from .attention import Llama4Attention, Llama4FlexAttention, create_llama4_attention
from .config import Llama4Config, Llama4TextConfig, Llama4VisionConfig
from .model import Llama4, Llama4Block, Llama4ForCausalLM, Llama4Model
from .moe import Llama4MLP, Llama4MoE, create_llama4_moe

__all__ = [
    # Config
    "Llama4Config",
    "Llama4TextConfig",
    "Llama4VisionConfig",
    # Model
    "Llama4",
    "Llama4ForCausalLM",
    "Llama4Model",
    "Llama4Block",
    # Components
    "Llama4Attention",
    "Llama4FlexAttention",
    "Llama4MoE",
    "Llama4MLP",
    # Factories
    "create_llama4_attention",
    "create_llama4_moe",
]
