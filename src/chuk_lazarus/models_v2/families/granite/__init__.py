"""
IBM Granite model family.

Supports:
- Granite 3.0/3.1: Dense transformer with multipliers
- Granite 4.0: Hybrid Mamba-2/Transformer with optional MoE

Key features:
- Embedding/attention/residual multipliers
- Logits scaling
- Mamba-2 blocks for efficient long-context
- Fine-grained MoE with shared experts

Reference: https://huggingface.co/ibm-granite
"""

from .config import GraniteConfig, GraniteHybridConfig
from .hybrid import (
    Granite4,
    GraniteHybrid,
    GraniteHybridAttention,
    GraniteHybridBlock,
    GraniteHybridForCausalLM,
    GraniteHybridModel,
    GraniteHybridMoE,
    GraniteMamba2Block,
)
from .model import (
    Granite,
    GraniteAttention,
    GraniteBlock,
    GraniteForCausalLM,
    GraniteModel,
)

__all__ = [
    # Config
    "GraniteConfig",
    "GraniteHybridConfig",
    # Granite 3.x
    "Granite",
    "GraniteForCausalLM",
    "GraniteModel",
    "GraniteBlock",
    "GraniteAttention",
    # Granite 4.x
    "Granite4",
    "GraniteHybrid",
    "GraniteHybridForCausalLM",
    "GraniteHybridModel",
    "GraniteHybridBlock",
    "GraniteHybridAttention",
    "GraniteHybridMoE",
    "GraniteMamba2Block",
]
