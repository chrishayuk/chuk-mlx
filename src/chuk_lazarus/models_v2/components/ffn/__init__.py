"""
Feed-Forward Network (FFN) components.

Provides:
- MLP: Standard two-layer MLP
- SwiGLU: Gated MLP with SiLU activation (Llama, Mistral)
- GEGLU: Gated MLP with GELU activation
- MoE: Mixture of Experts (multiple MLPs with router)
"""

from .geglu import GEGLU
from .mlp import MLP
from .moe import MoE, MoERouter
from .swiglu import SwiGLU

__all__ = [
    "MLP",
    "SwiGLU",
    "GEGLU",
    "MoE",
    "MoERouter",
]
