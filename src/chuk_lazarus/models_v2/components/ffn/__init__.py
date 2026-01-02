"""
Feed-Forward Network (FFN) components.

Provides:
- MLP: Standard two-layer MLP with configurable activation
- GLU: Gated Linear Unit with configurable activation
  - SwiGLU: GLU with SiLU activation (Llama, Mistral)
  - GEGLU: GLU with GELU activation
- MoE: Mixture of Experts (multiple MLPs with router)
"""

from .glu import GEGLU, GLU, SwiGLU, create_geglu, create_glu, create_swiglu
from .mlp import MLP, create_mlp
from .moe import MoE, MoERouter

__all__ = [
    # Standard MLP
    "MLP",
    "create_mlp",
    # Gated Linear Units
    "GLU",
    "SwiGLU",
    "GEGLU",
    "create_glu",
    "create_swiglu",
    "create_geglu",
    # Mixture of Experts
    "MoE",
    "MoERouter",
]
