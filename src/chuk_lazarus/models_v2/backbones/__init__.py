"""
Backbone stacks for sequence models.

Backbones combine embeddings + blocks into a complete encoder/decoder.
They produce hidden states that can be fed to different heads.

Provides:
- TransformerBackbone: Stack of transformer blocks
- MambaBackbone: Stack of Mamba blocks
- RecurrentBackbone: Stack of RNN blocks
- HybridBackbone: Mixed block types
"""

from .base import Backbone, BackboneOutput
from .hybrid import HybridBackbone, create_hybrid_backbone
from .mamba import MambaBackbone, create_mamba_backbone
from .recurrent import RecurrentBackbone, create_recurrent_backbone
from .transformer import TransformerBackbone, create_transformer_backbone

__all__ = [
    # Base
    "Backbone",
    "BackboneOutput",
    # Implementations
    "TransformerBackbone",
    "create_transformer_backbone",
    "MambaBackbone",
    "create_mamba_backbone",
    "RecurrentBackbone",
    "create_recurrent_backbone",
    "HybridBackbone",
    "create_hybrid_backbone",
]
