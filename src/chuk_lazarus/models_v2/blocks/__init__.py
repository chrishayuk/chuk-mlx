"""
Composable blocks for sequence models.

Blocks combine components (attention/SSM/RNN + FFN + normalization)
into reusable units that form the layers of a model.

Provides:
- TransformerBlock: Standard attention + FFN block
- MambaBlock: SSM-based block (imported from components)
- RecurrentBlock: RNN-based block
- HybridBlock: Combines multiple block types
"""

from .base import Block, BlockOutput
from .hybrid import HybridBlock, create_hybrid_block
from .mamba import MambaBlockWrapper, create_mamba_block_wrapper
from .recurrent import RecurrentBlockWrapper, RecurrentWithFFN, create_recurrent_block
from .transformer import TransformerBlock, create_transformer_block

__all__ = [
    # Base
    "Block",
    "BlockOutput",
    # Transformer
    "TransformerBlock",
    "create_transformer_block",
    # Mamba
    "MambaBlockWrapper",
    "create_mamba_block_wrapper",
    # Recurrent
    "RecurrentBlockWrapper",
    "RecurrentWithFFN",
    "create_recurrent_block",
    # Hybrid
    "HybridBlock",
    "create_hybrid_block",
]
