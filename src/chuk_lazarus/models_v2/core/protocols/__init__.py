"""
Protocol definitions for model components.

Protocols define the interfaces that components must implement.
This enables composition and type-safe dependency injection.

All protocols use abstract Tensor types - actual implementation
depends on the backend (MLX, PyTorch, JAX).

Architecture:
    Components (atomic) → Blocks (composed) → Backbones → Heads → Models

Design Principles:
- Async-native: Protocols support async where needed
- Pydantic-native: Configs use Pydantic for validation
- No magic strings: Use enums throughout
- Structural subtyping: Duck typing with type hints
"""

# Components
# Blocks and backbones
from .blocks import (
    Backbone,
    Block,
    Head,
)
from .components import (
    FFN,
    SSM,
    Attention,
    Embedding,
    Norm,
    PositionEmbedding,
    RecurrentCell,
)

# Complete models
from .models import Model

# Types
from .types import (
    AttentionCache,
    BlockCache,
    LayerCache,
)

__all__ = [
    # Components
    "Embedding",
    "PositionEmbedding",
    "Norm",
    "FFN",
    "Attention",
    "SSM",
    "RecurrentCell",
    # Blocks
    "Block",
    "Backbone",
    "Head",
    # Models
    "Model",
    # Types
    "AttentionCache",
    "BlockCache",
    "LayerCache",
]
