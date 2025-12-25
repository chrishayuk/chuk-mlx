"""
Core abstractions for the unified model architecture.

Provides:
- Enums for type-safe configuration
- Protocols defining component interfaces
- Pydantic configs for validation
- Registry for architecture discovery
- Backend abstraction for MLX/PyTorch/JAX
"""

from .backend import (
    Backend,
    BackendType,
    MLXBackend,
    Tensor,
    TorchBackend,
    get_backend,
    reset_backend,
    set_backend,
)
from .config import (
    AttentionConfig,
    BackboneConfig,
    BlockConfig,
    EmbeddingConfig,
    FFNConfig,
    HeadConfig,
    ModelConfig,
    NormConfig,
    PositionConfig,
    RoPEConfig,
    SSMConfig,
)
from .enums import (
    ActivationType,
    AttentionType,
    BackboneType,
    BlockType,
    ClassificationTask,
    DtRank,
    FFNType,
    HeadType,
    HybridCombineMode,
    HybridMixStrategy,
    InitType,
    ModelMode,
    NormType,
    PoolingType,
    PositionEmbeddingType,
    RecurrentType,
    SSMType,
)
from .protocols import (
    FFN,
    SSM,
    Attention,
    AttentionCache,
    Backbone,
    Block,
    BlockCache,
    Embedding,
    Head,
    LayerCache,
    Model,
    Norm,
    PositionEmbedding,
    RecurrentCell,
)
from .registry import (
    ModelCapability,
    ModelRegistry,
    create_model,
    find_models_by_capability,
    get_model_capabilities,
    get_model_class,
    list_models,
    register_model,
)
from .registry import (
    get_model_class as get_factory,  # Alias for convenience
)

__all__ = [
    # Enums
    "ModelMode",
    "BlockType",
    "BackboneType",
    "HeadType",
    "AttentionType",
    "NormType",
    "ActivationType",
    "PositionEmbeddingType",
    "PoolingType",
    "FFNType",
    "SSMType",
    "RecurrentType",
    "InitType",
    "HybridMixStrategy",
    "HybridCombineMode",
    "ClassificationTask",
    "DtRank",
    # Configs
    "ModelConfig",
    "EmbeddingConfig",
    "PositionConfig",
    "RoPEConfig",
    "AttentionConfig",
    "FFNConfig",
    "NormConfig",
    "BlockConfig",
    "BackboneConfig",
    "HeadConfig",
    "SSMConfig",
    # Protocols
    "Embedding",
    "PositionEmbedding",
    "Attention",
    "FFN",
    "SSM",
    "RecurrentCell",
    "Norm",
    "Block",
    "Backbone",
    "Head",
    "Model",
    # Cache types
    "AttentionCache",
    "BlockCache",
    "LayerCache",
    # Registry
    "ModelRegistry",
    "ModelCapability",
    "register_model",
    "get_model_class",
    "get_factory",
    "create_model",
    "list_models",
    "get_model_capabilities",
    "find_models_by_capability",
    # Backend
    "Tensor",
    "Backend",
    "BackendType",
    "MLXBackend",
    "TorchBackend",
    "get_backend",
    "set_backend",
    "reset_backend",
]
