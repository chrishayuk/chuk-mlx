"""
Research playground for experimental tokenization techniques.

Modules:
- soft_tokens: Learnable soft token embeddings
- token_morphing: Token interpolation and blending
- embedding_analysis: Embedding space visualization and analysis
"""

from .embedding_analysis import (
    ClusterInfo,
    ClusterMethod,
    DistanceMetric,
    EmbeddingAnalysis,
    EmbeddingConfig,
    NeighborInfo,
    ProjectionMethod,
    ProjectionResult,
    analyze_embeddings,
    cluster_tokens,
    compute_embedding_quality,
    find_analogies,
    find_nearest_neighbors,
    project_embeddings,
)
from .soft_tokens import (
    InitializationMethod,
    SoftToken,
    SoftTokenBank,
    SoftTokenConfig,
    SoftTokenEmbedding,
    create_control_token,
    create_prompt_tuning_bank,
    create_soft_token,
    initialize_soft_embedding,
    interpolate_embeddings,
)
from .token_morphing import (
    BlendMode,
    MorphConfig,
    MorphMethod,
    MorphResult,
    MorphSequence,
    TokenBlend,
    blend_tokens,
    compute_path_length,
    compute_straightness,
    create_morph_sequence,
    find_midpoint,
    morph_token,
)

__all__ = [
    # Soft tokens
    "InitializationMethod",
    "SoftToken",
    "SoftTokenBank",
    "SoftTokenConfig",
    "SoftTokenEmbedding",
    "create_control_token",
    "create_prompt_tuning_bank",
    "create_soft_token",
    "initialize_soft_embedding",
    "interpolate_embeddings",
    # Token morphing
    "BlendMode",
    "MorphConfig",
    "MorphMethod",
    "MorphResult",
    "MorphSequence",
    "TokenBlend",
    "blend_tokens",
    "compute_path_length",
    "compute_straightness",
    "create_morph_sequence",
    "find_midpoint",
    "morph_token",
    # Embedding analysis
    "ClusterInfo",
    "ClusterMethod",
    "DistanceMetric",
    "EmbeddingAnalysis",
    "EmbeddingConfig",
    "NeighborInfo",
    "ProjectionMethod",
    "ProjectionResult",
    "analyze_embeddings",
    "cluster_tokens",
    "compute_embedding_quality",
    "find_analogies",
    "find_nearest_neighbors",
    "project_embeddings",
]
