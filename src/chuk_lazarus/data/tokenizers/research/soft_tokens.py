"""
Soft token embeddings for learnable vocabulary extension.

Soft tokens are continuous embeddings that can be optimized during training,
allowing the model to learn task-specific "virtual" tokens without modifying
the base vocabulary.

Use cases:
- Prompt tuning: Learn soft prompts instead of discrete tokens
- Domain adaptation: Add learnable domain-specific tokens
- Controllable generation: Learn control tokens for style/sentiment
- Few-shot learning: Learn task-specific prefix tokens
"""

from enum import Enum
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field


class EmbeddingProtocol(Protocol):
    """Protocol for embedding layer compatibility."""

    def __call__(self, token_ids: list[int]) -> np.ndarray:
        """Get embeddings for token IDs."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension."""
        ...

    @property
    def num_embeddings(self) -> int:
        """Number of embeddings (vocab size)."""
        ...


class InitializationMethod(str, Enum):
    """Methods for initializing soft token embeddings."""

    RANDOM_NORMAL = "random_normal"
    RANDOM_UNIFORM = "random_uniform"
    FROM_TOKENS = "from_tokens"  # Average of existing tokens
    FROM_TEXT = "from_text"  # Average of text embeddings
    ZEROS = "zeros"
    ONES = "ones"


class SoftTokenConfig(BaseModel):
    """Configuration for soft token creation."""

    # Embedding dimension (must match model)
    embedding_dim: int = Field(ge=1, description="Embedding dimension")

    # Initialization
    init_method: InitializationMethod = Field(
        default=InitializationMethod.RANDOM_NORMAL,
        description="Initialization method for embeddings",
    )

    # For random initialization
    init_std: float = Field(default=0.02, ge=0, description="Std dev for random init")
    init_mean: float = Field(default=0.0, description="Mean for random init")

    # For uniform initialization
    init_min: float = Field(default=-0.1, description="Min for uniform init")
    init_max: float = Field(default=0.1, description="Max for uniform init")

    # Training settings
    trainable: bool = Field(default=True, description="Whether embedding is trainable")
    learning_rate_multiplier: float = Field(
        default=1.0, ge=0, description="LR multiplier for this embedding"
    )

    # Regularization
    l2_regularization: float = Field(default=0.0, ge=0, description="L2 regularization weight")
    dropout: float = Field(default=0.0, ge=0, le=1, description="Dropout rate")


class SoftToken(BaseModel):
    """A soft token with learnable embedding."""

    # Identity
    name: str = Field(description="Unique name for this soft token")
    token_id: int = Field(ge=0, description="Virtual token ID (outside vocab range)")

    # Description
    description: str = Field(default="", description="Human-readable description")
    purpose: str = Field(default="general", description="Purpose category")

    # Configuration
    config: SoftTokenConfig = Field(description="Token configuration")

    # Metadata
    created_from: str | None = Field(
        default=None, description="Source tokens/text if initialized from existing"
    )

    model_config = {"arbitrary_types_allowed": True}


class SoftTokenEmbedding(BaseModel):
    """Soft token with its embedding vector."""

    token: SoftToken = Field(description="Soft token metadata")
    embedding: list[float] = Field(description="Embedding vector as list")

    @property
    def embedding_array(self) -> np.ndarray:
        """Get embedding as numpy array."""
        return np.array(self.embedding, dtype=np.float32)

    @classmethod
    def from_array(cls, token: SoftToken, embedding: np.ndarray) -> "SoftTokenEmbedding":
        """Create from numpy array."""
        return cls(token=token, embedding=embedding.tolist())


class SoftTokenBank(BaseModel):
    """Collection of soft tokens for a model."""

    # Bank metadata
    name: str = Field(description="Bank name")
    model_name: str = Field(default="", description="Associated model name")
    embedding_dim: int = Field(ge=1, description="Embedding dimension")

    # Tokens
    tokens: list[SoftTokenEmbedding] = Field(
        default_factory=list, description="Soft tokens in bank"
    )

    # ID management
    base_token_id: int = Field(default=100000, ge=0, description="Starting ID for soft tokens")

    def add_token(self, token: SoftTokenEmbedding) -> None:
        """Add a soft token to the bank."""
        if token.token.token_id in self.token_ids:
            raise ValueError(f"Token ID {token.token.token_id} already exists")
        self.tokens.append(token)

    def get_token(self, name: str) -> SoftTokenEmbedding | None:
        """Get token by name."""
        for token in self.tokens:
            if token.token.name == name:
                return token
        return None

    def get_by_id(self, token_id: int) -> SoftTokenEmbedding | None:
        """Get token by ID."""
        for token in self.tokens:
            if token.token.token_id == token_id:
                return token
        return None

    @property
    def token_ids(self) -> set[int]:
        """Get all token IDs."""
        return {t.token.token_id for t in self.tokens}

    @property
    def token_names(self) -> set[str]:
        """Get all token names."""
        return {t.token.name for t in self.tokens}

    def get_embeddings_matrix(self) -> np.ndarray:
        """Get all embeddings as a matrix (num_tokens, embedding_dim)."""
        if not self.tokens:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return np.stack([t.embedding_array for t in self.tokens])

    def next_token_id(self) -> int:
        """Get next available token ID."""
        if not self.tokens:
            return self.base_token_id
        return max(self.token_ids) + 1


def initialize_soft_embedding(
    config: SoftTokenConfig,
    source_embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """
    Initialize a soft token embedding.

    Args:
        config: Soft token configuration
        source_embeddings: Optional source embeddings for FROM_TOKENS/FROM_TEXT init

    Returns:
        Initialized embedding vector
    """
    dim = config.embedding_dim

    if config.init_method == InitializationMethod.RANDOM_NORMAL:
        return np.random.normal(config.init_mean, config.init_std, dim).astype(np.float32)

    elif config.init_method == InitializationMethod.RANDOM_UNIFORM:
        return np.random.uniform(config.init_min, config.init_max, dim).astype(np.float32)

    elif config.init_method == InitializationMethod.ZEROS:
        return np.zeros(dim, dtype=np.float32)

    elif config.init_method == InitializationMethod.ONES:
        return np.ones(dim, dtype=np.float32)

    elif config.init_method in (InitializationMethod.FROM_TOKENS, InitializationMethod.FROM_TEXT):
        if source_embeddings is None:
            raise ValueError(f"{config.init_method} requires source_embeddings")
        # Average the source embeddings
        if source_embeddings.ndim == 1:
            return source_embeddings.astype(np.float32)
        return source_embeddings.mean(axis=0).astype(np.float32)

    else:
        raise ValueError(f"Unknown initialization method: {config.init_method}")


def create_soft_token(
    name: str,
    config: SoftTokenConfig,
    token_id: int | None = None,
    description: str = "",
    purpose: str = "general",
    source_embeddings: np.ndarray | None = None,
) -> SoftTokenEmbedding:
    """
    Create a soft token with initialized embedding.

    Args:
        name: Unique name for the token
        config: Token configuration
        token_id: Token ID (auto-assigned if None)
        description: Human-readable description
        purpose: Purpose category
        source_embeddings: Source embeddings for initialization

    Returns:
        SoftTokenEmbedding with initialized vector
    """
    if token_id is None:
        token_id = 100000  # Default base

    token = SoftToken(
        name=name,
        token_id=token_id,
        description=description,
        purpose=purpose,
        config=config,
    )

    embedding = initialize_soft_embedding(config, source_embeddings)

    return SoftTokenEmbedding.from_array(token, embedding)


def interpolate_embeddings(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    alpha: float = 0.5,
    method: str = "linear",
) -> np.ndarray:
    """
    Interpolate between two embeddings.

    Args:
        embedding1: First embedding
        embedding2: Second embedding
        alpha: Interpolation factor (0 = embedding1, 1 = embedding2)
        method: Interpolation method ("linear" or "spherical")

    Returns:
        Interpolated embedding
    """
    if method == "linear":
        return (1 - alpha) * embedding1 + alpha * embedding2

    elif method == "spherical":
        # Spherical linear interpolation (slerp)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return (1 - alpha) * embedding1 + alpha * embedding2

        unit1 = embedding1 / norm1
        unit2 = embedding2 / norm2

        dot = np.clip(np.dot(unit1, unit2), -1.0, 1.0)
        theta = np.arccos(dot)

        if theta < 1e-8:
            return (1 - alpha) * embedding1 + alpha * embedding2

        sin_theta = np.sin(theta)
        result = (np.sin((1 - alpha) * theta) / sin_theta) * unit1 + (
            np.sin(alpha * theta) / sin_theta
        ) * unit2

        # Interpolate norm
        norm = (1 - alpha) * norm1 + alpha * norm2
        return result * norm

    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def create_prompt_tuning_bank(
    num_tokens: int,
    embedding_dim: int,
    prefix: str = "prompt",
    init_method: InitializationMethod = InitializationMethod.RANDOM_NORMAL,
    init_std: float = 0.02,
) -> SoftTokenBank:
    """
    Create a bank of soft tokens for prompt tuning.

    Args:
        num_tokens: Number of soft prompt tokens
        embedding_dim: Embedding dimension
        prefix: Name prefix for tokens
        init_method: Initialization method
        init_std: Std dev for random init

    Returns:
        SoftTokenBank with initialized tokens
    """
    bank = SoftTokenBank(
        name=f"{prefix}_tuning",
        embedding_dim=embedding_dim,
        base_token_id=100000,
    )

    config = SoftTokenConfig(
        embedding_dim=embedding_dim,
        init_method=init_method,
        init_std=init_std,
        trainable=True,
    )

    for i in range(num_tokens):
        token = create_soft_token(
            name=f"{prefix}_{i}",
            config=config,
            token_id=bank.next_token_id(),
            description=f"Soft prompt token {i}",
            purpose="prompt_tuning",
        )
        bank.add_token(token)

    return bank


def create_control_token(
    name: str,
    embedding_dim: int,
    source_embeddings: np.ndarray | None = None,
    description: str = "",
) -> SoftTokenEmbedding:
    """
    Create a control token for controllable generation.

    Args:
        name: Token name (e.g., "positive_sentiment", "formal_style")
        embedding_dim: Embedding dimension
        source_embeddings: Optional source embeddings for initialization
        description: Human-readable description

    Returns:
        SoftTokenEmbedding for control
    """
    init_method = (
        InitializationMethod.FROM_TOKENS
        if source_embeddings is not None
        else InitializationMethod.RANDOM_NORMAL
    )

    config = SoftTokenConfig(
        embedding_dim=embedding_dim,
        init_method=init_method,
        trainable=True,
        learning_rate_multiplier=0.1,  # Slower learning for control tokens
    )

    return create_soft_token(
        name=name,
        config=config,
        description=description or f"Control token: {name}",
        purpose="control",
        source_embeddings=source_embeddings,
    )
