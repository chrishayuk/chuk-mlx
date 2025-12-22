"""
Token morphing and blending for embedding space exploration.

Token morphing allows smooth transitions between tokens in embedding space,
useful for:
- Understanding semantic relationships
- Generating intermediate representations
- Style interpolation
- Exploring the embedding manifold
"""

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field


class MorphMethod(str, Enum):
    """Methods for morphing between tokens."""

    LINEAR = "linear"  # Linear interpolation
    SPHERICAL = "spherical"  # Spherical linear interpolation (slerp)
    BEZIER = "bezier"  # Quadratic Bezier curve
    CUBIC = "cubic"  # Cubic spline interpolation


class BlendMode(str, Enum):
    """Modes for blending multiple tokens."""

    AVERAGE = "average"  # Simple average
    WEIGHTED = "weighted"  # Weighted average
    GEOMETRIC = "geometric"  # Geometric mean (normalized)
    ATTENTION = "attention"  # Attention-weighted blend


class MorphConfig(BaseModel):
    """Configuration for token morphing."""

    # Morphing method
    method: MorphMethod = Field(default=MorphMethod.LINEAR, description="Morphing method")

    # Number of intermediate steps
    num_steps: int = Field(default=10, ge=2, description="Number of intermediate steps")

    # Whether to include endpoints
    include_endpoints: bool = Field(default=True, description="Include start and end embeddings")

    # Normalization
    normalize_output: bool = Field(default=False, description="Normalize output embeddings")


class MorphResult(BaseModel):
    """Result of morphing between two tokens."""

    # Source info
    source_token: str = Field(description="Source token name/string")
    target_token: str = Field(description="Target token name/string")

    # Morphing config
    method: MorphMethod = Field(description="Method used")
    num_steps: int = Field(description="Number of steps")

    # Results (stored as lists for JSON serialization)
    alphas: list[float] = Field(description="Interpolation factors")
    embeddings: list[list[float]] = Field(description="Intermediate embeddings")

    def get_embeddings_array(self) -> np.ndarray:
        """Get embeddings as numpy array."""
        return np.array(self.embeddings, dtype=np.float32)

    def get_embedding_at(self, alpha: float) -> np.ndarray:
        """Get embedding at specific alpha (approximate if not exact match)."""
        alphas = np.array(self.alphas)
        idx = np.argmin(np.abs(alphas - alpha))
        return np.array(self.embeddings[idx], dtype=np.float32)


class MorphSequence(BaseModel):
    """Sequence of morphs through multiple tokens."""

    # Token sequence
    tokens: list[str] = Field(description="Token sequence")

    # Morph results between each pair
    morphs: list[MorphResult] = Field(description="Morph results between pairs")

    # Total steps
    total_steps: int = Field(description="Total number of embedding steps")

    def get_full_sequence(self) -> np.ndarray:
        """Get full embedding sequence as array."""
        if not self.morphs:
            return np.array([], dtype=np.float32)

        embeddings = []
        for i, morph in enumerate(self.morphs):
            arr = morph.get_embeddings_array()
            if i > 0:
                # Skip first embedding (duplicate from previous morph)
                arr = arr[1:]
            embeddings.append(arr)

        return np.vstack(embeddings) if embeddings else np.array([], dtype=np.float32)


class TokenBlend(BaseModel):
    """Result of blending multiple tokens."""

    # Source tokens
    tokens: list[str] = Field(description="Source token names/strings")
    weights: list[float] = Field(description="Blend weights")

    # Blend mode
    mode: BlendMode = Field(description="Blend mode used")

    # Result
    embedding: list[float] = Field(description="Blended embedding")

    def get_embedding_array(self) -> np.ndarray:
        """Get embedding as numpy array."""
        return np.array(self.embedding, dtype=np.float32)


def _linear_interpolate(e1: np.ndarray, e2: np.ndarray, alpha: float) -> np.ndarray:
    """Linear interpolation."""
    return (1 - alpha) * e1 + alpha * e2


def _spherical_interpolate(e1: np.ndarray, e2: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation (slerp)."""
    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)

    if norm1 < 1e-8 or norm2 < 1e-8:
        return _linear_interpolate(e1, e2, alpha)

    unit1 = e1 / norm1
    unit2 = e2 / norm2

    dot = np.clip(np.dot(unit1, unit2), -1.0, 1.0)
    theta = np.arccos(dot)

    if theta < 1e-8:
        return _linear_interpolate(e1, e2, alpha)

    sin_theta = np.sin(theta)
    result = (np.sin((1 - alpha) * theta) / sin_theta) * unit1 + (
        np.sin(alpha * theta) / sin_theta
    ) * unit2

    # Interpolate norm
    norm = (1 - alpha) * norm1 + alpha * norm2
    return result * norm


def _bezier_interpolate(
    e1: np.ndarray, e2: np.ndarray, alpha: float, control: np.ndarray | None = None
) -> np.ndarray:
    """Quadratic Bezier interpolation."""
    if control is None:
        # Use midpoint with slight perturbation as control
        control = (e1 + e2) / 2

    # Quadratic Bezier: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
    t = alpha
    return (1 - t) ** 2 * e1 + 2 * (1 - t) * t * control + t**2 * e2


def morph_token(
    source_embedding: np.ndarray,
    target_embedding: np.ndarray,
    source_name: str = "source",
    target_name: str = "target",
    config: MorphConfig | None = None,
    control_point: np.ndarray | None = None,
) -> MorphResult:
    """
    Morph between two token embeddings.

    Args:
        source_embedding: Source token embedding
        target_embedding: Target token embedding
        source_name: Name for source token
        target_name: Name for target token
        config: Morphing configuration
        control_point: Optional control point for Bezier interpolation

    Returns:
        MorphResult with intermediate embeddings
    """
    if config is None:
        config = MorphConfig()

    # Generate alpha values
    if config.include_endpoints:
        alphas = np.linspace(0, 1, config.num_steps).tolist()
    else:
        alphas = np.linspace(0, 1, config.num_steps + 2)[1:-1].tolist()

    # Generate intermediate embeddings
    embeddings: list[list[float]] = []

    for alpha in alphas:
        if config.method == MorphMethod.LINEAR:
            emb = _linear_interpolate(source_embedding, target_embedding, alpha)
        elif config.method == MorphMethod.SPHERICAL:
            emb = _spherical_interpolate(source_embedding, target_embedding, alpha)
        elif config.method == MorphMethod.BEZIER:
            emb = _bezier_interpolate(source_embedding, target_embedding, alpha, control_point)
        elif config.method == MorphMethod.CUBIC:
            # For cubic, we use a smooth step function
            # Hermite interpolation with zero derivatives at endpoints
            t = alpha
            h = 3 * t**2 - 2 * t**3  # Smooth step
            emb = (1 - h) * source_embedding + h * target_embedding
        else:
            emb = _linear_interpolate(source_embedding, target_embedding, alpha)

        if config.normalize_output:
            norm = np.linalg.norm(emb)
            if norm > 1e-8:
                emb = emb / norm

        embeddings.append(emb.tolist())

    return MorphResult(
        source_token=source_name,
        target_token=target_name,
        method=config.method,
        num_steps=len(alphas),
        alphas=alphas,
        embeddings=embeddings,
    )


def create_morph_sequence(
    embeddings: list[np.ndarray],
    names: list[str],
    config: MorphConfig | None = None,
) -> MorphSequence:
    """
    Create a morphing sequence through multiple tokens.

    Args:
        embeddings: List of token embeddings
        names: List of token names
        config: Morphing configuration

    Returns:
        MorphSequence with all intermediate embeddings
    """
    if len(embeddings) != len(names):
        raise ValueError("Embeddings and names must have same length")
    if len(embeddings) < 2:
        raise ValueError("Need at least 2 tokens for morphing")

    if config is None:
        config = MorphConfig()

    morphs: list[MorphResult] = []
    total_steps = 0

    for i in range(len(embeddings) - 1):
        morph = morph_token(
            embeddings[i],
            embeddings[i + 1],
            names[i],
            names[i + 1],
            config,
        )
        morphs.append(morph)

        # Count steps (avoid double-counting overlap)
        if i == 0:
            total_steps += morph.num_steps
        else:
            total_steps += morph.num_steps - 1

    return MorphSequence(
        tokens=names,
        morphs=morphs,
        total_steps=total_steps,
    )


def blend_tokens(
    embeddings: list[np.ndarray],
    names: list[str],
    weights: list[float] | None = None,
    mode: BlendMode = BlendMode.AVERAGE,
    normalize_weights: bool = True,
) -> TokenBlend:
    """
    Blend multiple token embeddings.

    Args:
        embeddings: List of token embeddings
        names: List of token names
        weights: Blend weights (None = equal weights)
        mode: Blending mode
        normalize_weights: Whether to normalize weights to sum to 1

    Returns:
        TokenBlend with blended embedding
    """
    if len(embeddings) != len(names):
        raise ValueError("Embeddings and names must have same length")
    if len(embeddings) == 0:
        raise ValueError("Need at least 1 token")

    if weights is None:
        weights = [1.0 / len(embeddings)] * len(embeddings)
    elif len(weights) != len(embeddings):
        raise ValueError("Weights must match number of embeddings")

    weights = list(weights)  # Copy
    if normalize_weights:
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

    embeddings_arr = np.stack(embeddings)
    weights_arr = np.array(weights)

    if mode == BlendMode.AVERAGE:
        blended = np.mean(embeddings_arr, axis=0)

    elif mode == BlendMode.WEIGHTED:
        blended = np.sum(embeddings_arr * weights_arr[:, np.newaxis], axis=0)

    elif mode == BlendMode.GEOMETRIC:
        # Geometric mean of normalized embeddings
        norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        normalized = embeddings_arr / norms

        # Weighted geometric mean
        weighted = normalized * weights_arr[:, np.newaxis]
        blended = np.exp(np.mean(np.log(np.abs(weighted) + 1e-8), axis=0))
        # Restore signs (use majority vote)
        signs = np.sign(np.sum(normalized, axis=0))
        blended = blended * signs

    elif mode == BlendMode.ATTENTION:
        # Compute attention-like weights based on embedding similarities
        mean_emb = np.mean(embeddings_arr, axis=0)
        similarities = np.dot(embeddings_arr, mean_emb)
        attn_weights = np.exp(similarities) / np.sum(np.exp(similarities))
        # Combine with explicit weights
        combined_weights = attn_weights * weights_arr
        combined_weights = combined_weights / np.sum(combined_weights)
        blended = np.sum(embeddings_arr * combined_weights[:, np.newaxis], axis=0)

    else:
        blended = np.mean(embeddings_arr, axis=0)

    return TokenBlend(
        tokens=names,
        weights=weights,
        mode=mode,
        embedding=blended.tolist(),
    )


def find_midpoint(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    method: MorphMethod = MorphMethod.LINEAR,
) -> np.ndarray:
    """
    Find the midpoint between two embeddings.

    Args:
        embedding1: First embedding
        embedding2: Second embedding
        method: Interpolation method

    Returns:
        Midpoint embedding
    """
    config = MorphConfig(method=method, num_steps=3, include_endpoints=True)
    result = morph_token(embedding1, embedding2, "a", "b", config)
    return np.array(result.embeddings[1], dtype=np.float32)


def compute_path_length(morph_result: MorphResult) -> float:
    """
    Compute the path length through embedding space.

    Args:
        morph_result: Morphing result

    Returns:
        Total Euclidean path length
    """
    embeddings = morph_result.get_embeddings_array()
    if len(embeddings) < 2:
        return 0.0

    diffs = embeddings[1:] - embeddings[:-1]
    distances = np.linalg.norm(diffs, axis=1)
    return float(np.sum(distances))


def compute_straightness(morph_result: MorphResult) -> float:
    """
    Compute how straight the morph path is (1.0 = perfectly straight).

    Args:
        morph_result: Morphing result

    Returns:
        Straightness score (0 to 1)
    """
    embeddings = morph_result.get_embeddings_array()
    if len(embeddings) < 2:
        return 1.0

    # Direct distance
    direct = np.linalg.norm(embeddings[-1] - embeddings[0])
    if direct < 1e-8:
        return 1.0

    # Path length
    path = compute_path_length(morph_result)

    # Straightness = direct / path (1.0 when perfectly straight)
    return float(direct / path) if path > 0 else 1.0
