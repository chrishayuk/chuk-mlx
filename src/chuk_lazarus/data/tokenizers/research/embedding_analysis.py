"""
Embedding space analysis and visualization utilities.

Tools for understanding token embedding spaces:
- Nearest neighbor search
- Clustering
- Dimensionality reduction (PCA, t-SNE projection)
- Analogy completion
- Embedding quality metrics
"""

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field


class ProjectionMethod(str, Enum):
    """Methods for dimensionality reduction."""

    PCA = "pca"
    RANDOM = "random"  # Random projection
    CENTERED = "centered"  # Centered PCA (mean subtracted)


class ClusterMethod(str, Enum):
    """Methods for clustering tokens."""

    KMEANS = "kmeans"
    AGGLOMERATIVE = "agglomerative"
    SPECTRAL = "spectral"


class DistanceMetric(str, Enum):
    """Distance metrics for similarity."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


class EmbeddingConfig(BaseModel):
    """Configuration for embedding analysis."""

    # Distance metric
    metric: DistanceMetric = Field(default=DistanceMetric.COSINE, description="Distance metric")

    # Nearest neighbors
    k_neighbors: int = Field(default=10, ge=1, description="Number of neighbors")

    # Clustering
    num_clusters: int = Field(default=10, ge=2, description="Number of clusters")
    cluster_method: ClusterMethod = Field(
        default=ClusterMethod.KMEANS, description="Clustering method"
    )

    # Projection
    projection_dim: int = Field(default=2, ge=1, description="Projection dimensionality")
    projection_method: ProjectionMethod = Field(
        default=ProjectionMethod.PCA, description="Projection method"
    )


class NeighborInfo(BaseModel):
    """Information about a nearest neighbor."""

    token_id: int = Field(description="Token ID")
    token_str: str = Field(description="Token string")
    distance: float = Field(description="Distance from query")
    similarity: float = Field(description="Similarity score (1 - normalized distance)")


class ClusterInfo(BaseModel):
    """Information about a cluster."""

    cluster_id: int = Field(description="Cluster ID")
    size: int = Field(description="Number of tokens in cluster")
    centroid: list[float] = Field(description="Cluster centroid")
    token_ids: list[int] = Field(description="Token IDs in cluster")
    token_strs: list[str] = Field(description="Token strings in cluster")
    intra_cluster_distance: float = Field(description="Average distance within cluster")


class ProjectionResult(BaseModel):
    """Result of dimensionality reduction."""

    # Method used
    method: ProjectionMethod = Field(description="Projection method")
    original_dim: int = Field(description="Original dimensionality")
    projected_dim: int = Field(description="Projected dimensionality")

    # Projected embeddings
    token_ids: list[int] = Field(description="Token IDs")
    token_strs: list[str] = Field(description="Token strings")
    coordinates: list[list[float]] = Field(description="Projected coordinates")

    # For PCA: explained variance
    explained_variance_ratio: list[float] = Field(
        default_factory=list, description="Variance explained by each component"
    )

    def get_coordinates_array(self) -> np.ndarray:
        """Get coordinates as numpy array."""
        return np.array(self.coordinates, dtype=np.float32)


class EmbeddingAnalysis(BaseModel):
    """Comprehensive embedding analysis results."""

    # Basic stats
    num_tokens: int = Field(description="Number of tokens analyzed")
    embedding_dim: int = Field(description="Embedding dimension")

    # Norms and distributions
    mean_norm: float = Field(description="Mean embedding norm")
    std_norm: float = Field(description="Std dev of embedding norms")
    min_norm: float = Field(description="Minimum norm")
    max_norm: float = Field(description="Maximum norm")

    # Isotropy (how uniformly distributed in space)
    isotropy_score: float = Field(description="Isotropy score (0-1)")

    # Clustering quality
    silhouette_score: float | None = Field(
        default=None, description="Silhouette score if clustered"
    )

    # Average similarity
    mean_pairwise_similarity: float = Field(description="Mean pairwise cosine similarity")


def _compute_distance(
    query: np.ndarray,
    embeddings: np.ndarray,
    metric: DistanceMetric,
) -> np.ndarray:
    """Compute distances from query to all embeddings."""
    if metric == DistanceMetric.COSINE:
        # Cosine distance = 1 - cosine similarity
        query_norm = np.linalg.norm(query)
        emb_norms = np.linalg.norm(embeddings, axis=1)
        if query_norm < 1e-8:
            return np.ones(len(embeddings))
        similarities = np.dot(embeddings, query) / (emb_norms * query_norm + 1e-8)
        return 1 - similarities

    elif metric == DistanceMetric.EUCLIDEAN:
        return np.linalg.norm(embeddings - query, axis=1)

    elif metric == DistanceMetric.DOT_PRODUCT:
        # Negative dot product (so lower = more similar)
        return -np.dot(embeddings, query)

    else:
        return np.linalg.norm(embeddings - query, axis=1)


def find_nearest_neighbors(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    token_ids: list[int],
    token_strs: list[str],
    k: int = 10,
    metric: DistanceMetric = DistanceMetric.COSINE,
    exclude_self: bool = True,
) -> list[NeighborInfo]:
    """
    Find k nearest neighbors to a query embedding.

    Args:
        query_embedding: Query embedding vector
        embeddings: Matrix of all embeddings (num_tokens, dim)
        token_ids: Token IDs corresponding to embeddings
        token_strs: Token strings corresponding to embeddings
        k: Number of neighbors to return
        metric: Distance metric to use
        exclude_self: Whether to exclude exact match

    Returns:
        List of NeighborInfo for k nearest neighbors
    """
    distances = _compute_distance(query_embedding, embeddings, metric)

    # Get sorted indices
    indices = np.argsort(distances)

    neighbors: list[NeighborInfo] = []
    for idx in indices:
        if exclude_self and distances[idx] < 1e-8:
            continue

        # Normalize distance to 0-1 range for similarity
        max_dist = np.max(distances)
        if max_dist > 0:
            similarity = 1 - (distances[idx] / max_dist)
        else:
            similarity = 1.0

        neighbors.append(
            NeighborInfo(
                token_id=token_ids[idx],
                token_str=token_strs[idx],
                distance=float(distances[idx]),
                similarity=float(similarity),
            )
        )

        if len(neighbors) >= k:
            break

    return neighbors


def cluster_tokens(
    embeddings: np.ndarray,
    token_ids: list[int],
    token_strs: list[str],
    num_clusters: int = 10,
    method: ClusterMethod = ClusterMethod.KMEANS,
    random_seed: int = 42,
) -> list[ClusterInfo]:
    """
    Cluster tokens based on embeddings.

    Args:
        embeddings: Embedding matrix (num_tokens, dim)
        token_ids: Token IDs
        token_strs: Token strings
        num_clusters: Number of clusters
        method: Clustering method
        random_seed: Random seed for reproducibility

    Returns:
        List of ClusterInfo for each cluster
    """
    np.random.seed(random_seed)
    n_tokens = len(embeddings)

    if n_tokens < num_clusters:
        num_clusters = n_tokens

    if method == ClusterMethod.KMEANS:
        # Simple k-means implementation
        labels = _kmeans_cluster(embeddings, num_clusters, random_seed)
    else:
        # Default to k-means for other methods (would need scipy for full impl)
        labels = _kmeans_cluster(embeddings, num_clusters, random_seed)

    # Build cluster info
    clusters: list[ClusterInfo] = []
    for c_id in range(num_clusters):
        mask = labels == c_id
        if not np.any(mask):
            continue

        cluster_embeddings = embeddings[mask]
        cluster_ids = [token_ids[i] for i in range(n_tokens) if mask[i]]
        cluster_strs = [token_strs[i] for i in range(n_tokens) if mask[i]]

        centroid = cluster_embeddings.mean(axis=0)

        # Intra-cluster distance
        if len(cluster_embeddings) > 1:
            dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            intra_dist = float(np.mean(dists))
        else:
            intra_dist = 0.0

        clusters.append(
            ClusterInfo(
                cluster_id=c_id,
                size=len(cluster_ids),
                centroid=centroid.tolist(),
                token_ids=cluster_ids,
                token_strs=cluster_strs,
                intra_cluster_distance=intra_dist,
            )
        )

    return clusters


def _kmeans_cluster(
    embeddings: np.ndarray, k: int, seed: int = 42, max_iter: int = 100
) -> np.ndarray:
    """Simple k-means clustering."""
    np.random.seed(seed)
    n_samples = len(embeddings)

    # Initialize centroids randomly
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = embeddings[indices].copy()

    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        # Assign to nearest centroid
        old_labels = labels.copy()
        for i in range(n_samples):
            distances = np.linalg.norm(centroids - embeddings[i], axis=1)
            labels[i] = np.argmin(distances)

        # Update centroids
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centroids[c] = embeddings[mask].mean(axis=0)

        # Check convergence
        if np.array_equal(labels, old_labels):
            break

    return labels


def project_embeddings(
    embeddings: np.ndarray,
    token_ids: list[int],
    token_strs: list[str],
    dim: int = 2,
    method: ProjectionMethod = ProjectionMethod.PCA,
) -> ProjectionResult:
    """
    Project embeddings to lower dimensions for visualization.

    Args:
        embeddings: Embedding matrix (num_tokens, dim)
        token_ids: Token IDs
        token_strs: Token strings
        dim: Target dimensionality
        method: Projection method

    Returns:
        ProjectionResult with projected coordinates
    """
    original_dim = embeddings.shape[1]
    explained_variance: list[float] = []

    if method == ProjectionMethod.PCA or method == ProjectionMethod.CENTERED:
        # Center the data
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean

        # SVD for PCA
        U, S, Vh = np.linalg.svd(centered, full_matrices=False)

        # Project to dim dimensions
        projected = centered @ Vh[:dim].T

        # Explained variance
        total_var = np.sum(S**2)
        if total_var > 0:
            explained_variance = (S[:dim] ** 2 / total_var).tolist()

    elif method == ProjectionMethod.RANDOM:
        # Random projection
        np.random.seed(42)
        projection_matrix = np.random.randn(original_dim, dim)
        projection_matrix /= np.linalg.norm(projection_matrix, axis=0)
        projected = embeddings @ projection_matrix

    else:
        # Default to PCA
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean
        U, S, Vh = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ Vh[:dim].T

    return ProjectionResult(
        method=method,
        original_dim=original_dim,
        projected_dim=dim,
        token_ids=token_ids,
        token_strs=token_strs,
        coordinates=projected.tolist(),
        explained_variance_ratio=explained_variance,
    )


def find_analogies(
    embeddings: np.ndarray,
    token_ids: list[int],
    token_strs: list[str],
    a_idx: int,
    b_idx: int,
    c_idx: int,
    k: int = 5,
) -> list[NeighborInfo]:
    """
    Find analogy completions: a is to b as c is to ?

    Uses the classic word2vec analogy formula: d = b - a + c

    Args:
        embeddings: Embedding matrix
        token_ids: Token IDs
        token_strs: Token strings
        a_idx: Index of token a
        b_idx: Index of token b
        c_idx: Index of token c
        k: Number of results to return

    Returns:
        List of candidate completions
    """
    # Compute target vector: b - a + c
    target = embeddings[b_idx] - embeddings[a_idx] + embeddings[c_idx]

    # Exclude input tokens
    exclude_indices = {a_idx, b_idx, c_idx}

    # Find nearest neighbors
    neighbors = find_nearest_neighbors(
        target,
        embeddings,
        token_ids,
        token_strs,
        k=k + len(exclude_indices),  # Get extra to filter
        metric=DistanceMetric.COSINE,
        exclude_self=False,
    )

    # Filter out input tokens
    filtered = []
    for n in neighbors:
        idx = token_ids.index(n.token_id) if n.token_id in token_ids else -1
        if idx not in exclude_indices:
            filtered.append(n)
        if len(filtered) >= k:
            break

    return filtered


def analyze_embeddings(
    embeddings: np.ndarray,
    sample_size: int = 1000,
    num_clusters: int = 10,
) -> EmbeddingAnalysis:
    """
    Comprehensive analysis of embedding space.

    Args:
        embeddings: Embedding matrix (num_tokens, dim)
        sample_size: Sample size for pairwise similarity (for efficiency)
        num_clusters: Number of clusters for silhouette score

    Returns:
        EmbeddingAnalysis with various metrics
    """
    n_tokens, dim = embeddings.shape

    # Compute norms
    norms = np.linalg.norm(embeddings, axis=1)

    # Sample for pairwise similarity (expensive for large vocab)
    if n_tokens > sample_size:
        indices = np.random.choice(n_tokens, sample_size, replace=False)
        sample = embeddings[indices]
    else:
        sample = embeddings

    # Pairwise cosine similarity
    norms_sample = np.linalg.norm(sample, axis=1, keepdims=True)
    normalized = sample / (norms_sample + 1e-8)
    similarity_matrix = normalized @ normalized.T
    # Exclude diagonal
    mask = ~np.eye(len(sample), dtype=bool)
    mean_similarity = float(np.mean(similarity_matrix[mask]))

    # Isotropy: measure how uniformly distributed embeddings are
    # Based on eigenvalue distribution of centered embeddings
    centered = embeddings - embeddings.mean(axis=0)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = S**2 / np.sum(S**2)
    # Isotropy = 1 - normalized entropy (uniform = 1, concentrated = 0)
    entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
    max_entropy = np.log(len(eigenvalues))
    isotropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # Silhouette score (simplified)
    silhouette = None
    if n_tokens >= num_clusters * 2:
        try:
            labels = _kmeans_cluster(embeddings, num_clusters)
            silhouette = _compute_silhouette(embeddings, labels)
        except Exception:
            silhouette = None

    return EmbeddingAnalysis(
        num_tokens=n_tokens,
        embedding_dim=dim,
        mean_norm=float(np.mean(norms)),
        std_norm=float(np.std(norms)),
        min_norm=float(np.min(norms)),
        max_norm=float(np.max(norms)),
        isotropy_score=isotropy,
        silhouette_score=silhouette,
        mean_pairwise_similarity=mean_similarity,
    )


def _compute_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute simplified silhouette score."""
    n = len(embeddings)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 0.0

    silhouette_vals = []

    for i in range(n):
        # Intra-cluster distance (a)
        same_cluster = labels == labels[i]
        same_cluster[i] = False
        if np.sum(same_cluster) == 0:
            continue
        a = np.mean(np.linalg.norm(embeddings[same_cluster] - embeddings[i], axis=1))

        # Nearest cluster distance (b)
        b = float("inf")
        for label in unique_labels:
            if label == labels[i]:
                continue
            other_cluster = labels == label
            if np.sum(other_cluster) == 0:
                continue
            dist = np.mean(np.linalg.norm(embeddings[other_cluster] - embeddings[i], axis=1))
            b = min(b, dist)

        if b == float("inf"):
            continue

        # Silhouette for this point
        s = (b - a) / max(a, b)
        silhouette_vals.append(s)

    return float(np.mean(silhouette_vals)) if silhouette_vals else 0.0


def compute_embedding_quality(
    embeddings: np.ndarray,
    token_strs: list[str],
) -> dict[str, float]:
    """
    Compute various embedding quality metrics.

    Args:
        embeddings: Embedding matrix
        token_strs: Token strings

    Returns:
        Dictionary of quality metrics
    """
    analysis = analyze_embeddings(embeddings)

    # Additional metrics
    metrics = {
        "isotropy": analysis.isotropy_score,
        "mean_similarity": analysis.mean_pairwise_similarity,
        "norm_std": analysis.std_norm / analysis.mean_norm if analysis.mean_norm > 0 else 0,
    }

    if analysis.silhouette_score is not None:
        metrics["silhouette"] = analysis.silhouette_score

    # Self-similarity (should be 1.0 for good embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    self_sim = np.mean(np.sum(normalized * normalized, axis=1))
    metrics["self_similarity"] = float(self_sim)

    return metrics
