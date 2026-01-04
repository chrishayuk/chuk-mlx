"""Embedding analysis research command handler."""

import json
import logging

import numpy as np

from .._types import ResearchEmbeddingsConfig

logger = logging.getLogger(__name__)


def research_analyze_embeddings(config: ResearchEmbeddingsConfig) -> None:
    """Analyze embedding space from a file.

    Args:
        config: Embeddings analysis configuration.
    """
    from .....data.tokenizers.research import (
        analyze_embeddings,
        cluster_tokens,
        project_embeddings,
    )

    # Load embeddings from file
    logger.info(f"Loading embeddings from: {config.file}")
    with open(config.file) as f:
        data = json.load(f)

    if "embeddings" in data:
        embeddings = np.array(data["embeddings"], dtype=np.float32)
        token_ids = data.get("token_ids", list(range(len(embeddings))))
        token_strs = data.get("token_strs", [f"token_{i}" for i in range(len(embeddings))])
    else:
        logger.error("File must contain 'embeddings' key")
        return

    print("\n=== Embedding Analysis ===")
    analysis = analyze_embeddings(embeddings, num_clusters=config.num_clusters)

    print(f"Num tokens:      {analysis.num_tokens}")
    print(f"Embedding dim:   {analysis.embedding_dim}")
    print(f"Mean norm:       {analysis.mean_norm:.4f}")
    print(f"Norm std:        {analysis.std_norm:.4f}")
    print(f"Isotropy:        {analysis.isotropy_score:.4f}")
    print(f"Mean similarity: {analysis.mean_pairwise_similarity:.4f}")
    if analysis.silhouette_score is not None:
        print(f"Silhouette:      {analysis.silhouette_score:.4f}")

    if config.cluster:
        print(f"\n=== Clustering ({config.num_clusters} clusters) ===")
        clusters = cluster_tokens(embeddings, token_ids, token_strs, config.num_clusters)
        for c in clusters:
            sample = c.token_strs[:3]
            print(f"  Cluster {c.cluster_id}: {c.size} tokens")
            print(f"    Intra-dist: {c.intra_cluster_distance:.4f}")
            print(f"    Sample: {sample}")

    if config.project:
        print("\n=== 2D Projection ===")
        projection = project_embeddings(embeddings, token_ids, token_strs, dim=2)
        print(f"Variance explained: {sum(projection.explained_variance_ratio):.2%}")
        coords = projection.get_coordinates_array()
        print(f"X range: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
        print(f"Y range: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
