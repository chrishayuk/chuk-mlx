#!/usr/bin/env python3
"""
Research Playground Examples

Demonstrates experimental tokenization techniques:
- Soft tokens for prompt tuning
- Token morphing and blending
- Embedding space analysis
"""

import numpy as np

from chuk_lazarus.data.tokenizers.research import (
    # Soft tokens
    BlendMode,
    InitializationMethod,
    MorphConfig,
    MorphMethod,
    SoftTokenConfig,
    analyze_embeddings,
    blend_tokens,
    cluster_tokens,
    create_control_token,
    create_morph_sequence,
    create_prompt_tuning_bank,
    create_soft_token,
    find_analogies,
    find_nearest_neighbors,
    interpolate_embeddings,
    morph_token,
    project_embeddings,
)


def demo_soft_tokens():
    """Demonstrate soft token creation for prompt tuning."""
    print("=" * 60)
    print("SOFT TOKENS FOR PROMPT TUNING")
    print("=" * 60)

    # Create a bank of soft prompt tokens
    embedding_dim = 256
    bank = create_prompt_tuning_bank(
        num_tokens=5,
        embedding_dim=embedding_dim,
        prefix="task",
        init_method=InitializationMethod.RANDOM_NORMAL,
        init_std=0.02,
    )

    print(f"\nCreated soft token bank: {bank.name}")
    print(f"  Embedding dimension: {bank.embedding_dim}")
    print(f"  Number of tokens: {len(bank.tokens)}")

    for token in bank.tokens:
        emb = token.embedding_array
        print(f"  - {token.token.name} (ID: {token.token.token_id})")
        print(f"    Norm: {np.linalg.norm(emb):.4f}, Mean: {emb.mean():.4f}")

    # Get embeddings matrix for training
    embeddings_matrix = bank.get_embeddings_matrix()
    print(f"\nEmbeddings matrix shape: {embeddings_matrix.shape}")

    # Create control tokens for controllable generation
    print("\n--- Control Tokens ---")
    positive = create_control_token(
        "positive_sentiment",
        embedding_dim=embedding_dim,
        description="Encourages positive output tone",
    )
    negative = create_control_token(
        "negative_sentiment",
        embedding_dim=embedding_dim,
        description="Encourages negative output tone",
    )

    print(f"Created control tokens:")
    print(f"  - {positive.token.name}: {positive.token.description}")
    print(f"  - {negative.token.name}: {negative.token.description}")

    # Create soft token from existing embeddings
    print("\n--- Soft Token from Existing Embeddings ---")
    source_embeddings = np.random.randn(10, embedding_dim).astype(np.float32)
    config = SoftTokenConfig(
        embedding_dim=embedding_dim,
        init_method=InitializationMethod.FROM_TOKENS,
        trainable=True,
    )
    derived = create_soft_token(
        "derived_token",
        config,
        token_id=100010,
        description="Token initialized from average of 10 embeddings",
        source_embeddings=source_embeddings,
    )
    print(f"Created: {derived.token.name}")
    print(f"  Initialized from mean of {len(source_embeddings)} embeddings")


def demo_token_morphing():
    """Demonstrate token morphing and blending."""
    print("\n" + "=" * 60)
    print("TOKEN MORPHING AND BLENDING")
    print("=" * 60)

    dim = 64

    # Create two embeddings to morph between
    e1 = np.random.randn(dim).astype(np.float32)
    e2 = np.random.randn(dim).astype(np.float32)

    # Normalize for spherical interpolation demo
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)

    print("\n--- Linear vs Spherical Interpolation ---")
    print(f"e1 norm: {np.linalg.norm(e1):.4f}")
    print(f"e2 norm: {np.linalg.norm(e2):.4f}")

    # Compare interpolation methods at midpoint
    linear_mid = interpolate_embeddings(e1, e2, alpha=0.5, method="linear")
    spherical_mid = interpolate_embeddings(e1, e2, alpha=0.5, method="spherical")

    print(f"\nAt alpha=0.5:")
    print(f"  Linear midpoint norm: {np.linalg.norm(linear_mid):.4f}")
    print(f"  Spherical midpoint norm: {np.linalg.norm(spherical_mid):.4f}")
    print("  (Spherical preserves unit norm along geodesic)")

    # Morph with multiple steps
    print("\n--- Token Morphing Trajectory ---")
    config = MorphConfig(
        method=MorphMethod.SPHERICAL,
        num_steps=5,
        include_endpoints=True,
    )
    result = morph_token(e1, e2, "start_token", "end_token", config)

    print(f"Morph from '{result.source_token}' to '{result.target_token}'")
    print(f"Method: {result.method.value}")
    print(f"Steps: {result.num_steps}")

    trajectory = result.get_embeddings_array()
    print(f"\nTrajectory norms (should all be ~1.0 for spherical):")
    for i, alpha in enumerate(result.alphas):
        norm = np.linalg.norm(trajectory[i])
        print(f"  alpha={alpha:.2f}: norm={norm:.4f}")

    # Multi-token morphing sequence
    print("\n--- Multi-Token Morph Sequence ---")
    embeddings = [np.random.randn(dim).astype(np.float32) for _ in range(4)]
    names = ["token_A", "token_B", "token_C", "token_D"]

    seq = create_morph_sequence(embeddings, names, config)
    print(f"Sequence through: {' -> '.join(seq.tokens)}")
    print(f"Total steps: {seq.total_steps}")
    print(f"Number of morph segments: {len(seq.morphs)}")

    # Token blending
    print("\n--- Token Blending ---")
    e3 = np.random.randn(dim).astype(np.float32)

    for mode in [BlendMode.AVERAGE, BlendMode.WEIGHTED, BlendMode.ATTENTION]:
        blend = blend_tokens(
            [e1, e2, e3],
            ["alpha", "beta", "gamma"],
            weights=[0.5, 0.3, 0.2],
            mode=mode,
        )
        blended = blend.get_embedding_array()
        print(f"  {mode.value}: norm={np.linalg.norm(blended):.4f}")


def demo_embedding_analysis():
    """Demonstrate embedding space analysis."""
    print("\n" + "=" * 60)
    print("EMBEDDING SPACE ANALYSIS")
    print("=" * 60)

    # Create synthetic embeddings with structure
    np.random.seed(42)
    dim = 64
    n_tokens = 100

    # Create 4 clusters
    cluster_centers = [
        np.array([5, 0, 0, 0] + [0] * (dim - 4)),
        np.array([0, 5, 0, 0] + [0] * (dim - 4)),
        np.array([0, 0, 5, 0] + [0] * (dim - 4)),
        np.array([0, 0, 0, 5] + [0] * (dim - 4)),
    ]

    embeddings = []
    for i in range(n_tokens):
        center = cluster_centers[i % 4]
        emb = center + np.random.randn(dim) * 0.5
        embeddings.append(emb.astype(np.float32))

    embeddings = np.stack(embeddings)
    token_ids = list(range(n_tokens))
    token_strs = [f"word_{i}" for i in range(n_tokens)]

    # Find nearest neighbors
    print("\n--- Nearest Neighbors ---")
    query_idx = 0
    neighbors = find_nearest_neighbors(
        embeddings[query_idx],
        embeddings,
        token_ids,
        token_strs,
        k=5,
    )

    print(f"Neighbors of '{token_strs[query_idx]}':")
    for n in neighbors:
        print(f"  {n.token_str}: distance={n.distance:.4f}, similarity={n.similarity:.4f}")

    # Cluster tokens
    print("\n--- Token Clustering ---")
    clusters = cluster_tokens(
        embeddings,
        token_ids,
        token_strs,
        num_clusters=4,
    )

    print(f"Found {len(clusters)} clusters:")
    for c in clusters:
        sample_tokens = c.token_strs[:3]
        print(f"  Cluster {c.cluster_id}: {c.size} tokens, intra-dist={c.intra_cluster_distance:.4f}")
        print(f"    Sample: {sample_tokens}")

    # Project to 2D
    print("\n--- 2D Projection (PCA) ---")
    projection = project_embeddings(embeddings, token_ids, token_strs, dim=2)

    print(f"Original dim: {projection.original_dim}")
    print(f"Projected dim: {projection.projected_dim}")
    if projection.explained_variance_ratio:
        total_var = sum(projection.explained_variance_ratio)
        print(f"Variance explained: {total_var:.2%}")

    coords = projection.get_coordinates_array()
    print(f"Coordinates shape: {coords.shape}")
    print(f"X range: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
    print(f"Y range: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")

    # Analogy finding
    print("\n--- Analogy Completion ---")
    # Create embeddings with analogy structure: a:b :: c:d
    analogy_embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # man
            [1.0, 1.0, 0.0, 0.0],  # woman (man + gender)
            [2.0, 0.0, 0.0, 0.0],  # king (man + royalty)
            [2.0, 1.0, 0.0, 0.0],  # queen (king + gender)
            [0.0, 0.0, 1.0, 0.0],  # other
        ],
        dtype=np.float32,
    )
    analogy_ids = [0, 1, 2, 3, 4]
    analogy_strs = ["man", "woman", "king", "queen", "other"]

    # man:woman :: king:?
    results = find_analogies(
        analogy_embeddings,
        analogy_ids,
        analogy_strs,
        a_idx=0,  # man
        b_idx=1,  # woman
        c_idx=2,  # king
        k=2,
    )

    print("Analogy: man:woman :: king:?")
    print("Results:")
    for r in results:
        print(f"  {r.token_str} (similarity={r.similarity:.4f})")

    # Comprehensive analysis
    print("\n--- Comprehensive Embedding Analysis ---")
    analysis = analyze_embeddings(embeddings, num_clusters=4)

    print(f"Number of tokens: {analysis.num_tokens}")
    print(f"Embedding dimension: {analysis.embedding_dim}")
    print(f"Mean norm: {analysis.mean_norm:.4f}")
    print(f"Norm std: {analysis.std_norm:.4f}")
    print(f"Isotropy score: {analysis.isotropy_score:.4f}")
    print(f"Mean pairwise similarity: {analysis.mean_pairwise_similarity:.4f}")
    if analysis.silhouette_score is not None:
        print(f"Silhouette score: {analysis.silhouette_score:.4f}")


def main():
    """Run all research playground demos."""
    print("RESEARCH PLAYGROUND EXAMPLES")
    print("Experimental tokenization techniques\n")

    demo_soft_tokens()
    demo_token_morphing()
    demo_embedding_analysis()

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
