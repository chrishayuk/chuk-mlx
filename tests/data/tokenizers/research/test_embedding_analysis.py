"""Tests for embedding analysis utilities."""

import numpy as np

from chuk_lazarus.data.tokenizers.research.embedding_analysis import (
    ClusterInfo,
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


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig model."""

    def test_default_values(self):
        config = EmbeddingConfig()
        assert config.metric == DistanceMetric.COSINE
        assert config.k_neighbors == 10
        assert config.num_clusters == 10

    def test_custom_values(self):
        config = EmbeddingConfig(
            metric=DistanceMetric.EUCLIDEAN,
            k_neighbors=5,
            projection_dim=3,
        )
        assert config.metric == DistanceMetric.EUCLIDEAN
        assert config.k_neighbors == 5


class TestNeighborInfo:
    """Tests for NeighborInfo model."""

    def test_valid_neighbor(self):
        neighbor = NeighborInfo(
            token_id=100,
            token_str="hello",
            distance=0.1,
            similarity=0.9,
        )
        assert neighbor.token_id == 100
        assert neighbor.similarity == 0.9


class TestClusterInfo:
    """Tests for ClusterInfo model."""

    def test_valid_cluster(self):
        cluster = ClusterInfo(
            cluster_id=0,
            size=5,
            centroid=[0.0, 1.0, 2.0],
            token_ids=[1, 2, 3, 4, 5],
            token_strs=["a", "b", "c", "d", "e"],
            intra_cluster_distance=0.5,
        )
        assert cluster.size == 5
        assert len(cluster.token_ids) == 5


class TestProjectionResult:
    """Tests for ProjectionResult model."""

    def test_get_coordinates_array(self):
        result = ProjectionResult(
            method=ProjectionMethod.PCA,
            original_dim=768,
            projected_dim=2,
            token_ids=[1, 2, 3],
            token_strs=["a", "b", "c"],
            coordinates=[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
        )
        arr = result.get_coordinates_array()
        assert arr.shape == (3, 2)


class TestEmbeddingAnalysis:
    """Tests for EmbeddingAnalysis model."""

    def test_valid_analysis(self):
        analysis = EmbeddingAnalysis(
            num_tokens=1000,
            embedding_dim=768,
            mean_norm=1.0,
            std_norm=0.1,
            min_norm=0.5,
            max_norm=1.5,
            isotropy_score=0.8,
            mean_pairwise_similarity=0.3,
        )
        assert analysis.num_tokens == 1000
        assert analysis.isotropy_score == 0.8


class TestFindNearestNeighbors:
    """Tests for find_nearest_neighbors function."""

    def test_basic_neighbors(self):
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        token_ids = [0, 1, 2, 3]
        token_strs = ["a", "b", "c", "d"]

        query = np.array([1.0, 0.0, 0.0])
        neighbors = find_nearest_neighbors(query, embeddings, token_ids, token_strs, k=2)

        assert len(neighbors) == 2
        # First neighbor should be "b" (closest to "a")
        assert neighbors[0].token_str == "b"

    def test_euclidean_distance(self):
        embeddings = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        token_ids = [0, 1, 2, 3]
        token_strs = ["origin", "x", "y", "xy"]

        query = np.array([0.1, 0.1])
        neighbors = find_nearest_neighbors(
            query,
            embeddings,
            token_ids,
            token_strs,
            k=2,
            metric=DistanceMetric.EUCLIDEAN,
        )

        # Origin should be closest
        assert neighbors[0].token_str == "origin"

    def test_dot_product(self):
        embeddings = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],  # Larger magnitude, higher dot product
                [0.5, 0.0],
            ]
        )
        token_ids = [0, 1, 2]
        token_strs = ["a", "b", "c"]

        query = np.array([1.0, 0.0])
        neighbors = find_nearest_neighbors(
            query,
            embeddings,
            token_ids,
            token_strs,
            k=3,
            metric=DistanceMetric.DOT_PRODUCT,
            exclude_self=False,  # Don't exclude any
        )

        # "b" has highest dot product
        assert neighbors[0].token_str == "b"

    def test_exclude_self(self):
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
            ]
        )
        token_ids = [0, 1]
        token_strs = ["a", "b"]

        query = np.array([1.0, 0.0])  # Exact match with "a"
        neighbors = find_nearest_neighbors(
            query,
            embeddings,
            token_ids,
            token_strs,
            k=2,
            exclude_self=True,
        )

        # Should not include exact match
        assert all(n.token_str != "a" or n.distance > 0.01 for n in neighbors)


class TestClusterTokens:
    """Tests for cluster_tokens function."""

    def test_basic_clustering(self):
        # Create embeddings with clear clusters
        np.random.seed(42)
        cluster1 = np.random.randn(10, 4) + np.array([5, 0, 0, 0])
        cluster2 = np.random.randn(10, 4) + np.array([0, 5, 0, 0])
        embeddings = np.vstack([cluster1, cluster2])
        token_ids = list(range(20))
        token_strs = [f"token_{i}" for i in range(20)]

        clusters = cluster_tokens(embeddings, token_ids, token_strs, num_clusters=2)

        assert len(clusters) == 2
        assert sum(c.size for c in clusters) == 20

    def test_single_cluster(self):
        embeddings = np.random.randn(10, 4)
        token_ids = list(range(10))
        token_strs = [f"t_{i}" for i in range(10)]

        # If fewer tokens than clusters, should adjust
        clusters = cluster_tokens(embeddings, token_ids, token_strs, num_clusters=20)
        assert len(clusters) <= 10


class TestProjectEmbeddings:
    """Tests for project_embeddings function."""

    def test_pca_projection(self):
        embeddings = np.random.randn(50, 100)
        token_ids = list(range(50))
        token_strs = [f"t_{i}" for i in range(50)]

        result = project_embeddings(
            embeddings,
            token_ids,
            token_strs,
            dim=2,
            method=ProjectionMethod.PCA,
        )

        assert result.original_dim == 100
        assert result.projected_dim == 2
        assert len(result.coordinates) == 50
        assert len(result.coordinates[0]) == 2

    def test_explained_variance(self):
        embeddings = np.random.randn(100, 50)
        token_ids = list(range(100))
        token_strs = [f"t_{i}" for i in range(100)]

        result = project_embeddings(
            embeddings,
            token_ids,
            token_strs,
            dim=5,
            method=ProjectionMethod.PCA,
        )

        # Should have variance ratios
        assert len(result.explained_variance_ratio) == 5
        # Sum should be <= 1
        assert sum(result.explained_variance_ratio) <= 1.0

    def test_random_projection(self):
        embeddings = np.random.randn(30, 200)
        token_ids = list(range(30))
        token_strs = [f"t_{i}" for i in range(30)]

        result = project_embeddings(
            embeddings,
            token_ids,
            token_strs,
            dim=3,
            method=ProjectionMethod.RANDOM,
        )

        assert result.method == ProjectionMethod.RANDOM
        assert result.projected_dim == 3


class TestFindAnalogies:
    """Tests for find_analogies function."""

    def test_simple_analogy(self):
        # Create embeddings where man:woman :: king:?
        # Simplified: just test that function works
        embeddings = np.array(
            [
                [1.0, 0.0],  # 0: "man"
                [1.0, 1.0],  # 1: "woman" (man + gender)
                [2.0, 0.0],  # 2: "king" (man + royalty)
                [2.0, 1.0],  # 3: "queen" (king + gender = woman + royalty)
                [0.0, 0.0],  # 4: "other"
            ]
        )
        token_ids = [0, 1, 2, 3, 4]
        token_strs = ["man", "woman", "king", "queen", "other"]

        # man:woman :: king:? should give queen
        results = find_analogies(
            embeddings,
            token_ids,
            token_strs,
            a_idx=0,
            b_idx=1,
            c_idx=2,
            k=1,
        )

        assert len(results) >= 1
        # "queen" should be top result
        assert results[0].token_str == "queen"


class TestAnalyzeEmbeddings:
    """Tests for analyze_embeddings function."""

    def test_basic_analysis(self):
        embeddings = np.random.randn(100, 64).astype(np.float32)
        analysis = analyze_embeddings(embeddings)

        assert analysis.num_tokens == 100
        assert analysis.embedding_dim == 64
        assert analysis.mean_norm > 0
        assert 0 <= analysis.isotropy_score <= 1
        assert -1 <= analysis.mean_pairwise_similarity <= 1

    def test_small_sample(self):
        embeddings = np.random.randn(10, 32).astype(np.float32)
        analysis = analyze_embeddings(embeddings, sample_size=5)

        assert analysis.num_tokens == 10

    def test_silhouette_computed(self):
        # Create well-separated clusters
        np.random.seed(42)
        cluster1 = np.random.randn(20, 8) + np.array([10, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(20, 8) + np.array([0, 10, 0, 0, 0, 0, 0, 0])
        embeddings = np.vstack([cluster1, cluster2]).astype(np.float32)

        analysis = analyze_embeddings(embeddings, num_clusters=2)

        # Should have positive silhouette for well-separated clusters
        assert analysis.silhouette_score is not None


class TestComputeEmbeddingQuality:
    """Tests for compute_embedding_quality function."""

    def test_basic_quality(self):
        embeddings = np.random.randn(50, 32).astype(np.float32)
        token_strs = [f"t_{i}" for i in range(50)]

        quality = compute_embedding_quality(embeddings, token_strs)

        assert "isotropy" in quality
        assert "mean_similarity" in quality
        assert "self_similarity" in quality
        assert quality["self_similarity"] > 0.99  # Should be ~1.0
