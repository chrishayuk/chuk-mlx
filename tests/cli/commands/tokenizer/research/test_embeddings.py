"""Tests for research_analyze_embeddings command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import ResearchEmbeddingsConfig
from chuk_lazarus.cli.commands.tokenizer.research.embeddings import (
    research_analyze_embeddings,
)


class TestResearchEmbeddingsConfig:
    """Tests for ResearchEmbeddingsConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.file = "/path/to/embeddings.json"
        args.num_clusters = 10
        args.cluster = False
        args.project = False

        config = ResearchEmbeddingsConfig.from_args(args)

        assert config.file == Path("/path/to/embeddings.json")
        assert config.num_clusters == 10
        assert config.cluster is False
        assert config.project is False

    def test_from_args_with_options(self):
        """Test config with all options."""
        args = MagicMock()
        args.file = "/path/to/embeddings.json"
        args.num_clusters = 20
        args.cluster = True
        args.project = True

        config = ResearchEmbeddingsConfig.from_args(args)

        assert config.num_clusters == 20
        assert config.cluster is True
        assert config.project is True


class TestResearchAnalyzeEmbeddings:
    """Tests for research_analyze_embeddings function."""

    @patch("chuk_lazarus.data.tokenizers.research.analyze_embeddings")
    @patch("builtins.open")
    @patch("json.load")
    def test_analyze_embeddings_basic(
        self, mock_json_load, mock_open, mock_analyze, capsys, tmp_path
    ):
        """Test basic embedding analysis."""
        # Setup mock embeddings data
        mock_json_load.return_value = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "token_ids": [0, 1],
            "token_strs": ["hello", "world"],
        }

        mock_analysis = MagicMock()
        mock_analysis.num_tokens = 2
        mock_analysis.embedding_dim = 3
        mock_analysis.mean_norm = 0.5
        mock_analysis.std_norm = 0.1
        mock_analysis.isotropy_score = 0.8
        mock_analysis.mean_pairwise_similarity = 0.3
        mock_analysis.silhouette_score = 0.6
        mock_analyze.return_value = mock_analysis

        config = ResearchEmbeddingsConfig(
            file=tmp_path / "embeddings.json",
            num_clusters=5,
            cluster=False,
            project=False,
        )
        research_analyze_embeddings(config)

        captured = capsys.readouterr()
        assert "Embedding Analysis" in captured.out
        assert "Num tokens:" in captured.out
        assert "Embedding dim:" in captured.out
        assert "Mean norm:" in captured.out
        assert "Isotropy:" in captured.out

    @patch("chuk_lazarus.data.tokenizers.research.analyze_embeddings")
    @patch("chuk_lazarus.data.tokenizers.research.cluster_tokens")
    @patch("builtins.open")
    @patch("json.load")
    def test_analyze_embeddings_with_clustering(
        self, mock_json_load, mock_open, mock_cluster, mock_analyze, capsys, tmp_path
    ):
        """Test embedding analysis with clustering."""
        mock_json_load.return_value = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "token_ids": [0, 1],
            "token_strs": ["hello", "world"],
        }

        mock_analysis = MagicMock()
        mock_analysis.num_tokens = 2
        mock_analysis.embedding_dim = 3
        mock_analysis.mean_norm = 0.5
        mock_analysis.std_norm = 0.1
        mock_analysis.isotropy_score = 0.8
        mock_analysis.mean_pairwise_similarity = 0.3
        mock_analysis.silhouette_score = None
        mock_analyze.return_value = mock_analysis

        mock_cluster_result = MagicMock()
        mock_cluster_result.cluster_id = 0
        mock_cluster_result.size = 2
        mock_cluster_result.intra_cluster_distance = 0.2
        mock_cluster_result.token_strs = ["hello", "world"]
        mock_cluster.return_value = [mock_cluster_result]

        config = ResearchEmbeddingsConfig(
            file=tmp_path / "embeddings.json",
            num_clusters=5,
            cluster=True,
            project=False,
        )
        research_analyze_embeddings(config)

        captured = capsys.readouterr()
        assert "Clustering" in captured.out
        assert "Cluster 0:" in captured.out
