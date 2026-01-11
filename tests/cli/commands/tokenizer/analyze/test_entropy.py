"""Tests for analyze_entropy command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import AnalyzeEntropyConfig
from chuk_lazarus.cli.commands.tokenizer.analyze.entropy import analyze_entropy


class TestAnalyzeEntropyConfig:
    """Tests for AnalyzeEntropyConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.file = None
        args.top_n = 20

        config = AnalyzeEntropyConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.file is None
        assert config.top_n == 20

    def test_from_args_with_file(self):
        """Test config with file and custom top_n."""
        args = MagicMock()
        args.tokenizer = "llama"
        args.file = Path("/path/to/corpus.txt")
        args.top_n = 50

        config = AnalyzeEntropyConfig.from_args(args)

        assert config.tokenizer == "llama"
        assert config.file == Path("/path/to/corpus.txt")
        assert config.top_n == 50


class TestAnalyzeEntropy:
    """Tests for analyze_entropy function."""

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.entropy.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    def test_analyze_entropy_no_texts(self, mock_load_tokenizer, mock_load_texts, capsys):
        """Test with no texts provided."""
        mock_load_tokenizer.return_value = MagicMock()
        mock_load_texts.return_value = []

        config = AnalyzeEntropyConfig(tokenizer="gpt2")
        analyze_entropy(config)

        # Should return early with no output for the report
        captured = capsys.readouterr()
        assert "Entropy Report" not in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.entropy.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.analyze.analyze_entropy")
    def test_analyze_entropy_basic(
        self, mock_do_analyze, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test basic entropy analysis."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Hello world", "Test text"]

        mock_report = MagicMock()
        mock_report.entropy = 8.5
        mock_report.perplexity = 256.0
        mock_report.normalized_entropy = 0.85
        mock_report.uniformity_score = 0.72
        mock_report.concentration_ratio = 0.15
        mock_report.distribution = None
        mock_do_analyze.return_value = mock_report

        config = AnalyzeEntropyConfig(tokenizer="gpt2")
        analyze_entropy(config)

        captured = capsys.readouterr()
        assert "Entropy Report" in captured.out
        assert "Entropy:" in captured.out
        assert "Perplexity:" in captured.out
        assert "Normalized:" in captured.out
        assert "Uniformity:" in captured.out
        assert "Concentration:" in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.entropy.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.analyze.analyze_entropy")
    def test_analyze_entropy_with_distribution(
        self, mock_do_analyze, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test entropy analysis with token distribution."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Hello world test text"]

        mock_distribution = MagicMock()
        mock_distribution.top_tokens = {
            "the": 100,
            "a": 80,
            "is": 60,
            "and": 50,
            "to": 40,
        }

        mock_report = MagicMock()
        mock_report.entropy = 9.2
        mock_report.perplexity = 512.0
        mock_report.normalized_entropy = 0.92
        mock_report.uniformity_score = 0.85
        mock_report.concentration_ratio = 0.08
        mock_report.distribution = mock_distribution
        mock_do_analyze.return_value = mock_report

        config = AnalyzeEntropyConfig(tokenizer="gpt2", top_n=20)
        analyze_entropy(config)

        captured = capsys.readouterr()
        assert "Top" in captured.out
        assert "tokens:" in captured.out
