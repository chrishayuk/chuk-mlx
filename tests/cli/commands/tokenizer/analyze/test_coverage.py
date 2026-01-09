"""Tests for analyze_coverage command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import AnalyzeCoverageConfig
from chuk_lazarus.cli.commands.tokenizer.analyze.coverage import analyze_coverage


class TestAnalyzeCoverageConfig:
    """Tests for AnalyzeCoverageConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.file = None
        args.fragments = False

        config = AnalyzeCoverageConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.file is None
        assert config.fragments is False

    def test_from_args_with_file(self):
        """Test config with file."""
        args = MagicMock()
        args.tokenizer = "llama"
        args.file = Path("/path/to/corpus.txt")
        args.fragments = True

        config = AnalyzeCoverageConfig.from_args(args)

        assert config.tokenizer == "llama"
        assert config.file == Path("/path/to/corpus.txt")
        assert config.fragments is True


class TestAnalyzeCoverage:
    """Tests for analyze_coverage function."""

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.coverage.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    def test_analyze_coverage_no_texts(self, mock_load_tokenizer, mock_load_texts, capsys):
        """Test with no texts provided."""
        mock_load_tokenizer.return_value = MagicMock()
        mock_load_texts.return_value = []

        config = AnalyzeCoverageConfig(tokenizer="gpt2")
        analyze_coverage(config)

        # Should return early with no output for the report
        captured = capsys.readouterr()
        assert "Coverage Report" not in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.coverage.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.analyze.analyze_coverage")
    def test_analyze_coverage_basic(
        self, mock_do_analyze, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test basic coverage analysis."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Hello world", "Test text"]

        # Mock the analysis function
        mock_report = MagicMock()
        mock_report.total_tokens = 10
        mock_report.unique_tokens = 8
        mock_report.unk_rate = 0.05
        mock_report.tokens_per_word = 1.25
        mock_report.vocab_utilization = 0.0025
        mock_report.warnings = []
        mock_report.fragments = None
        mock_do_analyze.return_value = mock_report

        config = AnalyzeCoverageConfig(tokenizer="gpt2", fragments=False)
        analyze_coverage(config)

        captured = capsys.readouterr()
        assert "Coverage Report" in captured.out
        assert "Total tokens:" in captured.out
        assert "Unique tokens:" in captured.out
        assert "UNK rate:" in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.coverage.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.analyze.analyze_coverage")
    def test_analyze_coverage_with_warnings(
        self, mock_do_analyze, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test coverage analysis with warnings."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Hello world"]

        mock_report = MagicMock()
        mock_report.total_tokens = 5
        mock_report.unique_tokens = 4
        mock_report.unk_rate = 0.2
        mock_report.tokens_per_word = 2.5
        mock_report.vocab_utilization = 0.001
        mock_report.warnings = ["High UNK rate detected", "Poor coverage"]
        mock_report.fragments = None
        mock_do_analyze.return_value = mock_report

        config = AnalyzeCoverageConfig(tokenizer="gpt2")
        analyze_coverage(config)

        captured = capsys.readouterr()
        assert "Warnings:" in captured.out
        assert "High UNK rate detected" in captured.out
        assert "Poor coverage" in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.coverage.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.analyze.analyze_coverage")
    def test_analyze_coverage_with_fragments(
        self, mock_do_analyze, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test coverage analysis with fragments enabled."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Testing tokenization"]

        mock_fragments = MagicMock()
        mock_fragments.top_fragmented = [
            "tokenization (3 pieces)",
            "fragmented (2 pieces)",
            "analysis (2 pieces)",
        ]

        mock_report = MagicMock()
        mock_report.total_tokens = 5
        mock_report.unique_tokens = 4
        mock_report.unk_rate = 0.0
        mock_report.tokens_per_word = 2.5
        mock_report.vocab_utilization = 0.001
        mock_report.warnings = []
        mock_report.fragments = mock_fragments
        mock_do_analyze.return_value = mock_report

        config = AnalyzeCoverageConfig(tokenizer="gpt2", fragments=True)
        analyze_coverage(config)

        captured = capsys.readouterr()
        assert "Top Fragmented Words:" in captured.out
        assert "tokenization" in captured.out
