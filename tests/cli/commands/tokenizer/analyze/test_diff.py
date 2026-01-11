"""Tests for analyze_diff command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import AnalyzeDiffConfig
from chuk_lazarus.cli.commands.tokenizer.analyze.diff import analyze_diff


class TestAnalyzeDiffConfig:
    """Tests for AnalyzeDiffConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.tokenizer1 = "gpt2"
        args.tokenizer2 = "llama"
        args.file = None

        config = AnalyzeDiffConfig.from_args(args)

        assert config.tokenizer1 == "gpt2"
        assert config.tokenizer2 == "llama"
        assert config.file is None

    def test_from_args_with_file(self):
        """Test config with file."""
        args = MagicMock()
        args.tokenizer1 = "bert"
        args.tokenizer2 = "roberta"
        args.file = Path("/path/to/corpus.txt")

        config = AnalyzeDiffConfig.from_args(args)

        assert config.tokenizer1 == "bert"
        assert config.tokenizer2 == "roberta"
        assert config.file == Path("/path/to/corpus.txt")


class TestAnalyzeDiff:
    """Tests for analyze_diff function."""

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.diff.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    def test_analyze_diff_no_texts(self, mock_load_tokenizer, mock_load_texts, capsys):
        """Test with no texts provided."""
        mock_load_tokenizer.return_value = MagicMock()
        mock_load_texts.return_value = []

        config = AnalyzeDiffConfig(tokenizer1="gpt2", tokenizer2="llama")
        analyze_diff(config)

        # Should return early with no output for the report
        captured = capsys.readouterr()
        assert "Corpus Diff Report" not in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.diff.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.analyze.diff_corpus")
    def test_analyze_diff_basic(
        self, mock_diff_corpus, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test basic diff analysis."""
        mock_tokenizer1 = MagicMock()
        mock_tokenizer2 = MagicMock()
        mock_load_tokenizer.side_effect = [mock_tokenizer1, mock_tokenizer2]
        mock_load_texts.return_value = ["Hello world", "Test text"]

        mock_diff = MagicMock()
        mock_diff.total_texts = 2
        mock_diff.avg_length_delta = -1.5
        mock_diff.compression_improvement = 0.12
        mock_diff.tokenizer1_total = 10
        mock_diff.tokenizer2_total = 8
        mock_diff.worst_regressions = []
        mock_diff_corpus.return_value = mock_diff

        config = AnalyzeDiffConfig(tokenizer1="gpt2", tokenizer2="llama")
        analyze_diff(config)

        captured = capsys.readouterr()
        assert "Corpus Diff Report" in captured.out
        assert "Texts compared:" in captured.out
        assert "Avg length delta:" in captured.out
        assert "Compression improved:" in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.diff.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.analyze.diff_corpus")
    def test_analyze_diff_with_regressions(
        self, mock_diff_corpus, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test diff analysis with regressions."""
        mock_tokenizer1 = MagicMock()
        mock_tokenizer2 = MagicMock()
        mock_load_tokenizer.side_effect = [mock_tokenizer1, mock_tokenizer2]
        mock_load_texts.return_value = ["Hello world"]

        mock_regression = MagicMock()
        mock_regression.length_delta = 5
        mock_regression.text = "This is a problematic text that regressed badly"

        mock_diff = MagicMock()
        mock_diff.total_texts = 1
        mock_diff.avg_length_delta = 5.0
        mock_diff.compression_improvement = -0.25
        mock_diff.tokenizer1_total = 10
        mock_diff.tokenizer2_total = 15
        mock_diff.worst_regressions = [mock_regression]
        mock_diff_corpus.return_value = mock_diff

        config = AnalyzeDiffConfig(tokenizer1="gpt2", tokenizer2="llama")
        analyze_diff(config)

        captured = capsys.readouterr()
        assert "Worst Regressions" in captured.out
        assert "Delta: +5" in captured.out
