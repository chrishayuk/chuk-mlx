"""Tests for analyze_fit_score command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import AnalyzeFitScoreConfig
from chuk_lazarus.cli.commands.tokenizer.analyze.fit_score import analyze_fit_score


class TestAnalyzeFitScoreConfig:
    """Tests for AnalyzeFitScoreConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.file = None

        config = AnalyzeFitScoreConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.file is None

    def test_from_args_with_file(self):
        """Test config with file."""
        args = MagicMock()
        args.tokenizer = "llama"
        args.file = Path("/path/to/corpus.txt")

        config = AnalyzeFitScoreConfig.from_args(args)

        assert config.tokenizer == "llama"
        assert config.file == Path("/path/to/corpus.txt")


class TestAnalyzeFitScore:
    """Tests for analyze_fit_score function."""

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.fit_score.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    def test_analyze_fit_score_no_texts(self, mock_load_tokenizer, mock_load_texts, capsys):
        """Test with no texts provided."""
        mock_load_tokenizer.return_value = MagicMock()
        mock_load_texts.return_value = []

        config = AnalyzeFitScoreConfig(tokenizer="gpt2")
        analyze_fit_score(config)

        # Should return early with no output for the report
        captured = capsys.readouterr()
        assert "Fit Score Report" not in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.fit_score.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.analyze.calculate_fit_score")
    def test_analyze_fit_score_basic(
        self, mock_calculate, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test basic fit score analysis."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Hello world", "Test text"]

        mock_score = MagicMock()
        mock_score.score = 85.5
        mock_score.grade = "A"
        mock_score.recommendations = []
        mock_score.details = {}
        mock_calculate.return_value = mock_score

        config = AnalyzeFitScoreConfig(tokenizer="gpt2")
        analyze_fit_score(config)

        captured = capsys.readouterr()
        assert "Fit Score Report" in captured.out
        assert "Overall Score:" in captured.out
        assert "Grade:" in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.fit_score.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.analyze.calculate_fit_score")
    def test_analyze_fit_score_with_recommendations(
        self, mock_calculate, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test fit score with recommendations."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Hello world"]

        mock_score = MagicMock()
        mock_score.score = 65.0
        mock_score.grade = "C"
        mock_score.recommendations = [
            "Consider fine-tuning tokenizer",
            "High fragmentation detected",
        ]
        mock_score.details = {}
        mock_calculate.return_value = mock_score

        config = AnalyzeFitScoreConfig(tokenizer="gpt2")
        analyze_fit_score(config)

        captured = capsys.readouterr()
        assert "Recommendations:" in captured.out
        assert "Consider fine-tuning tokenizer" in captured.out
        assert "High fragmentation detected" in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.fit_score.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.analyze.calculate_fit_score")
    def test_analyze_fit_score_with_details(
        self, mock_calculate, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test fit score with details."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Test text"]

        mock_score = MagicMock()
        mock_score.score = 90.0
        mock_score.grade = "A+"
        mock_score.recommendations = []
        mock_score.details = {
            "vocab_utilization": "95%",
            "unk_rate": "0.1%",
            "avg_tokens_per_word": "1.2",
        }
        mock_calculate.return_value = mock_score

        config = AnalyzeFitScoreConfig(tokenizer="gpt2")
        analyze_fit_score(config)

        captured = capsys.readouterr()
        assert "Details:" in captured.out
        assert "vocab_utilization" in captured.out
        assert "unk_rate" in captured.out
