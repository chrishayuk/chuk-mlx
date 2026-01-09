"""Tests for analyze_vocab_suggest command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import AnalyzeVocabSuggestConfig
from chuk_lazarus.cli.commands.tokenizer.analyze.vocab_suggest import analyze_vocab_suggest


class TestAnalyzeVocabSuggestConfig:
    """Tests for AnalyzeVocabSuggestConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.file = None
        args.min_freq = 5
        args.min_frag = 2
        args.limit = 100
        args.show = 20

        config = AnalyzeVocabSuggestConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.file is None
        assert config.min_freq == 5
        assert config.min_frag == 2
        assert config.limit == 100
        assert config.show == 20

    def test_from_args_with_file(self):
        """Test config with file and custom settings."""
        args = MagicMock()
        args.tokenizer = "llama"
        args.file = Path("/path/to/corpus.txt")
        args.min_freq = 10
        args.min_frag = 3
        args.limit = 50
        args.show = 10

        config = AnalyzeVocabSuggestConfig.from_args(args)

        assert config.tokenizer == "llama"
        assert config.file == Path("/path/to/corpus.txt")
        assert config.min_freq == 10
        assert config.min_frag == 3
        assert config.limit == 50
        assert config.show == 10


class TestAnalyzeVocabSuggest:
    """Tests for analyze_vocab_suggest function."""

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.vocab_suggest.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    def test_analyze_vocab_suggest_no_texts(self, mock_load_tokenizer, mock_load_texts, capsys):
        """Test with no texts provided."""
        mock_load_tokenizer.return_value = MagicMock()
        mock_load_texts.return_value = []

        config = AnalyzeVocabSuggestConfig(tokenizer="gpt2")
        analyze_vocab_suggest(config)

        # Should return early with no output for the report
        captured = capsys.readouterr()
        assert "Vocabulary Induction Report" not in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.vocab_suggest.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.analyze.analyze_vocab_induction")
    def test_analyze_vocab_suggest_basic(
        self, mock_analyze, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test basic vocab suggest analysis."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Hello world tokenization test"]

        # Create mock candidate
        mock_candidate = MagicMock()
        mock_candidate.token_str = "tokenization"
        mock_candidate.frequency = 50
        mock_candidate.current_tokens = 3
        mock_candidate.total_savings = 100

        mock_report = MagicMock()
        mock_report.total_candidates = 5
        mock_report.total_potential_savings = 500
        mock_report.savings_percent = 2.5
        mock_report.domain_breakdown = None
        mock_report.candidates = [mock_candidate]
        mock_report.recommendations = []
        mock_analyze.return_value = mock_report

        config = AnalyzeVocabSuggestConfig(tokenizer="gpt2")
        analyze_vocab_suggest(config)

        captured = capsys.readouterr()
        assert "Vocabulary Induction Report" in captured.out
        assert "Candidates found:" in captured.out
        assert "Potential savings:" in captured.out
        assert "Savings percent:" in captured.out
        assert "tokenization" in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.vocab_suggest.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.analyze.analyze_vocab_induction")
    def test_analyze_vocab_suggest_with_domains(
        self, mock_analyze, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test vocab suggest with domain breakdown."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Test text"]

        mock_report = MagicMock()
        mock_report.total_candidates = 10
        mock_report.total_potential_savings = 1000
        mock_report.savings_percent = 5.0
        mock_report.domain_breakdown = {"code": 5, "math": 3, "general": 2}
        mock_report.candidates = []
        mock_report.recommendations = []
        mock_analyze.return_value = mock_report

        config = AnalyzeVocabSuggestConfig(tokenizer="gpt2")
        analyze_vocab_suggest(config)

        captured = capsys.readouterr()
        assert "By domain:" in captured.out
        assert "code" in captured.out
        assert "math" in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.vocab_suggest.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.analyze.analyze_vocab_induction")
    def test_analyze_vocab_suggest_with_recommendations(
        self, mock_analyze, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test vocab suggest with recommendations."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Test"]

        mock_report = MagicMock()
        mock_report.total_candidates = 3
        mock_report.total_potential_savings = 200
        mock_report.savings_percent = 1.0
        mock_report.domain_breakdown = None
        mock_report.candidates = []
        mock_report.recommendations = [
            "Consider adding domain-specific tokens",
            "High fragmentation in code terms",
        ]
        mock_analyze.return_value = mock_report

        config = AnalyzeVocabSuggestConfig(tokenizer="gpt2")
        analyze_vocab_suggest(config)

        captured = capsys.readouterr()
        assert "Recommendations" in captured.out
        assert "Consider adding domain-specific tokens" in captured.out
