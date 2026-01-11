"""Tests for analyze_efficiency command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import AnalyzeEfficiencyConfig
from chuk_lazarus.cli.commands.tokenizer.analyze.efficiency import analyze_efficiency


class TestAnalyzeEfficiencyConfig:
    """Tests for AnalyzeEfficiencyConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.file = None

        config = AnalyzeEfficiencyConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.file is None

    def test_from_args_with_file(self):
        """Test config with file."""
        args = MagicMock()
        args.tokenizer = "llama"
        args.file = Path("/path/to/corpus.txt")

        config = AnalyzeEfficiencyConfig.from_args(args)

        assert config.tokenizer == "llama"
        assert config.file == Path("/path/to/corpus.txt")


class TestAnalyzeEfficiency:
    """Tests for analyze_efficiency function."""

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.efficiency.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    def test_analyze_efficiency_no_texts(self, mock_load_tokenizer, mock_load_texts, capsys):
        """Test with no texts provided."""
        mock_load_tokenizer.return_value = MagicMock()
        mock_load_texts.return_value = []

        config = AnalyzeEfficiencyConfig(tokenizer="gpt2")
        analyze_efficiency(config)

        # Should return early with no output for the report
        captured = capsys.readouterr()
        assert "Efficiency Report" not in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.efficiency.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.analyze.analyze_efficiency")
    def test_analyze_efficiency_basic(
        self, mock_do_analyze, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test basic efficiency analysis."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Hello world", "Test text"]

        # Create mock sample stats
        mock_sample_stats = MagicMock()
        mock_sample_stats.count = 2
        mock_sample_stats.total_tokens = 20
        mock_sample_stats.mean = 10.0
        mock_sample_stats.median = 10.0
        mock_sample_stats.std = 2.0
        mock_sample_stats.p5 = 6.0
        mock_sample_stats.p95 = 14.0
        mock_sample_stats.min_tokens = 8
        mock_sample_stats.max_tokens = 12

        # Create mock fragmentation
        mock_fragmentation = MagicMock()
        mock_fragmentation.fragmentation_score = 0.15
        mock_fragmentation.single_char_tokens = 5
        mock_fragmentation.subword_tokens = 10
        mock_fragmentation.fragmented_words = []

        mock_report = MagicMock()
        mock_report.efficiency_score = 85.5
        mock_report.sample_stats = mock_sample_stats
        mock_report.reasoning_steps = None
        mock_report.equations = None
        mock_report.tool_calls = None
        mock_report.fragmentation = mock_fragmentation
        mock_report.recommendations = []
        mock_do_analyze.return_value = mock_report

        config = AnalyzeEfficiencyConfig(tokenizer="gpt2")
        analyze_efficiency(config)

        captured = capsys.readouterr()
        assert "Efficiency Report" in captured.out
        assert "Efficiency Score:" in captured.out
        assert "Sample Statistics" in captured.out
        assert "Fragmentation" in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.analyze.efficiency.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.analyze.analyze_efficiency")
    def test_analyze_efficiency_with_special_sections(
        self, mock_do_analyze, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test efficiency analysis with reasoning/equations/tool calls."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Complex math reasoning"]

        # Create mock sections
        mock_sample_stats = MagicMock()
        mock_sample_stats.count = 1
        mock_sample_stats.total_tokens = 100
        mock_sample_stats.mean = 100.0
        mock_sample_stats.median = 100.0
        mock_sample_stats.std = 0.0
        mock_sample_stats.p5 = 100.0
        mock_sample_stats.p95 = 100.0
        mock_sample_stats.min_tokens = 100
        mock_sample_stats.max_tokens = 100

        mock_reasoning = MagicMock()
        mock_reasoning.count = 3
        mock_reasoning.mean_tokens = 15.0

        mock_equations = MagicMock()
        mock_equations.count = 5
        mock_equations.mean_tokens = 8.0

        mock_tool_calls = MagicMock()
        mock_tool_calls.count = 2
        mock_tool_calls.mean_tokens = 20.0

        mock_fragmentation = MagicMock()
        mock_fragmentation.fragmentation_score = 0.05
        mock_fragmentation.single_char_tokens = 2
        mock_fragmentation.subword_tokens = 5
        mock_fragmentation.fragmented_words = [
            {"word": "tokenization", "tokens": 3},
            {"word": "analysis", "tokens": 2},
        ]

        mock_report = MagicMock()
        mock_report.efficiency_score = 92.0
        mock_report.sample_stats = mock_sample_stats
        mock_report.reasoning_steps = mock_reasoning
        mock_report.equations = mock_equations
        mock_report.tool_calls = mock_tool_calls
        mock_report.fragmentation = mock_fragmentation
        mock_report.recommendations = ["Consider using a specialized math tokenizer"]
        mock_do_analyze.return_value = mock_report

        config = AnalyzeEfficiencyConfig(tokenizer="gpt2")
        analyze_efficiency(config)

        captured = capsys.readouterr()
        assert "Reasoning Steps" in captured.out
        assert "Equations" in captured.out
        assert "Tool Calls" in captured.out
        assert "Most fragmented words" in captured.out
        assert "Recommendations" in captured.out
