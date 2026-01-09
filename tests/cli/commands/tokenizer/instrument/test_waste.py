"""Tests for instrument_waste command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import InstrumentWasteConfig
from chuk_lazarus.cli.commands.tokenizer.instrument.waste import instrument_waste


class TestInstrumentWasteConfig:
    """Tests for InstrumentWasteConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.file = None
        args.max_length = 2048

        config = InstrumentWasteConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.file is None
        assert config.max_length == 2048

    def test_from_args_with_options(self):
        """Test config with custom options."""
        args = MagicMock()
        args.tokenizer = "llama"
        args.file = Path("/path/to/corpus.txt")
        args.max_length = 4096

        config = InstrumentWasteConfig.from_args(args)

        assert config.tokenizer == "llama"
        assert config.file == Path("/path/to/corpus.txt")
        assert config.max_length == 4096


class TestInstrumentWaste:
    """Tests for instrument_waste function."""

    @patch("chuk_lazarus.cli.commands.tokenizer.instrument.waste.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    def test_waste_no_texts(self, mock_load_tokenizer, mock_load_texts, capsys):
        """Test with no texts provided."""
        mock_load_tokenizer.return_value = MagicMock()
        mock_load_texts.return_value = []

        config = InstrumentWasteConfig(tokenizer="gpt2")
        instrument_waste(config)

        captured = capsys.readouterr()
        assert "Token Waste Report" not in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.instrument.waste.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.instrumentation.analyze_waste")
    def test_waste_basic(self, mock_analyze_waste, mock_load_tokenizer, mock_load_texts, capsys):
        """Test basic waste analysis."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Hello world", "Test text"]

        # Create mock padding analysis
        mock_padding = MagicMock()
        mock_padding.total_positions = 4096
        mock_padding.total_content_tokens = 3500
        mock_padding.total_padding_tokens = 596
        mock_padding.padding_rate = 0.145
        mock_padding.efficiency = 0.855
        mock_padding.mean_padding_per_sample = 298.0
        mock_padding.max_padding = 500

        # Create mock truncation analysis
        mock_truncation = MagicMock()
        mock_truncation.truncated_samples = 0
        mock_truncation.total_samples = 2
        mock_truncation.truncation_rate = 0.0
        mock_truncation.total_tokens_lost = 0
        mock_truncation.content_loss_rate = 0.0
        mock_truncation.minor_truncation = 0
        mock_truncation.major_truncation = 0
        mock_truncation.severe_truncation = 0

        # Create mock report
        mock_report = MagicMock()
        mock_report.max_length = 2048
        mock_report.total_samples = 2
        mock_report.overall_efficiency = 0.855
        mock_report.padding = mock_padding
        mock_report.truncation = mock_truncation
        mock_report.recommendations = []
        mock_analyze_waste.return_value = mock_report

        config = InstrumentWasteConfig(tokenizer="gpt2", max_length=2048)
        instrument_waste(config)

        captured = capsys.readouterr()
        assert "Token Waste Report" in captured.out
        assert "Max length:" in captured.out
        assert "Total samples:" in captured.out
        assert "Overall efficiency:" in captured.out
        assert "Padding Analysis" in captured.out
        assert "Truncation Analysis" in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.instrument.waste.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.instrumentation.analyze_waste")
    def test_waste_with_recommendations(
        self, mock_analyze_waste, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test waste analysis with recommendations."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Short", "Very long text " * 500]

        mock_padding = MagicMock()
        mock_padding.total_positions = 4096
        mock_padding.total_content_tokens = 2000
        mock_padding.total_padding_tokens = 2096
        mock_padding.padding_rate = 0.51
        mock_padding.efficiency = 0.49
        mock_padding.mean_padding_per_sample = 1048.0
        mock_padding.max_padding = 2040

        mock_truncation = MagicMock()
        mock_truncation.truncated_samples = 1
        mock_truncation.total_samples = 2
        mock_truncation.truncation_rate = 0.5
        mock_truncation.total_tokens_lost = 500
        mock_truncation.content_loss_rate = 0.2
        mock_truncation.minor_truncation = 0
        mock_truncation.major_truncation = 1
        mock_truncation.severe_truncation = 0

        mock_report = MagicMock()
        mock_report.max_length = 2048
        mock_report.total_samples = 2
        mock_report.overall_efficiency = 0.49
        mock_report.padding = mock_padding
        mock_report.truncation = mock_truncation
        mock_report.recommendations = [
            "Consider increasing max_length to reduce truncation",
            "Consider dynamic padding for efficiency",
        ]
        mock_analyze_waste.return_value = mock_report

        config = InstrumentWasteConfig(tokenizer="gpt2", max_length=2048)
        instrument_waste(config)

        captured = capsys.readouterr()
        assert "Recommendations" in captured.out
        assert "Consider increasing max_length" in captured.out
