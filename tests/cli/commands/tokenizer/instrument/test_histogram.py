"""Tests for instrument_histogram command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import InstrumentHistogramConfig
from chuk_lazarus.cli.commands.tokenizer.instrument.histogram import (
    instrument_histogram,
)


class TestInstrumentHistogramConfig:
    """Tests for InstrumentHistogramConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.file = None
        args.bins = 20
        args.width = 60
        args.quick = False

        config = InstrumentHistogramConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.file is None
        assert config.bins == 20
        assert config.width == 60
        assert config.quick is False

    def test_from_args_with_file_and_quick(self):
        """Test config with file and quick mode."""
        args = MagicMock()
        args.tokenizer = "llama"
        args.file = Path("/path/to/corpus.txt")
        args.bins = 50
        args.width = 80
        args.quick = True

        config = InstrumentHistogramConfig.from_args(args)

        assert config.tokenizer == "llama"
        assert config.file == Path("/path/to/corpus.txt")
        assert config.bins == 50
        assert config.width == 80
        assert config.quick is True


class TestInstrumentHistogram:
    """Tests for instrument_histogram function."""

    @patch("chuk_lazarus.cli.commands.tokenizer.instrument.histogram.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    def test_histogram_no_texts(self, mock_load_tokenizer, mock_load_texts, capsys):
        """Test with no texts provided."""
        mock_load_tokenizer.return_value = MagicMock()
        mock_load_texts.return_value = []

        config = InstrumentHistogramConfig(tokenizer="gpt2")
        instrument_histogram(config)

        captured = capsys.readouterr()
        assert "Quick Length Stats" not in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.instrument.histogram.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.instrumentation.get_length_stats")
    def test_histogram_quick_mode(
        self, mock_get_stats, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test quick mode histogram."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Hello world", "Test text"]

        mock_get_stats.return_value = {
            "count": 2,
            "mean": 5.5,
            "min": 3,
            "max": 8,
            "std": 2.1,
        }

        config = InstrumentHistogramConfig(tokenizer="gpt2", quick=True)
        instrument_histogram(config)

        captured = capsys.readouterr()
        assert "Quick Length Stats" in captured.out
        assert "count:" in captured.out
        assert "mean:" in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.instrument.histogram.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.instrumentation.compute_length_histogram")
    @patch("chuk_lazarus.data.tokenizers.instrumentation.format_histogram_ascii")
    def test_histogram_full_mode(
        self,
        mock_format_histogram,
        mock_compute_histogram,
        mock_load_tokenizer,
        mock_load_texts,
        capsys,
    ):
        """Test full histogram mode."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Hello world", "Test text", "Another text"]

        mock_histogram = MagicMock()
        mock_compute_histogram.return_value = mock_histogram
        mock_format_histogram.return_value = "ASCII histogram output"

        config = InstrumentHistogramConfig(tokenizer="gpt2", bins=10, width=40, quick=False)
        instrument_histogram(config)

        captured = capsys.readouterr()
        assert "ASCII histogram output" in captured.out
        mock_compute_histogram.assert_called_once_with(
            ["Hello world", "Test text", "Another text"], mock_tokenizer, num_bins=10
        )
        mock_format_histogram.assert_called_once_with(mock_histogram, width=40)
