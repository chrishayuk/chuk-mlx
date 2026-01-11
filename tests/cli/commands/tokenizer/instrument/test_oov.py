"""Tests for instrument_oov command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import InstrumentOovConfig
from chuk_lazarus.cli.commands.tokenizer.instrument.oov import instrument_oov


class TestInstrumentOovConfig:
    """Tests for InstrumentOovConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.file = None
        args.vocab_size = None
        args.show_rare = False
        args.max_freq = 5
        args.top_k = 20

        config = InstrumentOovConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.file is None
        assert config.vocab_size is None
        assert config.show_rare is False
        assert config.max_freq == 5
        assert config.top_k == 20

    def test_from_args_with_options(self):
        """Test config with all options."""
        args = MagicMock()
        args.tokenizer = "llama"
        args.file = Path("/path/to/corpus.txt")
        args.vocab_size = 32000
        args.show_rare = True
        args.max_freq = 10
        args.top_k = 50

        config = InstrumentOovConfig.from_args(args)

        assert config.tokenizer == "llama"
        assert config.file == Path("/path/to/corpus.txt")
        assert config.vocab_size == 32000
        assert config.show_rare is True
        assert config.max_freq == 10
        assert config.top_k == 50


class TestInstrumentOov:
    """Tests for instrument_oov function."""

    @patch("chuk_lazarus.cli.commands.tokenizer.instrument.oov.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    def test_oov_no_texts(self, mock_load_tokenizer, mock_load_texts, capsys):
        """Test with no texts provided."""
        mock_load_tokenizer.return_value = MagicMock()
        mock_load_texts.return_value = []

        config = InstrumentOovConfig(tokenizer="gpt2")
        instrument_oov(config)

        captured = capsys.readouterr()
        assert "OOV Report" not in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.instrument.oov.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.instrumentation.get_frequency_bands")
    @patch("chuk_lazarus.data.tokenizers.instrumentation.analyze_oov")
    def test_oov_basic(
        self,
        mock_analyze_oov,
        mock_get_bands,
        mock_load_tokenizer,
        mock_load_texts,
        capsys,
    ):
        """Test basic OOV analysis."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Hello world", "Test text"]

        # Create mock frequency band enum
        mock_band = MagicMock()
        mock_band.value = "common"
        mock_get_bands.return_value = {mock_band: 100}

        # Create mock OOV report
        mock_report = MagicMock()
        mock_report.total_tokens = 1000
        mock_report.unique_tokens = 200
        mock_report.unk_rate = 0.01
        mock_report.singleton_rate = 0.05
        mock_report.vocab_utilization = 0.15
        mock_report.recommendations = []
        mock_analyze_oov.return_value = mock_report

        config = InstrumentOovConfig(tokenizer="gpt2", show_rare=False)
        instrument_oov(config)

        captured = capsys.readouterr()
        assert "Token Frequency Bands" in captured.out
        assert "OOV Report" in captured.out
        assert "Total tokens:" in captured.out
        assert "Unique tokens:" in captured.out
        assert "UNK rate:" in captured.out
        assert "Singleton rate:" in captured.out
        assert "Vocab utilization:" in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.instrument.oov.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.instrumentation.get_frequency_bands")
    @patch("chuk_lazarus.data.tokenizers.instrumentation.analyze_oov")
    @patch("chuk_lazarus.data.tokenizers.instrumentation.find_rare_tokens")
    def test_oov_with_rare_tokens(
        self,
        mock_find_rare,
        mock_analyze_oov,
        mock_get_bands,
        mock_load_tokenizer,
        mock_load_texts,
        capsys,
    ):
        """Test OOV analysis with rare tokens display."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Hello world", "Test text"]

        mock_band = MagicMock()
        mock_band.value = "common"
        mock_get_bands.return_value = {mock_band: 100}

        mock_report = MagicMock()
        mock_report.total_tokens = 1000
        mock_report.unique_tokens = 200
        mock_report.unk_rate = 0.01
        mock_report.singleton_rate = 0.05
        mock_report.vocab_utilization = 0.15
        mock_report.recommendations = ["Consider expanding vocab"]
        mock_analyze_oov.return_value = mock_report

        # Create mock rare token
        mock_rare_token = MagicMock()
        mock_rare_token.token_str = "xyz"
        mock_rare_token.count = 3
        mock_rare_token.band = mock_band
        mock_find_rare.return_value = [mock_rare_token]

        config = InstrumentOovConfig(tokenizer="gpt2", show_rare=True, max_freq=5, top_k=10)
        instrument_oov(config)

        captured = capsys.readouterr()
        assert "Rare Tokens" in captured.out
        assert "Recommendations:" in captured.out
        mock_find_rare.assert_called_once_with(
            ["Hello world", "Test text"], mock_tokenizer, max_frequency=5, top_k=10
        )
