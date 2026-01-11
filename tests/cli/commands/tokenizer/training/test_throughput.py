"""Tests for training_throughput command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import TrainingThroughputConfig
from chuk_lazarus.cli.commands.tokenizer.training.throughput import training_throughput


class TestTrainingThroughputConfig:
    """Tests for TrainingThroughputConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.file = None
        args.batch_size = 32
        args.iterations = 10

        config = TrainingThroughputConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.file is None
        assert config.batch_size == 32
        assert config.iterations == 10

    def test_from_args_with_options(self):
        """Test config with all options."""
        args = MagicMock()
        args.tokenizer = "llama"
        args.file = Path("/path/to/corpus.txt")
        args.batch_size = 64
        args.iterations = 20

        config = TrainingThroughputConfig.from_args(args)

        assert config.tokenizer == "llama"
        assert config.file == Path("/path/to/corpus.txt")
        assert config.batch_size == 64
        assert config.iterations == 20


class TestTrainingThroughput:
    """Tests for training_throughput function."""

    @patch("chuk_lazarus.cli.commands.tokenizer.training.throughput.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    def test_throughput_no_texts(self, mock_load_tokenizer, mock_load_texts, capsys):
        """Test with no texts provided."""
        mock_load_tokenizer.return_value = MagicMock()
        mock_load_texts.return_value = []

        config = TrainingThroughputConfig(tokenizer="gpt2")
        training_throughput(config)

        captured = capsys.readouterr()
        assert "Throughput Profile" not in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.training.throughput.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.training.ThroughputProfiler")
    def test_throughput_basic(
        self, mock_profiler_cls, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test basic throughput profiling."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Text 1", "Text 2"]

        mock_metrics = MagicMock()
        mock_metrics.tokens_per_second = 10000.0
        mock_metrics.texts_per_second = 500.0
        mock_metrics.avg_batch_time_ms = 64.0
        mock_metrics.total_tokens = 50000
        mock_metrics.total_time_seconds = 5.0

        mock_profiler = MagicMock()
        mock_profiler.profile.return_value = mock_metrics
        mock_profiler_cls.return_value = mock_profiler

        config = TrainingThroughputConfig(tokenizer="gpt2", batch_size=32, iterations=10)
        training_throughput(config)

        captured = capsys.readouterr()
        assert "Throughput Profile" in captured.out
        assert "Tokens/second:" in captured.out
        assert "Texts/second:" in captured.out
