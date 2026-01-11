"""Tests for tokenizer_benchmark command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import BenchmarkConfig
from chuk_lazarus.cli.commands.tokenizer.health.benchmark import tokenizer_benchmark


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.samples = 1000
        args.avg_length = 100
        args.seed = None
        args.workers = 1
        args.file = None
        args.compare = False
        args.special_tokens = False
        args.warmup = 10

        config = BenchmarkConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.samples == 1000
        assert config.avg_length == 100
        assert config.workers == 1
        assert config.compare is False

    def test_from_args_with_options(self):
        """Test config with all options."""
        args = MagicMock()
        args.tokenizer = "llama"
        args.samples = 5000
        args.avg_length = 200
        args.seed = 42
        args.workers = 4
        args.file = Path("/path/to/corpus.txt")
        args.compare = True
        args.special_tokens = True
        args.warmup = 50

        config = BenchmarkConfig.from_args(args)

        assert config.tokenizer == "llama"
        assert config.samples == 5000
        assert config.avg_length == 200
        assert config.seed == 42
        assert config.workers == 4
        assert config.file == Path("/path/to/corpus.txt")
        assert config.compare is True
        assert config.special_tokens is True


class TestTokenizerBenchmark:
    """Tests for tokenizer_benchmark function."""

    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.backends.benchmark.generate_benchmark_corpus")
    @patch("chuk_lazarus.data.tokenizers.backends.benchmark.benchmark_tokenizer")
    def test_benchmark_basic(self, mock_benchmark, mock_gen_corpus, mock_load_tokenizer, capsys):
        """Test basic benchmark."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_gen_corpus.return_value = ["Hello world"] * 100

        mock_result = MagicMock()
        mock_result.backend_type = "hf"
        mock_result.total_tokens = 500
        mock_result.elapsed_seconds = 0.5
        mock_result.tokens_per_second = 1000.0
        mock_result.samples_per_second = 200.0
        mock_result.avg_tokens_per_sample = 5.0
        mock_benchmark.return_value = mock_result

        config = BenchmarkConfig(tokenizer="gpt2", samples=100)
        result = tokenizer_benchmark(config)

        captured = capsys.readouterr()
        assert "Tokenizer Benchmark" in captured.out
        assert "Throughput:" in captured.out
        assert result is not None
        assert result.total_tokens == 500

    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.backends.benchmark.generate_benchmark_corpus")
    @patch("chuk_lazarus.data.tokenizers.backends.benchmark.compare_backends")
    def test_benchmark_compare_mode(
        self, mock_compare, mock_gen_corpus, mock_load_tokenizer, capsys
    ):
        """Test benchmark comparison mode."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_gen_corpus.return_value = ["Hello world"] * 100

        mock_comparison = MagicMock()
        mock_comparison.summary.return_value = "Comparison results..."
        mock_compare.return_value = mock_comparison

        config = BenchmarkConfig(tokenizer="gpt2", samples=100, compare=True)
        result = tokenizer_benchmark(config)

        captured = capsys.readouterr()
        assert "Comparison results..." in captured.out
        assert result is None
