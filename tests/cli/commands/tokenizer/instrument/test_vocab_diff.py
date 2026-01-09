"""Tests for instrument_vocab_diff command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import InstrumentVocabDiffConfig
from chuk_lazarus.cli.commands.tokenizer.instrument.vocab_diff import (
    instrument_vocab_diff,
)


class TestInstrumentVocabDiffConfig:
    """Tests for InstrumentVocabDiffConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.tokenizer1 = "gpt2"
        args.tokenizer2 = "llama"
        args.file = None
        args.examples = 5
        args.cost = False

        config = InstrumentVocabDiffConfig.from_args(args)

        assert config.tokenizer1 == "gpt2"
        assert config.tokenizer2 == "llama"
        assert config.file is None
        assert config.examples == 5
        assert config.cost is False

    def test_from_args_with_options(self):
        """Test config with all options."""
        args = MagicMock()
        args.tokenizer1 = "gpt2"
        args.tokenizer2 = "bert-base"
        args.file = Path("/path/to/corpus.txt")
        args.examples = 10
        args.cost = True

        config = InstrumentVocabDiffConfig.from_args(args)

        assert config.tokenizer1 == "gpt2"
        assert config.tokenizer2 == "bert-base"
        assert config.file == Path("/path/to/corpus.txt")
        assert config.examples == 10
        assert config.cost is True


class TestInstrumentVocabDiff:
    """Tests for instrument_vocab_diff function."""

    @patch("chuk_lazarus.cli.commands.tokenizer.instrument.vocab_diff.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    def test_vocab_diff_no_texts(self, mock_load_tokenizer, mock_load_texts, capsys):
        """Test with no texts provided."""
        mock_load_tokenizer.return_value = MagicMock()
        mock_load_texts.return_value = []

        config = InstrumentVocabDiffConfig(tokenizer1="gpt2", tokenizer2="llama")
        instrument_vocab_diff(config)

        captured = capsys.readouterr()
        assert "Vocabulary Comparison" not in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.instrument.vocab_diff.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.instrumentation.compare_vocab_impact")
    def test_vocab_diff_basic(
        self, mock_compare_vocab, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test basic vocabulary diff."""
        mock_tok1 = MagicMock()
        mock_tok2 = MagicMock()
        mock_load_tokenizer.side_effect = [mock_tok1, mock_tok2]
        mock_load_texts.return_value = ["Hello world", "Test text"]

        # Create mock report
        mock_report = MagicMock()
        mock_report.tokenizer1_name = "gpt2"
        mock_report.tokenizer2_name = "llama"
        mock_report.tokenizer1_vocab_size = 50257
        mock_report.tokenizer2_vocab_size = 32000
        mock_report.tokens1_total = 100
        mock_report.tokens2_total = 90
        mock_report.token_count_diff = -10
        mock_report.token_count_ratio = 0.9
        mock_report.chars_per_token1 = 4.2
        mock_report.chars_per_token2 = 4.7
        mock_report.compression_improvement = 1.12
        mock_report.samples_improved = 1
        mock_report.samples_same = 0
        mock_report.samples_worse = 1
        mock_report.improvement_rate = 0.5
        mock_report.training_speedup = 1.1
        mock_report.memory_reduction = 0.1
        mock_report.recommendations = []
        mock_compare_vocab.return_value = mock_report

        config = InstrumentVocabDiffConfig(tokenizer1="gpt2", tokenizer2="llama")
        instrument_vocab_diff(config)

        captured = capsys.readouterr()
        assert "Vocabulary Comparison" in captured.out
        assert "Tokenizer 1:" in captured.out
        assert "Tokenizer 2:" in captured.out
        assert "Vocab size 1:" in captured.out
        assert "Vocab size 2:" in captured.out
        assert "Token Counts" in captured.out
        assert "Compression" in captured.out
        assert "Per-Sample Analysis" in captured.out
        assert "Training Impact" in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.instrument.vocab_diff.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.instrumentation.compare_vocab_impact")
    @patch("chuk_lazarus.data.tokenizers.instrumentation.estimate_retokenization_cost")
    def test_vocab_diff_with_cost(
        self,
        mock_estimate_cost,
        mock_compare_vocab,
        mock_load_tokenizer,
        mock_load_texts,
        capsys,
    ):
        """Test vocabulary diff with cost estimation."""
        mock_tok1 = MagicMock()
        mock_tok2 = MagicMock()
        mock_load_tokenizer.side_effect = [mock_tok1, mock_tok2]
        mock_load_texts.return_value = ["Hello world", "Test text"]

        mock_report = MagicMock()
        mock_report.tokenizer1_name = "gpt2"
        mock_report.tokenizer2_name = "llama"
        mock_report.tokenizer1_vocab_size = 50257
        mock_report.tokenizer2_vocab_size = 32000
        mock_report.tokens1_total = 100
        mock_report.tokens2_total = 90
        mock_report.token_count_diff = -10
        mock_report.token_count_ratio = 0.9
        mock_report.chars_per_token1 = 4.2
        mock_report.chars_per_token2 = 4.7
        mock_report.compression_improvement = 1.12
        mock_report.samples_improved = 1
        mock_report.samples_same = 0
        mock_report.samples_worse = 1
        mock_report.improvement_rate = 0.5
        mock_report.training_speedup = 1.1
        mock_report.memory_reduction = 0.1
        mock_report.recommendations = ["Consider switching to llama for math"]
        mock_compare_vocab.return_value = mock_report

        mock_estimate_cost.return_value = {
            "vocab_overlap": 25000,
            "vocab_overlap_rate": 0.78,
            "new_tokens": 7000,
            "removed_tokens": 25257,
            "embedding_reuse_rate": 0.78,
        }

        config = InstrumentVocabDiffConfig(tokenizer1="gpt2", tokenizer2="llama", cost=True)
        instrument_vocab_diff(config)

        captured = capsys.readouterr()
        assert "Retokenization Cost" in captured.out
        assert "Vocab overlap:" in captured.out
        assert "New tokens:" in captured.out
        assert "Removed tokens:" in captured.out
        assert "Embedding reuse:" in captured.out
        assert "Recommendations" in captured.out
