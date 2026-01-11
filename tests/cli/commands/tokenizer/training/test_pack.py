"""Tests for training_pack command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import TrainingPackConfig
from chuk_lazarus.cli.commands.tokenizer.training.pack import training_pack


class TestTrainingPackConfig:
    """Tests for TrainingPackConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.file = None
        args.max_length = 2048
        args.output = None

        config = TrainingPackConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.file is None
        assert config.max_length == 2048
        assert config.output is None

    def test_from_args_with_options(self):
        """Test config with all options."""
        args = MagicMock()
        args.tokenizer = "llama"
        args.file = Path("/path/to/corpus.txt")
        args.max_length = 4096
        args.output = Path("/path/to/output.jsonl")

        config = TrainingPackConfig.from_args(args)

        assert config.tokenizer == "llama"
        assert config.file == Path("/path/to/corpus.txt")
        assert config.max_length == 4096
        assert config.output == Path("/path/to/output.jsonl")


class TestTrainingPack:
    """Tests for training_pack function."""

    @patch("chuk_lazarus.cli.commands.tokenizer.training.pack.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    def test_pack_no_texts(self, mock_load_tokenizer, mock_load_texts, capsys):
        """Test with no texts provided."""
        mock_load_tokenizer.return_value = MagicMock()
        mock_load_texts.return_value = []

        config = TrainingPackConfig(tokenizer="gpt2")
        result = training_pack(config)

        assert result.input_sequences == 0
        assert result.packed_sequences == 0

    @patch("chuk_lazarus.cli.commands.tokenizer.training.pack.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.training.pack_sequences")
    @patch("chuk_lazarus.data.tokenizers.training.PackingConfig")
    def test_pack_basic(
        self,
        mock_packing_config_cls,
        mock_pack,
        mock_load_tokenizer,
        mock_load_texts,
        capsys,
    ):
        """Test basic packing."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Text 1", "Text 2", "Text 3", "Text 4"]

        # Create mock packed sequences
        mock_packed = MagicMock()
        mock_packed.token_ids = [1, 2, 3, 4, 5] * 100  # 500 tokens
        mock_pack.return_value = [mock_packed, mock_packed]  # 2 packed sequences

        config = TrainingPackConfig(tokenizer="gpt2", max_length=2048)
        result = training_pack(config)

        captured = capsys.readouterr()
        assert "Packing Results" in captured.out
        assert "Input sequences:" in captured.out
        assert "Packed sequences:" in captured.out
        assert result.input_sequences == 4
        assert result.packed_sequences == 2
