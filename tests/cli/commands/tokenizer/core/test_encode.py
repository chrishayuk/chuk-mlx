"""Tests for tokenizer_encode command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import EncodeConfig
from chuk_lazarus.cli.commands.tokenizer.core.encode import tokenizer_encode


class TestEncodeConfig:
    """Tests for EncodeConfig."""

    def test_from_args_with_text(self):
        """Test config with text."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.text = "Hello world"
        args.file = None
        args.special_tokens = True

        config = EncodeConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.text == "Hello world"
        assert config.file is None
        assert config.special_tokens is True

    def test_from_args_with_file(self):
        """Test config with file."""
        args = MagicMock()
        args.tokenizer = "llama"
        args.text = None
        args.file = Path("/path/to/file.txt")
        args.special_tokens = False

        config = EncodeConfig.from_args(args)

        assert config.tokenizer == "llama"
        assert config.text is None
        assert config.file == Path("/path/to/file.txt")
        assert config.special_tokens is False


class TestTokenizerEncode:
    """Tests for tokenizer_encode function."""

    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.token_display.TokenDisplayUtility")
    def test_encode_with_text(
        self, mock_display_cls, mock_load_tokenizer, capsys
    ):
        """Test encoding text."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        mock_display = MagicMock()
        mock_display_cls.return_value = mock_display

        config = EncodeConfig(tokenizer="gpt2", text="Hello world")
        tokenizer_encode(config)

        captured = capsys.readouterr()
        assert "Text: Hello world" in captured.out
        assert "Length:" in captured.out
        mock_display.display_tokens_from_prompt.assert_called_once_with(
            "Hello world", add_special_tokens=True
        )

    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.token_display.TokenDisplayUtility")
    @patch("builtins.open")
    def test_encode_with_file(
        self, mock_open, mock_display_cls, mock_load_tokenizer, capsys, tmp_path
    ):
        """Test encoding from file."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        mock_display = MagicMock()
        mock_display_cls.return_value = mock_display

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read.return_value = "File content here"
        mock_open.return_value = mock_file

        test_file = tmp_path / "test.txt"
        config = EncodeConfig(tokenizer="gpt2", file=test_file, special_tokens=False)
        tokenizer_encode(config)

        captured = capsys.readouterr()
        assert "Text: File content" in captured.out
        mock_display.display_tokens_from_prompt.assert_called_once_with(
            "File content here", add_special_tokens=False
        )
