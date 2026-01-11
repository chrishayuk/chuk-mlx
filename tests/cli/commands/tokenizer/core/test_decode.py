"""Tests for tokenizer decode command."""

from unittest.mock import patch

from chuk_lazarus.cli.commands.tokenizer._types import DecodeConfig
from chuk_lazarus.cli.commands.tokenizer.core.decode import tokenizer_decode

LOAD_TOKENIZER_PATCH = "chuk_lazarus.utils.tokenizer_loader.load_tokenizer"


class TestTokenizerDecode:
    """Tests for tokenizer_decode command."""

    def test_decode_comma_separated_ids(self, mock_tokenizer):
        """Test decoding comma-separated token IDs."""
        config = DecodeConfig(tokenizer="gpt2", ids="1,2,3,4,5")

        with patch(LOAD_TOKENIZER_PATCH, return_value=mock_tokenizer):
            result = tokenizer_decode(config)

        assert result.token_ids == [1, 2, 3, 4, 5]
        assert result.decoded == "Hello world"
        mock_tokenizer.decode.assert_called_once_with([1, 2, 3, 4, 5])

    def test_decode_space_separated_ids(self, mock_tokenizer):
        """Test decoding space-separated token IDs."""
        config = DecodeConfig(tokenizer="gpt2", ids="10 20 30")

        with patch(LOAD_TOKENIZER_PATCH, return_value=mock_tokenizer):
            result = tokenizer_decode(config)

        assert result.token_ids == [10, 20, 30]
        mock_tokenizer.decode.assert_called_once_with([10, 20, 30])

    def test_decode_mixed_separators(self, mock_tokenizer):
        """Test decoding with mixed separators."""
        config = DecodeConfig(tokenizer="gpt2", ids="1, 2, 3")

        with patch(LOAD_TOKENIZER_PATCH, return_value=mock_tokenizer):
            result = tokenizer_decode(config)

        assert result.token_ids == [1, 2, 3]

    def test_result_display(self, mock_tokenizer):
        """Test result display formatting."""
        config = DecodeConfig(tokenizer="gpt2", ids="1,2,3")

        with patch(LOAD_TOKENIZER_PATCH, return_value=mock_tokenizer):
            result = tokenizer_decode(config)

        display = result.to_display()
        assert "Token IDs: [1, 2, 3]" in display
        assert "Decoded: Hello world" in display
