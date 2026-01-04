"""Tests for tokenizer compare command."""

from unittest.mock import MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.tokenizer._types import CompareConfig
from chuk_lazarus.cli.commands.tokenizer.core.compare import tokenizer_compare

LOAD_TOKENIZER_PATCH = "chuk_lazarus.utils.tokenizer_loader.load_tokenizer"
DISPLAY_UTILITY_PATCH = "chuk_lazarus.data.tokenizers.token_display.TokenDisplayUtility"


class TestTokenizerCompare:
    """Tests for tokenizer_compare command."""

    def test_compare_basic(self):
        """Test basic tokenizer comparison."""
        tok1 = MagicMock()
        tok1.encode.return_value = [1, 2, 3, 4, 5]
        tok2 = MagicMock()
        tok2.encode.return_value = [1, 2, 3]

        config = CompareConfig(
            tokenizer1="gpt2",
            tokenizer2="llama",
            text="Hello world",
            verbose=False,
        )

        with patch(LOAD_TOKENIZER_PATCH, side_effect=[tok1, tok2]):
            result = tokenizer_compare(config)

        assert result.tokenizer1_count == 5
        assert result.tokenizer2_count == 3
        assert result.difference == 2
        assert result.ratio == pytest.approx(5 / 3)

    def test_compare_equal_tokenizers(self):
        """Test comparison with equal token counts."""
        tok1 = MagicMock()
        tok1.encode.return_value = [1, 2, 3]
        tok2 = MagicMock()
        tok2.encode.return_value = [4, 5, 6]

        config = CompareConfig(
            tokenizer1="gpt2",
            tokenizer2="llama",
            text="Test",
            verbose=False,
        )

        with patch(LOAD_TOKENIZER_PATCH, side_effect=[tok1, tok2]):
            result = tokenizer_compare(config)

        assert result.difference == 0
        assert result.ratio == pytest.approx(1.0)

    def test_compare_verbose_mode(self):
        """Test comparison in verbose mode calls display utilities."""
        tok1 = MagicMock()
        tok1.encode.return_value = [1, 2]
        tok2 = MagicMock()
        tok2.encode.return_value = [3, 4, 5]

        mock_display = MagicMock()

        config = CompareConfig(
            tokenizer1="gpt2",
            tokenizer2="llama",
            text="Verbose test",
            verbose=True,
        )

        with (
            patch(LOAD_TOKENIZER_PATCH, side_effect=[tok1, tok2]),
            patch(DISPLAY_UTILITY_PATCH, return_value=mock_display),
        ):
            result = tokenizer_compare(config)

        # Display should be called for both tokenizers
        assert mock_display.display_tokens_from_prompt.call_count == 2

    def test_result_to_display(self):
        """Test result display formatting."""
        tok1 = MagicMock()
        tok1.encode.return_value = [1, 2, 3, 4]
        tok2 = MagicMock()
        tok2.encode.return_value = [1, 2]

        config = CompareConfig(
            tokenizer1="gpt2",
            tokenizer2="llama",
            text="Test",
            verbose=False,
        )

        with patch(LOAD_TOKENIZER_PATCH, side_effect=[tok1, tok2]):
            result = tokenizer_compare(config)

        display = result.to_display()
        assert "Token count 1: 4" in display
        assert "Token count 2: 2" in display
        assert "+2 tokens" in display
