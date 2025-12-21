"""Tests for TokenDisplayUtility class."""

import pytest

from chuk_lazarus.data.tokenizers.token_display import TokenDisplayUtility


class MockTokenizer:
    """Mock tokenizer for testing TokenDisplayUtility."""

    def __init__(self):
        self.vocab = {
            "hello": 0,
            "world": 1,
            "test": 2,
            "<pad>": 3,
            "<unk>": 4,
            "<bos>": 5,
            "<eos>": 6,
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def encode(self, text, add_special_tokens=True):
        """Simple encode that splits by whitespace."""
        tokens = text.split()
        ids = [self.vocab.get(t, 4) for t in tokens]  # 4 is <unk>
        if add_special_tokens:
            ids = [5] + ids + [6]  # <bos> and <eos>
        return ids

    def decode(self, ids):
        """Decode token IDs to string."""
        if isinstance(ids, int):
            ids = [ids]
        tokens = [self.id_to_token.get(i, "<unk>") for i in ids]
        return " ".join(tokens)

    def get_vocab(self):
        """Return vocabulary."""
        return self.vocab


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def display_util(mock_tokenizer):
    """Create a TokenDisplayUtility instance."""
    return TokenDisplayUtility(mock_tokenizer)


class TestSafeStrConversion:
    """Tests for safe_str_conversion method."""

    def test_none_value(self, display_util):
        """Test conversion of None."""
        assert display_util.safe_str_conversion(None) == "None"

    def test_bool_true(self, display_util):
        """Test conversion of True."""
        assert display_util.safe_str_conversion(True) == "True"

    def test_bool_false(self, display_util):
        """Test conversion of False."""
        assert display_util.safe_str_conversion(False) == "False"

    def test_integer(self, display_util):
        """Test conversion of integer."""
        assert display_util.safe_str_conversion(42) == "42"

    def test_float(self, display_util):
        """Test conversion of float."""
        assert display_util.safe_str_conversion(3.14) == "3.14"

    def test_string(self, display_util):
        """Test conversion of string."""
        assert display_util.safe_str_conversion("hello") == "hello"

    def test_list(self, display_util):
        """Test conversion of list (uses repr)."""
        result = display_util.safe_str_conversion([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_dict(self, display_util):
        """Test conversion of dict (uses repr)."""
        result = display_util.safe_str_conversion({"key": "value"})
        assert "key" in result and "value" in result


class TestTruncateString:
    """Tests for truncate_string method."""

    def test_short_string(self, display_util):
        """Test string shorter than max length."""
        result = display_util.truncate_string("hello", max_length=30)
        assert result == "hello"

    def test_exact_length(self, display_util):
        """Test string exactly at max length."""
        result = display_util.truncate_string("hello", max_length=5)
        assert result == "hello"

    def test_long_string(self, display_util):
        """Test string longer than max length."""
        result = display_util.truncate_string("this is a very long string", max_length=10)
        assert len(result) == 10
        assert result.endswith("...")

    def test_truncate_none(self, display_util):
        """Test truncating None value."""
        result = display_util.truncate_string(None, max_length=10)
        assert result == "None"

    def test_truncate_number(self, display_util):
        """Test truncating a number."""
        result = display_util.truncate_string(12345, max_length=30)
        assert result == "12345"


class TestDisplayTokensFromPrompt:
    """Tests for display_tokens_from_prompt method."""

    def test_display_tokens_from_prompt(self, display_util, capsys):
        """Test displaying tokens from a prompt."""
        display_util.display_tokens_from_prompt("hello world")
        captured = capsys.readouterr()
        assert "Index" in captured.out
        assert "Token ID" in captured.out

    def test_display_without_special_tokens(self, display_util, capsys):
        """Test displaying tokens without special tokens."""
        display_util.display_tokens_from_prompt("hello", add_special_tokens=False)
        captured = capsys.readouterr()
        assert "Index" in captured.out


class TestDisplayTokensFromIds:
    """Tests for display_tokens_from_ids method."""

    def test_display_tokens_from_ids(self, display_util, capsys):
        """Test displaying tokens from IDs."""
        display_util.display_tokens_from_ids([0, 1, 2])
        captured = capsys.readouterr()
        assert "Index" in captured.out
        assert "Token ID" in captured.out


class TestDisplayTokens:
    """Tests for display_tokens method."""

    def test_display_tokens(self, display_util, capsys):
        """Test displaying tokens."""
        display_util.display_tokens([0, 1])
        captured = capsys.readouterr()
        assert "Index" in captured.out
        assert "Token ID" in captured.out
        assert "Decoded Token" in captured.out

    def test_display_single_token(self, display_util, capsys):
        """Test displaying single token."""
        display_util.display_tokens([0])
        captured = capsys.readouterr()
        assert "Index" in captured.out
        assert "hello" in captured.out


class TestDisplayFullVocabulary:
    """Tests for display_full_vocabulary method."""

    def test_display_full_vocabulary(self, display_util, capsys):
        """Test displaying full vocabulary."""
        display_util.display_full_vocabulary(chunk_size=100)
        captured = capsys.readouterr()
        assert "Vocabulary Chunk" in captured.out
        assert "Index" in captured.out

    def test_display_full_vocabulary_small_chunks(self, display_util, capsys):
        """Test displaying vocabulary in small chunks."""
        display_util.display_full_vocabulary(chunk_size=3)
        captured = capsys.readouterr()
        # Should have multiple chunks since vocab has 7 items
        assert "Vocabulary Chunk 1" in captured.out
        assert "Vocabulary Chunk 2" in captured.out


class TestManualFormatTable:
    """Tests for manual_format_table method."""

    def test_manual_format_table(self, display_util):
        """Test manual table formatting."""
        table_data = [
            [0, 100, "hello"],
            [1, 101, "world"],
        ]
        headers = ["Index", "Token ID", "Token"]

        result = display_util.manual_format_table(table_data, headers)

        assert "Index" in result
        assert "Token ID" in result
        assert "Token" in result
        assert "hello" in result
        assert "world" in result

    def test_manual_format_empty_table(self, display_util):
        """Test manual formatting of empty table."""
        result = display_util.manual_format_table([], ["A", "B", "C"])
        assert "A" in result
        assert "B" in result
        assert "C" in result

    def test_manual_format_with_special_values(self, display_util):
        """Test manual formatting with special values."""
        table_data = [
            [0, None, True],
            [1, 3.14, [1, 2]],
        ]
        headers = ["Index", "Value1", "Value2"]

        result = display_util.manual_format_table(table_data, headers)

        assert "None" in result
        assert "True" in result


class TestTokenDisplayUtilityInit:
    """Tests for TokenDisplayUtility initialization."""

    def test_init_stores_tokenizer(self, mock_tokenizer):
        """Test that initialization stores the tokenizer."""
        util = TokenDisplayUtility(mock_tokenizer)
        assert util.tokenizer is mock_tokenizer
