"""Tests for vocab_utils module."""

import json

import pytest
from pydantic import ValidationError

from chuk_lazarus.data.tokenizers.types import VocabularyData
from chuk_lazarus.data.tokenizers.vocab_utils import (
    load_vocabulary,
    load_vocabulary_async,
    save_vocabulary,
    save_vocabulary_async,
)


class TestLoadVocabulary:
    """Tests for load_vocabulary function."""

    def test_load_vocabulary_success(self, tmp_path):
        """Test loading a valid vocabulary file."""
        vocab_file = tmp_path / "tokenizer.json"
        vocab_data = {
            "vocab": {"hello": 0, "world": 1, "<unk>": 2},
            "special_tokens": {"<pad>": 3, "<unk>": 2, "<bos>": 4, "<eos>": 5},
            "added_tokens": ["<custom>"],
        }
        with open(vocab_file, "w") as f:
            json.dump(vocab_data, f)

        result = load_vocabulary(vocab_file)

        assert isinstance(result, VocabularyData)
        assert result.vocab == {"hello": 0, "world": 1, "<unk>": 2}
        assert result.special_tokens == {"<pad>": 3, "<unk>": 2, "<bos>": 4, "<eos>": 5}
        assert result.added_tokens == ["<custom>"]

    def test_load_vocabulary_empty_optional_fields(self, tmp_path):
        """Test loading vocabulary with missing optional fields."""
        vocab_file = tmp_path / "tokenizer.json"
        vocab_data = {"vocab": {"test": 0}}
        with open(vocab_file, "w") as f:
            json.dump(vocab_data, f)

        result = load_vocabulary(vocab_file)

        assert result.vocab == {"test": 0}
        assert result.special_tokens == {}
        assert result.added_tokens == []

    def test_load_vocabulary_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(ValueError, match="Vocabulary file not found"):
            load_vocabulary("/nonexistent/path/tokenizer.json")

    def test_load_vocabulary_with_path_object(self, tmp_path):
        """Test loading with Path object."""
        vocab_file = tmp_path / "tokenizer.json"
        vocab_data = {"vocab": {"test": 0}}
        with open(vocab_file, "w") as f:
            json.dump(vocab_data, f)

        result = load_vocabulary(vocab_file)  # Path object, not string

        assert result.vocab == {"test": 0}


class TestSaveVocabulary:
    """Tests for save_vocabulary function."""

    def test_save_vocabulary_success(self, tmp_path):
        """Test saving vocabulary to a file."""
        vocab_data = VocabularyData(
            vocab={"hello": 0, "world": 1},
            special_tokens={"<pad>": 2, "<unk>": 3, "<bos>": 4, "<eos>": 5},
            added_tokens=["<custom>"],
        )

        result = save_vocabulary(vocab_data, tmp_path)

        expected_path = tmp_path / "tokenizer.json"
        assert result == expected_path
        assert result.exists()

        with open(result) as f:
            saved_data = json.load(f)

        assert saved_data["vocab"] == {"hello": 0, "world": 1}
        assert saved_data["special_tokens"] == {"<pad>": 2, "<unk>": 3, "<bos>": 4, "<eos>": 5}
        assert saved_data["added_tokens"] == ["<custom>"]
        assert saved_data["version"] == "1.0"

    def test_save_vocabulary_creates_directory(self, tmp_path):
        """Test that save_vocabulary creates directory if it doesn't exist."""
        new_dir = tmp_path / "new_subdir" / "nested"
        vocab_data = VocabularyData(vocab={"test": 0})

        result = save_vocabulary(vocab_data, new_dir)

        assert new_dir.exists()
        assert result.exists()

    def test_save_vocabulary_custom_version(self, tmp_path):
        """Test saving vocabulary with custom version."""
        vocab_data = VocabularyData(vocab={"test": 0})

        result = save_vocabulary(vocab_data, tmp_path, version="2.0")

        with open(result) as f:
            saved_data = json.load(f)

        assert saved_data["version"] == "2.0"

    def test_save_and_load_roundtrip(self, tmp_path):
        """Test that saving and loading produces identical data."""
        vocab_data = VocabularyData(
            vocab={"hello": 0, "world": 1, "test": 2},
            special_tokens={"<pad>": 3, "<unk>": 4, "<bos>": 5, "<eos>": 6},
            added_tokens=["<custom1>", "<custom2>"],
        )

        save_vocabulary(vocab_data, tmp_path)
        loaded = load_vocabulary(tmp_path / "tokenizer.json")

        assert loaded.vocab == vocab_data.vocab
        assert loaded.special_tokens == vocab_data.special_tokens
        assert loaded.added_tokens == list(vocab_data.added_tokens)


class TestAsyncVocabularyIO:
    """Tests for async vocabulary I/O functions."""

    @pytest.mark.asyncio
    async def test_save_and_load_async(self, tmp_path):
        """Test async save and load roundtrip."""
        vocab_data = VocabularyData(
            vocab={"hello": 0, "world": 1},
            special_tokens={"<pad>": 2, "<unk>": 3},
            added_tokens=["<custom>"],
        )

        await save_vocabulary_async(vocab_data, tmp_path)
        loaded = await load_vocabulary_async(tmp_path / "tokenizer.json")

        assert loaded.vocab == vocab_data.vocab
        assert loaded.special_tokens == vocab_data.special_tokens
        assert loaded.added_tokens == list(vocab_data.added_tokens)

    @pytest.mark.asyncio
    async def test_load_async_file_not_found(self):
        """Test async loading from non-existent file."""
        with pytest.raises(ValueError, match="Vocabulary file not found"):
            await load_vocabulary_async("/nonexistent/path/tokenizer.json")


class TestVocabularyData:
    """Tests for VocabularyData model."""

    def test_get_special_token_id(self):
        """Test getting special token ID by enum or string."""
        from chuk_lazarus.data.tokenizers.types import SpecialTokenName

        vocab_data = VocabularyData(
            vocab={"test": 0},
            special_tokens={"<pad>": 1, "<unk>": 2},
        )

        # By enum
        assert vocab_data.get_special_token_id(SpecialTokenName.PAD) == 1
        assert vocab_data.get_special_token_id(SpecialTokenName.UNK) == 2

        # By string
        assert vocab_data.get_special_token_id("<pad>") == 1
        assert vocab_data.get_special_token_id("<unk>") == 2

        # Missing
        assert vocab_data.get_special_token_id(SpecialTokenName.BOS) is None
        assert vocab_data.get_special_token_id("<nonexistent>") is None

    def test_frozen(self):
        """Test VocabularyData is immutable."""
        vocab_data = VocabularyData(vocab={"test": 0})

        with pytest.raises(ValidationError):  # Pydantic frozen model
            vocab_data.vocab = {"new": 1}
