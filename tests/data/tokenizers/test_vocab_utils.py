"""Tests for vocab_utils module."""

import json
import os

import pytest

from chuk_lazarus.data.tokenizers.vocab_utils import load_vocabulary, save_vocabulary


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

        vocab, special_tokens, added_tokens = load_vocabulary(str(vocab_file))

        assert vocab == {"hello": 0, "world": 1, "<unk>": 2}
        assert special_tokens == {"<pad>": 3, "<unk>": 2, "<bos>": 4, "<eos>": 5}
        assert added_tokens == ["<custom>"]

    def test_load_vocabulary_empty_optional_fields(self, tmp_path):
        """Test loading vocabulary with missing optional fields."""
        vocab_file = tmp_path / "tokenizer.json"
        vocab_data = {"vocab": {"test": 0}}
        with open(vocab_file, "w") as f:
            json.dump(vocab_data, f)

        vocab, special_tokens, added_tokens = load_vocabulary(str(vocab_file))

        assert vocab == {"test": 0}
        assert special_tokens == {}
        assert added_tokens == []

    def test_load_vocabulary_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(ValueError, match="valid vocab_file path"):
            load_vocabulary("/nonexistent/path/tokenizer.json")

    def test_load_vocabulary_none_path(self):
        """Test loading with None path."""
        with pytest.raises(ValueError, match="valid vocab_file path"):
            load_vocabulary(None)

    def test_load_vocabulary_empty_path(self):
        """Test loading with empty path."""
        with pytest.raises(ValueError, match="valid vocab_file path"):
            load_vocabulary("")


class TestSaveVocabulary:
    """Tests for save_vocabulary function."""

    def test_save_vocabulary_success(self, tmp_path):
        """Test saving vocabulary to a file."""
        vocab = {"hello": 0, "world": 1}
        special_tokens = {"<pad>": 2, "<unk>": 3, "<bos>": 4, "<eos>": 5}
        added_tokens = ["<custom>"]

        result = save_vocabulary(vocab, special_tokens, added_tokens, str(tmp_path))

        assert result == str(tmp_path / "tokenizer.json")
        assert os.path.exists(result)

        with open(result) as f:
            saved_data = json.load(f)

        assert saved_data["vocab"] == vocab
        assert saved_data["special_tokens"] == special_tokens
        assert saved_data["added_tokens"] == added_tokens
        assert saved_data["version"] == "1.0"

    def test_save_vocabulary_creates_directory(self, tmp_path):
        """Test that save_vocabulary creates directory if it doesn't exist."""
        new_dir = tmp_path / "new_subdir" / "nested"
        vocab = {"test": 0}
        special_tokens = {}
        added_tokens = []

        result = save_vocabulary(vocab, special_tokens, added_tokens, str(new_dir))

        assert os.path.exists(new_dir)
        assert os.path.exists(result)

    def test_save_vocabulary_custom_version(self, tmp_path):
        """Test saving vocabulary with custom version."""
        vocab = {"test": 0}
        special_tokens = {}
        added_tokens = []

        result = save_vocabulary(vocab, special_tokens, added_tokens, str(tmp_path), version="2.0")

        with open(result) as f:
            saved_data = json.load(f)

        assert saved_data["version"] == "2.0"

    def test_save_and_load_roundtrip(self, tmp_path):
        """Test that saving and loading produces identical data."""
        vocab = {"hello": 0, "world": 1, "test": 2}
        special_tokens = {"<pad>": 3, "<unk>": 4, "<bos>": 5, "<eos>": 6}
        added_tokens = ["<custom1>", "<custom2>"]

        saved_file = save_vocabulary(vocab, special_tokens, added_tokens, str(tmp_path))
        loaded_vocab, loaded_special, loaded_added = load_vocabulary(saved_file)

        assert loaded_vocab == vocab
        assert loaded_special == special_tokens
        assert loaded_added == added_tokens
