"""Shared fixtures for tokenizer command tests."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "Hello world"
    tokenizer.get_vocab.return_value = {
        "<pad>": 0,
        "<eos>": 1,
        "hello": 2,
        "world": 3,
        "test": 4,
    }
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.bos_token_id = None
    tokenizer.unk_token_id = None
    tokenizer.convert_ids_to_tokens.return_value = ["<pad>"]
    return tokenizer


@pytest.fixture
def mock_fingerprint():
    """Create a mock fingerprint result."""
    fp = MagicMock()
    fp.fingerprint = "abc123"
    fp.vocab_size = 32000
    fp.vocab_hash = "hash_vocab"
    fp.full_hash = "hash_full"
    fp.special_tokens_hash = "hash_special"
    fp.merges_hash = "hash_merges"
    fp.special_tokens = {"pad_token_id": 0, "eos_token_id": 1}
    return fp


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Testing tokenization 123.",
    ]


@pytest.fixture
def sample_texts_file(tmp_path, sample_texts):
    """Create a temporary file with sample texts."""
    file_path = tmp_path / "texts.txt"
    with open(file_path, "w") as f:
        for text in sample_texts:
            f.write(text + "\n")
    return file_path
