"""Tests for tokenizers __init__.py exports."""

from chuk_lazarus.data.tokenizers import (
    CustomTokenizer,
    TokenDisplayUtility,
    load_vocabulary,
    save_vocabulary,
)


def test_custom_tokenizer_export():
    """Test that CustomTokenizer is exported."""
    assert CustomTokenizer is not None


def test_token_display_utility_export():
    """Test that TokenDisplayUtility is exported."""
    assert TokenDisplayUtility is not None


def test_load_vocabulary_export():
    """Test that load_vocabulary is exported."""
    assert callable(load_vocabulary)


def test_save_vocabulary_export():
    """Test that save_vocabulary is exported."""
    assert callable(save_vocabulary)
