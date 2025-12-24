"""Tests for BoWCharacterTokenizer."""

import pytest

from chuk_lazarus.data.tokenizers import BoWCharacterTokenizer, BoWTokenizerConfig


class TestBoWTokenizerCreation:
    """Tests for BoWCharacterTokenizer creation."""

    def test_from_charset(self):
        """Test creation from explicit charset."""
        tokenizer = BoWCharacterTokenizer("abc")

        assert tokenizer.vocab_size == 3
        assert "a" in tokenizer
        assert "b" in tokenizer
        assert "c" in tokenizer
        assert "d" not in tokenizer

    def test_from_corpus(self):
        """Test learning vocabulary from corpus."""
        texts = ["hello", "world"]
        tokenizer = BoWCharacterTokenizer.from_corpus(texts)

        # Unique chars: d, e, h, l, o, r, w = 7
        assert tokenizer.vocab_size == 7
        assert "h" in tokenizer
        assert "w" in tokenizer
        assert "a" not in tokenizer

    def test_from_corpus_with_config(self):
        """Test corpus learning with config."""
        texts = ["Hello", "WORLD"]
        config = BoWTokenizerConfig(lowercase=True)
        tokenizer = BoWCharacterTokenizer.from_corpus(texts, config)

        # All lowercase: d, e, h, l, o, r, w = 7
        assert tokenizer.vocab_size == 7
        assert "h" in tokenizer
        assert "H" not in tokenizer

    def test_from_ascii(self):
        """Test creation with ASCII charset."""
        tokenizer = BoWCharacterTokenizer.from_ascii()

        # Should have all printable ASCII
        assert tokenizer.vocab_size == 100  # printable ASCII chars
        assert "a" in tokenizer
        assert "Z" in tokenizer
        assert "0" in tokenizer
        assert " " in tokenizer

    def test_from_ascii_lowercase(self):
        """Test creation with lowercase ASCII."""
        tokenizer = BoWCharacterTokenizer.from_ascii_lowercase()

        assert "a" in tokenizer
        assert "z" in tokenizer
        assert " " in tokenizer
        assert "." in tokenizer

    def test_default_config(self):
        """Test default config values."""
        tokenizer = BoWCharacterTokenizer("abc")

        assert tokenizer.config.lowercase is True
        assert tokenizer.config.normalize is True


class TestBoWTokenizerEncoding:
    """Tests for encoding functionality."""

    def test_encode_basic(self):
        """Test basic encoding returns float vector."""
        tokenizer = BoWCharacterTokenizer("abc")
        vec = tokenizer.encode("abc")

        assert len(vec) == 3
        assert all(isinstance(v, float) for v in vec)

    def test_encode_normalized(self):
        """Test encoding with normalization."""
        config = BoWTokenizerConfig(normalize=True)
        tokenizer = BoWCharacterTokenizer("abc", config)

        vec = tokenizer.encode("aaa")

        # Should be normalized (sums to 1.0)
        assert abs(sum(vec) - 1.0) < 1e-6

    def test_encode_unnormalized(self):
        """Test encoding without normalization."""
        config = BoWTokenizerConfig(normalize=False)
        tokenizer = BoWCharacterTokenizer("abc", config)

        vec = tokenizer.encode("aaa")

        # Should have raw counts
        assert sum(vec) == 3.0

    def test_encode_character_counts(self):
        """Test that encoding counts characters correctly."""
        config = BoWTokenizerConfig(normalize=False)
        tokenizer = BoWCharacterTokenizer("abc", config)

        vec = tokenizer.encode("aabbc")

        # Sorted charset: a, b, c -> indices 0, 1, 2
        assert vec[0] == 2.0  # 'a' appears twice
        assert vec[1] == 2.0  # 'b' appears twice
        assert vec[2] == 1.0  # 'c' appears once

    def test_encode_unknown_chars_ignored(self):
        """Test that unknown characters are ignored."""
        config = BoWTokenizerConfig(normalize=False)
        tokenizer = BoWCharacterTokenizer("abc", config)

        vec = tokenizer.encode("axyz")

        # Only 'a' should be counted
        assert vec[0] == 1.0  # 'a'
        assert vec[1] == 0.0  # 'b'
        assert vec[2] == 0.0  # 'c'

    def test_encode_empty_string(self):
        """Test encoding empty string."""
        config = BoWTokenizerConfig(normalize=True)
        tokenizer = BoWCharacterTokenizer("abc", config)

        vec = tokenizer.encode("")

        # All zeros (no normalization when total is 0)
        assert all(v == 0.0 for v in vec)

    def test_encode_with_lowercase(self):
        """Test encoding with lowercase config."""
        config = BoWTokenizerConfig(lowercase=True, normalize=False)
        tokenizer = BoWCharacterTokenizer("abc", config)

        vec_upper = tokenizer.encode("ABC")
        vec_lower = tokenizer.encode("abc")

        assert vec_upper == vec_lower

    def test_encode_without_lowercase(self):
        """Test encoding without lowercase."""
        config = BoWTokenizerConfig(lowercase=False, normalize=False)
        tokenizer = BoWCharacterTokenizer("abc", config)

        vec = tokenizer.encode("ABC")

        # Uppercase not in vocab, so all zeros
        assert all(v == 0.0 for v in vec)


class TestBoWTokenizerBatch:
    """Tests for batch operations."""

    def test_encode_batch(self):
        """Test batch encoding."""
        tokenizer = BoWCharacterTokenizer.from_corpus(["cat", "dog"])
        texts = ["cat", "dog", "cog"]

        batch = tokenizer.encode_batch(texts)

        assert len(batch) == 3
        assert all(len(vec) == tokenizer.vocab_size for vec in batch)
        assert all(isinstance(v, float) for vec in batch for v in vec)

    def test_encode_batch_same_as_individual(self):
        """Test batch encoding matches individual encoding."""
        tokenizer = BoWCharacterTokenizer.from_corpus(["hello", "world"])
        texts = ["hello", "world"]

        batch = tokenizer.encode_batch(texts)
        individual = [tokenizer.encode(t) for t in texts]

        assert batch == individual


class TestBoWTokenizerUtilities:
    """Tests for utility methods."""

    def test_get_vocab(self):
        """Test getting vocabulary."""
        tokenizer = BoWCharacterTokenizer("abc")
        vocab = tokenizer.get_vocab()

        assert isinstance(vocab, dict)
        assert len(vocab) == 3
        assert "a" in vocab
        assert vocab["a"] == 0  # First in sorted order

    def test_get_charset(self):
        """Test getting charset."""
        tokenizer = BoWCharacterTokenizer("cba")  # Unsorted input
        charset = tokenizer.get_charset()

        # Should be sorted
        assert charset == "abc"

    def test_len(self):
        """Test __len__."""
        tokenizer = BoWCharacterTokenizer("abc")
        assert len(tokenizer) == 3

    def test_contains(self):
        """Test __contains__."""
        tokenizer = BoWCharacterTokenizer("abc")

        assert "a" in tokenizer
        assert "d" not in tokenizer

    def test_repr(self):
        """Test __repr__."""
        tokenizer = BoWCharacterTokenizer("abc")
        repr_str = repr(tokenizer)

        assert "BoWCharacterTokenizer" in repr_str
        assert "vocab_size=3" in repr_str

    def test_repr_long_charset(self):
        """Test __repr__ with long charset truncates."""
        tokenizer = BoWCharacterTokenizer.from_ascii()
        repr_str = repr(tokenizer)

        assert "..." in repr_str


class TestBoWTokenizerConfig:
    """Tests for BoWTokenizerConfig."""

    def test_defaults(self):
        """Test default config values."""
        config = BoWTokenizerConfig()

        assert config.lowercase is True
        assert config.normalize is True

    def test_frozen(self):
        """Test config is frozen."""
        from pydantic import ValidationError

        config = BoWTokenizerConfig()
        with pytest.raises(ValidationError):
            config.lowercase = False

    def test_custom_values(self):
        """Test custom config values."""
        config = BoWTokenizerConfig(
            lowercase=False,
            normalize=False,
        )

        assert config.lowercase is False
        assert config.normalize is False


class TestBoWTokenizerClassification:
    """Tests simulating classification use cases."""

    def test_sentiment_classification_workflow(self):
        """Test typical sentiment classification workflow."""
        train_texts = [
            "great movie loved it",
            "terrible waste of time",
            "amazing performance",
            "boring and slow",
        ]
        tokenizer = BoWCharacterTokenizer.from_corpus(train_texts)

        # Encode training data
        encoded = tokenizer.encode_batch(train_texts)

        # All same length (vocab_size)
        assert all(len(e) == tokenizer.vocab_size for e in encoded)

        # Each vector sums to ~1.0 (normalized)
        for vec in encoded:
            assert abs(sum(vec) - 1.0) < 1e-6

    def test_consistent_encoding(self):
        """Test that same text always produces same encoding."""
        tokenizer = BoWCharacterTokenizer.from_corpus(["hello", "world"])

        vec1 = tokenizer.encode("hello world")
        vec2 = tokenizer.encode("hello world")

        assert vec1 == vec2

    def test_order_invariance(self):
        """Test that word order doesn't affect encoding (bag-of-words)."""
        config = BoWTokenizerConfig(normalize=False)
        tokenizer = BoWCharacterTokenizer.from_corpus(["abc"], config)

        vec1 = tokenizer.encode("abc")
        vec2 = tokenizer.encode("cba")
        vec3 = tokenizer.encode("bca")

        assert vec1 == vec2 == vec3


class TestBoWTokenizerEdgeCases:
    """Tests for edge cases."""

    def test_single_char_vocab(self):
        """Test with single character vocabulary."""
        tokenizer = BoWCharacterTokenizer("a")

        vec = tokenizer.encode("aaa")
        assert len(vec) == 1
        assert vec[0] == 1.0  # Normalized

    def test_unicode_chars(self):
        """Test with unicode characters."""
        tokenizer = BoWCharacterTokenizer("αβγ")

        assert tokenizer.vocab_size == 3
        assert "α" in tokenizer

        vec = tokenizer.encode("αβ")
        assert len(vec) == 3

    def test_whitespace_handling(self):
        """Test whitespace is handled correctly."""
        config = BoWTokenizerConfig(normalize=False)
        tokenizer = BoWCharacterTokenizer(" \t\n", config)

        assert " " in tokenizer
        assert "\t" in tokenizer
        assert "\n" in tokenizer

        vec = tokenizer.encode("  \t")
        assert sum(vec) == 3.0

    def test_duplicate_chars_in_init(self):
        """Test that duplicate characters are handled."""
        tokenizer = BoWCharacterTokenizer("aaabbbccc")

        # Should only have unique chars
        assert tokenizer.vocab_size == 3
