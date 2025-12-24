"""Tests for CharacterTokenizer."""

import pytest

from chuk_lazarus.data.tokenizers import CharacterTokenizer, CharacterTokenizerConfig


class TestCharacterTokenizerCreation:
    """Tests for CharacterTokenizer creation."""

    def test_from_charset(self):
        """Test creation from explicit charset."""
        tokenizer = CharacterTokenizer.from_charset("abc")

        assert tokenizer.vocab_size == 7  # 4 special + 3 chars
        assert "a" in tokenizer
        assert "b" in tokenizer
        assert "c" in tokenizer
        assert "d" not in tokenizer

    def test_from_ascii(self):
        """Test creation with ASCII charset."""
        tokenizer = CharacterTokenizer.from_ascii()

        # Should have all printable ASCII + 4 special tokens
        assert tokenizer.vocab_size == 104  # 100 printable + 4 special
        assert "a" in tokenizer
        assert "Z" in tokenizer
        assert "0" in tokenizer
        assert " " in tokenizer

    def test_from_ascii_lowercase(self):
        """Test creation with lowercase ASCII."""
        tokenizer = CharacterTokenizer.from_ascii_lowercase()

        assert "a" in tokenizer
        assert "z" in tokenizer
        assert "A" not in tokenizer  # No uppercase
        assert " " in tokenizer
        assert "." in tokenizer

    def test_from_digits(self):
        """Test creation with digits only."""
        tokenizer = CharacterTokenizer.from_digits()

        assert tokenizer.vocab_size == 14  # 4 special + 10 digits
        assert "0" in tokenizer
        assert "9" in tokenizer
        assert "a" not in tokenizer

    def test_from_corpus(self):
        """Test learning vocabulary from corpus."""
        texts = ["hello", "world"]
        tokenizer = CharacterTokenizer.from_corpus(texts)

        # Unique chars: h, e, l, o, w, r, d = 7 + 4 special = 11
        assert tokenizer.vocab_size == 11
        assert "h" in tokenizer
        assert "w" in tokenizer
        assert "a" not in tokenizer

    def test_from_corpus_with_lowercase(self):
        """Test corpus learning with lowercase config."""
        texts = ["Hello", "WORLD"]
        config = CharacterTokenizerConfig(lowercase=True)
        tokenizer = CharacterTokenizer.from_corpus(texts, config)

        # All lowercase: h, e, l, o, w, r, d = 7 + 4 special = 11
        assert "h" in tokenizer
        assert "H" not in tokenizer

    def test_custom_config(self):
        """Test creation with custom config."""
        config = CharacterTokenizerConfig(
            pad_token_id=10,
            unk_token_id=11,
            bos_token_id=12,
            eos_token_id=13,
        )
        tokenizer = CharacterTokenizer.from_charset("ab", config)

        assert tokenizer.pad_token_id == 10
        assert tokenizer.unk_token_id == 11
        assert tokenizer.bos_token_id == 12
        assert tokenizer.eos_token_id == 13


class TestCharacterTokenizerEncoding:
    """Tests for encoding functionality."""

    def test_encode_basic(self):
        """Test basic encoding."""
        tokenizer = CharacterTokenizer.from_charset("abc")
        ids = tokenizer.encode("abc", add_special_tokens=False)

        assert len(ids) == 3
        assert all(isinstance(i, int) for i in ids)

    def test_encode_with_special_tokens(self):
        """Test encoding with BOS/EOS."""
        tokenizer = CharacterTokenizer.from_charset("abc")
        ids = tokenizer.encode("abc", add_special_tokens=True)

        assert len(ids) == 5  # BOS + 3 chars + EOS
        assert ids[0] == tokenizer.bos_token_id
        assert ids[-1] == tokenizer.eos_token_id

    def test_encode_unknown_chars(self):
        """Test encoding with unknown characters."""
        tokenizer = CharacterTokenizer.from_charset("abc")
        ids = tokenizer.encode("axyz", add_special_tokens=False)

        # 'a' should be known, 'xyz' should be UNK
        assert ids[0] != tokenizer.unk_token_id
        assert ids[1] == tokenizer.unk_token_id
        assert ids[2] == tokenizer.unk_token_id
        assert ids[3] == tokenizer.unk_token_id

    def test_encode_with_max_length(self):
        """Test encoding with max_length truncation."""
        tokenizer = CharacterTokenizer.from_charset("abcdef")
        ids = tokenizer.encode("abcdef", add_special_tokens=True, max_length=4)

        assert len(ids) == 4
        # Should preserve EOS at end
        assert ids[-1] == tokenizer.eos_token_id

    def test_encode_empty_string(self):
        """Test encoding empty string."""
        tokenizer = CharacterTokenizer.from_ascii()
        ids = tokenizer.encode("", add_special_tokens=True)

        assert len(ids) == 2  # Just BOS and EOS
        assert ids[0] == tokenizer.bos_token_id
        assert ids[1] == tokenizer.eos_token_id

    def test_encode_with_lowercase_config(self):
        """Test encoding with lowercase config."""
        config = CharacterTokenizerConfig(lowercase=True)
        tokenizer = CharacterTokenizer.from_charset("abc", config)

        # "ABC" should be lowercased and encoded
        ids = tokenizer.encode("ABC", add_special_tokens=False)
        ids_lower = tokenizer.encode("abc", add_special_tokens=False)

        assert ids == ids_lower


class TestCharacterTokenizerDecoding:
    """Tests for decoding functionality."""

    def test_decode_basic(self):
        """Test basic decoding."""
        tokenizer = CharacterTokenizer.from_charset("abc")
        ids = tokenizer.encode("abc", add_special_tokens=False)
        text = tokenizer.decode(ids)

        assert text == "abc"

    def test_decode_skip_special_tokens(self):
        """Test decoding with special token skipping."""
        tokenizer = CharacterTokenizer.from_charset("abc")
        ids = tokenizer.encode("abc", add_special_tokens=True)

        text = tokenizer.decode(ids, skip_special_tokens=True)
        assert text == "abc"

        text_with_special = tokenizer.decode(ids, skip_special_tokens=False)
        assert "<s>" in text_with_special or len(text_with_special) > 3

    def test_decode_empty(self):
        """Test decoding empty list."""
        tokenizer = CharacterTokenizer.from_ascii()
        text = tokenizer.decode([])

        assert text == ""

    def test_roundtrip(self):
        """Test encode-decode roundtrip."""
        tokenizer = CharacterTokenizer.from_ascii()
        original = "Hello, World! 123"

        ids = tokenizer.encode(original, add_special_tokens=False)
        decoded = tokenizer.decode(ids)

        assert decoded == original

    def test_roundtrip_with_special_tokens(self):
        """Test roundtrip with special tokens."""
        tokenizer = CharacterTokenizer.from_ascii()
        original = "test"

        ids = tokenizer.encode(original, add_special_tokens=True)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)

        assert decoded == original


class TestCharacterTokenizerBatch:
    """Tests for batch operations."""

    def test_encode_batch(self):
        """Test batch encoding."""
        tokenizer = CharacterTokenizer.from_ascii()
        texts = ["hello", "world", "test"]

        batch = tokenizer.encode_batch(texts, add_special_tokens=False)

        assert len(batch) == 3
        assert len(batch[0]) == 5  # "hello"
        assert len(batch[1]) == 5  # "world"
        assert len(batch[2]) == 4  # "test"

    def test_encode_batch_with_padding(self):
        """Test batch encoding with padding."""
        tokenizer = CharacterTokenizer.from_ascii()
        texts = ["hi", "hello"]

        batch = tokenizer.encode_batch(
            texts, add_special_tokens=False, max_length=10, padding=True
        )

        assert len(batch[0]) == 10
        assert len(batch[1]) == 10
        # Check padding
        assert batch[0].count(tokenizer.pad_token_id) == 8  # 10 - 2

    def test_decode_batch(self):
        """Test batch decoding."""
        tokenizer = CharacterTokenizer.from_ascii()
        texts = ["hello", "world"]

        batch = tokenizer.encode_batch(texts, add_special_tokens=False)
        decoded = tokenizer.decode_batch(batch)

        assert decoded == texts


class TestCharacterTokenizerUtilities:
    """Tests for utility methods."""

    def test_get_vocab(self):
        """Test getting vocabulary."""
        tokenizer = CharacterTokenizer.from_charset("abc")
        vocab = tokenizer.get_vocab()

        assert isinstance(vocab, dict)
        assert len(vocab) == tokenizer.vocab_size
        assert "a" in vocab
        assert "<pad>" in vocab

    def test_get_charset(self):
        """Test getting charset."""
        tokenizer = CharacterTokenizer.from_charset("abc")
        charset = tokenizer.get_charset()

        assert "a" in charset
        assert "b" in charset
        assert "c" in charset
        assert "<pad>" not in charset

    def test_len(self):
        """Test __len__."""
        tokenizer = CharacterTokenizer.from_charset("abc")
        assert len(tokenizer) == tokenizer.vocab_size

    def test_contains(self):
        """Test __contains__."""
        tokenizer = CharacterTokenizer.from_charset("abc")

        assert "a" in tokenizer
        assert "d" not in tokenizer

    def test_repr(self):
        """Test __repr__."""
        tokenizer = CharacterTokenizer.from_charset("abc")
        repr_str = repr(tokenizer)

        assert "CharacterTokenizer" in repr_str
        assert "vocab_size=" in repr_str

    def test_tokenize(self):
        """Test tokenize method."""
        tokenizer = CharacterTokenizer.from_ascii()
        tokens = tokenizer.tokenize("hello")

        assert tokens == ["h", "e", "l", "l", "o"]


class TestCharacterTokenizerProperties:
    """Tests for tokenizer properties."""

    def test_special_token_ids(self):
        """Test special token ID properties."""
        tokenizer = CharacterTokenizer.from_ascii()

        assert tokenizer.pad_token_id == 0
        assert tokenizer.unk_token_id == 1
        assert tokenizer.bos_token_id == 2
        assert tokenizer.eos_token_id == 3

    def test_vocab_size(self):
        """Test vocab_size property."""
        tokenizer = CharacterTokenizer.from_charset("abc")
        assert tokenizer.vocab_size == 7  # 4 special + 3 chars


class TestCharacterTokenizerConfig:
    """Tests for CharacterTokenizerConfig."""

    def test_defaults(self):
        """Test default config values."""
        config = CharacterTokenizerConfig()

        assert config.pad_token_id == 0
        assert config.unk_token_id == 1
        assert config.bos_token_id == 2
        assert config.eos_token_id == 3
        assert config.lowercase is False

    def test_frozen(self):
        """Test config is frozen."""
        from pydantic import ValidationError

        config = CharacterTokenizerConfig()
        with pytest.raises(ValidationError):
            config.lowercase = True

    def test_custom_values(self):
        """Test custom config values."""
        config = CharacterTokenizerConfig(
            pad_token_id=100,
            unk_token_id=101,
            bos_token_id=102,
            eos_token_id=103,
            lowercase=True,
        )

        assert config.pad_token_id == 100
        assert config.lowercase is True


class TestCharacterTokenizerClassification:
    """Tests simulating classification use cases."""

    def test_sentiment_classification_workflow(self):
        """Test typical sentiment classification workflow."""
        # Train tokenizer on corpus
        train_texts = [
            "great movie loved it",
            "terrible waste of time",
            "amazing performance",
            "boring and slow",
        ]
        tokenizer = CharacterTokenizer.from_corpus(train_texts)

        # Encode training data
        encoded = tokenizer.encode_batch(
            train_texts, add_special_tokens=True, max_length=50, padding=True
        )

        # All same length after padding
        assert all(len(e) == 50 for e in encoded)

        # Can decode back
        decoded = tokenizer.decode_batch(encoded, skip_special_tokens=True)
        # Decoded should match (stripped of padding)
        assert all(d.strip("\x00") == t or d == t for d, t in zip(decoded, train_texts))

    def test_label_mapping(self):
        """Test creating label tokenizer for classification."""
        labels = ["positive", "negative", "neutral"]
        label_tokenizer = CharacterTokenizer.from_corpus(labels)

        # Each label should encode consistently
        for label in labels:
            ids1 = label_tokenizer.encode(label, add_special_tokens=False)
            ids2 = label_tokenizer.encode(label, add_special_tokens=False)
            assert ids1 == ids2
