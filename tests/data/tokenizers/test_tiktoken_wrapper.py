"""Tests for TiktokenWrapper."""

import pytest

from chuk_lazarus.data.tokenizers.tiktoken_wrapper import (
    TiktokenWrapper,
    is_tiktoken_model,
)


def _tiktoken_available() -> bool:
    """Check if tiktoken is available."""
    try:
        import tiktoken  # noqa: F401

        return True
    except ImportError:
        return False


class TestIsTiktokenModel:
    """Tests for is_tiktoken_model function."""

    def test_gpt4_models(self):
        """GPT-4 models should be detected."""
        assert is_tiktoken_model("gpt-4") is True
        assert is_tiktoken_model("gpt-4-turbo") is True
        assert is_tiktoken_model("gpt-4o") is True
        assert is_tiktoken_model("gpt-4o-mini") is True

    def test_gpt35_models(self):
        """GPT-3.5 models should be detected."""
        assert is_tiktoken_model("gpt-3.5-turbo") is True
        assert is_tiktoken_model("gpt-3.5-turbo-16k") is True

    def test_o1_models(self):
        """O1 models should be detected."""
        assert is_tiktoken_model("o1") is True
        assert is_tiktoken_model("o1-mini") is True
        assert is_tiktoken_model("o1-preview") is True

    def test_encoding_names(self):
        """Encoding names should be detected."""
        assert is_tiktoken_model("cl100k_base") is True
        assert is_tiktoken_model("o200k_base") is True
        assert is_tiktoken_model("p50k_base") is True
        assert is_tiktoken_model("r50k_base") is True

    def test_huggingface_models_not_detected(self):
        """HuggingFace models should not be detected as tiktoken."""
        # Note: "gpt2" is both a HuggingFace model AND a tiktoken encoding name
        # We treat it as tiktoken since tiktoken's gpt2 encoding is well-known
        assert is_tiktoken_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0") is False
        assert is_tiktoken_model("meta-llama/Llama-2-7b") is False
        assert is_tiktoken_model("mistralai/Mistral-7B-v0.1") is False
        assert is_tiktoken_model("bert-base-uncased") is False
        assert is_tiktoken_model("facebook/opt-125m") is False

    def test_case_insensitive(self):
        """Detection should be case-insensitive."""
        assert is_tiktoken_model("GPT-4") is True
        assert is_tiktoken_model("Gpt-3.5-Turbo") is True
        assert is_tiktoken_model("CL100K_BASE") is True


@pytest.mark.skipif(not _tiktoken_available(), reason="tiktoken not installed")
class TestTiktokenWrapperFromModel:
    """Tests for TiktokenWrapper.from_model."""

    def test_load_gpt4(self):
        """Should load GPT-4 tokenizer."""
        tokenizer = TiktokenWrapper.from_model("gpt-4")
        assert tokenizer is not None
        assert tokenizer.vocab_size > 0

    def test_load_gpt35(self):
        """Should load GPT-3.5 tokenizer."""
        tokenizer = TiktokenWrapper.from_model("gpt-3.5-turbo")
        assert tokenizer is not None
        assert tokenizer.vocab_size > 0

    def test_load_gpt4o(self):
        """Should load GPT-4o tokenizer."""
        tokenizer = TiktokenWrapper.from_model("gpt-4o")
        assert tokenizer is not None
        # o200k_base has ~200k tokens
        assert tokenizer.vocab_size >= 100000

    def test_load_encoding_by_name_via_from_model(self):
        """Should load encoding when passed as model name."""
        # This tests the fallback path in from_model when KeyError is raised
        # and the name is a valid encoding
        tokenizer = TiktokenWrapper.from_model("cl100k_base")
        assert tokenizer is not None
        assert tokenizer.vocab_size == 100277

    def test_invalid_model(self):
        """Should raise ValueError for invalid model."""
        with pytest.raises(ValueError, match="Unknown model"):
            TiktokenWrapper.from_model("invalid-model-xyz")


@pytest.mark.skipif(not _tiktoken_available(), reason="tiktoken not installed")
class TestTiktokenWrapperFromEncoding:
    """Tests for TiktokenWrapper.from_encoding."""

    def test_load_cl100k(self):
        """Should load cl100k_base encoding."""
        tokenizer = TiktokenWrapper.from_encoding("cl100k_base")
        assert tokenizer is not None
        assert tokenizer.vocab_size == 100277  # cl100k has exactly this many tokens

    def test_load_o200k(self):
        """Should load o200k_base encoding."""
        tokenizer = TiktokenWrapper.from_encoding("o200k_base")
        assert tokenizer is not None
        assert tokenizer.vocab_size == 200019  # o200k has exactly this many tokens

    def test_invalid_encoding(self):
        """Should raise ValueError for invalid encoding."""
        with pytest.raises(ValueError, match="Unknown encoding"):
            TiktokenWrapper.from_encoding("invalid_encoding")


@pytest.mark.skipif(not _tiktoken_available(), reason="tiktoken not installed")
class TestTiktokenWrapperEncode:
    """Tests for TiktokenWrapper.encode."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        return TiktokenWrapper.from_model("gpt-4")

    def test_encode_simple(self, tokenizer):
        """Should encode simple text."""
        tokens = tokenizer.encode("Hello, world!")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_encode_empty(self, tokenizer):
        """Should handle empty string."""
        tokens = tokenizer.encode("")
        assert tokens == []

    def test_encode_unicode(self, tokenizer):
        """Should handle Unicode text."""
        tokens = tokenizer.encode("Hello 世界 🌍")
        assert len(tokens) > 0

    def test_encode_long_text(self, tokenizer):
        """Should handle long text."""
        text = "The quick brown fox jumps over the lazy dog. " * 100
        tokens = tokenizer.encode(text)
        assert len(tokens) > 100


@pytest.mark.skipif(not _tiktoken_available(), reason="tiktoken not installed")
class TestTiktokenWrapperDecode:
    """Tests for TiktokenWrapper.decode."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        return TiktokenWrapper.from_model("gpt-4")

    def test_decode_simple(self, tokenizer):
        """Should decode tokens back to text."""
        original = "Hello, world!"
        tokens = tokenizer.encode(original)
        decoded = tokenizer.decode(tokens)
        assert decoded == original

    def test_decode_empty(self, tokenizer):
        """Should handle empty token list."""
        decoded = tokenizer.decode([])
        assert decoded == ""

    def test_roundtrip(self, tokenizer):
        """Encode/decode should be lossless."""
        texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming AI.",
            "def hello(): return 'world'",
            "1 + 1 = 2",
        ]
        for text in texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Roundtrip failed for: {text}"


@pytest.mark.skipif(not _tiktoken_available(), reason="tiktoken not installed")
class TestTiktokenWrapperVocab:
    """Tests for TiktokenWrapper vocabulary methods."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        return TiktokenWrapper.from_model("gpt-4")

    def test_vocab_size(self, tokenizer):
        """Should have correct vocab size."""
        assert tokenizer.vocab_size == 100277  # cl100k_base

    def test_get_vocab(self, tokenizer):
        """Should return vocabulary dict."""
        vocab = tokenizer.get_vocab()
        assert isinstance(vocab, dict)
        assert len(vocab) > 0
        # All values should be ints
        assert all(isinstance(v, int) for v in vocab.values())

    def test_get_vocab_cached(self, tokenizer):
        """Vocab should be cached."""
        vocab1 = tokenizer.get_vocab()
        vocab2 = tokenizer.get_vocab()
        assert vocab1 is vocab2  # Same object

    def test_convert_ids_to_tokens(self, tokenizer):
        """Should convert IDs to token strings."""
        tokens = tokenizer.encode("Hello")
        token_strs = tokenizer.convert_ids_to_tokens(tokens)
        assert len(token_strs) == len(tokens)
        assert all(isinstance(s, str) for s in token_strs)

    def test_convert_tokens_to_ids(self, tokenizer):
        """Should convert token strings to IDs."""
        tokens = tokenizer.encode("Hello")
        token_strs = tokenizer.convert_ids_to_tokens(tokens)
        ids = tokenizer.convert_tokens_to_ids(token_strs)
        assert ids == tokens

    def test_convert_ids_to_tokens_invalid_id(self, tokenizer):
        """Should handle invalid token IDs gracefully."""
        # Use a very large invalid token ID
        invalid_id = 999999999
        tokens = tokenizer.convert_ids_to_tokens([invalid_id])
        assert len(tokens) == 1
        assert "<UNK:" in tokens[0]

    def test_convert_tokens_to_ids_unknown_token(self, tokenizer):
        """Should return 0 for unknown tokens."""
        ids = tokenizer.convert_tokens_to_ids(["<NONEXISTENT_TOKEN>"])
        assert ids == [0]


@pytest.mark.skipif(not _tiktoken_available(), reason="tiktoken not installed")
class TestTiktokenWrapperSpecialTokens:
    """Tests for special token properties."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        return TiktokenWrapper.from_model("gpt-4")

    def test_has_special_token_properties(self, tokenizer):
        """Should have special token properties."""
        # These should exist even if None
        assert hasattr(tokenizer, "pad_token_id")
        assert hasattr(tokenizer, "unk_token_id")
        assert hasattr(tokenizer, "bos_token_id")
        assert hasattr(tokenizer, "eos_token_id")

    def test_pad_token_id_settable(self, tokenizer):
        """Should be able to set pad_token_id."""
        tokenizer.pad_token_id = 0
        assert tokenizer.pad_token_id == 0

    def test_name_or_path(self, tokenizer):
        """Should return model name via name_or_path property."""
        assert tokenizer.name_or_path == "gpt-4"

    def test_repr(self, tokenizer):
        """Should have a useful repr."""
        repr_str = repr(tokenizer)
        assert "TiktokenWrapper" in repr_str
        assert "gpt-4" in repr_str
        assert "vocab_size=" in repr_str


@pytest.mark.skipif(not _tiktoken_available(), reason="tiktoken not installed")
class TestTiktokenWrapperIntegration:
    """Integration tests with tokenizer analysis tools."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        return TiktokenWrapper.from_model("gpt-4")

    def test_works_with_analyze_coverage(self, tokenizer):
        """Should work with analyze_coverage."""
        from chuk_lazarus.data.tokenizers.analyze import analyze_coverage

        texts = ["Hello, world!", "The quick brown fox."]
        report = analyze_coverage(texts, tokenizer)
        assert report.total_tokens > 0
        # Check that the report has expected attributes
        assert report.total_texts == 2

    def test_works_with_fingerprint(self, tokenizer):
        """Should work with compute_fingerprint."""
        from chuk_lazarus.data.tokenizers.fingerprint import compute_fingerprint

        fp = compute_fingerprint(tokenizer)
        assert fp.fingerprint is not None
        # Fingerprint vocab_size may differ slightly from tokenizer.vocab_size
        # due to how get_vocab handles special tokens
        assert fp.vocab_size > 0
