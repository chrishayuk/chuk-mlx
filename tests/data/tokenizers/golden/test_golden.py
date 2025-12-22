"""Golden tests for popular tokenizer model families.

These tests verify that tokenization behavior matches known-good outputs
for specific model families. They serve as regression tests to detect:
- Changes in tokenizer behavior from library updates
- Model version mismatches
- Configuration errors

Golden tests are marked as 'slow' since they require downloading real tokenizers.
Run with: pytest -m golden
"""

import pytest

from chuk_lazarus.data.tokenizers.fingerprint import compute_fingerprint

# Mark all tests in this module as slow/golden
pytestmark = [pytest.mark.slow, pytest.mark.golden]


# =============================================================================
# Golden Test Fixtures - Known-good tokenization results
# =============================================================================

# Each fixture defines:
# - model: HuggingFace model identifier
# - fingerprint: Expected short fingerprint (first 16 chars of full hash)
# - vocab_size: Expected vocabulary size
# - special_tokens: Expected special token IDs
# - golden_texts: List of (input_text, expected_token_ids) tuples

TINYLLAMA_GOLDEN = {
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "fingerprint": "4fa65691bbbdf232",
    "vocab_size": 32000,
    "special_tokens": {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 2,
        "unk_token_id": 0,
    },
    "golden_texts": [
        # (input_text, expected_token_ids without special tokens)
        ("Hello", [15043]),
        ("Hello, world!", [15043, 29892, 3186, 29991]),
        ("The quick brown fox", [450, 4996, 17354, 1701, 29916]),
        ("1 + 1 = 2", [29871, 29896, 718, 29871, 29896, 353, 29871, 29906]),
    ],
}

# GPT-2 / Phi family (byte-level BPE)
GPT2_GOLDEN = {
    "model": "gpt2",
    "fingerprint": None,  # Will be computed on first run
    "vocab_size": 50257,
    "special_tokens": {
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "unk_token_id": 50256,  # GPT-2 uses EOS as UNK
    },
    "golden_texts": [
        ("Hello", [15496]),
        ("Hello, world!", [15496, 11, 995, 0]),
        ("The quick brown fox", [464, 2068, 7586, 21831]),
    ],
}


# =============================================================================
# Test Helper Functions
# =============================================================================


def load_tokenizer_for_test(model_name: str):
    """Load tokenizer, skipping test if download fails."""
    try:
        from chuk_lazarus.utils.tokenizer_loader import load_tokenizer

        return load_tokenizer(model_name)
    except Exception as e:
        pytest.skip(f"Could not load tokenizer {model_name}: {e}")


def assert_special_tokens(tokenizer, expected: dict):
    """Assert special token IDs match expected values."""
    for attr, expected_id in expected.items():
        actual_id = getattr(tokenizer, attr, None)
        assert actual_id == expected_id, f"{attr}: expected {expected_id}, got {actual_id}"


def assert_golden_tokenization(tokenizer, golden_texts: list):
    """Assert tokenization matches golden expected output."""
    for text, expected_ids in golden_texts:
        actual_ids = tokenizer.encode(text, add_special_tokens=False)
        assert actual_ids == expected_ids, (
            f"Tokenization mismatch for '{text}':\n"
            f"  Expected: {expected_ids}\n"
            f"  Actual:   {actual_ids}"
        )


def assert_roundtrip(tokenizer, texts: list[str]):
    """Assert encode/decode roundtrip preserves text."""
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        # Normalize whitespace for comparison
        normalized_text = " ".join(text.split())
        normalized_decoded = " ".join(decoded.split())
        assert normalized_text == normalized_decoded, (
            f"Roundtrip mismatch for '{text}':\n  Decoded: '{decoded}'"
        )


# =============================================================================
# TinyLlama / Llama Family Tests
# =============================================================================


class TestTinyLlamaGolden:
    """Golden tests for TinyLlama (Llama family representative)."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Load TinyLlama tokenizer once per test class."""
        return load_tokenizer_for_test(TINYLLAMA_GOLDEN["model"])

    def test_vocab_size(self, tokenizer):
        """Verify vocabulary size matches expected."""
        vocab = tokenizer.get_vocab()
        assert len(vocab) == TINYLLAMA_GOLDEN["vocab_size"], (
            f"Vocab size mismatch: expected {TINYLLAMA_GOLDEN['vocab_size']}, got {len(vocab)}"
        )

    def test_special_tokens(self, tokenizer):
        """Verify special token IDs match expected."""
        assert_special_tokens(tokenizer, TINYLLAMA_GOLDEN["special_tokens"])

    def test_fingerprint(self, tokenizer):
        """Verify fingerprint matches expected."""
        fp = compute_fingerprint(tokenizer)
        expected = TINYLLAMA_GOLDEN["fingerprint"]
        assert fp.fingerprint == expected, (
            f"Fingerprint mismatch:\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {fp.fingerprint}\n"
            "This may indicate a tokenizer library update or model change."
        )

    def test_golden_tokenization(self, tokenizer):
        """Verify tokenization matches known-good outputs."""
        assert_golden_tokenization(tokenizer, TINYLLAMA_GOLDEN["golden_texts"])

    def test_roundtrip_basic(self, tokenizer):
        """Verify encode/decode roundtrip for basic text."""
        test_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world.",
        ]
        assert_roundtrip(tokenizer, test_texts)

    def test_roundtrip_math(self, tokenizer):
        """Verify encode/decode roundtrip for math expressions."""
        test_texts = [
            "1 + 1 = 2",
            "x^2 + y^2 = r^2",
            "f(x) = 3x + 7",
        ]
        assert_roundtrip(tokenizer, test_texts)

    def test_roundtrip_code(self, tokenizer):
        """Verify encode/decode roundtrip for code snippets."""
        test_texts = [
            "def hello(): pass",
            "for i in range(10): print(i)",
            "import numpy as np",
        ]
        assert_roundtrip(tokenizer, test_texts)

    def test_chat_template_available(self, tokenizer):
        """Verify chat template is available."""
        assert hasattr(tokenizer, "chat_template"), "Missing chat_template attribute"
        assert tokenizer.chat_template is not None, "Chat template is None"

    def test_chat_template_works(self, tokenizer):
        """Verify chat template can format messages."""
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Hello!" in result
        assert "Hi there!" in result


# =============================================================================
# GPT-2 / Phi Family Tests (Optional - may not be installed)
# =============================================================================


@pytest.mark.skip(reason="GPT-2 download can be slow; enable for full testing")
class TestGPT2Golden:
    """Golden tests for GPT-2 (byte-level BPE representative)."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Load GPT-2 tokenizer once per test class."""
        return load_tokenizer_for_test(GPT2_GOLDEN["model"])

    def test_vocab_size(self, tokenizer):
        """Verify vocabulary size matches expected."""
        vocab = tokenizer.get_vocab()
        assert len(vocab) == GPT2_GOLDEN["vocab_size"]

    def test_special_tokens(self, tokenizer):
        """Verify special token IDs match expected."""
        assert_special_tokens(tokenizer, GPT2_GOLDEN["special_tokens"])

    def test_golden_tokenization(self, tokenizer):
        """Verify tokenization matches known-good outputs."""
        assert_golden_tokenization(tokenizer, GPT2_GOLDEN["golden_texts"])


# =============================================================================
# Cross-Model Consistency Tests
# =============================================================================


class TestTokenizerConsistency:
    """Tests for tokenizer behavior consistency."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Load TinyLlama for consistency tests."""
        return load_tokenizer_for_test(TINYLLAMA_GOLDEN["model"])

    def test_deterministic_encoding(self, tokenizer):
        """Verify encoding is deterministic (same input -> same output)."""
        text = "This is a test of deterministic encoding."
        ids1 = tokenizer.encode(text)
        ids2 = tokenizer.encode(text)
        ids3 = tokenizer.encode(text)
        assert ids1 == ids2 == ids3, "Encoding should be deterministic"

    def test_empty_string(self, tokenizer):
        """Verify handling of empty string."""
        ids = tokenizer.encode("", add_special_tokens=False)
        assert ids == [], "Empty string should encode to empty list"

    def test_whitespace_only(self, tokenizer):
        """Verify handling of whitespace-only input."""
        ids = tokenizer.encode("   ", add_special_tokens=False)
        # Should encode to at least one token (space tokens)
        assert isinstance(ids, list)

    def test_unicode_basic(self, tokenizer):
        """Verify handling of basic Unicode characters."""
        test_cases = [
            "café",
            "naïve",
            "100",  # Full-width characters
        ]
        for text in test_cases:
            ids = tokenizer.encode(text, add_special_tokens=False)
            assert len(ids) > 0, f"Should encode '{text}' to non-empty"
            decoded = tokenizer.decode(ids)
            assert text in decoded or decoded.strip() == text.strip()

    def test_long_text_encoding(self, tokenizer):
        """Verify handling of long text."""
        # Create a long text by repeating a sentence
        sentence = "The quick brown fox jumps over the lazy dog. "
        long_text = sentence * 100  # ~4500 characters

        ids = tokenizer.encode(long_text, add_special_tokens=False)
        assert len(ids) > 0, "Should encode long text"
        # Token count should be roughly proportional
        assert len(ids) < len(long_text), "Should compress text into fewer tokens"

    def test_special_characters(self, tokenizer):
        """Verify handling of special characters."""
        test_cases = [
            "Hello @world!",
            "Price: $99.99",
            "Email: user@example.com",
            "Path: /usr/local/bin",
        ]
        for text in test_cases:
            ids = tokenizer.encode(text, add_special_tokens=False)
            assert len(ids) > 0, f"Should encode '{text}'"


# =============================================================================
# Regression Detection Tests
# =============================================================================


class TestTokenizerRegression:
    """Tests designed to detect tokenizer regressions."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Load TinyLlama for regression tests."""
        return load_tokenizer_for_test(TINYLLAMA_GOLDEN["model"])

    def test_token_count_stability(self, tokenizer):
        """Verify token counts for standard texts are stable."""
        # These counts should NOT change between runs
        expected_counts = [
            ("Hello, world!", 4),
            ("The quick brown fox jumps over the lazy dog.", 12),
            ("Machine learning models require careful training.", 7),
        ]
        for text, expected_count in expected_counts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            assert len(ids) == expected_count, (
                f"Token count regression for '{text}':\n"
                f"  Expected: {expected_count}\n"
                f"  Actual:   {len(ids)}"
            )

    def test_specific_token_presence(self, tokenizer):
        """Verify specific tokens are present in vocabulary."""
        vocab = tokenizer.get_vocab()
        # These tokens should exist in Llama-family tokenizers
        expected_tokens = ["<s>", "</s>", "<unk>"]
        for token in expected_tokens:
            assert token in vocab, f"Expected token '{token}' not in vocabulary"

    def test_bos_eos_not_same(self, tokenizer):
        """Verify BOS and EOS tokens are distinct (or correctly aliased)."""
        bos_id = getattr(tokenizer, "bos_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        # TinyLlama uses different IDs for BOS (1) and EOS (2)
        assert bos_id == 1, f"BOS should be 1, got {bos_id}"
        assert eos_id == 2, f"EOS should be 2, got {eos_id}"
