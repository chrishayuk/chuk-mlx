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
    "tokenizer_type": "sentencepiece",  # Llama-style SentencePiece BPE
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
    "tokenizer_type": "bpe",
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

# Mistral (SentencePiece BPE) - Llama family
MISTRAL_GOLDEN = {
    "model": "mistralai/Mistral-7B-v0.1",
    "tokenizer_type": "sentencepiece",
    "fingerprint": None,
    "vocab_size": 32000,
    "special_tokens": {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "unk_token_id": 0,
    },
    "golden_texts": [
        ("Hello", [22557]),
        ("Hello, world!", [22557, 28725, 1526, 28808]),
    ],
}

# Qwen (BPE with ChatML template)
QWEN_GOLDEN = {
    "model": "Qwen/Qwen2-0.5B",
    "tokenizer_type": "bpe",
    "fingerprint": None,
    "vocab_size": 151936,
    "special_tokens": {
        "bos_token_id": None,  # Qwen doesn't use BOS
        "eos_token_id": 151645,  # <|endoftext|>
    },
    "golden_texts": [
        ("Hello", [9707]),
        ("Hello, world!", [9707, 11, 1879, 0]),
    ],
}

# Gemma (SentencePiece Unigram)
GEMMA_GOLDEN = {
    "model": "google/gemma-2b",
    "tokenizer_type": "sentencepiece",
    "fingerprint": None,
    "vocab_size": 256000,
    "special_tokens": {
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
    },
    "golden_texts": [
        ("Hello", [4521]),
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


# =============================================================================
# HuggingFace Parity Tests
# =============================================================================


class TestHuggingFaceParity:
    """Tests ensuring our tokenizer loading matches HuggingFace directly.

    These tests verify that load_tokenizer() produces identical results
    to loading directly via transformers.AutoTokenizer.
    """

    @pytest.fixture(scope="class")
    def hf_tokenizer(self):
        """Load TinyLlama via HuggingFace directly."""
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(TINYLLAMA_GOLDEN["model"])
        except Exception as e:
            pytest.skip(f"Could not load HF tokenizer: {e}")

    @pytest.fixture(scope="class")
    def our_tokenizer(self):
        """Load TinyLlama via our loader."""
        return load_tokenizer_for_test(TINYLLAMA_GOLDEN["model"])

    def test_encode_parity(self, hf_tokenizer, our_tokenizer):
        """Verify encode() produces identical results to HuggingFace."""
        test_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "def hello(): return 'world'",
            "x = 3.14159 * r ** 2",
            "Unicode: 你好世界 🎉",
        ]
        for text in test_texts:
            hf_ids = hf_tokenizer.encode(text, add_special_tokens=False)
            our_ids = our_tokenizer.encode(text, add_special_tokens=False)
            assert hf_ids == our_ids, (
                f"Encode parity mismatch for '{text[:50]}...':\n"
                f"  HF:  {hf_ids[:10]}...\n"
                f"  Ours: {our_ids[:10]}..."
            )

    def test_decode_parity(self, hf_tokenizer, our_tokenizer):
        """Verify decode() produces identical results to HuggingFace."""
        # Use known token IDs from TinyLlama
        test_ids = [
            [15043],  # "Hello"
            [15043, 29892, 3186, 29991],  # "Hello, world!"
            [450, 4996, 17354, 1701, 29916],  # "The quick brown fox"
        ]
        for ids in test_ids:
            hf_text = hf_tokenizer.decode(ids, skip_special_tokens=True)
            our_text = our_tokenizer.decode(ids, skip_special_tokens=True)
            assert hf_text == our_text, (
                f"Decode parity mismatch for {ids}:\n  HF:  '{hf_text}'\n  Ours: '{our_text}'"
            )

    def test_vocab_parity(self, hf_tokenizer, our_tokenizer):
        """Verify vocabulary matches HuggingFace."""
        hf_vocab = hf_tokenizer.get_vocab()
        our_vocab = our_tokenizer.get_vocab()
        assert len(hf_vocab) == len(our_vocab), (
            f"Vocab size mismatch: HF={len(hf_vocab)}, Ours={len(our_vocab)}"
        )

    def test_special_tokens_parity(self, hf_tokenizer, our_tokenizer):
        """Verify special token IDs match HuggingFace."""
        attrs = ["pad_token_id", "unk_token_id", "bos_token_id", "eos_token_id"]
        for attr in attrs:
            hf_val = getattr(hf_tokenizer, attr, None)
            our_val = getattr(our_tokenizer, attr, None)
            assert hf_val == our_val, (
                f"Special token parity mismatch for {attr}: HF={hf_val}, Ours={our_val}"
            )

    def test_chat_template_parity(self, hf_tokenizer, our_tokenizer):
        """Verify chat template rendering matches HuggingFace."""
        if not hasattr(hf_tokenizer, "chat_template") or not hf_tokenizer.chat_template:
            pytest.skip("Tokenizer has no chat template")

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "And 3+3?"},
        ]

        hf_result = hf_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        our_result = our_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        assert hf_result == our_result, (
            f"Chat template parity mismatch:\n"
            f"  HF:\n{hf_result[:200]}...\n"
            f"  Ours:\n{our_result[:200]}..."
        )


# =============================================================================
# Tiktoken / OpenAI Golden Tests
# =============================================================================


def _tiktoken_available() -> bool:
    """Check if tiktoken is available."""
    try:
        import tiktoken  # noqa: F401

        return True
    except ImportError:
        return False


TIKTOKEN_GPT4_GOLDEN = {
    "model": "gpt-4",
    "encoding": "cl100k_base",
    "vocab_size": 100277,
    "golden_texts": [
        # (input_text, expected_token_ids)
        ("Hello", [9906]),
        ("Hello, world!", [9906, 11, 1917, 0]),
        ("The quick brown fox", [791, 4062, 14198, 39935]),
    ],
}

TIKTOKEN_GPT4O_GOLDEN = {
    "model": "gpt-4o",
    "encoding": "o200k_base",
    "vocab_size": 200019,
    "golden_texts": [
        ("Hello", [13225]),
        ("Hello, world!", [13225, 11, 2375, 0]),
    ],
}


@pytest.mark.skipif(not _tiktoken_available(), reason="tiktoken not installed")
class TestTiktokenGolden:
    """Golden tests for tiktoken/OpenAI tokenizers."""

    @pytest.fixture(scope="class")
    def gpt4_tokenizer(self):
        """Load GPT-4 tokenizer via our wrapper."""
        from chuk_lazarus.data.tokenizers.tiktoken_wrapper import TiktokenWrapper

        return TiktokenWrapper.from_model("gpt-4")

    @pytest.fixture(scope="class")
    def gpt4o_tokenizer(self):
        """Load GPT-4o tokenizer via our wrapper."""
        from chuk_lazarus.data.tokenizers.tiktoken_wrapper import TiktokenWrapper

        return TiktokenWrapper.from_model("gpt-4o")

    def test_gpt4_vocab_size(self, gpt4_tokenizer):
        """Verify GPT-4 vocabulary size."""
        assert gpt4_tokenizer.vocab_size == TIKTOKEN_GPT4_GOLDEN["vocab_size"]

    def test_gpt4_golden_tokenization(self, gpt4_tokenizer):
        """Verify GPT-4 tokenization matches expected."""
        for text, expected_ids in TIKTOKEN_GPT4_GOLDEN["golden_texts"]:
            actual_ids = gpt4_tokenizer.encode(text)
            assert actual_ids == expected_ids, (
                f"GPT-4 tokenization mismatch for '{text}':\n"
                f"  Expected: {expected_ids}\n"
                f"  Actual:   {actual_ids}"
            )

    def test_gpt4o_vocab_size(self, gpt4o_tokenizer):
        """Verify GPT-4o vocabulary size."""
        assert gpt4o_tokenizer.vocab_size == TIKTOKEN_GPT4O_GOLDEN["vocab_size"]

    def test_gpt4o_golden_tokenization(self, gpt4o_tokenizer):
        """Verify GPT-4o tokenization matches expected."""
        for text, expected_ids in TIKTOKEN_GPT4O_GOLDEN["golden_texts"]:
            actual_ids = gpt4o_tokenizer.encode(text)
            assert actual_ids == expected_ids, (
                f"GPT-4o tokenization mismatch for '{text}':\n"
                f"  Expected: {expected_ids}\n"
                f"  Actual:   {actual_ids}"
            )

    def test_gpt4_roundtrip(self, gpt4_tokenizer):
        """Verify GPT-4 encode/decode roundtrip."""
        test_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "def hello(): return 42",
        ]
        for text in test_texts:
            encoded = gpt4_tokenizer.encode(text)
            decoded = gpt4_tokenizer.decode(encoded)
            assert decoded == text, f"Roundtrip failed: '{text}' -> '{decoded}'"


@pytest.mark.skipif(not _tiktoken_available(), reason="tiktoken not installed")
class TestTiktokenDirectParity:
    """Tests ensuring our TiktokenWrapper matches tiktoken directly."""

    def test_encode_parity_gpt4(self):
        """Verify our wrapper matches tiktoken.encode() for GPT-4."""
        import tiktoken

        from chuk_lazarus.data.tokenizers.tiktoken_wrapper import TiktokenWrapper

        direct = tiktoken.encoding_for_model("gpt-4")
        wrapper = TiktokenWrapper.from_model("gpt-4")

        test_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "import numpy as np\nprint(np.array([1,2,3]))",
            "Unicode: 你好 🌍 émoji",
        ]
        for text in test_texts:
            direct_ids = direct.encode(text)
            wrapper_ids = wrapper.encode(text)
            assert direct_ids == wrapper_ids, (
                f"Tiktoken parity mismatch for '{text[:30]}...':\n"
                f"  Direct:  {direct_ids[:10]}...\n"
                f"  Wrapper: {wrapper_ids[:10]}..."
            )

    def test_decode_parity_gpt4(self):
        """Verify our wrapper matches tiktoken.decode() for GPT-4."""
        import tiktoken

        from chuk_lazarus.data.tokenizers.tiktoken_wrapper import TiktokenWrapper

        direct = tiktoken.encoding_for_model("gpt-4")
        wrapper = TiktokenWrapper.from_model("gpt-4")

        test_ids = [
            [9906, 11, 1917, 0],  # "Hello, world!"
            [791, 4062, 14198, 39935],  # "The quick brown fox"
        ]
        for ids in test_ids:
            direct_text = direct.decode(ids)
            wrapper_text = wrapper.decode(ids)
            assert direct_text == wrapper_text, (
                f"Tiktoken decode parity mismatch for {ids}:\n"
                f"  Direct:  '{direct_text}'\n"
                f"  Wrapper: '{wrapper_text}'"
            )


# =============================================================================
# Multi-Model Tokenizer Type Tests (BPE / SentencePiece / Unigram)
# =============================================================================


# Parametrized test data for multiple tokenizer types
TOKENIZER_TYPE_FIXTURES = [
    pytest.param(
        TINYLLAMA_GOLDEN,
        id="tinyllama-sentencepiece",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        GPT2_GOLDEN,
        id="gpt2-bpe",
        marks=[pytest.mark.slow, pytest.mark.skip(reason="Slow download")],
    ),
]


@pytest.mark.parametrize("fixture", TOKENIZER_TYPE_FIXTURES)
class TestTokenizerTypeParity:
    """Parametrized tests across different tokenizer types."""

    def test_hf_encode_parity(self, fixture):
        """Verify encode() matches HuggingFace directly."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            pytest.skip("transformers not installed")

        try:
            hf_tok = AutoTokenizer.from_pretrained(fixture["model"])
        except Exception as e:
            pytest.skip(f"Could not load {fixture['model']}: {e}")

        our_tok = load_tokenizer_for_test(fixture["model"])

        test_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "def foo(x): return x * 2",
            "1 + 2 = 3",
            "Unicode test: 你好世界 🎉",
        ]

        for text in test_texts:
            hf_ids = hf_tok.encode(text, add_special_tokens=False)
            our_ids = our_tok.encode(text, add_special_tokens=False)
            assert hf_ids == our_ids, (
                f"[{fixture['model']}] Encode parity failed for '{text[:30]}...'\n"
                f"  HF:  {hf_ids[:10]}...\n"
                f"  Ours: {our_ids[:10]}..."
            )

    def test_hf_decode_parity(self, fixture):
        """Verify decode() matches HuggingFace directly."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            pytest.skip("transformers not installed")

        try:
            hf_tok = AutoTokenizer.from_pretrained(fixture["model"])
        except Exception as e:
            pytest.skip(f"Could not load {fixture['model']}: {e}")

        our_tok = load_tokenizer_for_test(fixture["model"])

        # Test golden text IDs
        for text, expected_ids in fixture["golden_texts"]:
            hf_text = hf_tok.decode(expected_ids, skip_special_tokens=True)
            our_text = our_tok.decode(expected_ids, skip_special_tokens=True)
            assert hf_text == our_text, (
                f"[{fixture['model']}] Decode parity failed for {expected_ids}\n"
                f"  HF:  '{hf_text}'\n"
                f"  Ours: '{our_text}'"
            )

    def test_roundtrip(self, fixture):
        """Verify encode/decode roundtrip."""
        tokenizer = load_tokenizer_for_test(fixture["model"])

        test_texts = [
            "Simple text",
            "The quick brown fox jumps over the lazy dog.",
            "Special chars: @#$%^&*()",
            "Math: x^2 + y^2 = z^2",
        ]

        for text in test_texts:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(encoded, skip_special_tokens=True)
            # Normalize whitespace
            normalized_text = " ".join(text.split())
            normalized_decoded = " ".join(decoded.split())
            assert normalized_text == normalized_decoded, (
                f"[{fixture['model']}] Roundtrip failed for '{text}'\n  Decoded: '{decoded}'"
            )

    def test_vocab_size(self, fixture):
        """Verify vocabulary size matches expected."""
        tokenizer = load_tokenizer_for_test(fixture["model"])
        vocab = tokenizer.get_vocab()
        assert len(vocab) == fixture["vocab_size"], (
            f"[{fixture['model']}] Vocab size mismatch: "
            f"expected {fixture['vocab_size']}, got {len(vocab)}"
        )


# =============================================================================
# Chat Template Tests
# =============================================================================


class TestChatTemplateValidation:
    """Tests for chat template detection and validation."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Load TinyLlama which has a chat template."""
        return load_tokenizer_for_test(TINYLLAMA_GOLDEN["model"])

    def test_chat_template_detection(self, tokenizer):
        """Verify chat template format detection works."""
        from chuk_lazarus.data.tokenizers.runtime.chat_templates import (
            ChatTemplateRegistry,
            TemplateFormat,
        )

        registry = ChatTemplateRegistry()
        template = getattr(tokenizer, "chat_template", None)

        assert template is not None, "TinyLlama should have chat template"

        format = registry.detect_format(template)
        # TinyLlama uses Phi-style or Llama-style depending on version
        assert format != TemplateFormat.UNKNOWN, f"Should detect known format, got {format.value}"

    def test_chat_template_validation(self, tokenizer):
        """Verify chat template validation works."""
        from chuk_lazarus.data.tokenizers.runtime.chat_templates import (
            validate_chat_template,
        )

        result = validate_chat_template(tokenizer)

        assert result.is_valid, (
            f"Template should be valid. Issues: {[i.message for i in result.issues]}"
        )

    def test_chat_template_rendering(self, tokenizer):
        """Verify chat template renders correctly."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "And 3+3?"},
        ]

        result = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # All content should appear in output
        assert "2+2" in result, "User message should appear"
        assert "4" in result, "Assistant response should appear"
        assert "3+3" in result, "Second user message should appear"


class TestChatTemplateRegistry:
    """Tests for the chat template registry."""

    def test_registry_has_common_formats(self):
        """Verify registry has common template formats."""
        from chuk_lazarus.data.tokenizers.runtime.chat_templates import (
            ChatTemplateRegistry,
            TemplateFormat,
        )

        registry = ChatTemplateRegistry()
        templates = registry.list_templates()

        formats = {t.format for t in templates}

        # Should have at least these common formats
        expected = {
            TemplateFormat.CHATML,
            TemplateFormat.LLAMA,
            TemplateFormat.PHI,
            TemplateFormat.GEMMA,
        }

        for fmt in expected:
            assert fmt in formats, f"Registry should have {fmt.value} template"

    def test_model_family_detection(self):
        """Verify model family to template mapping works."""
        from chuk_lazarus.data.tokenizers.runtime.chat_templates import (
            ChatTemplateRegistry,
            TemplateFormat,
        )

        registry = ChatTemplateRegistry()

        # Test various model name patterns
        test_cases = [
            ("meta-llama/Llama-2-7b-chat-hf", TemplateFormat.LLAMA),
            ("Qwen/Qwen2-0.5B", TemplateFormat.CHATML),
            ("microsoft/Phi-3-mini-4k-instruct", TemplateFormat.PHI),
            ("google/gemma-2b", TemplateFormat.GEMMA),
        ]

        for model_name, expected_format in test_cases:
            template = registry.get_for_model_family(model_name)
            if template:
                assert template.format == expected_format, (
                    f"Model {model_name} should map to {expected_format.value}, "
                    f"got {template.format.value}"
                )

    def test_jinja2_syntax_validation(self):
        """Verify Jinja2 syntax validation works."""
        from chuk_lazarus.data.tokenizers.runtime.chat_templates import (
            validate_jinja2_syntax,
        )

        # Valid template
        valid = "{% for m in messages %}{{ m.content }}{% endfor %}"
        is_valid, error = validate_jinja2_syntax(valid)
        assert is_valid, f"Valid template should pass: {error}"

        # Invalid template (unclosed tag)
        invalid = "{% for m in messages %}{{ m.content }"
        is_valid, error = validate_jinja2_syntax(invalid)
        assert not is_valid, "Invalid template should fail"
        assert error is not None
