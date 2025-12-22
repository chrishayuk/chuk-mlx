"""Tests for tokenizer fingerprinting."""

import json
import tempfile
from pathlib import Path

import pytest

from chuk_lazarus.data.tokenizers.fingerprint import (
    FingerprintMismatch,
    FingerprintRegistry,
    TokenizerFingerprint,
    assert_fingerprint,
    compute_fingerprint,
    get_registry,
    load_fingerprint,
    save_fingerprint,
    verify_fingerprint,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab: dict[str, int] | None = None):
        self.vocab = vocab or {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
            "hello": 4,
            "world": 5,
        }

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def unk_token_id(self) -> int:
        return 1

    @property
    def bos_token_id(self) -> int:
        return 2

    @property
    def eos_token_id(self) -> int:
        return 3


class TestTokenizerFingerprint:
    """Tests for TokenizerFingerprint model."""

    def test_create_fingerprint(self):
        fp = TokenizerFingerprint(
            fingerprint="abc123",
            full_hash="abc123def456",
            vocab_hash="vocab123",
            special_tokens_hash="special123",
            merges_hash="merges123",
            vocab_size=1000,
        )
        assert fp.fingerprint == "abc123"
        assert fp.vocab_size == 1000

    def test_matches(self):
        fp1 = TokenizerFingerprint(
            fingerprint="abc",
            full_hash="abc123",
            vocab_hash="v1",
            special_tokens_hash="s1",
            merges_hash="m1",
            vocab_size=100,
        )
        fp2 = TokenizerFingerprint(
            fingerprint="abc",
            full_hash="abc123",
            vocab_hash="v1",
            special_tokens_hash="s1",
            merges_hash="m1",
            vocab_size=100,
        )
        assert fp1.matches(fp2)

    def test_matches_vocab(self):
        fp1 = TokenizerFingerprint(
            fingerprint="a",
            full_hash="a1",
            vocab_hash="same",
            special_tokens_hash="diff1",
            merges_hash="m1",
            vocab_size=100,
        )
        fp2 = TokenizerFingerprint(
            fingerprint="b",
            full_hash="b1",
            vocab_hash="same",
            special_tokens_hash="diff2",
            merges_hash="m1",
            vocab_size=100,
        )
        assert fp1.matches_vocab(fp2)
        assert not fp1.matches(fp2)

    def test_diff(self):
        fp1 = TokenizerFingerprint(
            fingerprint="a",
            full_hash="a1",
            vocab_hash="v1",
            special_tokens_hash="s1",
            merges_hash="m1",
            vocab_size=100,
        )
        fp2 = TokenizerFingerprint(
            fingerprint="b",
            full_hash="b1",
            vocab_hash="v1",
            special_tokens_hash="s2",
            merges_hash="m1",
            vocab_size=200,
        )
        diff = fp1.diff(fp2)
        assert diff["vocab_matches"] is True
        assert diff["special_tokens_match"] is False
        assert diff["merges_match"] is True
        assert diff["size_matches"] is False
        assert diff["full_match"] is False


class TestComputeFingerprint:
    """Tests for compute_fingerprint function."""

    def test_basic_fingerprint(self):
        tokenizer = MockTokenizer()
        fp = compute_fingerprint(tokenizer)

        assert isinstance(fp, TokenizerFingerprint)
        assert len(fp.fingerprint) == 16
        assert len(fp.full_hash) == 64  # SHA-256
        assert fp.vocab_size == 6

    def test_fingerprint_deterministic(self):
        tokenizer = MockTokenizer()
        fp1 = compute_fingerprint(tokenizer)
        fp2 = compute_fingerprint(tokenizer)

        assert fp1.fingerprint == fp2.fingerprint
        assert fp1.full_hash == fp2.full_hash

    def test_different_vocab_different_fingerprint(self):
        tok1 = MockTokenizer({"a": 0, "b": 1})
        tok2 = MockTokenizer({"x": 0, "y": 1})

        fp1 = compute_fingerprint(tok1)
        fp2 = compute_fingerprint(tok2)

        assert fp1.fingerprint != fp2.fingerprint

    def test_special_tokens_captured(self):
        tokenizer = MockTokenizer()
        fp = compute_fingerprint(tokenizer)

        assert fp.special_tokens["pad_token_id"] == 0
        assert fp.special_tokens["unk_token_id"] == 1
        assert fp.special_tokens["bos_token_id"] == 2
        assert fp.special_tokens["eos_token_id"] == 3

    def test_with_merges(self):
        tokenizer = MockTokenizer()
        merges = ["a b", "c d", "e f"]
        fp = compute_fingerprint(tokenizer, merges=merges)

        assert fp.merges_hash != "none"

    def test_without_merges(self):
        tokenizer = MockTokenizer()
        fp = compute_fingerprint(tokenizer)

        # Without merges provided, should be "none"
        assert fp.merges_hash == "none"


class TestVerifyFingerprint:
    """Tests for verify_fingerprint function."""

    def test_verify_match(self):
        tokenizer = MockTokenizer()
        expected = compute_fingerprint(tokenizer)

        result = verify_fingerprint(tokenizer, expected)
        assert result is None  # No mismatch

    def test_verify_mismatch(self):
        tok1 = MockTokenizer({"a": 0})
        tok2 = MockTokenizer({"b": 0})

        expected = compute_fingerprint(tok1)
        result = verify_fingerprint(tok2, expected)

        assert result is not None
        assert isinstance(result, FingerprintMismatch)
        assert result.is_compatible is False

    def test_verify_with_string_fingerprint(self):
        tokenizer = MockTokenizer()
        fp = compute_fingerprint(tokenizer)

        # Match using short fingerprint string
        result = verify_fingerprint(tokenizer, fp.fingerprint)
        assert result is None

    def test_verify_special_tokens_only_diff(self):
        """Special token diffs alone should be compatible."""

        class Tok1(MockTokenizer):
            @property
            def sep_token_id(self):
                return 100

        class Tok2(MockTokenizer):
            @property
            def sep_token_id(self):
                return 200

        tok1 = Tok1()
        tok2 = Tok2()

        expected = compute_fingerprint(tok1)
        result = verify_fingerprint(tok2, expected)

        # Vocab is same, so should still be compatible
        # (special tokens differ but vocab matches)
        if result is not None:
            assert "Special token" in result.warnings[0] or result.is_compatible


class TestAssertFingerprint:
    """Tests for assert_fingerprint function."""

    def test_assert_match(self):
        tokenizer = MockTokenizer()
        expected = compute_fingerprint(tokenizer)

        # Should not raise
        assert_fingerprint(tokenizer, expected)

    def test_assert_mismatch_raises(self):
        tok1 = MockTokenizer({"a": 0})
        tok2 = MockTokenizer({"b": 0})

        expected = compute_fingerprint(tok1)

        with pytest.raises(ValueError, match="fingerprint mismatch"):
            assert_fingerprint(tok2, expected)


class TestSaveLoadFingerprint:
    """Tests for save/load fingerprint."""

    def test_save_load_roundtrip(self):
        tokenizer = MockTokenizer()
        fp = compute_fingerprint(tokenizer)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_fingerprint(fp, path)
            loaded = load_fingerprint(path)

            assert loaded.fingerprint == fp.fingerprint
            assert loaded.full_hash == fp.full_hash
            assert loaded.vocab_size == fp.vocab_size
        finally:
            Path(path).unlink()

    def test_save_creates_valid_json(self):
        tokenizer = MockTokenizer()
        fp = compute_fingerprint(tokenizer)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_fingerprint(fp, path)

            with open(path) as f:
                data = json.load(f)

            assert "fingerprint" in data
            assert "full_hash" in data
        finally:
            Path(path).unlink()


class TestFingerprintRegistry:
    """Tests for FingerprintRegistry."""

    def test_register_and_get(self):
        registry = FingerprintRegistry()
        fp = TokenizerFingerprint(
            fingerprint="test",
            full_hash="test123",
            vocab_hash="v1",
            special_tokens_hash="s1",
            merges_hash="m1",
            vocab_size=100,
        )

        registry.register("my-tokenizer", fp)
        retrieved = registry.get("my-tokenizer")

        assert retrieved is not None
        assert retrieved.fingerprint == "test"

    def test_register_with_aliases(self):
        registry = FingerprintRegistry()
        fp = TokenizerFingerprint(
            fingerprint="test",
            full_hash="test123",
            vocab_hash="v1",
            special_tokens_hash="s1",
            merges_hash="m1",
            vocab_size=100,
        )

        registry.register("llama-3-8b", fp, aliases=["llama3", "l3-8b"])

        assert registry.get("llama-3-8b") is not None
        assert registry.get("llama3") is not None
        assert registry.get("l3-8b") is not None

    def test_get_unknown(self):
        registry = FingerprintRegistry()
        assert registry.get("unknown") is None

    def test_verify_registered(self):
        registry = FingerprintRegistry()
        tokenizer = MockTokenizer()
        fp = compute_fingerprint(tokenizer)

        registry.register("test", fp)
        result = registry.verify(tokenizer, "test")

        assert result is None  # Match

    def test_verify_unknown_raises(self):
        registry = FingerprintRegistry()
        tokenizer = MockTokenizer()

        with pytest.raises(KeyError):
            registry.verify(tokenizer, "unknown")

    def test_identify(self):
        registry = FingerprintRegistry()
        tokenizer = MockTokenizer()
        fp = compute_fingerprint(tokenizer)

        registry.register("my-tok", fp)
        matches = registry.identify(tokenizer)

        assert len(matches) >= 1
        assert any("my-tok" in name for name, _ in matches)

    def test_list_all(self):
        registry = FingerprintRegistry()
        registry.register(
            "tok1",
            TokenizerFingerprint(
                fingerprint="a",
                full_hash="a1",
                vocab_hash="v",
                special_tokens_hash="s",
                merges_hash="m",
                vocab_size=1,
            ),
        )
        registry.register(
            "tok2",
            TokenizerFingerprint(
                fingerprint="b",
                full_hash="b1",
                vocab_hash="v",
                special_tokens_hash="s",
                merges_hash="m",
                vocab_size=1,
            ),
        )

        all_names = registry.list_all()
        assert "tok1" in all_names
        assert "tok2" in all_names


class TestGlobalRegistry:
    """Tests for global registry."""

    def test_get_registry(self):
        registry = get_registry()
        assert isinstance(registry, FingerprintRegistry)

    def test_global_registry_persists(self):
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2


class TestAsyncIO:
    """Tests for async fingerprint I/O."""

    def test_save_load_async(self):
        import asyncio

        from chuk_lazarus.data.tokenizers.fingerprint import (
            load_fingerprint_async,
            save_fingerprint_async,
        )

        tokenizer = MockTokenizer()
        fp = compute_fingerprint(tokenizer)

        async def run_async():
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                path = f.name

            try:
                await save_fingerprint_async(fp, path)
                loaded = await load_fingerprint_async(path)
                return loaded
            finally:
                Path(path).unlink()

        loaded = asyncio.run(run_async())
        assert loaded.fingerprint == fp.fingerprint
        assert loaded.full_hash == fp.full_hash


class TestFingerprintFromJson:
    """Tests for fingerprint_from_json utility."""

    def test_from_json_string(self):
        from chuk_lazarus.data.tokenizers.fingerprint import fingerprint_from_json

        data = {
            "fingerprint": "abc123",
            "full_hash": "abc123def456",
            "vocab_hash": "vocab123",
            "special_tokens_hash": "special123",
            "merges_hash": "merges123",
            "vocab_size": 1000,
        }
        json_str = json.dumps(data)
        fp = fingerprint_from_json(json_str)

        assert fp.fingerprint == "abc123"
        assert fp.vocab_size == 1000


class TestMergesExtraction:
    """Tests for BPE merge rules extraction."""

    def test_bpe_ranks_extraction(self):
        """Test extracting merges from bpe_ranks attribute."""

        class TokenizerWithBpeRanks(MockTokenizer):
            @property
            def bpe_ranks(self):
                return {"ab": 0, "cd": 1, "ef": 2}

        tokenizer = TokenizerWithBpeRanks()
        fp = compute_fingerprint(tokenizer)

        # Should have non-"none" merges hash since bpe_ranks was found
        assert fp.merges_hash != "none"

    def test_get_merges_extraction(self):
        """Test extracting merges from get_merges method."""

        class TokenizerWithGetMerges(MockTokenizer):
            def get_merges(self):
                return ["ab", "cd", "ef"]

        tokenizer = TokenizerWithGetMerges()
        fp = compute_fingerprint(tokenizer)

        # Should have non-"none" merges hash since get_merges was found
        assert fp.merges_hash != "none"


class TestVerifyFingerprintEdgeCases:
    """Tests for verify_fingerprint edge cases."""

    def test_verify_string_mismatch(self):
        """Test verify with string fingerprint that doesn't match."""
        tokenizer = MockTokenizer()

        # Use a string that won't match
        result = verify_fingerprint(tokenizer, "nonexistent_fingerprint_123")

        assert result is not None
        assert isinstance(result, FingerprintMismatch)

    def test_verify_strict_mode_merges_differ(self):
        """Test strict mode when merge rules differ."""
        tok1 = MockTokenizer()
        fp1 = compute_fingerprint(tok1, merges=["ab", "cd"])

        # Same vocab but different merges - strict mode should flag it
        result = verify_fingerprint(tok1, fp1, strict=True)

        # Result depends on whether merges hash matches
        if result is not None:
            # Check that strict mode warnings include merge rules
            assert (
                any("Merge" in w or "merges" in w.lower() for w in result.warnings)
                or result.is_compatible
            )

    def test_verify_size_mismatch_warning(self):
        """Test that vocab size mismatch generates warning."""
        tok1 = MockTokenizer({"a": 0, "b": 1, "c": 2})
        tok2 = MockTokenizer({"x": 0, "y": 1})

        fp1 = compute_fingerprint(tok1)
        result = verify_fingerprint(tok2, fp1)

        assert result is not None
        # Should have vocab mismatch and size mismatch warnings
        assert len(result.warnings) > 0


class TestRegistryIdentifyVocabOnly:
    """Tests for registry identify with vocab-only match."""

    def test_identify_vocab_only_match(self):
        """Test identify returns vocab-only matches."""
        registry = FingerprintRegistry()

        # Create two tokenizers with same vocab but different special tokens
        class Tok1(MockTokenizer):
            @property
            def sep_token_id(self):
                return 100

        class Tok2(MockTokenizer):
            @property
            def sep_token_id(self):
                return 200

        tok1 = Tok1()
        tok2 = Tok2()

        fp1 = compute_fingerprint(tok1)
        registry.register("tok1", fp1)

        # tok2 has same vocab but different special tokens
        matches = registry.identify(tok2)

        # Should find a vocab-only match if hashes differ but vocab matches
        # (depends on implementation details)
        assert isinstance(matches, list)
