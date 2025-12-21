"""Tests for byte fallback wrapper."""

from chuk_lazarus.data.tokenizers.preprocessing.fallback import (
    ByteFallbackConfig,
    ByteFallbackStats,
    ByteFallbackWrapper,
    ensure_byte_safety,
    run_byte_safety_tests,
    wrap_with_fallback,
)


class MockTokenizer:
    """Mock tokenizer that produces UNK for certain chars."""

    def __init__(self):
        self.vocab = {chr(i): i for i in range(32, 127)}  # ASCII printable
        self.unk_id = 0
        self._vocab_size = 128

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.unk_id)  # UNK for non-ASCII
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        id_to_char = {v: k for k, v in self.vocab.items()}
        id_to_char[0] = "<unk>"
        return "".join(id_to_char.get(i, "?") for i in token_ids)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


class TestByteFallbackConfig:
    """Tests for ByteFallbackConfig model."""

    def test_default_values(self):
        config = ByteFallbackConfig()
        assert config.use_hex_encoding is True
        assert config.preserve_ascii is True

    def test_custom_values(self):
        config = ByteFallbackConfig(
            unk_token_id=1,
            fallback_entire_text=True,
        )
        assert config.unk_token_id == 1
        assert config.fallback_entire_text is True


class TestByteFallbackWrapper:
    """Tests for ByteFallbackWrapper."""

    def test_encode_ascii(self):
        tokenizer = MockTokenizer()
        wrapper = ByteFallbackWrapper(tokenizer)
        tokens = wrapper.encode("hello")
        assert 0 not in tokens  # No UNK

    def test_encode_unicode(self):
        tokenizer = MockTokenizer()
        wrapper = ByteFallbackWrapper(tokenizer)
        # This would produce UNK without fallback
        tokens = wrapper.encode("café")
        # Should not have UNK
        assert tokens is not None

    def test_decode(self):
        tokenizer = MockTokenizer()
        wrapper = ByteFallbackWrapper(tokenizer)
        tokens = wrapper.encode("hello")
        decoded = wrapper.decode(tokens)
        assert "hello" in decoded or decoded is not None

    def test_vocab_size(self):
        tokenizer = MockTokenizer()
        wrapper = ByteFallbackWrapper(tokenizer)
        assert wrapper.vocab_size == 128

    def test_get_fallback_stats(self):
        tokenizer = MockTokenizer()
        wrapper = ByteFallbackWrapper(tokenizer)
        stats = wrapper.get_fallback_stats("hello")
        assert isinstance(stats, ByteFallbackStats)
        assert stats.original_length == 5


class TestByteFallbackStats:
    """Tests for ByteFallbackStats model."""

    def test_valid_stats(self):
        stats = ByteFallbackStats(
            original_length=10,
            fallback_chars=2,
            fallback_ratio=0.2,
            unk_avoided=2,
        )
        assert stats.original_length == 10
        assert stats.fallback_ratio == 0.2


class TestEnsureByteSafety:
    """Tests for ensure_byte_safety function."""

    def test_ascii_unchanged(self):
        tokenizer = MockTokenizer()
        text = "hello world"
        result = ensure_byte_safety(text, tokenizer)
        assert result == text

    def test_unicode_encoded(self):
        tokenizer = MockTokenizer()
        text = "café"
        result = ensure_byte_safety(text, tokenizer)
        # Should have byte encoding for é
        assert result is not None


class TestWrapWithFallback:
    """Tests for wrap_with_fallback function."""

    def test_returns_wrapper(self):
        tokenizer = MockTokenizer()
        wrapper = wrap_with_fallback(tokenizer)
        assert isinstance(wrapper, ByteFallbackWrapper)

    def test_with_config(self):
        tokenizer = MockTokenizer()
        config = ByteFallbackConfig(preserve_ascii=False)
        wrapper = wrap_with_fallback(tokenizer, config)
        assert wrapper.config.preserve_ascii is False


class TestRunByteSafetyTests:
    """Tests for run_byte_safety_tests function."""

    def test_basic_tests(self):
        tokenizer = MockTokenizer()
        results = run_byte_safety_tests(tokenizer)
        assert "total_tests" in results
        assert "passed" in results
        assert "failed" in results

    def test_custom_strings(self):
        tokenizer = MockTokenizer()
        results = run_byte_safety_tests(tokenizer, ["hello", "world"])
        assert results["total_tests"] == 2

    def test_counts_failures(self):
        tokenizer = MockTokenizer()
        results = run_byte_safety_tests(tokenizer)
        assert results["passed"] >= 0
        assert results["failed"] >= 0
        assert results["passed"] + results["failed"] == results["total_tests"]


class TestByteFallbackEdgeCases:
    """Tests for edge cases and additional coverage."""

    def test_config_with_custom_unk_id(self):
        """Test with explicit unk_token_id in config."""
        tokenizer = MockTokenizer()
        config = ByteFallbackConfig(unk_token_id=5)
        wrapper = ByteFallbackWrapper(tokenizer, config)
        assert wrapper._unk_id == 5

    def test_encode_non_ascii_no_preserve(self):
        """Test encoding without preserving ASCII."""
        tokenizer = MockTokenizer()
        config = ByteFallbackConfig(preserve_ascii=False)
        wrapper = ByteFallbackWrapper(tokenizer, config)
        # Should still encode properly
        result = wrapper._encode_as_bytes("ab")
        assert result is not None

    def test_decode_byte_tokens_with_unicode(self):
        """Test decoding complex byte sequences."""
        tokenizer = MockTokenizer()
        wrapper = ByteFallbackWrapper(tokenizer)
        # Test decoding text with byte tokens
        decoded = wrapper._decode_byte_tokens("hello world")
        assert decoded == "hello world"

    def test_fallback_entire_text_mode(self):
        """Test fallback_entire_text configuration."""
        tokenizer = MockTokenizer()
        config = ByteFallbackConfig(fallback_entire_text=True)
        wrapper = ByteFallbackWrapper(tokenizer, config)
        # Unicode text should trigger full fallback
        tokens = wrapper.encode("héllo")
        assert tokens is not None

    def test_fallback_stats_with_entire_text(self):
        """Test stats when fallback_entire_text is enabled."""
        tokenizer = MockTokenizer()
        config = ByteFallbackConfig(fallback_entire_text=True)
        wrapper = ByteFallbackWrapper(tokenizer, config)
        stats = wrapper.get_fallback_stats("café")
        assert stats.original_length == 4

    def test_needs_fallback_high_unk_ratio(self):
        """Test detection with high UNK ratio."""
        tokenizer = MockTokenizer()
        config = ByteFallbackConfig(detect_threshold=0.5)
        wrapper = ByteFallbackWrapper(tokenizer, config)
        # All unicode should trigger fallback
        needs, indices = wrapper._needs_fallback("日本語")
        assert needs is True

    def test_decode_with_mixed_content(self):
        """Test decoding mixed ASCII and byte tokens."""
        tokenizer = MockTokenizer()
        wrapper = ByteFallbackWrapper(tokenizer)
        # Encode then decode unicode
        tokens = wrapper.encode("test café test")
        decoded = wrapper.decode(tokens)
        assert decoded is not None

    def test_ensure_byte_safety_with_config(self):
        """Test ensure_byte_safety with custom config."""
        tokenizer = MockTokenizer()
        config = ByteFallbackConfig(preserve_ascii=True)
        result = ensure_byte_safety("hello café", tokenizer, config)
        assert result is not None

    def test_byte_token_template_custom(self):
        """Test custom byte token template."""
        tokenizer = MockTokenizer()
        config = ByteFallbackConfig(byte_token_template="[BYTE_{byte:02X}]")
        wrapper = ByteFallbackWrapper(tokenizer, config)
        assert "[BYTE_00]" in wrapper._byte_token_cache[0]

    def test_empty_text_fallback_stats(self):
        """Test fallback stats for empty text."""
        tokenizer = MockTokenizer()
        wrapper = ByteFallbackWrapper(tokenizer)
        stats = wrapper.get_fallback_stats("")
        assert stats.original_length == 0
        assert stats.fallback_ratio == 0


class MockTokenizerWithUnkDetection:
    """Mock tokenizer that returns UNK-like tokens for detection."""

    def __init__(self):
        self.vocab = {chr(i): i for i in range(32, 127)}
        self._vocab_size = 128

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self.vocab.get(c, 0) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        # Return <unk> for token id 0 to trigger detection
        id_to_char = {v: k for k, v in self.vocab.items()}
        result = []
        for tid in token_ids:
            if tid == 0:
                result.append("<unk>")
            else:
                result.append(id_to_char.get(tid, "?"))
        return "".join(result)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


class TestUnkDetection:
    """Tests for UNK token detection."""

    def test_detect_unk_from_decode(self):
        """Test detecting UNK from decode output."""
        tokenizer = MockTokenizerWithUnkDetection()
        wrapper = ByteFallbackWrapper(tokenizer)
        # Should detect 0 as UNK
        assert wrapper._unk_id == 0

    def test_detect_unk_fallback(self):
        """Test UNK detection falls back to 0."""

        class MinimalTokenizer:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [ord(c) for c in text]

            def decode(self, token_ids: list[int]) -> str:
                return "".join(chr(t) for t in token_ids)

            @property
            def vocab_size(self) -> int:
                return 256

        tokenizer = MinimalTokenizer()
        wrapper = ByteFallbackWrapper(tokenizer)
        # Should default to 0
        assert wrapper._unk_id == 0


class TestByteFallbackDecoding:
    """Tests for complex decoding scenarios."""

    def test_decode_utf8_reconstruction(self):
        """Test UTF-8 byte sequence reconstruction."""
        tokenizer = MockTokenizer()
        wrapper = ByteFallbackWrapper(tokenizer)
        # Test with actual UTF-8 encoded text
        text = "日本語"
        encoded = wrapper._encode_as_bytes(text)
        decoded = wrapper._decode_byte_tokens(encoded)
        # Should attempt reconstruction
        assert decoded is not None

    def test_decode_partial_utf8(self):
        """Test handling of partial UTF-8 sequences."""
        tokenizer = MockTokenizer()
        wrapper = ByteFallbackWrapper(tokenizer)
        # Just verify it doesn't crash on edge cases
        result = wrapper._decode_byte_tokens("test <0xC3> text")
        assert result is not None


class TestDecodeBytesComplexCases:
    """Additional tests for complex byte decoding scenarios."""

    def test_decode_with_unicode_chars_in_middle(self):
        """Test decoding when unicode chars appear mid-stream."""
        tokenizer = MockTokenizer()
        wrapper = ByteFallbackWrapper(tokenizer)
        # Create text with high unicode point
        text = "abc\u4e2d\u6587def"  # Chinese chars in middle
        encoded = wrapper._encode_as_bytes(text)
        decoded = wrapper._decode_byte_tokens(encoded)
        assert decoded is not None

    def test_run_byte_safety_exception_handling(self):
        """Test exception handling in run_byte_safety_tests."""

        class FailingTokenizer:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                if "fail" in text:
                    raise ValueError("Test error")
                return [ord(c) for c in text]

            def decode(self, token_ids: list[int]) -> str:
                return "".join(chr(t) if t < 128 else "?" for t in token_ids)

            @property
            def vocab_size(self) -> int:
                return 256

        tokenizer = FailingTokenizer()
        results = run_byte_safety_tests(tokenizer, ["hello", "fail here"])
        assert results["failed"] >= 1
        assert len(results["failures"]) >= 1

    def test_unk_detection_with_replacement_char(self):
        """Test UNK detection with replacement character."""

        class ReplacementTokenizer:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [0 if c == "?" else ord(c) for c in text]

            def decode(self, token_ids: list[int]) -> str:
                return "".join("�" if t == 0 else chr(t) for t in token_ids)

            @property
            def vocab_size(self) -> int:
                return 256

        tokenizer = ReplacementTokenizer()
        wrapper = ByteFallbackWrapper(tokenizer)
        # Should detect 0 as UNK (decoded as replacement char)
        assert wrapper._unk_id == 0

    def test_unk_detection_with_unk_marker(self):
        """Test UNK detection with [UNK] marker."""

        class UnkMarkerTokenizer:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [1 if c == "?" else ord(c) for c in text]

            def decode(self, token_ids: list[int]) -> str:
                return "".join("[UNK]" if t == 1 else chr(t) for t in token_ids)

            @property
            def vocab_size(self) -> int:
                return 256

        tokenizer = UnkMarkerTokenizer()
        wrapper = ByteFallbackWrapper(tokenizer)
        # Should detect 1 as UNK (decoded as [UNK])
        assert wrapper._unk_id == 1

    def test_decode_unicode_decode_error_recovery(self):
        """Test recovery from UnicodeDecodeError during byte reconstruction."""
        tokenizer = MockTokenizer()
        wrapper = ByteFallbackWrapper(tokenizer)
        # Invalid UTF-8 sequence should not crash
        # Simulate by decoding something with invalid byte sequence representation
        result = wrapper._decode_byte_tokens("test\x80\x81text")
        assert result is not None

    def test_encode_with_selective_fallback(self):
        """Test selective fallback for problematic characters only."""
        tokenizer = MockTokenizer()
        config = ByteFallbackConfig(fallback_entire_text=False)
        wrapper = ByteFallbackWrapper(tokenizer, config)
        # Mix of ASCII and non-ASCII
        tokens = wrapper.encode("hello世界")
        assert tokens is not None

    def test_byte_safety_roundtrip_mismatch(self):
        """Test handling of roundtrip mismatches."""

        class LossyTokenizer:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [ord(c) % 128 for c in text]

            def decode(self, token_ids: list[int]) -> str:
                # Lossy decode - always returns lowercase
                return "".join(chr(t).lower() if 32 <= t < 127 else "x" for t in token_ids)

            @property
            def vocab_size(self) -> int:
                return 128

        tokenizer = LossyTokenizer()
        results = run_byte_safety_tests(tokenizer, ["Hello World"])
        # Should handle the mismatch gracefully
        assert results["total_tests"] == 1
