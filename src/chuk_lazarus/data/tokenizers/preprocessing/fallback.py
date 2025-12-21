"""
Byte fallback wrapper for tokenizers.

Ensures any byte sequence can be tokenized without [UNK] explosion:
- Detects characters that would produce UNK tokens
- Encodes them as byte tokens
- Restores on decode
"""

from typing import Protocol

from pydantic import BaseModel, Field


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...

    @property
    def vocab_size(self) -> int: ...


class ByteFallbackConfig(BaseModel):
    """Configuration for byte fallback."""

    # Detection
    unk_token_id: int | None = Field(default=None, description="UNK token ID (auto-detect if None)")
    detect_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Max acceptable UNK ratio before fallback",
    )

    # Encoding
    byte_token_template: str = Field(
        default="<0x{byte:02X}>", description="Template for byte tokens"
    )
    use_hex_encoding: bool = Field(default=True, description="Use hex encoding for bytes")

    # Fallback behavior
    fallback_entire_text: bool = Field(
        default=False,
        description="Fallback entire text if any UNK detected",
    )
    preserve_ascii: bool = Field(
        default=True, description="Keep ASCII chars as-is even in fallback mode"
    )


class ByteFallbackStats(BaseModel):
    """Statistics from byte fallback encoding."""

    original_length: int = Field(description="Original text length")
    fallback_chars: int = Field(description="Characters encoded as bytes")
    fallback_ratio: float = Field(description="Ratio of fallback chars")
    unk_avoided: int = Field(description="UNK tokens avoided")


class ByteFallbackWrapper:
    """Wrapper that adds byte fallback to any tokenizer."""

    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        config: ByteFallbackConfig | None = None,
    ):
        self.tokenizer = tokenizer
        self.config = config or ByteFallbackConfig()
        self._unk_id = self._detect_unk_id()
        self._byte_token_cache: dict[int, str] = {}
        self._reverse_cache: dict[str, int] = {}
        self._build_byte_cache()

    def _detect_unk_id(self) -> int:
        """Detect the UNK token ID."""
        if self.config.unk_token_id is not None:
            return self.config.unk_token_id

        # Try common UNK token IDs
        for test_id in [0, 1, 2, 3]:
            try:
                decoded = self.tokenizer.decode([test_id])
                if "unk" in decoded.lower() or decoded in ["<unk>", "[UNK]", "�"]:
                    return test_id
            except Exception:
                pass

        # Default to 0
        return 0

    def _build_byte_cache(self) -> None:
        """Build cache of byte tokens."""
        for byte_val in range(256):
            token = self.config.byte_token_template.format(byte=byte_val)
            self._byte_token_cache[byte_val] = token
            self._reverse_cache[token] = byte_val

    def _encode_as_bytes(self, text: str) -> str:
        """Encode text as byte tokens."""
        result_parts: list[str] = []
        text_bytes = text.encode("utf-8")

        for byte_val in text_bytes:
            if self.config.preserve_ascii and 32 <= byte_val < 127:
                # Keep printable ASCII as-is
                result_parts.append(chr(byte_val))
            else:
                result_parts.append(self._byte_token_cache[byte_val])

        return "".join(result_parts)

    def _decode_byte_tokens(self, text: str) -> str:
        """Decode byte tokens back to text."""
        result = text

        # Find and replace byte tokens
        for token, byte_val in self._reverse_cache.items():
            if token in result:
                result = result.replace(token, chr(byte_val))

        # Handle UTF-8 reconstruction
        try:
            # Try to decode as UTF-8 encoded bytes
            byte_list = []
            i = 0
            while i < len(result):
                if ord(result[i]) < 256:
                    byte_list.append(ord(result[i]))
                    i += 1
                else:
                    # Non-byte character, keep as-is
                    if byte_list:
                        try:
                            decoded = bytes(byte_list).decode("utf-8")
                            result = result[: i - len(byte_list)] + decoded + result[i:]
                            i = i - len(byte_list) + len(decoded)
                        except UnicodeDecodeError:
                            pass
                        byte_list = []
                    i += 1

            # Handle remaining bytes
            if byte_list:
                try:
                    decoded = bytes(byte_list).decode("utf-8")
                    result = result[: -len(byte_list)] + decoded
                except UnicodeDecodeError:
                    pass

        except Exception:
            pass

        return result

    def _needs_fallback(self, text: str) -> tuple[bool, list[int]]:
        """
        Check if text needs byte fallback.

        Returns:
            Tuple of (needs_fallback, problematic_char_indices)
        """
        # Encode normally
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # Check for UNK tokens
        unk_count = sum(1 for tid in token_ids if tid == self._unk_id)

        if unk_count == 0:
            return False, []

        unk_ratio = unk_count / len(token_ids) if token_ids else 0
        if unk_ratio > self.config.detect_threshold:
            return True, []

        # Find which characters cause UNK
        problematic: list[int] = []
        for i, char in enumerate(text):
            char_tokens = self.tokenizer.encode(char, add_special_tokens=False)
            if self._unk_id in char_tokens:
                problematic.append(i)

        return len(problematic) > 0, problematic

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """
        Encode text with byte fallback for problematic characters.

        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens

        Returns:
            Token IDs with no UNK tokens
        """
        needs_fallback, problematic = self._needs_fallback(text)

        if not needs_fallback:
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

        if self.config.fallback_entire_text:
            # Encode entire text as bytes
            byte_text = self._encode_as_bytes(text)
            return self.tokenizer.encode(byte_text, add_special_tokens=add_special_tokens)

        # Selective fallback - only encode problematic characters
        result_text = list(text)
        for idx in problematic:
            char = text[idx]
            byte_encoded = self._encode_as_bytes(char)
            result_text[idx] = byte_encoded

        return self.tokenizer.encode("".join(result_text), add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode tokens, restoring byte-encoded characters.

        Args:
            token_ids: Token IDs to decode

        Returns:
            Decoded text with byte tokens restored
        """
        decoded = self.tokenizer.decode(token_ids)
        return self._decode_byte_tokens(decoded)

    def get_fallback_stats(self, text: str) -> ByteFallbackStats:
        """
        Get statistics about byte fallback for a text.

        Args:
            text: Input text

        Returns:
            ByteFallbackStats
        """
        needs_fallback, problematic = self._needs_fallback(text)

        fallback_chars = (
            len(problematic)
            if not self.config.fallback_entire_text
            else (len(text) if needs_fallback else 0)
        )

        return ByteFallbackStats(
            original_length=len(text),
            fallback_chars=fallback_chars,
            fallback_ratio=fallback_chars / len(text) if text else 0,
            unk_avoided=fallback_chars,
        )

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.vocab_size


def ensure_byte_safety(
    text: str,
    tokenizer: TokenizerProtocol,
    config: ByteFallbackConfig | None = None,
) -> str:
    """
    Ensure text can be safely tokenized without UNK.

    Args:
        text: Input text
        tokenizer: Tokenizer to check against
        config: Fallback configuration

    Returns:
        Text with problematic chars encoded as bytes
    """
    wrapper = ByteFallbackWrapper(tokenizer, config)

    # Check if fallback needed
    needs_fallback, problematic = wrapper._needs_fallback(text)

    if not needs_fallback:
        return text

    # Apply byte encoding to problematic chars
    result = list(text)
    for idx in problematic:
        result[idx] = wrapper._encode_as_bytes(text[idx])

    return "".join(result)


def wrap_with_fallback(
    tokenizer: TokenizerProtocol,
    config: ByteFallbackConfig | None = None,
) -> ByteFallbackWrapper:
    """
    Wrap a tokenizer with byte fallback.

    Args:
        tokenizer: Tokenizer to wrap
        config: Fallback configuration

    Returns:
        ByteFallbackWrapper
    """
    return ByteFallbackWrapper(tokenizer, config)


def run_byte_safety_tests(
    tokenizer: TokenizerProtocol,
    test_strings: list[str] | None = None,
) -> dict:
    """
    Test tokenizer byte safety with edge cases.

    Args:
        tokenizer: Tokenizer to test
        test_strings: Custom test strings (uses defaults if None)

    Returns:
        Dict with test results
    """
    if test_strings is None:
        test_strings = [
            # Basic
            "Hello, world!",
            # Unicode
            "Héllo wörld",
            "你好世界",
            "مرحبا بالعالم",
            "🎉🎊🎁",
            # Edge cases
            "\x00\x01\x02",  # Null bytes
            "foo\ufffdbar",  # Replacement char
            "a\u200bb",  # Zero-width space
            # Mixed
            "Price: €100 or ¥1000",
            "Temperature: 25°C",
        ]

    results = {
        "total_tests": len(test_strings),
        "passed": 0,
        "failed": 0,
        "failures": [],
    }

    wrapper = ByteFallbackWrapper(tokenizer)

    for test_str in test_strings:
        try:
            # Encode and decode
            token_ids = wrapper.encode(test_str, add_special_tokens=False)
            decoded = wrapper.decode(token_ids)

            # Check roundtrip
            if decoded == test_str or test_str in decoded:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["failures"].append(
                    {
                        "input": test_str[:50],
                        "decoded": decoded[:50],
                        "error": "roundtrip mismatch",
                    }
                )
        except Exception as e:
            results["failed"] += 1
            results["failures"].append(
                {
                    "input": test_str[:50],
                    "error": str(e),
                }
            )

    return results
