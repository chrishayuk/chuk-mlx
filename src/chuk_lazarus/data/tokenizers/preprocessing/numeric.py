"""
Numeric-aware tokenization utilities.

Detects and normalizes numbers to prevent token explosion:
- Integers, floats, scientific notation
- Reversible encoding/decoding
- Configurable normalization strategies
"""

import re
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field


class NumericFormat(str, Enum):
    """Format of detected number."""

    INTEGER = "integer"
    FLOAT = "float"
    SCIENTIFIC = "scientific"
    HEXADECIMAL = "hex"
    BINARY = "binary"
    PERCENTAGE = "percentage"
    FRACTION = "fraction"


class NumericConfig(BaseModel):
    """Configuration for numeric detection and encoding."""

    # Detection settings
    detect_integers: bool = Field(default=True, description="Detect integer numbers")
    detect_floats: bool = Field(default=True, description="Detect floating point")
    detect_scientific: bool = Field(default=True, description="Detect scientific notation")
    detect_hex: bool = Field(default=True, description="Detect hexadecimal (0x...)")
    detect_binary: bool = Field(default=False, description="Detect binary (0b...)")
    detect_percentages: bool = Field(default=True, description="Detect percentages")
    detect_fractions: bool = Field(default=True, description="Detect fractions (1/2)")

    # Encoding settings
    use_placeholder: bool = Field(default=True, description="Replace numbers with placeholders")
    placeholder_template: str = Field(
        default="<NUM_{idx}>", description="Template for placeholder tokens"
    )
    preserve_sign: bool = Field(default=True, description="Keep sign separate from number")
    max_integer_digits: int = Field(
        default=12, description="Max digits before scientific conversion"
    )
    decimal_precision: int = Field(default=6, description="Max decimal places to preserve")

    # Bucket encoding (alternative to placeholders)
    use_buckets: bool = Field(
        default=False, description="Use magnitude buckets instead of placeholders"
    )
    bucket_base: int = Field(default=10, description="Base for magnitude buckets")


class NumericSpan(BaseModel):
    """A detected numeric span in text."""

    start: int = Field(ge=0, description="Start position in text")
    end: int = Field(ge=0, description="End position in text")
    original: str = Field(description="Original text of the number")
    format: NumericFormat = Field(description="Detected format")
    value: float | None = Field(default=None, description="Parsed numeric value")
    sign: str = Field(default="", description="Sign if present (+/-)")


class NumericEncoding(BaseModel):
    """Result of encoding numbers in text."""

    encoded_text: str = Field(description="Text with numbers replaced")
    spans: list[NumericSpan] = Field(default_factory=list, description="Detected numeric spans")
    mapping: dict[str, str] = Field(
        default_factory=dict, description="Placeholder -> original mapping"
    )


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...


# Regex patterns for number detection
PATTERNS = {
    NumericFormat.SCIENTIFIC: re.compile(r"[+-]?\d+\.?\d*[eE][+-]?\d+", re.IGNORECASE),
    NumericFormat.HEXADECIMAL: re.compile(r"0[xX][0-9a-fA-F]+"),
    NumericFormat.BINARY: re.compile(r"0[bB][01]+"),
    NumericFormat.PERCENTAGE: re.compile(r"[+-]?\d+\.?\d*%"),
    NumericFormat.FRACTION: re.compile(r"\d+/\d+"),
    NumericFormat.FLOAT: re.compile(r"[+-]?\d+\.\d+"),
    NumericFormat.INTEGER: re.compile(r"[+-]?\d+"),
}


def _parse_value(text: str, fmt: NumericFormat) -> float | None:
    """Parse numeric value from text."""
    try:
        if fmt == NumericFormat.HEXADECIMAL:
            return float(int(text, 16))
        elif fmt == NumericFormat.BINARY:
            return float(int(text, 2))
        elif fmt == NumericFormat.PERCENTAGE:
            return float(text.rstrip("%")) / 100
        elif fmt == NumericFormat.FRACTION:
            num, denom = text.split("/")
            return float(num) / float(denom) if float(denom) != 0 else None
        else:
            return float(text)
    except (ValueError, ZeroDivisionError):
        return None


def detect_numbers(
    text: str,
    config: NumericConfig | None = None,
) -> list[NumericSpan]:
    """
    Detect all numeric spans in text.

    Args:
        text: Input text to scan
        config: Detection configuration

    Returns:
        List of NumericSpan objects, sorted by position
    """
    if config is None:
        config = NumericConfig()

    spans: list[NumericSpan] = []
    used_positions: set[int] = set()

    # Order matters - check more specific patterns first
    format_order = [
        (NumericFormat.SCIENTIFIC, config.detect_scientific),
        (NumericFormat.HEXADECIMAL, config.detect_hex),
        (NumericFormat.BINARY, config.detect_binary),
        (NumericFormat.PERCENTAGE, config.detect_percentages),
        (NumericFormat.FRACTION, config.detect_fractions),
        (NumericFormat.FLOAT, config.detect_floats),
        (NumericFormat.INTEGER, config.detect_integers),
    ]

    for fmt, enabled in format_order:
        if not enabled:
            continue

        pattern = PATTERNS[fmt]
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()

            # Skip if this position is already covered
            if any(pos in used_positions for pos in range(start, end)):
                continue

            original = match.group()
            sign = ""
            if original and original[0] in "+-":
                sign = original[0]

            span = NumericSpan(
                start=start,
                end=end,
                original=original,
                format=fmt,
                value=_parse_value(original, fmt),
                sign=sign,
            )
            spans.append(span)

            # Mark positions as used
            for pos in range(start, end):
                used_positions.add(pos)

    # Sort by position
    spans.sort(key=lambda s: s.start)
    return spans


def encode_number(
    value: float,
    fmt: NumericFormat = NumericFormat.FLOAT,
    config: NumericConfig | None = None,
) -> str:
    """
    Encode a single number to a normalized string representation.

    Args:
        value: Numeric value to encode
        fmt: Original format hint
        config: Encoding configuration

    Returns:
        Encoded string representation
    """
    if config is None:
        config = NumericConfig()

    if config.use_buckets:
        # Magnitude bucket encoding
        if value == 0:
            return "<NUM_ZERO>"
        magnitude = int(abs(value)).bit_length() if value != 0 else 0
        bucket = magnitude // config.bucket_base
        sign = "NEG_" if value < 0 else ""
        return f"<NUM_{sign}MAG_{bucket}>"

    # Standard encoding - preserve precision
    if fmt == NumericFormat.INTEGER and value == int(value):
        int_val = int(value)
        if abs(int_val) >= 10**config.max_integer_digits:
            # Convert large integers to scientific
            return f"{value:.{config.decimal_precision}e}"
        return str(int_val)
    elif fmt == NumericFormat.SCIENTIFIC:
        return f"{value:.{config.decimal_precision}e}"
    else:
        # Float with controlled precision
        formatted = f"{value:.{config.decimal_precision}f}"
        # Strip trailing zeros but keep at least one decimal
        formatted = formatted.rstrip("0").rstrip(".")
        if "." not in formatted:
            formatted += ".0"
        return formatted


def decode_number(encoded: str, config: NumericConfig | None = None) -> str:
    """
    Decode a normalized number back to string.

    For placeholder encoding, this is handled by restore_numbers.
    For bucket encoding, returns a representative value.

    Args:
        encoded: Encoded number string
        config: Decoding configuration

    Returns:
        Decoded string representation
    """
    if config is None:
        config = NumericConfig()

    # Handle bucket tokens
    if encoded.startswith("<NUM_") and encoded.endswith(">"):
        inner = encoded[5:-1]
        if inner == "ZERO":
            return "0"
        if inner.startswith("NEG_MAG_"):
            mag = int(inner[8:])
            return f"-1e{mag * config.bucket_base}"
        if inner.startswith("MAG_"):
            mag = int(inner[4:])
            return f"1e{mag * config.bucket_base}"

    return encoded


def normalize_numbers(
    text: str,
    config: NumericConfig | None = None,
) -> NumericEncoding:
    """
    Normalize all numbers in text with placeholders or bucket tokens.

    Args:
        text: Input text
        config: Normalization configuration

    Returns:
        NumericEncoding with encoded text and mapping for restoration
    """
    if config is None:
        config = NumericConfig()

    spans = detect_numbers(text, config)

    if not spans:
        return NumericEncoding(encoded_text=text, spans=[], mapping={})

    mapping: dict[str, str] = {}
    parts: list[str] = []
    last_end = 0

    for idx, span in enumerate(spans):
        # Add text before this number
        parts.append(text[last_end : span.start])

        if config.use_placeholder:
            # Placeholder encoding
            placeholder = config.placeholder_template.format(idx=idx)
            parts.append(placeholder)
            mapping[placeholder] = span.original
        elif config.use_buckets and span.value is not None:
            # Bucket encoding
            encoded = encode_number(span.value, span.format, config)
            parts.append(encoded)
            mapping[encoded] = span.original
        else:
            # Just normalize the representation
            if span.value is not None:
                normalized = encode_number(span.value, span.format, config)
                parts.append(normalized)
                mapping[normalized] = span.original
            else:
                parts.append(span.original)

        last_end = span.end

    # Add remaining text
    parts.append(text[last_end:])

    return NumericEncoding(
        encoded_text="".join(parts),
        spans=spans,
        mapping=mapping,
    )


def restore_numbers(
    encoded_text: str,
    mapping: dict[str, str],
) -> str:
    """
    Restore original numbers from encoded text.

    Args:
        encoded_text: Text with number placeholders/encodings
        mapping: Placeholder -> original mapping from normalize_numbers

    Returns:
        Text with original numbers restored
    """
    result = encoded_text
    for placeholder, original in mapping.items():
        result = result.replace(placeholder, original)
    return result


def get_numeric_token_savings(
    text: str,
    tokenizer: TokenizerProtocol,
    config: NumericConfig | None = None,
) -> dict:
    """
    Calculate token savings from numeric normalization.

    Args:
        text: Input text
        tokenizer: Tokenizer to measure tokens
        config: Normalization configuration

    Returns:
        Dict with original_tokens, normalized_tokens, savings
    """
    if config is None:
        config = NumericConfig()

    original_tokens = len(tokenizer.encode(text, add_special_tokens=False))

    encoding = normalize_numbers(text, config)
    normalized_tokens = len(tokenizer.encode(encoding.encoded_text, add_special_tokens=False))

    return {
        "original_tokens": original_tokens,
        "normalized_tokens": normalized_tokens,
        "savings": original_tokens - normalized_tokens,
        "savings_percent": (
            (original_tokens - normalized_tokens) / original_tokens * 100
            if original_tokens > 0
            else 0
        ),
        "numbers_detected": len(encoding.spans),
    }
