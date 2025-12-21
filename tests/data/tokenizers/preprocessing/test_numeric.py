"""Tests for numeric preprocessing."""

from chuk_lazarus.data.tokenizers.preprocessing.numeric import (
    NumericConfig,
    NumericFormat,
    NumericSpan,
    detect_numbers,
    encode_number,
    normalize_numbers,
    restore_numbers,
)


class TestNumericConfig:
    """Tests for NumericConfig model."""

    def test_default_values(self):
        config = NumericConfig()
        assert config.detect_integers is True
        assert config.detect_floats is True
        assert config.detect_scientific is True
        assert config.use_placeholder is True

    def test_custom_values(self):
        config = NumericConfig(
            detect_integers=False,
            use_buckets=True,
        )
        assert config.detect_integers is False
        assert config.use_buckets is True


class TestDetectNumbers:
    """Tests for detect_numbers function."""

    def test_detect_integer(self):
        spans = detect_numbers("The answer is 42")
        assert len(spans) == 1
        assert spans[0].original == "42"
        assert spans[0].format == NumericFormat.INTEGER
        assert spans[0].value == 42.0

    def test_detect_float(self):
        spans = detect_numbers("Pi is 3.14159")
        assert len(spans) == 1
        assert spans[0].original == "3.14159"
        assert spans[0].format == NumericFormat.FLOAT

    def test_detect_scientific(self):
        spans = detect_numbers("Avogadro: 6.022e23")
        assert len(spans) == 1
        assert spans[0].format == NumericFormat.SCIENTIFIC
        assert abs(spans[0].value - 6.022e23) < 1e20

    def test_detect_hex(self):
        spans = detect_numbers("Color: 0xFF00FF")
        assert len(spans) == 1
        assert spans[0].format == NumericFormat.HEXADECIMAL

    def test_detect_percentage(self):
        spans = detect_numbers("Success rate: 95%")
        assert len(spans) == 1
        assert spans[0].format == NumericFormat.PERCENTAGE
        assert spans[0].value == 0.95

    def test_detect_fraction(self):
        spans = detect_numbers("Half is 1/2")
        assert len(spans) == 1
        assert spans[0].format == NumericFormat.FRACTION
        assert spans[0].value == 0.5

    def test_detect_multiple(self):
        spans = detect_numbers("From 10 to 20.5")
        assert len(spans) == 2

    def test_detect_negative(self):
        spans = detect_numbers("Temperature: -40")
        assert len(spans) == 1
        assert spans[0].sign == "-"
        assert spans[0].value == -40.0

    def test_empty_text(self):
        spans = detect_numbers("")
        assert len(spans) == 0

    def test_no_numbers(self):
        spans = detect_numbers("Hello world")
        assert len(spans) == 0

    def test_config_disables_detection(self):
        config = NumericConfig(detect_integers=False)
        spans = detect_numbers("Value: 42", config)
        assert len(spans) == 0


class TestEncodeNumber:
    """Tests for encode_number function."""

    def test_encode_integer(self):
        result = encode_number(42, NumericFormat.INTEGER)
        assert result == "42"

    def test_encode_float(self):
        result = encode_number(3.14, NumericFormat.FLOAT)
        assert "3.14" in result

    def test_encode_large_integer(self):
        config = NumericConfig(max_integer_digits=6)
        result = encode_number(1234567890, NumericFormat.INTEGER, config)
        assert "e" in result  # Should be scientific

    def test_encode_with_buckets(self):
        config = NumericConfig(use_buckets=True)
        result = encode_number(1000, NumericFormat.INTEGER, config)
        assert result.startswith("<NUM_")

    def test_encode_zero(self):
        config = NumericConfig(use_buckets=True)
        result = encode_number(0, NumericFormat.INTEGER, config)
        assert result == "<NUM_ZERO>"

    def test_encode_negative_bucket(self):
        config = NumericConfig(use_buckets=True)
        result = encode_number(-100, NumericFormat.INTEGER, config)
        assert "NEG" in result


class TestNormalizeNumbers:
    """Tests for normalize_numbers function."""

    def test_placeholder_encoding(self):
        result = normalize_numbers("Value is 42")
        assert "<NUM_0>" in result.encoded_text
        assert "<NUM_0>" in result.mapping
        assert result.mapping["<NUM_0>"] == "42"

    def test_multiple_placeholders(self):
        result = normalize_numbers("Sum: 10 + 20 = 30")
        assert "<NUM_0>" in result.encoded_text
        assert "<NUM_1>" in result.encoded_text
        assert "<NUM_2>" in result.encoded_text
        assert len(result.spans) == 3

    def test_restore_numbers(self):
        text = "Pi is 3.14159"
        encoding = normalize_numbers(text)
        restored = restore_numbers(encoding.encoded_text, encoding.mapping)
        assert restored == text

    def test_complex_text(self):
        text = "The price is $99.99 (50% off from $199.98)"
        encoding = normalize_numbers(text)
        restored = restore_numbers(encoding.encoded_text, encoding.mapping)
        assert restored == text

    def test_empty_text(self):
        result = normalize_numbers("")
        assert result.encoded_text == ""
        assert len(result.spans) == 0

    def test_no_numbers(self):
        text = "Hello world"
        result = normalize_numbers(text)
        assert result.encoded_text == text
        assert len(result.spans) == 0


class TestNumericSpan:
    """Tests for NumericSpan model."""

    def test_valid_span(self):
        span = NumericSpan(
            start=10,
            end=12,
            original="42",
            format=NumericFormat.INTEGER,
            value=42.0,
        )
        assert span.start == 10
        assert span.end == 12
        assert span.format == NumericFormat.INTEGER

    def test_span_with_sign(self):
        span = NumericSpan(
            start=0,
            end=3,
            original="-42",
            format=NumericFormat.INTEGER,
            value=-42.0,
            sign="-",
        )
        assert span.sign == "-"


class TestNumericEdgeCases:
    """Tests for edge cases and additional coverage."""

    def test_detect_binary(self):
        """Test binary number detection."""
        config = NumericConfig(detect_binary=True)
        spans = detect_numbers("Binary: 0b1010", config)
        assert len(spans) == 1
        assert spans[0].format == NumericFormat.BINARY
        assert spans[0].value == 10.0

    def test_parse_invalid_value(self):
        """Test parsing with invalid numeric values."""
        # Division by zero in fraction
        config = NumericConfig(detect_fractions=True)
        spans = detect_numbers("Invalid: 5/0", config)
        assert len(spans) == 1
        assert spans[0].value is None

    def test_encode_scientific_format(self):
        """Test encoding with scientific format."""
        result = encode_number(6.022e23, NumericFormat.SCIENTIFIC)
        assert "e" in result.lower()

    def test_encode_float_with_trailing_zeros(self):
        """Test float encoding strips trailing zeros properly."""
        result = encode_number(3.10, NumericFormat.FLOAT)
        assert result == "3.1"

    def test_encode_whole_float(self):
        """Test float that's actually a whole number."""
        result = encode_number(5.0, NumericFormat.FLOAT)
        assert "5" in result

    def test_normalize_with_buckets(self):
        """Test bucket encoding mode."""
        config = NumericConfig(use_placeholder=False, use_buckets=True)
        result = normalize_numbers("Value: 1000", config)
        assert "<NUM_" in result.encoded_text
        assert "MAG_" in result.encoded_text

    def test_normalize_no_placeholder_no_bucket(self):
        """Test normalization without placeholder or bucket."""
        config = NumericConfig(use_placeholder=False, use_buckets=False)
        result = normalize_numbers("Pi is 3.14159", config)
        # Should normalize the representation
        assert result.encoded_text is not None

    def test_normalize_with_none_value(self):
        """Test normalization when value parsing fails."""
        # This is harder to trigger, but we can test the path
        config = NumericConfig(use_placeholder=False, use_buckets=False)
        result = normalize_numbers("Test 5/0", config)
        # Should keep original for unparseable fractions
        assert result.encoded_text is not None

    def test_positive_sign_detection(self):
        """Test detection of positive sign."""
        spans = detect_numbers("Value: +42")
        assert len(spans) == 1
        assert spans[0].sign == "+"
        assert spans[0].value == 42.0

    def test_overlapping_patterns(self):
        """Test that overlapping patterns are handled correctly."""
        # Scientific notation should be detected over float
        spans = detect_numbers("1.5e10")
        assert len(spans) == 1
        assert spans[0].format == NumericFormat.SCIENTIFIC

    def test_config_all_disabled(self):
        """Test with all detection disabled."""
        config = NumericConfig(
            detect_integers=False,
            detect_floats=False,
            detect_scientific=False,
            detect_hex=False,
            detect_binary=False,
            detect_percentages=False,
            detect_fractions=False,
        )
        spans = detect_numbers("Value: 42 and 3.14 and 1e10", config)
        assert len(spans) == 0


class MockTokenizerForNumeric:
    """Mock tokenizer for testing token savings."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Simple tokenization: one token per character
        return list(range(len(text)))

    def decode(self, token_ids: list[int]) -> str:
        return "x" * len(token_ids)


class TestTokenSavings:
    """Tests for get_numeric_token_savings function."""

    def test_token_savings_calculation(self):
        """Test token savings calculation."""
        from chuk_lazarus.data.tokenizers.preprocessing.numeric import (
            get_numeric_token_savings,
        )

        tokenizer = MockTokenizerForNumeric()
        result = get_numeric_token_savings("The value is 3.14159", tokenizer)
        assert "original_tokens" in result
        assert "normalized_tokens" in result
        assert "savings" in result
        assert "savings_percent" in result
        assert "numbers_detected" in result

    def test_token_savings_no_numbers(self):
        """Test token savings with no numbers."""
        from chuk_lazarus.data.tokenizers.preprocessing.numeric import (
            get_numeric_token_savings,
        )

        tokenizer = MockTokenizerForNumeric()
        result = get_numeric_token_savings("Hello world", tokenizer)
        assert result["savings"] == 0
        assert result["numbers_detected"] == 0

    def test_token_savings_with_config(self):
        """Test token savings with custom config."""
        from chuk_lazarus.data.tokenizers.preprocessing.numeric import (
            get_numeric_token_savings,
        )

        tokenizer = MockTokenizerForNumeric()
        config = NumericConfig(use_placeholder=True)
        result = get_numeric_token_savings("Value: 42", tokenizer, config)
        assert result["numbers_detected"] == 1


class TestDecodeNumber:
    """Tests for decode_number function."""

    def test_decode_zero_bucket(self):
        """Test decoding zero bucket token."""
        from chuk_lazarus.data.tokenizers.preprocessing.numeric import decode_number

        result = decode_number("<NUM_ZERO>")
        assert result == "0"

    def test_decode_negative_bucket(self):
        """Test decoding negative magnitude bucket."""
        from chuk_lazarus.data.tokenizers.preprocessing.numeric import decode_number

        result = decode_number("<NUM_NEG_MAG_2>")
        assert "-" in result and "e" in result

    def test_decode_positive_bucket(self):
        """Test decoding positive magnitude bucket."""
        from chuk_lazarus.data.tokenizers.preprocessing.numeric import decode_number

        result = decode_number("<NUM_MAG_3>")
        assert "e" in result

    def test_decode_non_bucket(self):
        """Test decoding non-bucket token."""
        from chuk_lazarus.data.tokenizers.preprocessing.numeric import decode_number

        result = decode_number("42")
        assert result == "42"

    def test_decode_with_config(self):
        """Test decoding with custom config."""
        from chuk_lazarus.data.tokenizers.preprocessing.numeric import decode_number

        config = NumericConfig(bucket_base=5)
        result = decode_number("<NUM_MAG_2>", config)
        assert "e" in result
