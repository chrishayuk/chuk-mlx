"""Tests for structure token injection."""

from chuk_lazarus.data.tokenizers.preprocessing.structure import (
    StructureConfig,
    StructureSpan,
    StructureType,
    create_math_aware_config,
    create_tool_aware_config,
    detect_structures,
    get_structure_stats,
    inject_structure_tokens,
    restore_structures,
)


class TestStructureConfig:
    """Tests for StructureConfig model."""

    def test_default_values(self):
        config = StructureConfig()
        assert config.detect_json_keys is True
        assert config.detect_uuids is True
        assert config.detect_urls is True

    def test_custom_values(self):
        config = StructureConfig(
            detect_json_keys=False,
            detect_uuids=False,
        )
        assert config.detect_json_keys is False
        assert config.detect_uuids is False


class TestDetectStructures:
    """Tests for detect_structures function."""

    def test_detect_uuid(self):
        text = "ID: 550e8400-e29b-41d4-a716-446655440000"
        spans = detect_structures(text)
        assert len(spans) == 1
        assert spans[0].structure_type == StructureType.UUID

    def test_detect_url(self):
        text = "Visit https://example.com/path"
        spans = detect_structures(text)
        assert len(spans) == 1
        assert spans[0].structure_type == StructureType.URL

    def test_detect_email(self):
        text = "Contact: user@example.com"
        spans = detect_structures(text)
        assert len(spans) == 1
        assert spans[0].structure_type == StructureType.EMAIL

    def test_detect_ip_address(self):
        text = "Server: 192.168.1.1"
        spans = detect_structures(text)
        assert len(spans) == 1
        assert spans[0].structure_type == StructureType.IP_ADDRESS

    def test_detect_date(self):
        text = "Date: 2024-01-15"
        spans = detect_structures(text)
        assert len(spans) == 1
        assert spans[0].structure_type == StructureType.DATE

    def test_detect_time(self):
        text = "Time: 14:30:00"
        spans = detect_structures(text)
        assert len(spans) == 1
        assert spans[0].structure_type == StructureType.TIME

    def test_detect_datetime(self):
        text = "Timestamp: 2024-01-15T14:30:00Z"
        spans = detect_structures(text)
        assert len(spans) == 1
        assert spans[0].structure_type == StructureType.DATETIME

    def test_detect_path(self):
        text = "File: /usr/local/bin/python"
        spans = detect_structures(text)
        assert len(spans) == 1
        assert spans[0].structure_type == StructureType.PATH

    def test_detect_json_key(self):
        text = '{"name": "value"}'
        spans = detect_structures(text)
        # Should detect the key including colon
        assert any(s.structure_type == StructureType.JSON_KEY for s in spans)

    def test_detect_variable(self):
        text = "Value is ${VAR_NAME}"
        spans = detect_structures(text)
        assert len(spans) == 1
        assert spans[0].structure_type == StructureType.VARIABLE

    def test_detect_multiple(self):
        text = "User user@test.com at 192.168.1.1"
        spans = detect_structures(text)
        assert len(spans) == 2

    def test_empty_text(self):
        spans = detect_structures("")
        assert len(spans) == 0

    def test_no_structures(self):
        spans = detect_structures("Hello world")
        assert len(spans) == 0

    def test_config_disables_detection(self):
        config = StructureConfig(detect_uuids=False)
        text = "ID: 550e8400-e29b-41d4-a716-446655440000"
        spans = detect_structures(text, config)
        assert len(spans) == 0


class TestInjectStructureTokens:
    """Tests for inject_structure_tokens function."""

    def test_inject_uuid(self):
        text = "ID: 550e8400-e29b-41d4-a716-446655440000"
        result = inject_structure_tokens(text)
        assert "<UUID_0>" in result.encoded_text
        assert "550e8400" not in result.encoded_text

    def test_inject_multiple(self):
        text = "Email: a@b.com and IP: 1.2.3.4"
        result = inject_structure_tokens(text)
        assert "<EMAIL_0>" in result.encoded_text
        assert "<IP_0>" in result.encoded_text

    def test_restore_structures(self):
        text = "Visit https://example.com"
        encoding = inject_structure_tokens(text)
        restored = restore_structures(encoding.encoded_text, encoding.mapping)
        assert restored == text

    def test_complex_text(self):
        text = "User 550e8400-e29b-41d4-a716-446655440000 at 192.168.1.1 sent email to test@example.com"
        encoding = inject_structure_tokens(text)
        restored = restore_structures(encoding.encoded_text, encoding.mapping)
        assert restored == text

    def test_empty_text(self):
        result = inject_structure_tokens("")
        assert result.encoded_text == ""
        assert len(result.spans) == 0


class TestGetStructureStats:
    """Tests for get_structure_stats function."""

    def test_basic_stats(self):
        text = "Email: a@b.com and IP: 1.2.3.4"
        stats = get_structure_stats(text)
        assert stats["total_structures"] == 2
        assert "email" in stats["by_type"]
        assert "ip_address" in stats["by_type"]

    def test_empty_text(self):
        stats = get_structure_stats("")
        assert stats["total_structures"] == 0
        assert stats["coverage"] == 0

    def test_no_structures(self):
        stats = get_structure_stats("Hello world")
        assert stats["total_structures"] == 0


class TestConfigFactories:
    """Tests for configuration factory functions."""

    def test_tool_aware_config(self):
        config = create_tool_aware_config()
        assert config.detect_json_keys is True
        assert config.detect_tool_syntax is True
        assert config.detect_variables is True

    def test_math_aware_config(self):
        config = create_math_aware_config()
        assert config.detect_json_keys is False
        assert config.detect_variables is True


class TestStructureSpan:
    """Tests for StructureSpan model."""

    def test_valid_span(self):
        span = StructureSpan(
            start=0,
            end=36,
            original="550e8400-e29b-41d4-a716-446655440000",
            structure_type=StructureType.UUID,
        )
        assert span.start == 0
        assert span.end == 36
        assert span.structure_type == StructureType.UUID

    def test_span_with_metadata(self):
        span = StructureSpan(
            start=0,
            end=10,
            original="test@a.com",
            structure_type=StructureType.EMAIL,
            metadata={"domain": "a.com"},
        )
        assert span.metadata["domain"] == "a.com"
