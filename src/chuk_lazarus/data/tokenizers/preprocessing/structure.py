"""
Structure token injection utilities.

Detects and replaces structured data patterns with atomic tokens:
- JSON keys and values
- UUIDs
- File paths
- URLs
- Dates and timestamps
- Tool call syntax
"""

import re
from enum import Enum

from pydantic import BaseModel, Field


class StructureType(str, Enum):
    """Types of structure patterns."""

    JSON_KEY = "json_key"
    JSON_STRING = "json_string"
    UUID = "uuid"
    PATH = "path"
    URL = "url"
    EMAIL = "email"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    IP_ADDRESS = "ip_address"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    VARIABLE = "variable"
    FUNCTION_NAME = "function_name"


class StructureConfig(BaseModel):
    """Configuration for structure detection."""

    # Detection toggles
    detect_json_keys: bool = Field(default=True, description="Detect JSON keys")
    detect_uuids: bool = Field(default=True, description="Detect UUIDs")
    detect_paths: bool = Field(default=True, description="Detect file paths")
    detect_urls: bool = Field(default=True, description="Detect URLs")
    detect_emails: bool = Field(default=True, description="Detect email addresses")
    detect_dates: bool = Field(default=True, description="Detect dates")
    detect_times: bool = Field(default=True, description="Detect times")
    detect_ip_addresses: bool = Field(default=True, description="Detect IP addresses")
    detect_tool_syntax: bool = Field(default=True, description="Detect tool call syntax")
    detect_variables: bool = Field(default=True, description="Detect variable patterns")

    # Token templates
    token_templates: dict[str, str] = Field(
        default_factory=lambda: {
            StructureType.JSON_KEY: "<JSON_KEY_{idx}>",
            StructureType.JSON_STRING: "<JSON_STR_{idx}>",
            StructureType.UUID: "<UUID_{idx}>",
            StructureType.PATH: "<PATH_{idx}>",
            StructureType.URL: "<URL_{idx}>",
            StructureType.EMAIL: "<EMAIL_{idx}>",
            StructureType.DATE: "<DATE_{idx}>",
            StructureType.TIME: "<TIME_{idx}>",
            StructureType.DATETIME: "<DATETIME_{idx}>",
            StructureType.IP_ADDRESS: "<IP_{idx}>",
            StructureType.TOOL_CALL: "<TOOL_CALL_{idx}>",
            StructureType.TOOL_RESULT: "<TOOL_RESULT_{idx}>",
            StructureType.VARIABLE: "<VAR_{idx}>",
            StructureType.FUNCTION_NAME: "<FUNC_{idx}>",
        },
        description="Token templates for each structure type",
    )

    # Preservation
    preserve_json_structure: bool = Field(
        default=False, description="Keep JSON brackets/colons, only replace values"
    )
    min_path_segments: int = Field(default=2, description="Minimum path segments to detect")
    min_uuid_confidence: float = Field(
        default=0.9, description="Confidence threshold for UUID detection"
    )


class StructureSpan(BaseModel):
    """A detected structure span in text."""

    start: int = Field(ge=0, description="Start position in text")
    end: int = Field(ge=0, description="End position in text")
    original: str = Field(description="Original text")
    structure_type: StructureType = Field(description="Type of structure")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class StructureEncoding(BaseModel):
    """Result of structure token injection."""

    encoded_text: str = Field(description="Text with structures replaced")
    spans: list[StructureSpan] = Field(default_factory=list, description="Detected structure spans")
    mapping: dict[str, str] = Field(default_factory=dict, description="Token -> original mapping")


# Regex patterns for structure detection
PATTERNS: dict[StructureType, re.Pattern] = {
    # UUID: 8-4-4-4-12 hex digits
    StructureType.UUID: re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    ),
    # URL
    StructureType.URL: re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+"),
    # Email
    StructureType.EMAIL: re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    # IP address (IPv4)
    StructureType.IP_ADDRESS: re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    ),
    # ISO date: YYYY-MM-DD
    StructureType.DATE: re.compile(r"\b\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b"),
    # Time: HH:MM:SS or HH:MM
    StructureType.TIME: re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\b"),
    # ISO datetime
    StructureType.DATETIME: re.compile(
        r"\b\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])T(?:[01]?\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
    ),
    # Unix-style path
    StructureType.PATH: re.compile(
        r"(?:/[a-zA-Z0-9._-]+){2,}|(?:[a-zA-Z]:\\(?:[a-zA-Z0-9._-]+\\)+[a-zA-Z0-9._-]+)"
    ),
    # JSON key (quoted string followed by colon)
    StructureType.JSON_KEY: re.compile(r'"([^"\\]|\\.)*"\s*:'),
    # Tool call syntax: <tool_name>(...) or [tool_name](...)
    StructureType.TOOL_CALL: re.compile(
        r"<([a-zA-Z_][a-zA-Z0-9_]*)>\s*\([^)]*\)|"
        r"\[([a-zA-Z_][a-zA-Z0-9_]*)\]\s*\([^)]*\)"
    ),
    # Variable pattern: ${var} or {{var}} or $var
    StructureType.VARIABLE: re.compile(
        r"\$\{[a-zA-Z_][a-zA-Z0-9_]*\}|"
        r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}|"
        r"\$[a-zA-Z_][a-zA-Z0-9_]*"
    ),
    # Function name: word followed by parentheses
    StructureType.FUNCTION_NAME: re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\("),
}


def detect_structures(
    text: str,
    config: StructureConfig | None = None,
) -> list[StructureSpan]:
    """
    Detect all structure patterns in text.

    Args:
        text: Input text to scan
        config: Detection configuration

    Returns:
        List of StructureSpan objects, sorted by position
    """
    if config is None:
        config = StructureConfig()

    spans: list[StructureSpan] = []
    used_positions: set[int] = set()

    # Detection order (more specific first)
    detection_order = [
        (StructureType.DATETIME, config.detect_dates),  # Before DATE/TIME
        (StructureType.UUID, config.detect_uuids),
        (StructureType.URL, config.detect_urls),
        (StructureType.EMAIL, config.detect_emails),
        (StructureType.IP_ADDRESS, config.detect_ip_addresses),
        (StructureType.DATE, config.detect_dates),
        (StructureType.TIME, config.detect_times),
        (StructureType.PATH, config.detect_paths),
        (StructureType.JSON_KEY, config.detect_json_keys),
        (StructureType.TOOL_CALL, config.detect_tool_syntax),
        (StructureType.VARIABLE, config.detect_variables),
    ]

    for struct_type, enabled in detection_order:
        if not enabled:
            continue

        pattern = PATTERNS.get(struct_type)
        if pattern is None:
            continue

        for match in pattern.finditer(text):
            start, end = match.start(), match.end()

            # Skip if overlapping with existing span
            if any(pos in used_positions for pos in range(start, end)):
                continue

            # Additional validation for paths
            if struct_type == StructureType.PATH:
                path = match.group()
                segments = path.replace("\\", "/").split("/")
                if len([s for s in segments if s]) < config.min_path_segments:
                    continue

            span = StructureSpan(
                start=start,
                end=end,
                original=match.group(),
                structure_type=struct_type,
                metadata={"groups": match.groups()},
            )
            spans.append(span)

            # Mark positions as used
            for pos in range(start, end):
                used_positions.add(pos)

    # Sort by position
    spans.sort(key=lambda s: s.start)
    return spans


def inject_structure_tokens(
    text: str,
    config: StructureConfig | None = None,
) -> StructureEncoding:
    """
    Replace detected structures with atomic tokens.

    Args:
        text: Input text
        config: Injection configuration

    Returns:
        StructureEncoding with encoded text and mapping for restoration
    """
    if config is None:
        config = StructureConfig()

    spans = detect_structures(text, config)

    if not spans:
        return StructureEncoding(encoded_text=text, spans=[], mapping={})

    mapping: dict[str, str] = {}
    parts: list[str] = []
    last_end = 0

    # Track indices per type for unique tokens
    type_indices: dict[StructureType, int] = {}

    for span in spans:
        # Add text before this structure
        parts.append(text[last_end : span.start])

        # Get index for this type
        idx = type_indices.get(span.structure_type, 0)
        type_indices[span.structure_type] = idx + 1

        # Generate token
        template = config.token_templates.get(
            span.structure_type, f"<{span.structure_type.value.upper()}_{{idx}}>"
        )
        token = template.format(idx=idx)

        parts.append(token)
        mapping[token] = span.original

        last_end = span.end

    # Add remaining text
    parts.append(text[last_end:])

    return StructureEncoding(
        encoded_text="".join(parts),
        spans=spans,
        mapping=mapping,
    )


def restore_structures(
    encoded_text: str,
    mapping: dict[str, str],
) -> str:
    """
    Restore original structures from encoded text.

    Args:
        encoded_text: Text with structure tokens
        mapping: Token -> original mapping from inject_structure_tokens

    Returns:
        Text with original structures restored
    """
    result = encoded_text
    for token, original in mapping.items():
        result = result.replace(token, original)
    return result


def get_structure_stats(
    text: str,
    config: StructureConfig | None = None,
) -> dict:
    """
    Get statistics about structures in text.

    Args:
        text: Input text
        config: Detection configuration

    Returns:
        Dict with counts by structure type
    """
    spans = detect_structures(text, config)

    stats: dict[str, int] = {}
    for span in spans:
        key = span.structure_type.value
        stats[key] = stats.get(key, 0) + 1

    return {
        "total_structures": len(spans),
        "by_type": stats,
        "coverage": sum(s.end - s.start for s in spans) / len(text) if text else 0,
    }


def create_tool_aware_config() -> StructureConfig:
    """
    Create a configuration optimized for tool/agent traces.

    Returns:
        StructureConfig tuned for tool use scenarios
    """
    return StructureConfig(
        detect_json_keys=True,
        detect_uuids=True,
        detect_paths=True,
        detect_urls=True,
        detect_emails=True,
        detect_dates=True,
        detect_times=True,
        detect_ip_addresses=True,
        detect_tool_syntax=True,
        detect_variables=True,
        preserve_json_structure=True,
    )


def create_math_aware_config() -> StructureConfig:
    """
    Create a configuration optimized for math/reasoning traces.

    Returns:
        StructureConfig tuned for math scenarios
    """
    return StructureConfig(
        detect_json_keys=False,
        detect_uuids=False,
        detect_paths=False,
        detect_urls=False,
        detect_emails=False,
        detect_dates=False,
        detect_times=False,
        detect_ip_addresses=False,
        detect_tool_syntax=False,
        detect_variables=True,  # Keep variables for math
    )
