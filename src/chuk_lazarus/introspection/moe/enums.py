"""MoE-specific enums."""

from enum import Enum


class MoEArchitecture(str, Enum):
    """Supported MoE architecture types."""

    GPT_OSS = "gpt_oss"
    """GPT-OSS: 32 experts, 4 active, MXFP4 quantized."""

    LLAMA4 = "llama4"
    """Llama 4: Shared expert (always active) + routed experts."""

    GRANITE_HYBRID = "granite_hybrid"
    """Granite Hybrid: MoE with Mamba-2/Attention hybrid."""

    MIXTRAL = "mixtral"
    """Mixtral: 8 experts, 2 active, standard routing."""

    GENERIC = "generic"
    """Generic MoE: Uses standard MoE component."""


class ExpertCategory(str, Enum):
    """Categories of expert specialization."""

    CODE = "code"
    MATH = "math"
    LANGUAGE = "language"
    PUNCTUATION = "punctuation"
    PROPER_NOUNS = "proper_nouns"
    FUNCTION_WORDS = "function_words"
    NUMBERS = "numbers"
    POSITION_FIRST = "position_first"
    POSITION_LAST = "position_last"
    GENERALIST = "generalist"
    UNKNOWN = "unknown"


class ExpertRole(str, Enum):
    """Roles that experts can play."""

    SPECIALIST = "specialist"
    """Expert specializes in specific token/domain type."""

    GENERALIST = "generalist"
    """Expert activates across many domains."""

    POSITIONAL = "positional"
    """Expert specializes by position (first/last tokens)."""

    RARE = "rare"
    """Expert rarely activates."""
