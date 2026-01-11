"""Shared constants for introspection module.

These constants are used by both the introspection services and CLI commands.
They are placed here to avoid circular imports between introspection.moe
and cli.commands modules.
"""

from __future__ import annotations

from enum import Enum


class LayerPhase(str, Enum):
    """Layer phase classifications for MoE analysis."""

    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"


class LayerPhaseDefaults:
    """Default layer boundaries for phase classification."""

    EARLY_END: int = 8
    MIDDLE_END: int = 16


class PatternCategory(str, Enum):
    """Pattern categories for MoE trigram analysis."""

    ARITHMETIC = "arithmetic"
    CODE = "code"
    SYNONYM = "synonym"
    ANTONYM = "antonym"
    ANALOGY = "analogy"
    HYPERNYM = "hypernym"
    COMPARISON = "comparison"
    CAUSATION = "causation"
    CONDITIONAL = "conditional"
    QUESTION = "question"
    NEGATION = "negation"
    TEMPORAL = "temporal"
    QUANTIFICATION = "quantification"
    CONTEXT_SWITCH = "context_switch"
    POSITION = "position"
    COORDINATION = "coordination"


class Domain(str, Enum):
    """Domain categories for expert analysis."""

    MATH = "math"
    CODE = "code"
    LANGUAGE = "language"
    REASONING = "reasoning"


class TokenType(str, Enum):
    """Semantic token type classifications for MoE analysis."""

    # Numbers and operators
    NUM = "NUM"
    OP = "OP"
    BR = "BR"  # Brackets
    PN = "PN"  # Punctuation
    QUOTE = "QUOTE"

    # Code-related
    KW = "KW"  # Keywords
    BOOL = "BOOL"  # Boolean literals
    TYPE = "TYPE"  # Type keywords
    VAR = "VAR"  # Variables

    # Semantic relations
    SYN = "SYN"  # Synonym markers
    ANT = "ANT"  # Antonym markers
    AS = "AS"  # "as" marker
    TO = "TO"  # "to" marker
    CAUSE = "CAUSE"  # Causation markers
    COND = "COND"  # Conditional markers
    THAN = "THAN"  # Comparison marker

    # Question/answer
    QW = "QW"  # Question words
    ANS = "ANS"  # Answer words

    # Modifiers
    NEG = "NEG"  # Negation
    TIME = "TIME"  # Temporal
    QUANT = "QUANT"  # Quantifiers
    COMP = "COMP"  # Comparatives
    COORD = "COORD"  # Coordination

    # Parts of speech
    NOUN = "NOUN"
    ADJ = "ADJ"
    VERB = "VERB"
    FUNC = "FUNC"  # Function words

    # Other
    CAP = "CAP"  # Capitalized (proper noun)
    CW = "CW"  # Content word (default)
    WS = "WS"  # Whitespace


__all__ = [
    "Domain",
    "LayerPhase",
    "LayerPhaseDefaults",
    "PatternCategory",
    "TokenType",
]
