"""Reasoning complexity scoring for curriculum learning."""

import re
from typing import Protocol

from pydantic import BaseModel, Field


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...


class ReasoningConfig(BaseModel):
    """Configuration for reasoning density scoring."""

    # Weight for each signal
    math_symbol_weight: float = Field(default=0.2, ge=0.0)
    bracket_depth_weight: float = Field(default=0.2, ge=0.0)
    variable_weight: float = Field(default=0.15, ge=0.0)
    numeric_weight: float = Field(default=0.15, ge=0.0)
    operator_weight: float = Field(default=0.15, ge=0.0)
    length_weight: float = Field(default=0.15, ge=0.0)

    # Patterns to detect
    math_symbols: str = Field(
        default=r"[∑∏∫∂∇√∞±×÷≤≥≠≈∈∉⊂⊃∪∩∀∃σμλθαβγδε]",
        description="Regex for math symbols",
    )
    variable_pattern: str = Field(
        default=r"\b[a-zA-Z]_?\{?[a-zA-Z0-9]*\}?\b",
        description="Regex for variables",
    )


class ReasoningDensityScore(BaseModel):
    """Reasoning density score for a single text."""

    text_index: int = Field(ge=0, description="Original index in corpus")
    overall_score: float = Field(ge=0.0, description="Overall reasoning density")
    math_symbol_score: float = Field(ge=0.0, description="Math symbol density")
    bracket_depth_score: float = Field(ge=0.0, description="Nesting complexity")
    variable_score: float = Field(ge=0.0, description="Variable usage density")
    numeric_score: float = Field(ge=0.0, description="Numeric content density")
    operator_score: float = Field(ge=0.0, description="Operator density")
    length_score: float = Field(ge=0.0, description="Length-based score")
    token_count: int = Field(ge=0, description="Number of tokens")


class DifficultyPercentiles(BaseModel):
    """Difficulty percentile breakdown."""

    p10: float = Field(description="10th percentile score")
    p25: float = Field(description="25th percentile score")
    p50: float = Field(description="Median score")
    p75: float = Field(description="75th percentile score")
    p90: float = Field(description="90th percentile score")
    min_score: float = Field(description="Minimum score")
    max_score: float = Field(description="Maximum score")
    mean_score: float = Field(description="Mean score")


def _calculate_bracket_depth(text: str) -> int:
    """Calculate maximum bracket nesting depth."""
    max_depth = 0
    current_depth = 0
    brackets = {"(": 1, ")": -1, "[": 1, "]": -1, "{": 1, "}": -1}

    for char in text:
        if char in brackets:
            current_depth += brackets[char]
            max_depth = max(max_depth, current_depth)

    return max_depth


def _count_operators(text: str) -> int:
    """Count mathematical and logical operators."""
    operators = r"[+\-*/=<>≤≥≠≈∧∨¬→↔]"
    return len(re.findall(operators, text))


def _count_numbers(text: str) -> int:
    """Count numeric values."""
    # Match integers, floats, scientific notation
    number_pattern = r"\b\d+\.?\d*(?:[eE][+-]?\d+)?\b"
    return len(re.findall(number_pattern, text))


def score_reasoning_density(
    text: str,
    index: int,
    tokenizer: TokenizerProtocol,
    config: ReasoningConfig | None = None,
) -> ReasoningDensityScore:
    """
    Score reasoning complexity of a single text.

    Higher scores indicate more "thinking heavy" content.

    Args:
        text: Text to score
        index: Original index in corpus
        tokenizer: Tokenizer instance
        config: Scoring configuration

    Returns:
        ReasoningDensityScore with component scores
    """
    if config is None:
        config = ReasoningConfig()

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_count = len(token_ids)
    text_len = len(text)

    if text_len == 0:
        return ReasoningDensityScore(
            text_index=index,
            overall_score=0.0,
            math_symbol_score=0.0,
            bracket_depth_score=0.0,
            variable_score=0.0,
            numeric_score=0.0,
            operator_score=0.0,
            length_score=0.0,
            token_count=token_count,
        )

    # Math symbols
    math_matches = len(re.findall(config.math_symbols, text))
    math_score = min(1.0, math_matches / (text_len / 50))  # Normalize

    # Bracket depth
    max_depth = _calculate_bracket_depth(text)
    bracket_score = min(1.0, max_depth / 5)  # Cap at depth 5

    # Variables
    var_matches = len(re.findall(config.variable_pattern, text))
    var_score = min(1.0, var_matches / (text_len / 20))

    # Numerics
    num_count = _count_numbers(text)
    numeric_score = min(1.0, num_count / (text_len / 30))

    # Operators
    op_count = _count_operators(text)
    operator_score = min(1.0, op_count / (text_len / 20))

    # Length (longer = potentially more complex)
    length_score = min(1.0, token_count / 500)

    # Weighted overall score
    overall = (
        math_score * config.math_symbol_weight
        + bracket_score * config.bracket_depth_weight
        + var_score * config.variable_weight
        + numeric_score * config.numeric_weight
        + operator_score * config.operator_weight
        + length_score * config.length_weight
    )

    return ReasoningDensityScore(
        text_index=index,
        overall_score=overall,
        math_symbol_score=math_score,
        bracket_depth_score=bracket_score,
        variable_score=var_score,
        numeric_score=numeric_score,
        operator_score=operator_score,
        length_score=length_score,
        token_count=token_count,
    )


def sort_by_reasoning_density(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    config: ReasoningConfig | None = None,
    reverse: bool = False,
) -> list[ReasoningDensityScore]:
    """
    Sort texts by reasoning density.

    Args:
        texts: List of texts
        tokenizer: Tokenizer instance
        config: Scoring configuration
        reverse: If True, sort hardest first

    Returns:
        List of scores sorted by overall_score
    """
    scores = [score_reasoning_density(text, i, tokenizer, config) for i, text in enumerate(texts)]
    return sorted(scores, key=lambda x: x.overall_score, reverse=reverse)


def get_difficulty_percentiles(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    config: ReasoningConfig | None = None,
) -> DifficultyPercentiles:
    """
    Calculate difficulty percentiles for curriculum planning.

    Args:
        texts: List of texts
        tokenizer: Tokenizer instance
        config: Scoring configuration

    Returns:
        DifficultyPercentiles with distribution stats
    """
    scores = [
        score_reasoning_density(text, i, tokenizer, config).overall_score
        for i, text in enumerate(texts)
    ]

    if not scores:
        return DifficultyPercentiles(
            p10=0.0,
            p25=0.0,
            p50=0.0,
            p75=0.0,
            p90=0.0,
            min_score=0.0,
            max_score=0.0,
            mean_score=0.0,
        )

    sorted_scores = sorted(scores)
    n = len(sorted_scores)

    def percentile(p: float) -> float:
        idx = int(p * n / 100)
        return sorted_scores[min(idx, n - 1)]

    return DifficultyPercentiles(
        p10=percentile(10),
        p25=percentile(25),
        p50=percentile(50),
        p75=percentile(75),
        p90=percentile(90),
        min_score=sorted_scores[0],
        max_score=sorted_scores[-1],
        mean_score=sum(scores) / n,
    )
