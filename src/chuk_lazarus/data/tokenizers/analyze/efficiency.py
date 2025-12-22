"""
Token efficiency analysis.

Measures token waste and efficiency across different content types:
- Tokens per sample (mean/p95)
- Tokens per reasoning step
- Tokens per equation
- Tokens per tool call
- Fragmentation score
"""

import re
from typing import Protocol

from pydantic import BaseModel, Field


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...


class EfficiencyConfig(BaseModel):
    """Configuration for efficiency analysis."""

    # Reasoning step detection
    step_patterns: list[str] = Field(
        default=[
            r"Step \d+:",
            r"\d+\.",
            r"First,",
            r"Second,",
            r"Third,",
            r"Then,",
            r"Next,",
            r"Finally,",
            r"Therefore,",
            r"Thus,",
            r"So,",
            r"Hence,",
        ],
        description="Patterns that indicate reasoning steps",
    )

    # Equation detection
    equation_patterns: list[str] = Field(
        default=[
            r"[=<>≤≥≠]+",  # Operators
            r"\d+\s*[+\-*/×÷]\s*\d+",  # Simple arithmetic
            r"[a-z]\s*=\s*[^,;]+",  # Variable assignments
            r"\([^)]+\)",  # Parenthesized expressions
            r"\\[a-z]+\{",  # LaTeX commands
            r"\$[^$]+\$",  # LaTeX inline math
        ],
        description="Patterns that indicate equations",
    )

    # Tool call detection
    tool_patterns: list[str] = Field(
        default=[
            r"<TOOL_CALL>",
            r"<tool_call>",
            r"\[TOOL:",
            r"```tool",
            r"Action:",
            r"Observation:",
            r"<function=",
            r"<|plugin|>",
        ],
        description="Patterns that indicate tool calls",
    )


class SampleStats(BaseModel):
    """Statistics for tokens per sample."""

    count: int = Field(description="Number of samples")
    total_tokens: int = Field(description="Total tokens across all samples")
    mean: float = Field(description="Mean tokens per sample")
    median: float = Field(description="Median tokens per sample")
    std: float = Field(description="Standard deviation")
    p5: float = Field(description="5th percentile")
    p25: float = Field(description="25th percentile")
    p75: float = Field(description="75th percentile")
    p95: float = Field(description="95th percentile")
    p99: float = Field(description="99th percentile")
    min_tokens: int = Field(description="Minimum tokens in a sample")
    max_tokens: int = Field(description="Maximum tokens in a sample")


class ContentTypeStats(BaseModel):
    """Statistics for a specific content type."""

    content_type: str = Field(description="Type of content (step, equation, tool)")
    count: int = Field(description="Number of instances detected")
    total_tokens: int = Field(description="Total tokens for this content type")
    mean_tokens: float = Field(description="Mean tokens per instance")
    examples: list[dict] = Field(
        default_factory=list, description="Example instances with token counts"
    )


class FragmentationStats(BaseModel):
    """Fragmentation analysis results."""

    fragmentation_score: float = Field(
        ge=0.0, le=1.0, description="Overall fragmentation score (0=perfect, 1=worst)"
    )
    single_char_tokens: int = Field(description="Number of single-character tokens")
    subword_tokens: int = Field(description="Number of subword tokens (##, Ġ, etc.)")
    continuation_tokens: int = Field(description="Number of continuation tokens")
    total_tokens: int = Field(description="Total tokens analyzed")
    fragmented_words: list[dict] = Field(default_factory=list, description="Most fragmented words")


class EfficiencyReport(BaseModel):
    """Complete efficiency analysis report."""

    # Sample-level stats
    sample_stats: SampleStats = Field(description="Tokens per sample statistics")

    # Content-type stats
    reasoning_steps: ContentTypeStats | None = Field(
        default=None, description="Tokens per reasoning step"
    )
    equations: ContentTypeStats | None = Field(default=None, description="Tokens per equation")
    tool_calls: ContentTypeStats | None = Field(default=None, description="Tokens per tool call")

    # Fragmentation
    fragmentation: FragmentationStats = Field(description="Fragmentation analysis")

    # Overall metrics
    efficiency_score: float = Field(
        ge=0.0, le=100.0, description="Overall efficiency score (100=best)"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations for improvement"
    )


def _percentile(values: list[float], p: float) -> float:
    """Calculate percentile of a sorted list."""
    if not values:
        return 0.0
    k = (len(values) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(values) else f
    return values[f] + (k - f) * (values[c] - values[f]) if f != c else values[f]


def _std_dev(values: list[float], mean: float) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance**0.5


def analyze_sample_stats(
    texts: list[str],
    tokenizer: TokenizerProtocol,
) -> SampleStats:
    """
    Analyze tokens per sample statistics.

    Args:
        texts: List of texts to analyze
        tokenizer: Tokenizer to use

    Returns:
        SampleStats with distribution metrics
    """
    if not texts:
        return SampleStats(
            count=0,
            total_tokens=0,
            mean=0.0,
            median=0.0,
            std=0.0,
            p5=0.0,
            p25=0.0,
            p75=0.0,
            p95=0.0,
            p99=0.0,
            min_tokens=0,
            max_tokens=0,
        )

    token_counts = [len(tokenizer.encode(text, add_special_tokens=False)) for text in texts]
    sorted_counts = sorted(token_counts)
    total = sum(token_counts)
    mean = total / len(token_counts)

    return SampleStats(
        count=len(texts),
        total_tokens=total,
        mean=mean,
        median=_percentile(sorted_counts, 50),
        std=_std_dev([float(c) for c in token_counts], mean),
        p5=_percentile(sorted_counts, 5),
        p25=_percentile(sorted_counts, 25),
        p75=_percentile(sorted_counts, 75),
        p95=_percentile(sorted_counts, 95),
        p99=_percentile(sorted_counts, 99),
        min_tokens=min(token_counts),
        max_tokens=max(token_counts),
    )


def analyze_content_type(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    content_type: str,
    patterns: list[str],
    max_examples: int = 5,
) -> ContentTypeStats | None:
    """
    Analyze tokens for a specific content type.

    Args:
        texts: List of texts to analyze
        tokenizer: Tokenizer to use
        content_type: Name of content type
        patterns: Regex patterns to detect content
        max_examples: Maximum examples to include

    Returns:
        ContentTypeStats or None if no instances found
    """
    instances: list[dict] = []

    combined_pattern = "|".join(f"({p})" for p in patterns)
    regex = re.compile(combined_pattern, re.IGNORECASE)

    for text in texts:
        matches = list(regex.finditer(text))
        for match in matches:
            # Get surrounding context
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 50)
            context = text[start:end]

            # Count tokens for the matched region
            matched_text = match.group()
            tokens = tokenizer.encode(matched_text, add_special_tokens=False)

            instances.append(
                {
                    "matched": matched_text,
                    "context": context,
                    "token_count": len(tokens),
                }
            )

    if not instances:
        return None

    total_tokens = sum(i["token_count"] for i in instances)

    return ContentTypeStats(
        content_type=content_type,
        count=len(instances),
        total_tokens=total_tokens,
        mean_tokens=total_tokens / len(instances),
        examples=instances[:max_examples],
    )


def analyze_fragmentation(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    max_fragmented: int = 10,
) -> FragmentationStats:
    """
    Analyze token fragmentation.

    Args:
        texts: List of texts to analyze
        tokenizer: Tokenizer to use
        max_fragmented: Maximum fragmented words to include

    Returns:
        FragmentationStats
    """
    single_char = 0
    subword = 0
    continuation = 0
    total = 0

    word_fragmentation: dict[str, int] = {}

    for text in texts:
        # Split into words
        words = re.findall(r"\b\w+\b", text)

        for word in words:
            tokens = tokenizer.encode(word, add_special_tokens=False)
            token_count = len(tokens)
            total += token_count

            if token_count > 1:
                # Word was fragmented
                word_fragmentation[word] = max(word_fragmentation.get(word, 0), token_count)

            for token_id in tokens:
                try:
                    decoded = tokenizer.decode([token_id])
                    if len(decoded) == 1:
                        single_char += 1
                    if decoded.startswith("##") or decoded.startswith("Ġ"):
                        subword += 1
                    if decoded.startswith("▁") or decoded.startswith("_"):
                        continuation += 1
                except Exception:
                    pass

    # Sort by fragmentation
    sorted_words = sorted(word_fragmentation.items(), key=lambda x: -x[1])
    fragmented_words = [{"word": w, "tokens": c} for w, c in sorted_words[:max_fragmented]]

    # Calculate fragmentation score
    # Higher score = more fragmentation = worse
    if total == 0:
        fragmentation_score = 0.0
    else:
        # Ratio of non-whole-word tokens
        fragmented_ratio = (single_char + subword + continuation) / total
        # Normalize to 0-1
        fragmentation_score = min(1.0, fragmented_ratio)

    return FragmentationStats(
        fragmentation_score=fragmentation_score,
        single_char_tokens=single_char,
        subword_tokens=subword,
        continuation_tokens=continuation,
        total_tokens=total,
        fragmented_words=fragmented_words,
    )


def analyze_efficiency(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    config: EfficiencyConfig | None = None,
) -> EfficiencyReport:
    """
    Comprehensive token efficiency analysis.

    Args:
        texts: List of texts to analyze
        tokenizer: Tokenizer to use
        config: Analysis configuration

    Returns:
        EfficiencyReport with all metrics
    """
    if config is None:
        config = EfficiencyConfig()

    # Sample stats
    sample_stats = analyze_sample_stats(texts, tokenizer)

    # Content type analysis
    reasoning_steps = analyze_content_type(texts, tokenizer, "reasoning_step", config.step_patterns)
    equations = analyze_content_type(texts, tokenizer, "equation", config.equation_patterns)
    tool_calls = analyze_content_type(texts, tokenizer, "tool_call", config.tool_patterns)

    # Fragmentation
    fragmentation = analyze_fragmentation(texts, tokenizer)

    # Calculate overall efficiency score
    # Start at 100, deduct for issues
    score = 100.0

    # Deduct for high fragmentation
    score -= fragmentation.fragmentation_score * 20

    # Deduct for high token variance (inconsistent encoding)
    if sample_stats.std > sample_stats.mean * 0.5:
        score -= 10

    # Deduct for very high p95/mean ratio
    if sample_stats.mean > 0 and sample_stats.p95 / sample_stats.mean > 3:
        score -= 10

    score = max(0.0, min(100.0, score))

    # Generate recommendations
    recommendations: list[str] = []

    if fragmentation.fragmentation_score > 0.3:
        recommendations.append(
            f"High fragmentation ({fragmentation.fragmentation_score:.1%}). "
            "Consider preprocessing to reduce subword splits."
        )

    if fragmentation.fragmented_words:
        top_word = fragmentation.fragmented_words[0]
        recommendations.append(
            f"Word '{top_word['word']}' splits into {top_word['tokens']} tokens. "
            "Consider adding domain tokens."
        )

    if reasoning_steps and reasoning_steps.mean_tokens > 10:
        recommendations.append(
            f"Reasoning steps average {reasoning_steps.mean_tokens:.1f} tokens. "
            "Consider structure-aware tokenization."
        )

    if equations and equations.mean_tokens > 15:
        recommendations.append(
            f"Equations average {equations.mean_tokens:.1f} tokens. Consider numeric normalization."
        )

    if tool_calls and tool_calls.mean_tokens > 20:
        recommendations.append(
            f"Tool calls average {tool_calls.mean_tokens:.1f} tokens. "
            "Consider structure token injection."
        )

    if sample_stats.p95 > sample_stats.mean * 3:
        recommendations.append(
            f"High variance: p95 ({sample_stats.p95:.0f}) is 3x+ mean ({sample_stats.mean:.0f}). "
            "Consider length-based batching."
        )

    return EfficiencyReport(
        sample_stats=sample_stats,
        reasoning_steps=reasoning_steps,
        equations=equations,
        tool_calls=tool_calls,
        fragmentation=fragmentation,
        efficiency_score=score,
        recommendations=recommendations,
    )
