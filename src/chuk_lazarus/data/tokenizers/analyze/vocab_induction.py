"""
Vocabulary induction for domain-specific tokenization.

Automatically identifies high-impact tokens that cause waste:
- Strings that fragment into many tokens
- Frequently occurring patterns
- Domain-specific vocabulary candidates

Provides suggestions for vocabulary extension without full retraining.
"""

import re
from collections import Counter
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...
    def get_vocab(self) -> dict[str, int]: ...


class TokenDomain(str, Enum):
    """Domain categories for suggested tokens."""

    MATH = "math"
    CODE = "code"
    SCIENCE = "science"
    TOOL = "tool"
    GENERAL = "general"
    CUSTOM = "custom"


class InductionConfig(BaseModel):
    """Configuration for vocabulary induction."""

    # Minimum occurrences to consider
    min_frequency: int = Field(
        default=5, ge=1, description="Minimum occurrences to consider a candidate"
    )

    # Fragmentation threshold
    min_fragmentation: int = Field(
        default=3, ge=2, description="Minimum tokens a string must split into"
    )

    # Token savings threshold
    min_savings: int = Field(default=2, ge=1, description="Minimum token savings to suggest")

    # Maximum candidates to return
    max_candidates: int = Field(
        default=50, ge=1, description="Maximum number of candidates to return"
    )

    # Word length bounds
    min_word_length: int = Field(default=4, ge=1, description="Minimum word length to consider")
    max_word_length: int = Field(default=30, ge=1, description="Maximum word length to consider")

    # Include domain-specific patterns
    include_math_symbols: bool = Field(default=True, description="Include math symbol candidates")
    include_code_patterns: bool = Field(default=True, description="Include code pattern candidates")
    include_tool_patterns: bool = Field(
        default=True, description="Include tool/agent pattern candidates"
    )


class TokenCandidate(BaseModel):
    """A candidate token for vocabulary extension."""

    token_str: str = Field(description="The string to add as a token")
    frequency: int = Field(description="Number of occurrences in corpus")
    current_tokens: int = Field(description="Current number of tokens needed")
    savings_per_occurrence: int = Field(description="Tokens saved per occurrence")
    total_savings: int = Field(description="Total tokens saved across corpus")
    domain: TokenDomain = Field(description="Suggested domain category")
    priority_score: float = Field(description="Priority score for selection")
    examples: list[str] = Field(default_factory=list, description="Example contexts")


class InductionReport(BaseModel):
    """Report from vocabulary induction analysis."""

    total_candidates: int = Field(description="Total candidates identified")
    total_potential_savings: int = Field(description="Total tokens that could be saved")
    savings_percent: float = Field(description="Percentage of tokens that could be saved")
    candidates: list[TokenCandidate] = Field(description="Ranked token candidates")
    domain_breakdown: dict[str, int] = Field(
        default_factory=dict, description="Candidates by domain"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations for implementation"
    )


class DomainVocab(BaseModel):
    """Pre-defined domain vocabulary for common use cases."""

    domain: TokenDomain = Field(description="Domain name")
    tokens: list[str] = Field(description="Suggested tokens for this domain")
    description: str = Field(description="Domain description")


# Pre-defined domain vocabularies
MATH_VOCAB = DomainVocab(
    domain=TokenDomain.MATH,
    description="Mathematical symbols and operators",
    tokens=[
        "∑",  # Summation
        "∏",  # Product
        "∫",  # Integral
        "∂",  # Partial derivative
        "√",  # Square root
        "∞",  # Infinity
        "≤",  # Less than or equal
        "≥",  # Greater than or equal
        "≠",  # Not equal
        "≈",  # Approximately equal
        "∈",  # Element of
        "∉",  # Not element of
        "⊂",  # Subset
        "∪",  # Union
        "∩",  # Intersection
        "×",  # Multiplication
        "÷",  # Division
        "±",  # Plus minus
        "π",  # Pi
        "θ",  # Theta
        "σ",  # Sigma
        "μ",  # Mu
        "λ",  # Lambda
        "α",  # Alpha
        "β",  # Beta
        "γ",  # Gamma
        "δ",  # Delta
        "ε",  # Epsilon
    ],
)

CODE_VOCAB = DomainVocab(
    domain=TokenDomain.CODE,
    description="Common programming patterns",
    tokens=[
        "def ",
        "class ",
        "import ",
        "from ",
        "return ",
        "self.",
        "async ",
        "await ",
        "lambda ",
        "yield ",
        "raise ",
        "except ",
        "finally ",
        "assert ",
        "__init__",
        "__name__",
        "__main__",
        "None",
        "True",
        "False",
        "isinstance",
        "enumerate",
        "TypeError",
        "ValueError",
        "KeyError",
        "IndexError",
        "AttributeError",
    ],
)

TOOL_VOCAB = DomainVocab(
    domain=TokenDomain.TOOL,
    description="Tool/agent interaction patterns",
    tokens=[
        "<TOOL_CALL>",
        "</TOOL_CALL>",
        "<TOOL_RESULT>",
        "</TOOL_RESULT>",
        "<FUNCTION>",
        "</FUNCTION>",
        "<OBSERVATION>",
        "</OBSERVATION>",
        "<THOUGHT>",
        "</THOUGHT>",
        "<ACTION>",
        "</ACTION>",
        "Action:",
        "Observation:",
        "Thought:",
        "Final Answer:",
    ],
)


def _extract_words(text: str, min_len: int, max_len: int) -> list[str]:
    """Extract words from text within length bounds."""
    # Match word-like sequences
    pattern = r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"
    words = re.findall(pattern, text)
    return [w for w in words if min_len <= len(w) <= max_len]


def _extract_patterns(text: str) -> list[str]:
    """Extract common patterns like camelCase, snake_case, etc."""
    patterns: list[str] = []

    # CamelCase words
    camel = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", text)
    patterns.extend(camel)

    # snake_case words
    snake = re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", text)
    patterns.extend(snake)

    # SCREAMING_SNAKE_CASE
    screaming = re.findall(r"\b[A-Z]+(?:_[A-Z]+)+\b", text)
    patterns.extend(screaming)

    return patterns


def _categorize_domain(token_str: str) -> TokenDomain:
    """Categorize a token into a domain."""
    # Math symbols
    if any(c in token_str for c in "∑∏∫∂√∞≤≥≠≈∈∉⊂∪∩×÷±πθσμλαβγδε"):
        return TokenDomain.MATH

    # Code patterns
    if any(
        kw in token_str for kw in ["def ", "class ", "import ", "self.", "__", "Error", "Exception"]
    ):
        return TokenDomain.CODE

    # Tool patterns
    if any(
        pattern in token_str
        for pattern in ["<TOOL", "</TOOL", "Action:", "Observation:", "<FUNCTION"]
    ):
        return TokenDomain.TOOL

    # Science (Greek letters, units)
    if any(c in token_str for c in "αβγδεζηθικλμνξοπρστυφχψω"):
        return TokenDomain.SCIENCE

    return TokenDomain.GENERAL


def find_fragmented_words(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    config: InductionConfig | None = None,
) -> list[TokenCandidate]:
    """
    Find words that fragment into many tokens.

    Args:
        texts: Corpus texts to analyze
        tokenizer: Tokenizer to check fragmentation
        config: Induction configuration

    Returns:
        List of fragmented word candidates
    """
    if config is None:
        config = InductionConfig()

    word_counts: Counter[str] = Counter()
    word_tokens: dict[str, int] = {}

    for text in texts:
        words = _extract_words(text, config.min_word_length, config.max_word_length)
        patterns = _extract_patterns(text)
        all_candidates = words + patterns

        for word in all_candidates:
            word_counts[word] += 1

            if word not in word_tokens:
                tokens = tokenizer.encode(word, add_special_tokens=False)
                word_tokens[word] = len(tokens)

    candidates: list[TokenCandidate] = []

    for word, count in word_counts.items():
        token_count = word_tokens.get(word, 1)

        if count < config.min_frequency:
            continue
        if token_count < config.min_fragmentation:
            continue

        savings = token_count - 1  # Would be 1 token if added to vocab
        if savings < config.min_savings:
            continue

        total_savings = savings * count
        priority = total_savings * (token_count / len(word))  # Favor high fragmentation

        candidates.append(
            TokenCandidate(
                token_str=word,
                frequency=count,
                current_tokens=token_count,
                savings_per_occurrence=savings,
                total_savings=total_savings,
                domain=_categorize_domain(word),
                priority_score=priority,
            )
        )

    # Sort by priority
    candidates.sort(key=lambda c: -c.priority_score)
    return candidates[: config.max_candidates]


def find_frequent_ngrams(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    n_range: tuple[int, int] = (2, 4),
    min_frequency: int = 10,
    max_candidates: int = 20,
) -> list[TokenCandidate]:
    """
    Find frequent n-grams that could benefit from single tokens.

    Args:
        texts: Corpus texts to analyze
        tokenizer: Tokenizer to check
        n_range: Range of n-gram sizes (min, max)
        min_frequency: Minimum occurrences
        max_candidates: Maximum candidates to return

    Returns:
        List of n-gram candidates
    """
    ngram_counts: Counter[str] = Counter()

    for text in texts:
        words = text.split()
        for n in range(n_range[0], n_range[1] + 1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i : i + n])
                if len(ngram) <= 50:  # Reasonable length
                    ngram_counts[ngram] += 1

    candidates: list[TokenCandidate] = []

    for ngram, count in ngram_counts.most_common(max_candidates * 2):
        if count < min_frequency:
            continue

        tokens = tokenizer.encode(ngram, add_special_tokens=False)
        token_count = len(tokens)

        if token_count < 2:
            continue

        savings = token_count - 1
        total_savings = savings * count

        candidates.append(
            TokenCandidate(
                token_str=ngram,
                frequency=count,
                current_tokens=token_count,
                savings_per_occurrence=savings,
                total_savings=total_savings,
                domain=_categorize_domain(ngram),
                priority_score=total_savings,
            )
        )

    candidates.sort(key=lambda c: -c.priority_score)
    return candidates[:max_candidates]


def suggest_domain_tokens(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    domains: list[TokenDomain] | None = None,
) -> list[TokenCandidate]:
    """
    Suggest domain-specific tokens based on corpus content.

    Args:
        texts: Corpus texts to analyze
        tokenizer: Tokenizer to check
        domains: Domains to check (None = all)

    Returns:
        List of domain token candidates
    """
    if domains is None:
        domains = [TokenDomain.MATH, TokenDomain.CODE, TokenDomain.TOOL]

    domain_vocabs = {
        TokenDomain.MATH: MATH_VOCAB,
        TokenDomain.CODE: CODE_VOCAB,
        TokenDomain.TOOL: TOOL_VOCAB,
    }

    corpus_text = " ".join(texts)
    candidates: list[TokenCandidate] = []
    vocab = tokenizer.get_vocab()

    for domain in domains:
        if domain not in domain_vocabs:
            continue

        domain_vocab = domain_vocabs[domain]

        for token_str in domain_vocab.tokens:
            # Skip if already in vocab
            if token_str in vocab:
                continue

            # Count occurrences
            count = corpus_text.count(token_str)
            if count < 2:
                continue

            # Check token count
            tokens = tokenizer.encode(token_str, add_special_tokens=False)
            token_count = len(tokens)

            if token_count < 2:
                continue

            savings = token_count - 1
            total_savings = savings * count

            candidates.append(
                TokenCandidate(
                    token_str=token_str,
                    frequency=count,
                    current_tokens=token_count,
                    savings_per_occurrence=savings,
                    total_savings=total_savings,
                    domain=domain,
                    priority_score=total_savings,
                )
            )

    candidates.sort(key=lambda c: -c.priority_score)
    return candidates


def analyze_vocab_induction(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    config: InductionConfig | None = None,
) -> InductionReport:
    """
    Comprehensive vocabulary induction analysis.

    Args:
        texts: Corpus texts to analyze
        tokenizer: Tokenizer to use
        config: Induction configuration

    Returns:
        InductionReport with all candidates and recommendations
    """
    if config is None:
        config = InductionConfig()

    # Collect candidates from all sources
    all_candidates: list[TokenCandidate] = []

    # Fragmented words
    fragmented = find_fragmented_words(texts, tokenizer, config)
    all_candidates.extend(fragmented)

    # Frequent n-grams
    ngrams = find_frequent_ngrams(texts, tokenizer)
    all_candidates.extend(ngrams)

    # Domain tokens
    domains_to_check = []
    if config.include_math_symbols:
        domains_to_check.append(TokenDomain.MATH)
    if config.include_code_patterns:
        domains_to_check.append(TokenDomain.CODE)
    if config.include_tool_patterns:
        domains_to_check.append(TokenDomain.TOOL)

    if domains_to_check:
        domain_tokens = suggest_domain_tokens(texts, tokenizer, domains_to_check)
        all_candidates.extend(domain_tokens)

    # Deduplicate by token_str
    seen: set[str] = set()
    unique_candidates: list[TokenCandidate] = []
    for c in all_candidates:
        if c.token_str not in seen:
            seen.add(c.token_str)
            unique_candidates.append(c)

    # Sort by priority
    unique_candidates.sort(key=lambda c: -c.priority_score)
    top_candidates = unique_candidates[: config.max_candidates]

    # Calculate totals
    total_savings = sum(c.total_savings for c in top_candidates)

    # Calculate total tokens in corpus for percentage
    total_tokens = sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in texts)
    savings_percent = (total_savings / total_tokens * 100) if total_tokens > 0 else 0

    # Domain breakdown
    domain_counts: dict[str, int] = {}
    for c in top_candidates:
        domain_counts[c.domain.value] = domain_counts.get(c.domain.value, 0) + 1

    # Generate recommendations
    recommendations: list[str] = []

    if total_savings > 0:
        recommendations.append(
            f"Adding top {len(top_candidates)} tokens could save "
            f"{total_savings:,} tokens ({savings_percent:.1f}% of corpus)."
        )

    if domain_counts.get("math", 0) > 5:
        recommendations.append("Consider adding math domain tokens for better math representation.")

    if domain_counts.get("code", 0) > 5:
        recommendations.append("Consider adding code domain tokens for better code representation.")

    if domain_counts.get("tool", 0) > 3:
        recommendations.append("Consider adding tool/agent tokens for structured interactions.")

    high_impact = [c for c in top_candidates if c.total_savings > 100]
    if high_impact:
        recommendations.append(
            f"Found {len(high_impact)} high-impact candidates (>100 tokens saved each)."
        )

    return InductionReport(
        total_candidates=len(top_candidates),
        total_potential_savings=total_savings,
        savings_percent=savings_percent,
        candidates=top_candidates,
        domain_breakdown=domain_counts,
        recommendations=recommendations,
    )


def get_domain_vocab(domain: TokenDomain) -> DomainVocab | None:
    """
    Get pre-defined vocabulary for a domain.

    Args:
        domain: Domain to get vocabulary for

    Returns:
        DomainVocab or None if not available
    """
    vocabs = {
        TokenDomain.MATH: MATH_VOCAB,
        TokenDomain.CODE: CODE_VOCAB,
        TokenDomain.TOOL: TOOL_VOCAB,
    }
    return vocabs.get(domain)


def list_domain_vocabs() -> list[DomainVocab]:
    """
    List all available domain vocabularies.

    Returns:
        List of available DomainVocab
    """
    return [MATH_VOCAB, CODE_VOCAB, TOOL_VOCAB]
