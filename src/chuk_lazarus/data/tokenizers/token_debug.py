"""Token debugging and visualization utilities with Pydantic models."""

from typing import Protocol

from pydantic import BaseModel, Field


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def get_vocab(self) -> dict[str, int]: ...


class TokenInfo(BaseModel):
    """Information about a single token."""

    token_id: int = Field(description="Token ID")
    token_str: str = Field(description="Token string representation")
    byte_repr: str = Field(description="Byte representation")
    char_count: int = Field(ge=0, description="Number of characters")
    byte_count: int = Field(ge=0, description="Number of bytes (UTF-8)")


class TokenComparison(BaseModel):
    """Comparison of tokenizations from two tokenizers."""

    text: str = Field(description="Original text")
    tokenizer1_ids: list[int] = Field(description="Token IDs from first tokenizer")
    tokenizer2_ids: list[int] = Field(description="Token IDs from second tokenizer")
    tokenizer1_tokens: list[str] = Field(description="Decoded tokens from first tokenizer")
    tokenizer2_tokens: list[str] = Field(description="Decoded tokens from second tokenizer")
    tokenizer1_count: int = Field(ge=0, description="Token count from first tokenizer")
    tokenizer2_count: int = Field(ge=0, description="Token count from second tokenizer")


class UnknownTokenAnalysis(BaseModel):
    """Analysis of unknown tokens in text."""

    text: str = Field(description="Original text")
    unknown_count: int = Field(ge=0, description="Number of unknown tokens")
    total_count: int = Field(ge=0, description="Total token count")
    unknown_ratio: float = Field(ge=0.0, le=1.0, description="Ratio of unknown tokens")
    unknown_positions: list[int] = Field(description="Positions of unknown tokens")
    unknown_segments: list[str] = Field(description="Text segments that are unknown")


def get_token_info(
    token_id: int,
    tokenizer: TokenizerProtocol,
) -> TokenInfo:
    """
    Get detailed information about a single token.

    Args:
        token_id: Token ID to analyze
        tokenizer: Tokenizer instance

    Returns:
        TokenInfo with details about the token
    """
    try:
        token_str = tokenizer.decode([token_id])
    except Exception:
        token_str = f"<decode_error:{token_id}>"

    byte_repr = " ".join(f"{b:02x}" for b in token_str.encode("utf-8"))

    return TokenInfo(
        token_id=token_id,
        token_str=token_str,
        byte_repr=byte_repr,
        char_count=len(token_str),
        byte_count=len(token_str.encode("utf-8")),
    )


def get_tokens_info(
    token_ids: list[int],
    tokenizer: TokenizerProtocol,
) -> list[TokenInfo]:
    """
    Get detailed information about multiple tokens.

    Args:
        token_ids: List of token IDs
        tokenizer: Tokenizer instance

    Returns:
        List of TokenInfo for each token
    """
    return [get_token_info(tid, tokenizer) for tid in token_ids]


def compare_tokenizations(
    text: str,
    tokenizer1: TokenizerProtocol,
    tokenizer2: TokenizerProtocol,
    add_special_tokens: bool = False,
) -> TokenComparison:
    """
    Compare how two tokenizers tokenize the same text.

    Args:
        text: Text to tokenize
        tokenizer1: First tokenizer
        tokenizer2: Second tokenizer
        add_special_tokens: Whether to add special tokens

    Returns:
        TokenComparison with results from both tokenizers
    """
    ids1 = tokenizer1.encode(text, add_special_tokens=add_special_tokens)
    ids2 = tokenizer2.encode(text, add_special_tokens=add_special_tokens)

    tokens1 = [tokenizer1.decode([tid]) for tid in ids1]
    tokens2 = [tokenizer2.decode([tid]) for tid in ids2]

    return TokenComparison(
        text=text,
        tokenizer1_ids=ids1,
        tokenizer2_ids=ids2,
        tokenizer1_tokens=tokens1,
        tokenizer2_tokens=tokens2,
        tokenizer1_count=len(ids1),
        tokenizer2_count=len(ids2),
    )


def analyze_unknown_tokens(
    text: str,
    tokenizer: TokenizerProtocol,
    unk_token_id: int | None = None,
) -> UnknownTokenAnalysis:
    """
    Analyze which parts of text map to unknown tokens.

    Args:
        text: Text to analyze
        tokenizer: Tokenizer instance
        unk_token_id: Unknown token ID (auto-detected if None)

    Returns:
        UnknownTokenAnalysis with details about unknown tokens
    """
    if unk_token_id is None:
        vocab = tokenizer.get_vocab()
        unk_token_id = vocab.get("<unk>", vocab.get("[UNK]", -1))

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    total = len(token_ids)

    unknown_positions = [i for i, tid in enumerate(token_ids) if tid == unk_token_id]
    unknown_count = len(unknown_positions)

    # Try to find original text segments that became unknown
    unknown_segments = []
    tokens = text.split()
    for i, token in enumerate(tokens):
        if i < total and token_ids[i] == unk_token_id:
            unknown_segments.append(token)

    return UnknownTokenAnalysis(
        text=text,
        unknown_count=unknown_count,
        total_count=total,
        unknown_ratio=unknown_count / total if total > 0 else 0.0,
        unknown_positions=unknown_positions,
        unknown_segments=unknown_segments,
    )


def highlight_tokens(
    text: str,
    tokenizer: TokenizerProtocol,
    separator: str = "|",
    add_special_tokens: bool = False,
) -> str:
    """
    Show token boundaries in text.

    Args:
        text: Text to tokenize and highlight
        tokenizer: Tokenizer instance
        separator: Character(s) to use as token separator
        add_special_tokens: Whether to include special tokens

    Returns:
        Text with token boundaries marked
    """
    token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    return separator.join(tokens)


def token_to_bytes(
    token_id: int,
    tokenizer: TokenizerProtocol,
) -> bytes:
    """
    Get the raw bytes for a token.

    Args:
        token_id: Token ID
        tokenizer: Tokenizer instance

    Returns:
        Raw bytes of the decoded token
    """
    token_str = tokenizer.decode([token_id])
    return token_str.encode("utf-8")


def format_token_table(
    token_ids: list[int],
    tokenizer: TokenizerProtocol,
) -> str:
    """
    Format tokens as a readable table.

    Args:
        token_ids: Token IDs to display
        tokenizer: Tokenizer instance

    Returns:
        Formatted table string
    """
    lines = ["Index | ID     | Token          | Bytes"]
    lines.append("-" * 50)

    for i, tid in enumerate(token_ids):
        info = get_token_info(tid, tokenizer)
        token_display = info.token_str[:14].ljust(14)
        byte_display = (
            info.byte_repr[:20] if len(info.byte_repr) <= 20 else info.byte_repr[:17] + "..."
        )
        lines.append(f"{i:5} | {tid:6} | {token_display} | {byte_display}")

    return "\n".join(lines)


def find_token_by_string(
    token_str: str,
    tokenizer: TokenizerProtocol,
) -> list[int]:
    """
    Find token IDs that match a given string.

    Args:
        token_str: String to search for
        tokenizer: Tokenizer instance

    Returns:
        List of token IDs whose decoded form contains the string
    """
    vocab = tokenizer.get_vocab()
    matching_ids = []

    for token, token_id in vocab.items():
        if token_str in token:
            matching_ids.append(token_id)

    return matching_ids


def get_similar_tokens(
    token_id: int,
    tokenizer: TokenizerProtocol,
    max_results: int = 10,
) -> list[TokenInfo]:
    """
    Find tokens similar to a given token (by string prefix).

    Args:
        token_id: Reference token ID
        tokenizer: Tokenizer instance
        max_results: Maximum number of results

    Returns:
        List of TokenInfo for similar tokens
    """
    reference = tokenizer.decode([token_id])
    if not reference:
        return []

    prefix = reference[:2] if len(reference) >= 2 else reference
    matching = find_token_by_string(prefix, tokenizer)

    # Exclude the original token
    matching = [tid for tid in matching if tid != token_id][:max_results]

    return get_tokens_info(matching, tokenizer)
