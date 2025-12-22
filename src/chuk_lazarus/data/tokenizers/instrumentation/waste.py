"""
Token waste metrics for training efficiency analysis.

Measures:
- Padding waste: tokens spent on padding
- Truncation loss: content lost to truncation
- Attention waste: inefficient attention patterns
"""

from typing import Protocol

from pydantic import BaseModel, Field


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...

    @property
    def pad_token_id(self) -> int | None: ...


class PaddingStats(BaseModel):
    """Statistics about padding waste."""

    total_samples: int = Field(description="Total samples in batch")
    total_positions: int = Field(description="Total sequence positions (samples * max_len)")
    total_content_tokens: int = Field(description="Actual content tokens")
    total_padding_tokens: int = Field(description="Padding tokens")

    padding_rate: float = Field(description="Fraction of positions that are padding")
    efficiency: float = Field(description="Content efficiency (1 - padding_rate)")

    # Per-sample stats
    mean_padding_per_sample: float = Field(description="Average padding per sample")
    max_padding: int = Field(description="Maximum padding in a single sample")
    min_padding: int = Field(description="Minimum padding in a single sample")

    # Cost estimates
    wasted_compute_factor: float = Field(
        description="Factor of compute wasted on padding (approximate)"
    )


class TruncationStats(BaseModel):
    """Statistics about truncation loss."""

    total_samples: int = Field(description="Total samples")
    truncated_samples: int = Field(description="Samples that were truncated")
    truncation_rate: float = Field(description="Fraction of samples truncated")

    total_tokens_lost: int = Field(description="Total tokens lost to truncation")
    mean_tokens_lost: float = Field(description="Mean tokens lost per truncated sample")
    max_tokens_lost: int = Field(description="Maximum tokens lost in single sample")

    # Content loss estimates
    content_loss_rate: float = Field(description="Fraction of content lost overall")

    # Samples by severity
    minor_truncation: int = Field(description="Samples with <10% truncated")
    major_truncation: int = Field(description="Samples with 10-50% truncated")
    severe_truncation: int = Field(description="Samples with >50% truncated")


class WasteReport(BaseModel):
    """Combined waste analysis report."""

    # Input parameters
    max_length: int = Field(description="Maximum sequence length used")
    total_samples: int = Field(description="Total samples analyzed")

    # Padding analysis
    padding: PaddingStats = Field(description="Padding waste statistics")

    # Truncation analysis
    truncation: TruncationStats = Field(description="Truncation loss statistics")

    # Combined efficiency
    overall_efficiency: float = Field(
        description="Overall token efficiency (content / total possible)"
    )

    # Recommendations
    recommendations: list[str] = Field(
        default_factory=list, description="Suggestions for improvement"
    )


def analyze_padding_waste(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    max_length: int,
) -> PaddingStats:
    """
    Analyze padding waste for a batch configuration.

    Args:
        texts: List of text samples
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        PaddingStats with detailed padding analysis
    """
    if not texts:
        return PaddingStats(
            total_samples=0,
            total_positions=0,
            total_content_tokens=0,
            total_padding_tokens=0,
            padding_rate=0,
            efficiency=1,
            mean_padding_per_sample=0,
            max_padding=0,
            min_padding=0,
            wasted_compute_factor=0,
        )

    # Compute lengths
    lengths = [
        min(len(tokenizer.encode(text, add_special_tokens=True)), max_length) for text in texts
    ]

    total_samples = len(texts)
    total_positions = total_samples * max_length
    total_content = sum(lengths)
    total_padding = total_positions - total_content

    padding_per_sample = [max_length - length for length in lengths]

    padding_rate = total_padding / total_positions if total_positions > 0 else 0

    # Compute wasted factor (padding still requires some compute in attention)
    # Approximate: attention is O(n^2), so padding contributes to waste
    wasted_compute = padding_rate * 0.5  # Simplified estimate

    return PaddingStats(
        total_samples=total_samples,
        total_positions=total_positions,
        total_content_tokens=total_content,
        total_padding_tokens=total_padding,
        padding_rate=padding_rate,
        efficiency=1 - padding_rate,
        mean_padding_per_sample=total_padding / total_samples,
        max_padding=max(padding_per_sample),
        min_padding=min(padding_per_sample),
        wasted_compute_factor=wasted_compute,
    )


def analyze_truncation_loss(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    max_length: int,
) -> TruncationStats:
    """
    Analyze content loss due to truncation.

    Args:
        texts: List of text samples
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        TruncationStats with detailed truncation analysis
    """
    if not texts:
        return TruncationStats(
            total_samples=0,
            truncated_samples=0,
            truncation_rate=0,
            total_tokens_lost=0,
            mean_tokens_lost=0,
            max_tokens_lost=0,
            content_loss_rate=0,
            minor_truncation=0,
            major_truncation=0,
            severe_truncation=0,
        )

    # Compute full lengths (before truncation)
    full_lengths = [len(tokenizer.encode(text, add_special_tokens=True)) for text in texts]

    total_samples = len(texts)
    total_original = sum(full_lengths)

    # Compute truncation
    tokens_lost = [max(0, length - max_length) for length in full_lengths]
    truncated_samples = sum(1 for lost in tokens_lost if lost > 0)
    total_lost = sum(tokens_lost)

    # Severity categories
    minor = 0
    major = 0
    severe = 0

    for full_len, lost in zip(full_lengths, tokens_lost):
        if lost == 0:
            continue
        loss_rate = lost / full_len
        if loss_rate < 0.1:
            minor += 1
        elif loss_rate < 0.5:
            major += 1
        else:
            severe += 1

    return TruncationStats(
        total_samples=total_samples,
        truncated_samples=truncated_samples,
        truncation_rate=truncated_samples / total_samples if total_samples > 0 else 0,
        total_tokens_lost=total_lost,
        mean_tokens_lost=total_lost / truncated_samples if truncated_samples > 0 else 0,
        max_tokens_lost=max(tokens_lost) if tokens_lost else 0,
        content_loss_rate=total_lost / total_original if total_original > 0 else 0,
        minor_truncation=minor,
        major_truncation=major,
        severe_truncation=severe,
    )


def analyze_waste(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    max_length: int,
) -> WasteReport:
    """
    Comprehensive waste analysis combining padding and truncation.

    Args:
        texts: List of text samples
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        WasteReport with full analysis and recommendations
    """
    padding = analyze_padding_waste(texts, tokenizer, max_length)
    truncation = analyze_truncation_loss(texts, tokenizer, max_length)

    # Overall efficiency: content tokens / (content + padding + lost)
    total_possible = padding.total_positions + truncation.total_tokens_lost
    overall_efficiency = padding.total_content_tokens / total_possible if total_possible > 0 else 1

    # Generate recommendations
    recommendations: list[str] = []

    if padding.padding_rate > 0.3:
        recommendations.append(
            f"High padding rate ({padding.padding_rate:.1%}). "
            f"Consider sequence packing or shorter max_length."
        )

    if truncation.truncation_rate > 0.1:
        recommendations.append(
            f"Significant truncation ({truncation.truncation_rate:.1%} of samples). "
            f"Consider increasing max_length or chunking."
        )

    if truncation.severe_truncation > 0 and truncation.total_samples > 0:
        pct = truncation.severe_truncation / truncation.total_samples * 100
        recommendations.append(
            f"{truncation.severe_truncation} samples ({pct:.1f}%) lose >50% of content. "
            f"Review data preprocessing."
        )

    if padding.efficiency < 0.5:
        recommendations.append(
            f"Low efficiency ({padding.efficiency:.1%}). "
            f"Sequence packing could improve throughput by ~{1 / padding.efficiency:.1f}x."
        )

    # Compute optimal length only if we have samples
    if padding.total_samples > 0:
        optimal_length = int(padding.total_content_tokens / padding.total_samples * 1.2)
    else:
        optimal_length = max_length
    # Round to common value
    for candidate in [128, 256, 512, 1024, 2048, 4096]:
        if candidate >= optimal_length:
            optimal_length = candidate
            break

    if optimal_length < max_length * 0.7:
        recommendations.append(f"Consider max_length={optimal_length} (currently {max_length}).")

    return WasteReport(
        max_length=max_length,
        total_samples=len(texts),
        padding=padding,
        truncation=truncation,
        overall_efficiency=overall_efficiency,
        recommendations=recommendations,
    )
