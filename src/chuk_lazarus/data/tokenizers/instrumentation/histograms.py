"""
Token length histograms for dataset analysis.

Provides visual and statistical views of token length distributions
to inform training decisions (batch sizes, sequence lengths, etc).
"""

from typing import Protocol

from pydantic import BaseModel, Field


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...


class HistogramBin(BaseModel):
    """A single bin in a histogram."""

    min_value: int = Field(description="Minimum value (inclusive)")
    max_value: int = Field(description="Maximum value (exclusive)")
    count: int = Field(description="Number of samples in bin")
    percentage: float = Field(description="Percentage of total samples")


class PercentileStats(BaseModel):
    """Percentile statistics for a distribution."""

    p10: float = Field(description="10th percentile")
    p25: float = Field(description="25th percentile (Q1)")
    p50: float = Field(description="50th percentile (median)")
    p75: float = Field(description="75th percentile (Q3)")
    p90: float = Field(description="90th percentile")
    p95: float = Field(description="95th percentile")
    p99: float = Field(description="99th percentile")


class LengthHistogram(BaseModel):
    """Complete histogram with statistics."""

    # Basic stats
    total_samples: int = Field(description="Total number of samples")
    total_tokens: int = Field(description="Total tokens across all samples")
    min_length: int = Field(description="Minimum token length")
    max_length: int = Field(description="Maximum token length")
    mean_length: float = Field(description="Mean token length")
    std_length: float = Field(description="Standard deviation of lengths")

    # Percentiles
    percentiles: PercentileStats = Field(description="Percentile statistics")

    # Histogram bins
    bins: list[HistogramBin] = Field(description="Histogram bins")
    bin_width: int = Field(description="Width of each bin")

    # Training recommendations
    recommended_max_length: int = Field(description="Recommended max_length (covers 95% of data)")
    samples_over_2048: int = Field(description="Samples exceeding 2048 tokens")
    samples_over_4096: int = Field(description="Samples exceeding 4096 tokens")


def compute_percentiles(lengths: list[int]) -> PercentileStats:
    """
    Compute percentile statistics for token lengths.

    Args:
        lengths: List of token lengths

    Returns:
        PercentileStats with key percentiles
    """
    if not lengths:
        return PercentileStats(p10=0, p25=0, p50=0, p75=0, p90=0, p95=0, p99=0)

    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)

    def percentile(p: float) -> float:
        # Use (p/100) * (n-1) for linear interpolation matching numpy's default
        idx = int(p * (n - 1) / 100)
        idx = min(idx, n - 1)
        return float(sorted_lengths[idx])

    return PercentileStats(
        p10=percentile(10),
        p25=percentile(25),
        p50=percentile(50),
        p75=percentile(75),
        p90=percentile(90),
        p95=percentile(95),
        p99=percentile(99),
    )


def compute_length_histogram(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    num_bins: int = 20,
    max_display_length: int | None = None,
) -> LengthHistogram:
    """
    Compute token length histogram for a dataset.

    Args:
        texts: List of text samples
        tokenizer: Tokenizer to use
        num_bins: Number of histogram bins
        max_display_length: Maximum length for histogram (None = auto)

    Returns:
        LengthHistogram with full statistics
    """
    if not texts:
        return LengthHistogram(
            total_samples=0,
            total_tokens=0,
            min_length=0,
            max_length=0,
            mean_length=0,
            std_length=0,
            percentiles=PercentileStats(p10=0, p25=0, p50=0, p75=0, p90=0, p95=0, p99=0),
            bins=[],
            bin_width=0,
            recommended_max_length=512,
            samples_over_2048=0,
            samples_over_4096=0,
        )

    # Compute all lengths
    lengths = [len(tokenizer.encode(text, add_special_tokens=True)) for text in texts]

    # Basic stats
    total_samples = len(lengths)
    total_tokens = sum(lengths)
    min_length = min(lengths)
    max_length = max(lengths)
    mean_length = total_tokens / total_samples

    # Standard deviation
    variance = sum((length - mean_length) ** 2 for length in lengths) / total_samples
    std_length = variance**0.5

    # Percentiles
    percentiles = compute_percentiles(lengths)

    # Determine bin range
    if max_display_length is None:
        # Use p99 + some margin, capped at actual max
        max_display_length = min(int(percentiles.p99 * 1.2), max_length)
    max_display_length = max(max_display_length, min_length + 1)

    # Compute bins
    bin_width = max(1, (max_display_length - min_length) // num_bins)
    bins: list[HistogramBin] = []

    for i in range(num_bins):
        bin_min = min_length + i * bin_width
        bin_max = bin_min + bin_width
        if i == num_bins - 1:
            # Last bin includes everything above
            count = sum(1 for length in lengths if length >= bin_min)
        else:
            count = sum(1 for length in lengths if bin_min <= length < bin_max)

        bins.append(
            HistogramBin(
                min_value=bin_min,
                max_value=bin_max,
                count=count,
                percentage=count / total_samples * 100,
            )
        )

    # Training recommendations
    recommended_max_length = int(percentiles.p95)
    # Round up to power of 2 or common value
    for candidate in [128, 256, 512, 1024, 2048, 4096, 8192]:
        if candidate >= recommended_max_length:
            recommended_max_length = candidate
            break

    samples_over_2048 = sum(1 for length in lengths if length > 2048)
    samples_over_4096 = sum(1 for length in lengths if length > 4096)

    return LengthHistogram(
        total_samples=total_samples,
        total_tokens=total_tokens,
        min_length=min_length,
        max_length=max_length,
        mean_length=mean_length,
        std_length=std_length,
        percentiles=percentiles,
        bins=bins,
        bin_width=bin_width,
        recommended_max_length=recommended_max_length,
        samples_over_2048=samples_over_2048,
        samples_over_4096=samples_over_4096,
    )


def get_length_stats(
    texts: list[str],
    tokenizer: TokenizerProtocol,
) -> dict[str, float | int]:
    """
    Get quick length statistics without full histogram.

    Args:
        texts: List of text samples
        tokenizer: Tokenizer to use

    Returns:
        Dictionary of key statistics
    """
    if not texts:
        return {
            "total_samples": 0,
            "total_tokens": 0,
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "p95": 0,
        }

    lengths = [len(tokenizer.encode(text, add_special_tokens=True)) for text in texts]
    sorted_lengths = sorted(lengths)
    n = len(lengths)

    return {
        "total_samples": n,
        "total_tokens": sum(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / n,
        "median": sorted_lengths[n // 2],
        "p95": sorted_lengths[int(0.95 * n)],
    }


def format_histogram_ascii(
    histogram: LengthHistogram,
    width: int = 50,
    show_percentiles: bool = True,
) -> str:
    """
    Format histogram as ASCII art for terminal display.

    Args:
        histogram: LengthHistogram to format
        width: Width of the bar chart
        show_percentiles: Whether to show percentile markers

    Returns:
        Formatted string for terminal output
    """
    lines: list[str] = []

    # Header
    lines.append("=" * 60)
    lines.append("TOKEN LENGTH HISTOGRAM")
    lines.append("=" * 60)

    # Basic stats
    lines.append(f"Samples: {histogram.total_samples:,}")
    lines.append(f"Total tokens: {histogram.total_tokens:,}")
    lines.append(f"Length range: {histogram.min_length} - {histogram.max_length}")
    lines.append(f"Mean: {histogram.mean_length:.1f} (std: {histogram.std_length:.1f})")
    lines.append("")

    # Percentiles
    if show_percentiles:
        p = histogram.percentiles
        lines.append(
            f"Percentiles: p10={p.p10:.0f} p25={p.p25:.0f} p50={p.p50:.0f} "
            f"p75={p.p75:.0f} p90={p.p90:.0f} p95={p.p95:.0f} p99={p.p99:.0f}"
        )
        lines.append("")

    # Histogram bars
    max_count = max(b.count for b in histogram.bins) if histogram.bins else 1
    for bin in histogram.bins:
        bar_len = int(bin.count / max_count * width) if max_count > 0 else 0
        bar = "█" * bar_len
        label = f"{bin.min_value:5d}-{bin.max_value:5d}"
        count_str = f"{bin.count:6d} ({bin.percentage:5.1f}%)"
        lines.append(f"{label} │{bar:<{width}} {count_str}")

    lines.append("")

    # Recommendations
    lines.append("--- Training Recommendations ---")
    lines.append(f"Recommended max_length: {histogram.recommended_max_length}")
    if histogram.samples_over_2048 > 0:
        pct = histogram.samples_over_2048 / histogram.total_samples * 100
        lines.append(f"Samples > 2048: {histogram.samples_over_2048:,} ({pct:.1f}%)")
    if histogram.samples_over_4096 > 0:
        pct = histogram.samples_over_4096 / histogram.total_samples * 100
        lines.append(f"Samples > 4096: {histogram.samples_over_4096:,} ({pct:.1f}%)")

    return "\n".join(lines)
