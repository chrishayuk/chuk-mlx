"""
Batching analysis tools.

This module provides:
- LengthHistogram: Distribution of sequence lengths
- BucketAnalysis: Per-bucket efficiency metrics
- BatchingEfficiencyReport: Complete efficiency report
- suggest_bucket_edges: Optimal bucket edge suggestions
- analyze_padding_waste: Per-batch padding analysis
- visualize_buckets: ASCII visualization of bucket distribution

Mirrors the analysis capabilities in tokenizers/analyze/ and tokenizers/instrumentation/.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from ..core.buckets import BucketSpec


class HistogramBin(BaseModel):
    """A single histogram bin."""

    model_config = ConfigDict(frozen=True)

    min_length: int = Field(description="Minimum length (inclusive)")
    max_length: int = Field(description="Maximum length (exclusive)")
    count: int = Field(description="Number of samples in bin")
    percentage: float = Field(description="Percentage of total samples")

    @property
    def label(self) -> str:
        """Human-readable bin label."""
        return f"{self.min_length}-{self.max_length - 1}"


class LengthHistogram(BaseModel):
    """
    Distribution of sequence lengths.

    Provides insights into the length distribution of a dataset,
    useful for choosing bucket edges.
    """

    model_config = ConfigDict(frozen=True)

    bins: tuple[HistogramBin, ...] = Field(description="Histogram bins")
    total_samples: int = Field(description="Total number of samples")
    min_length: int = Field(description="Minimum sequence length")
    max_length: int = Field(description="Maximum sequence length")
    mean_length: float = Field(description="Mean sequence length")
    median_length: int = Field(description="Median sequence length")
    std_length: float = Field(description="Standard deviation of lengths")

    # Percentiles for bucket edge suggestions
    p25: int = Field(description="25th percentile")
    p50: int = Field(description="50th percentile (median)")
    p75: int = Field(description="75th percentile")
    p90: int = Field(description="90th percentile")
    p95: int = Field(description="95th percentile")
    p99: int = Field(description="99th percentile")

    def to_ascii(self, width: int = 50) -> str:
        """
        Create ASCII histogram visualization.

        Args:
            width: Maximum bar width in characters

        Returns:
            Multi-line string with ASCII histogram
        """
        if not self.bins:
            return "No data"

        max_count = max(b.count for b in self.bins)
        lines = ["Length Distribution", "=" * (width + 20)]

        for b in self.bins:
            bar_len = int((b.count / max_count) * width) if max_count > 0 else 0
            bar = "█" * bar_len
            lines.append(f"{b.label:>12s} | {bar:<{width}s} {b.count:>6d} ({b.percentage:5.1f}%)")

        lines.append("=" * (width + 20))
        lines.append(f"Total: {self.total_samples:,} samples")
        lines.append(f"Range: {self.min_length} - {self.max_length}")
        lines.append(f"Mean:  {self.mean_length:.1f} ± {self.std_length:.1f}")

        return "\n".join(lines)


class BucketEfficiency(BaseModel):
    """Efficiency metrics for a single bucket."""

    model_config = ConfigDict(frozen=True)

    bucket_id: int = Field(description="Bucket ID")
    min_length: int = Field(description="Bucket minimum length")
    max_length: int = Field(description="Bucket maximum length")
    sample_count: int = Field(description="Number of samples in bucket")
    total_tokens: int = Field(description="Total tokens (actual content)")
    padded_tokens: int = Field(description="Total tokens after padding")
    efficiency: float = Field(description="Token efficiency (content / padded)")
    waste_percentage: float = Field(description="Padding waste percentage")

    @property
    def waste_tokens(self) -> int:
        """Number of wasted padding tokens."""
        return self.padded_tokens - self.total_tokens


class BucketAnalysis(BaseModel):
    """
    Per-bucket efficiency analysis.

    Shows how efficiently each bucket is being utilized.
    """

    model_config = ConfigDict(frozen=True)

    bucket_spec: BucketSpec = Field(description="Bucket configuration")
    buckets: tuple[BucketEfficiency, ...] = Field(description="Per-bucket metrics")
    overall_efficiency: float = Field(description="Overall token efficiency")
    overall_waste: float = Field(description="Overall waste percentage")
    empty_buckets: int = Field(description="Number of empty buckets")

    def to_ascii(self) -> str:
        """Create ASCII table of bucket efficiencies."""
        lines = ["Bucket Efficiency Analysis", "=" * 70]
        lines.append(
            f"{'Bucket':>8s} {'Range':>15s} {'Samples':>10s} {'Efficiency':>12s} {'Waste':>10s}"
        )
        lines.append("-" * 70)

        for b in self.buckets:
            if b.sample_count > 0:
                range_str = f"{b.min_length}-{b.max_length}"
                lines.append(
                    f"{b.bucket_id:>8d} {range_str:>15s} {b.sample_count:>10,d} "
                    f"{b.efficiency:>11.1%} {b.waste_percentage:>8.1f}%"
                )

        lines.append("-" * 70)
        lines.append(
            f"{'Overall':>8s} {'':>15s} {'':>10s} "
            f"{self.overall_efficiency:>11.1%} {self.overall_waste:>8.1f}%"
        )
        lines.append(f"\nEmpty buckets: {self.empty_buckets}")

        return "\n".join(lines)


class OptimizationGoal(str, Enum):
    """Goal for bucket edge optimization."""

    MINIMIZE_WASTE = "minimize_waste"
    BALANCE_BUCKETS = "balance_buckets"
    MINIMIZE_MEMORY = "minimize_memory"


class BucketSuggestion(BaseModel):
    """Suggested bucket edges with rationale."""

    model_config = ConfigDict(frozen=True)

    edges: tuple[int, ...] = Field(description="Suggested bucket edges")
    overflow_max: int = Field(description="Suggested overflow max")
    rationale: str = Field(description="Explanation for suggestion")
    estimated_efficiency: float = Field(description="Estimated efficiency with these edges")
    optimization_goal: OptimizationGoal = Field(description="Goal used for optimization")


class BatchingEfficiencyReport(BaseModel):
    """
    Complete efficiency report for a batching configuration.

    Combines length histogram, bucket analysis, and recommendations.
    """

    model_config = ConfigDict(frozen=True)

    length_histogram: LengthHistogram = Field(description="Length distribution")
    bucket_analysis: BucketAnalysis = Field(description="Per-bucket efficiency")
    suggestions: tuple[BucketSuggestion, ...] = Field(description="Optimization suggestions")
    recommendations: tuple[str, ...] = Field(description="Text recommendations")

    # Summary metrics
    total_samples: int = Field(description="Total samples analyzed")
    total_tokens: int = Field(description="Total content tokens")
    total_padded: int = Field(description="Total tokens with padding")
    overall_efficiency: float = Field(description="Overall efficiency")

    def to_ascii(self) -> str:
        """Create complete ASCII report."""
        sections = [
            "=" * 70,
            "BATCHING EFFICIENCY REPORT",
            "=" * 70,
            "",
            self.length_histogram.to_ascii(),
            "",
            self.bucket_analysis.to_ascii(),
            "",
            "--- Recommendations ---",
        ]

        for rec in self.recommendations:
            sections.append(f"  • {rec}")

        if self.suggestions:
            sections.append("")
            sections.append("--- Suggested Bucket Edges ---")
            for sug in self.suggestions:
                sections.append(f"  Goal: {sug.optimization_goal.value}")
                sections.append(f"  Edges: {sug.edges}")
                sections.append(f"  Overflow: {sug.overflow_max}")
                sections.append(f"  Est. efficiency: {sug.estimated_efficiency:.1%}")
                sections.append(f"  Rationale: {sug.rationale}")
                sections.append("")

        sections.append("=" * 70)
        sections.append(
            f"Summary: {self.total_samples:,} samples, "
            f"{self.overall_efficiency:.1%} efficiency, "
            f"{100 - self.overall_efficiency * 100:.1f}% waste"
        )

        return "\n".join(sections)


def compute_length_histogram(
    lengths: Mapping[str, int],
    num_bins: int = 10,
    bin_width: int | None = None,
) -> LengthHistogram:
    """
    Compute length histogram for a dataset.

    Args:
        lengths: Dict mapping sample_id -> length
        num_bins: Number of histogram bins (if bin_width not specified)
        bin_width: Fixed bin width (overrides num_bins)

    Returns:
        LengthHistogram with distribution analysis
    """
    if not lengths:
        return LengthHistogram(
            bins=(),
            total_samples=0,
            min_length=0,
            max_length=0,
            mean_length=0.0,
            median_length=0,
            std_length=0.0,
            p25=0,
            p50=0,
            p75=0,
            p90=0,
            p95=0,
            p99=0,
        )

    values = sorted(lengths.values())
    n = len(values)

    # Basic stats
    min_len = values[0]
    max_len = values[-1]
    mean_len = sum(values) / n
    median_len = values[n // 2]

    # Standard deviation
    variance = sum((v - mean_len) ** 2 for v in values) / n
    std_len = math.sqrt(variance)

    # Percentiles
    def percentile(p: float) -> int:
        idx = int(n * p / 100)
        return values[min(idx, n - 1)]

    # Compute bins
    if bin_width is None:
        range_size = max_len - min_len + 1
        bin_width = max(1, range_size // num_bins)

    bins = []
    current = min_len
    while current <= max_len:
        bin_max = current + bin_width
        count = sum(1 for v in values if current <= v < bin_max)
        bins.append(
            HistogramBin(
                min_length=current,
                max_length=bin_max,
                count=count,
                percentage=(count / n) * 100,
            )
        )
        current = bin_max

    return LengthHistogram(
        bins=tuple(bins),
        total_samples=n,
        min_length=min_len,
        max_length=max_len,
        mean_length=mean_len,
        median_length=median_len,
        std_length=std_len,
        p25=percentile(25),
        p50=percentile(50),
        p75=percentile(75),
        p90=percentile(90),
        p95=percentile(95),
        p99=percentile(99),
    )


def analyze_bucket_efficiency(
    lengths: Mapping[str, int],
    bucket_spec: BucketSpec,
) -> BucketAnalysis:
    """
    Analyze efficiency of a bucket configuration.

    Args:
        lengths: Dict mapping sample_id -> length
        bucket_spec: Bucket configuration to analyze

    Returns:
        BucketAnalysis with per-bucket metrics
    """
    if not lengths:
        return BucketAnalysis(
            bucket_spec=bucket_spec,
            buckets=(),
            overall_efficiency=0.0,
            overall_waste=0.0,
            empty_buckets=len(bucket_spec.edges) + 1,
        )

    # Group by bucket
    bucket_lengths: dict[int, list[int]] = {}
    for length in lengths.values():
        bucket_id = bucket_spec.get_bucket_id(length)
        if bucket_id not in bucket_lengths:
            bucket_lengths[bucket_id] = []
        bucket_lengths[bucket_id].append(length)

    # Analyze each bucket
    bucket_efficiencies = []
    total_tokens = 0
    total_padded = 0
    empty_count = 0

    for bucket_id in range(len(bucket_spec.edges) + 1):
        min_len, max_len = bucket_spec.get_bucket_range(bucket_id)

        if bucket_id not in bucket_lengths:
            empty_count += 1
            bucket_efficiencies.append(
                BucketEfficiency(
                    bucket_id=bucket_id,
                    min_length=min_len,
                    max_length=max_len,
                    sample_count=0,
                    total_tokens=0,
                    padded_tokens=0,
                    efficiency=0.0,
                    waste_percentage=0.0,
                )
            )
            continue

        bucket_lens = bucket_lengths[bucket_id]
        sample_count = len(bucket_lens)
        bucket_total = sum(bucket_lens)
        # Padding to bucket max
        bucket_padded = sample_count * max_len

        efficiency = bucket_total / bucket_padded if bucket_padded > 0 else 0.0
        waste = 1.0 - efficiency

        total_tokens += bucket_total
        total_padded += bucket_padded

        bucket_efficiencies.append(
            BucketEfficiency(
                bucket_id=bucket_id,
                min_length=min_len,
                max_length=max_len,
                sample_count=sample_count,
                total_tokens=bucket_total,
                padded_tokens=bucket_padded,
                efficiency=efficiency,
                waste_percentage=waste * 100,
            )
        )

    overall_eff = total_tokens / total_padded if total_padded > 0 else 0.0

    return BucketAnalysis(
        bucket_spec=bucket_spec,
        buckets=tuple(bucket_efficiencies),
        overall_efficiency=overall_eff,
        overall_waste=(1.0 - overall_eff) * 100,
        empty_buckets=empty_count,
    )


def suggest_bucket_edges(
    lengths: Mapping[str, int],
    num_buckets: int = 4,
    goal: OptimizationGoal = OptimizationGoal.MINIMIZE_WASTE,
    max_length: int | None = None,
) -> BucketSuggestion:
    """
    Suggest optimal bucket edges based on length distribution.

    Args:
        lengths: Dict mapping sample_id -> length
        num_buckets: Number of buckets to create
        goal: Optimization goal
        max_length: Maximum sequence length (for overflow bucket)

    Returns:
        BucketSuggestion with recommended edges
    """
    if not lengths:
        return BucketSuggestion(
            edges=(),
            overflow_max=1024,
            rationale="No data to analyze",
            estimated_efficiency=0.0,
            optimization_goal=goal,
        )

    values = sorted(lengths.values())
    n = len(values)
    max_val = values[-1]
    overflow_max = max_length or int(max_val * 1.1)

    if goal == OptimizationGoal.BALANCE_BUCKETS:
        # Use quantiles to balance sample counts
        edges = []
        for i in range(1, num_buckets):
            idx = int(n * i / num_buckets)
            edge = values[idx]
            # Round to nice numbers
            edge = ((edge + 63) // 64) * 64
            if not edges or edge > edges[-1]:
                edges.append(edge)

        rationale = "Edges chosen to balance sample counts across buckets"

    elif goal == OptimizationGoal.MINIMIZE_MEMORY:
        # Use power-of-2 boundaries
        edges = []
        current = 64
        while current < max_val and len(edges) < num_buckets - 1:
            edges.append(current)
            current *= 2

        rationale = "Power-of-2 edges for memory alignment"

    else:  # MINIMIZE_WASTE
        # Use percentile-based edges optimized for efficiency
        histogram = compute_length_histogram(lengths)
        edges = [histogram.p25, histogram.p50, histogram.p75, histogram.p90]
        # Round to multiples of 64
        edges = [((e + 63) // 64) * 64 for e in edges]
        # Remove duplicates and limit to num_buckets - 1
        edges = sorted(set(edges))[: num_buckets - 1]

        rationale = "Percentile-based edges to minimize padding waste"

    # Filter out edges that would conflict with overflow_max
    edges = [e for e in edges if e < overflow_max]

    # Estimate efficiency
    bucket_spec = BucketSpec(edges=tuple(edges), overflow_max=overflow_max)
    analysis = analyze_bucket_efficiency(lengths, bucket_spec)

    return BucketSuggestion(
        edges=tuple(edges),
        overflow_max=overflow_max,
        rationale=rationale,
        estimated_efficiency=analysis.overall_efficiency,
        optimization_goal=goal,
    )


def create_efficiency_report(
    lengths: Mapping[str, int],
    bucket_spec: BucketSpec,
) -> BatchingEfficiencyReport:
    """
    Create a complete efficiency report.

    Args:
        lengths: Dict mapping sample_id -> length
        bucket_spec: Current bucket configuration

    Returns:
        BatchingEfficiencyReport with full analysis
    """
    histogram = compute_length_histogram(lengths)
    bucket_analysis = analyze_bucket_efficiency(lengths, bucket_spec)

    # Generate suggestions for different goals
    suggestions = []
    for goal in OptimizationGoal:
        suggestion = suggest_bucket_edges(
            lengths,
            num_buckets=len(bucket_spec.edges) + 1,
            goal=goal,
            max_length=bucket_spec.overflow_max,
        )
        suggestions.append(suggestion)

    # Generate recommendations
    recommendations = []

    if bucket_analysis.overall_efficiency < 0.5:
        recommendations.append("Efficiency is below 50%. Consider using sequence packing.")

    if bucket_analysis.empty_buckets > 0:
        recommendations.append(
            f"{bucket_analysis.empty_buckets} bucket(s) are empty. "
            "Consider adjusting bucket edges to match data distribution."
        )

    # Check for unbalanced buckets
    sample_counts = [b.sample_count for b in bucket_analysis.buckets if b.sample_count > 0]
    if sample_counts:
        max_count = max(sample_counts)
        min_count = min(sample_counts)
        if max_count > min_count * 10:
            recommendations.append(
                "Bucket sizes are highly unbalanced. "
                "Consider rebalancing edges for more even distribution."
            )

    if histogram.p99 > bucket_spec.overflow_max:
        recommendations.append(
            f"P99 length ({histogram.p99}) exceeds overflow max ({bucket_spec.overflow_max}). "
            "Consider increasing overflow_max or truncating long sequences."
        )

    # Calculate totals
    total_tokens = sum(lengths.values())
    total_padded = (
        bucket_analysis.buckets and sum(b.padded_tokens for b in bucket_analysis.buckets) or 0
    )

    return BatchingEfficiencyReport(
        length_histogram=histogram,
        bucket_analysis=bucket_analysis,
        suggestions=tuple(suggestions),
        recommendations=tuple(recommendations),
        total_samples=len(lengths),
        total_tokens=total_tokens,
        total_padded=total_padded,
        overall_efficiency=bucket_analysis.overall_efficiency,
    )


def visualize_buckets(
    lengths: Mapping[str, int],
    bucket_spec: BucketSpec,
    width: int = 50,
) -> str:
    """
    Create ASCII visualization of bucket distribution.

    Args:
        lengths: Dict mapping sample_id -> length
        bucket_spec: Bucket configuration
        width: Maximum bar width

    Returns:
        Multi-line ASCII visualization
    """
    analysis = analyze_bucket_efficiency(lengths, bucket_spec)
    return analysis.to_ascii()
