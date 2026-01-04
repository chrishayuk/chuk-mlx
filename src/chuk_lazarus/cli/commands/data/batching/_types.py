"""Types for batching commands."""

from __future__ import annotations

from argparse import Namespace
from enum import Enum
from pathlib import Path

from pydantic import Field

from chuk_lazarus.cli.commands._base import CommandConfig, CommandResult


class OptimizationGoalType(str, Enum):
    """Optimization goal for bucket suggestions."""

    WASTE = "waste"
    BALANCE = "balance"
    MEMORY = "memory"


class AnalyzeConfig(CommandConfig):
    """Configuration for batching analysis."""

    cache: Path = Field(..., description="Path to length cache")
    bucket_edges: str = Field(..., description="Comma-separated bucket edges")
    overflow_max: int = Field(..., gt=0, description="Maximum overflow length")
    output: Path | None = Field(default=None, description="Output path for JSON report")

    @classmethod
    def from_args(cls, args: Namespace) -> AnalyzeConfig:
        """Create config from argparse namespace."""
        return cls(
            cache=Path(args.cache),
            bucket_edges=args.bucket_edges,
            overflow_max=args.overflow_max,
            output=Path(args.output) if args.output else None,
        )

    def get_bucket_edges(self) -> tuple[int, ...]:
        """Parse bucket edges string to tuple."""
        return tuple(int(x.strip()) for x in self.bucket_edges.split(","))


class AnalyzeResult(CommandResult):
    """Result of batching analysis."""

    report_ascii: str = Field(..., description="ASCII formatted report")
    output_path: Path | None = Field(default=None, description="Output path if saved")

    def to_display(self) -> str:
        """Format result for display."""
        lines = [self.report_ascii]
        if self.output_path:
            lines.append(f"\nReport saved to: {self.output_path}")
        return "\n".join(lines)


class HistogramConfig(CommandConfig):
    """Configuration for histogram display."""

    cache: Path = Field(..., description="Path to length cache")
    bins: int = Field(default=20, gt=0, description="Number of histogram bins")
    width: int = Field(default=80, gt=0, description="Display width")

    @classmethod
    def from_args(cls, args: Namespace) -> HistogramConfig:
        """Create config from argparse namespace."""
        return cls(
            cache=Path(args.cache),
            bins=args.bins,
            width=args.width,
        )


class HistogramResult(CommandResult):
    """Result of histogram display."""

    histogram_ascii: str = Field(..., description="ASCII histogram")
    p25: int = Field(..., ge=0, description="25th percentile")
    p50: int = Field(..., ge=0, description="50th percentile")
    p75: int = Field(..., ge=0, description="75th percentile")
    p90: int = Field(..., ge=0, description="90th percentile")
    p95: int = Field(..., ge=0, description="95th percentile")
    p99: int = Field(..., ge=0, description="99th percentile")

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            self.histogram_ascii,
            "",
            "--- Percentiles ---",
            f"  P25: {self.p25}",
            f"  P50: {self.p50}",
            f"  P75: {self.p75}",
            f"  P90: {self.p90}",
            f"  P95: {self.p95}",
            f"  P99: {self.p99}",
        ]
        return "\n".join(lines)


class SuggestConfig(CommandConfig):
    """Configuration for bucket edge suggestions."""

    cache: Path = Field(..., description="Path to length cache")
    num_buckets: int = Field(default=5, gt=0, description="Number of buckets")
    goal: OptimizationGoalType = Field(
        default=OptimizationGoalType.WASTE, description="Optimization goal"
    )
    max_length: int = Field(default=2048, gt=0, description="Maximum sequence length")

    @classmethod
    def from_args(cls, args: Namespace) -> SuggestConfig:
        """Create config from argparse namespace."""
        goal_map = {
            "waste": OptimizationGoalType.WASTE,
            "balance": OptimizationGoalType.BALANCE,
            "memory": OptimizationGoalType.MEMORY,
        }
        return cls(
            cache=Path(args.cache),
            num_buckets=args.num_buckets,
            goal=goal_map.get(args.goal, OptimizationGoalType.WASTE),
            max_length=args.max_length,
        )


class SuggestResult(CommandResult):
    """Result of bucket edge suggestions."""

    goal: str = Field(..., description="Optimization goal")
    num_buckets: int = Field(..., description="Number of buckets")
    edges: list[int] = Field(..., description="Suggested bucket edges")
    overflow_max: int = Field(..., description="Suggested overflow max")
    estimated_efficiency: float = Field(..., ge=0, le=1, description="Estimated efficiency")
    rationale: str = Field(..., description="Explanation of suggestions")

    def to_display(self) -> str:
        """Format result for display."""
        edges_str = ",".join(str(e) for e in self.edges)
        lines = [
            "",
            "=" * 60,
            "Bucket Edge Suggestions",
            "=" * 60,
            f"  Goal:           {self.goal}",
            f"  Num buckets:    {self.num_buckets}",
            "",
            f"  Suggested edges:  {self.edges}",
            f"  Overflow max:     {self.overflow_max}",
            f"  Est. efficiency:  {self.estimated_efficiency:.1%}",
            "",
            f"  Rationale: {self.rationale}",
            "",
            "  Use with:",
            f"    lazarus data batchplan build --bucket-edges {edges_str} "
            f"--overflow-max {self.overflow_max} ...",
        ]
        return "\n".join(lines)


class GenerateConfig(CommandConfig):
    """Configuration for batch generation."""

    plan: Path = Field(..., description="Path to batch plan")
    dataset: Path = Field(..., description="Path to dataset")
    tokenizer: str = Field(..., description="Tokenizer to use")
    output: Path = Field(..., description="Output directory")

    @classmethod
    def from_args(cls, args: Namespace) -> GenerateConfig:
        """Create config from argparse namespace."""
        return cls(
            plan=Path(args.plan),
            dataset=Path(args.dataset),
            tokenizer=args.tokenizer,
            output=Path(args.output),
        )


class GenerateResult(CommandResult):
    """Result of batch generation."""

    batch_plan: str = Field(..., description="Path to batch plan")
    dataset: str = Field(..., description="Path to dataset")
    output_dir: Path = Field(..., description="Output directory")
    num_files: int = Field(..., ge=0, description="Number of files generated")
    num_epochs: int = Field(..., ge=0, description="Number of epochs")
    fingerprint: str | None = Field(default=None, description="Plan fingerprint")

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            "",
            "=" * 60,
            "Batch Generation Complete",
            "=" * 60,
            f"  Batch plan:   {self.batch_plan}",
            f"  Dataset:      {self.dataset}",
            f"  Output:       {self.output_dir}",
            f"  Files:        {self.num_files}",
            f"  Epochs:       {self.num_epochs}",
        ]
        if self.fingerprint:
            lines.append(f"  Fingerprint:  {self.fingerprint}")
        return "\n".join(lines)
