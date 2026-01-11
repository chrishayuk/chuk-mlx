"""Types for length cache commands."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from pydantic import Field, field_validator

from chuk_lazarus.cli.commands._base import CommandConfig, CommandResult


class LengthBuildConfig(CommandConfig):
    """Configuration for building a length cache."""

    tokenizer: str = Field(..., description="Tokenizer to use")
    dataset: Path = Field(..., description="Path to dataset file")
    output: Path = Field(..., description="Output path for length cache")

    @classmethod
    def from_args(cls, args: Namespace) -> LengthBuildConfig:
        """Create config from argparse namespace."""
        return cls(
            tokenizer=args.tokenizer,
            dataset=Path(args.dataset),
            output=Path(args.output),
        )


class LengthBuildResult(CommandResult):
    """Result of building a length cache."""

    dataset: str = Field(..., description="Path to source dataset")
    tokenizer: str = Field(..., description="Tokenizer used")
    samples_processed: int = Field(..., ge=0, description="Number of samples processed")
    output_path: Path = Field(..., description="Path to output cache")
    tokenizer_hash: str = Field(..., description="Tokenizer fingerprint hash")

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            "",
            "=" * 60,
            "Length Cache Built",
            "=" * 60,
            f"  Dataset:        {self.dataset}",
            f"  Tokenizer:      {self.tokenizer}",
            f"  Samples:        {self.samples_processed:,}",
            f"  Output:         {self.output_path}",
            f"  Tokenizer hash: {self.tokenizer_hash}",
        ]
        return "\n".join(lines)


class LengthStatsConfig(CommandConfig):
    """Configuration for showing length cache statistics."""

    cache: Path = Field(..., description="Path to length cache file")

    @classmethod
    def from_args(cls, args: Namespace) -> LengthStatsConfig:
        """Create config from argparse namespace."""
        return cls(cache=Path(args.cache))


class LengthStatsResult(CommandResult):
    """Result of length cache statistics."""

    cache_path: str = Field(..., description="Path to cache file")
    tokenizer_hash: str = Field(..., description="Tokenizer hash")
    total_samples: int = Field(..., ge=0, description="Total samples in cache")
    total_tokens: int = Field(..., ge=0, description="Total tokens in cache")
    min_length: int = Field(..., ge=0, description="Minimum sequence length")
    max_length: int = Field(..., ge=0, description="Maximum sequence length")
    mean_length: float = Field(..., ge=0, description="Mean sequence length")
    median_length: int = Field(..., ge=0, description="Median sequence length")
    p10: int = Field(..., ge=0, description="10th percentile")
    p25: int = Field(..., ge=0, description="25th percentile")
    p50: int = Field(..., ge=0, description="50th percentile")
    p75: int = Field(..., ge=0, description="75th percentile")
    p90: int = Field(..., ge=0, description="90th percentile")
    p95: int = Field(..., ge=0, description="95th percentile")
    p99: int = Field(..., ge=0, description="99th percentile")

    @field_validator("mean_length", mode="before")
    @classmethod
    def round_mean(cls, v: float) -> float:
        """Round mean to 1 decimal place."""
        return round(v, 1)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            "",
            "=" * 60,
            "Length Cache Statistics",
            "=" * 60,
            f"  Cache file:    {self.cache_path}",
            f"  Tokenizer:     {self.tokenizer_hash}",
            f"  Total samples: {self.total_samples:,}",
            f"  Total tokens:  {self.total_tokens:,}",
            "",
            f"  Min length:    {self.min_length}",
            f"  Max length:    {self.max_length}",
            f"  Mean length:   {self.mean_length:.1f}",
            f"  Median:        {self.median_length}",
            "",
            f"  P10:           {self.p10}",
            f"  P25:           {self.p25}",
            f"  P50:           {self.p50}",
            f"  P75:           {self.p75}",
            f"  P90:           {self.p90}",
            f"  P95:           {self.p95}",
            f"  P99:           {self.p99}",
        ]
        return "\n".join(lines)


class EmptyStatsResult(CommandResult):
    """Result when cache is empty."""

    def to_display(self) -> str:
        """Format result for display."""
        return "Cache is empty"
