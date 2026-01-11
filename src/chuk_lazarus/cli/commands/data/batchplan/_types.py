"""Types for batch plan commands."""

from __future__ import annotations

from argparse import Namespace
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field

from chuk_lazarus.cli.commands._base import CommandConfig, CommandResult


class BatchPlanMode(str, Enum):
    """Batch plan build mode."""

    PREDICTABLE = "predictable"
    THROUGHPUT = "throughput"


class BatchPlanBuildConfig(CommandConfig):
    """Configuration for building a batch plan."""

    lengths: Path = Field(..., description="Path to length cache")
    bucket_edges: str = Field(..., description="Comma-separated bucket edges")
    token_budget: int = Field(..., gt=0, description="Token budget per batch")
    overflow_max: int = Field(..., gt=0, description="Maximum overflow length")
    predictable: bool = Field(default=False, description="Use predictable mode")
    seed: int | None = Field(default=None, description="Random seed for predictable mode")
    epochs: int = Field(default=1, gt=0, description="Number of epochs")
    output: Path = Field(..., description="Output path for batch plan")
    dataset_hash: str | None = Field(default=None, description="Dataset hash")

    @classmethod
    def from_args(cls, args: Namespace) -> BatchPlanBuildConfig:
        """Create config from argparse namespace."""
        return cls(
            lengths=Path(args.lengths),
            bucket_edges=args.bucket_edges,
            token_budget=args.token_budget,
            overflow_max=args.overflow_max,
            predictable=args.predictable,
            seed=args.seed,
            epochs=args.epochs,
            output=Path(args.output),
            dataset_hash=args.dataset_hash,
        )

    def get_bucket_edges(self) -> tuple[int, ...]:
        """Parse bucket edges string to tuple."""
        return tuple(int(x.strip()) for x in self.bucket_edges.split(","))

    @property
    def mode(self) -> BatchPlanMode:
        """Get the build mode."""
        return BatchPlanMode.PREDICTABLE if self.predictable else BatchPlanMode.THROUGHPUT


class BatchPlanBuildResult(CommandResult):
    """Result of building a batch plan."""

    lengths_cache: str = Field(..., description="Path to lengths cache")
    epochs: int = Field(..., gt=0, description="Number of epochs")
    token_budget: int = Field(..., gt=0, description="Token budget")
    mode: BatchPlanMode = Field(..., description="Build mode")
    total_batches: int = Field(..., ge=0, description="Total microbatches")
    fingerprint: str = Field(..., description="Plan fingerprint")
    output_path: Path = Field(..., description="Output path")
    epoch_details: list[dict[str, Any]] = Field(
        default_factory=list, description="Per-epoch details"
    )

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            "",
            "=" * 60,
            "Batch Plan Built",
            "=" * 60,
            f"  Lengths cache: {self.lengths_cache}",
            f"  Epochs:        {self.epochs}",
            f"  Token budget:  {self.token_budget}",
            f"  Mode:          {self.mode.value}",
            "",
            f"  Total batches: {self.total_batches}",
            f"  Fingerprint:   {self.fingerprint}",
            "",
            f"  Output:        {self.output_path}",
            "",
            "  Per-epoch details:",
        ]
        for detail in self.epoch_details:
            lines.append(
                f"    Epoch {detail['epoch']}: {detail['batches']} batches, "
                f"{detail['samples']} samples, {detail['tokens']:,} tokens"
            )
        return "\n".join(lines)


class BatchPlanInfoConfig(CommandConfig):
    """Configuration for showing batch plan info."""

    plan: Path = Field(..., description="Path to batch plan")
    rank: int | None = Field(default=None, ge=0, description="Rank for sharding")
    world_size: int | None = Field(default=None, gt=0, description="World size for sharding")
    show_batches: int | None = Field(default=None, gt=0, description="Show sample batches")

    @classmethod
    def from_args(cls, args: Namespace) -> BatchPlanInfoConfig:
        """Create config from argparse namespace."""
        return cls(
            plan=Path(args.plan),
            rank=args.rank,
            world_size=args.world_size,
            show_batches=args.show_batches,
        )


class BatchPlanInfoResult(CommandResult):
    """Result of batch plan info."""

    plan_path: str = Field(..., description="Path to plan")
    fingerprint: str = Field(..., description="Plan fingerprint")
    created_at: str = Field(..., description="Creation timestamp")
    dataset_hash: str = Field(..., description="Dataset hash")
    tokenizer_hash: str = Field(..., description="Tokenizer hash")
    token_budget: int = Field(..., description="Token budget")
    bucket_edges: list[int] = Field(..., description="Bucket edges")
    epochs: int = Field(..., description="Number of epochs")
    total_batches: int = Field(..., description="Total batches")
    shard_info: str | None = Field(default=None, description="Shard information")
    epoch_details: list[dict[str, Any]] = Field(
        default_factory=list, description="Per-epoch details"
    )
    sample_batches: list[dict[str, Any]] = Field(
        default_factory=list, description="Sample batch details"
    )

    def to_display(self) -> str:
        """Format result for display."""
        shard_str = f" ({self.shard_info})" if self.shard_info else ""
        lines = [
            "",
            "=" * 60,
            f"Batch Plan Info{shard_str}",
            "=" * 60,
            f"  Plan path:     {self.plan_path}",
            f"  Fingerprint:   {self.fingerprint}",
            f"  Created:       {self.created_at}",
            "",
            f"  Dataset hash:  {self.dataset_hash}",
            f"  Tokenizer:     {self.tokenizer_hash}",
            f"  Token budget:  {self.token_budget}",
            f"  Bucket edges:  {self.bucket_edges}",
            "",
            f"  Epochs:        {self.epochs}",
            f"  Total batches: {self.total_batches}",
            "",
            "  Per-epoch details:",
        ]
        for detail in self.epoch_details:
            lines.append(
                f"    Epoch {detail['epoch']}: {detail['batches']} batches, "
                f"{detail['samples']} samples, {detail['tokens']:,} tokens"
            )

        if self.sample_batches:
            lines.append("")
            lines.append("  Sample batches from epoch 0:")
            for batch in self.sample_batches:
                lines.append(
                    f"    Batch {batch['index']}: {batch['size']} samples, "
                    f"bucket={batch['bucket_id']}, max_len={batch['max_len']}"
                )

        return "\n".join(lines)


class InvalidRankError(CommandResult):
    """Error when rank is invalid."""

    rank: int = Field(..., description="Invalid rank")
    world_size: int = Field(..., description="World size")

    def to_display(self) -> str:
        """Format error for display."""
        return f"Error: rank must be in range [0, {self.world_size})"


class BatchPlanVerifyConfig(CommandConfig):
    """Configuration for verifying a batch plan."""

    plan: Path = Field(..., description="Path to batch plan")
    lengths: Path = Field(..., description="Path to length cache")

    @classmethod
    def from_args(cls, args: Namespace) -> BatchPlanVerifyConfig:
        """Create config from argparse namespace."""
        return cls(plan=Path(args.plan), lengths=Path(args.lengths))


class BatchPlanVerifyResult(CommandResult):
    """Result of batch plan verification."""

    original_fingerprint: str = Field(..., description="Original plan fingerprint")
    rebuilt_fingerprint: str = Field(..., description="Rebuilt plan fingerprint")
    match: bool = Field(..., description="Whether fingerprints match")
    epoch_comparison: list[dict[str, Any]] = Field(
        default_factory=list, description="Per-epoch comparison"
    )

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            "",
            "=" * 60,
            "Batch Plan Verification",
            "=" * 60,
            f"  Original fingerprint: {self.original_fingerprint}",
            f"  Rebuilt fingerprint:  {self.rebuilt_fingerprint}",
        ]

        if self.match:
            lines.extend(["", "  Result: MATCH", "  The batch plan is reproducible."])
        else:
            lines.extend(
                [
                    "",
                    "  Result: MISMATCH",
                    "  Warning: Rebuilt plan differs from original!",
                ]
            )
            for comp in self.epoch_comparison:
                if comp.get("count_differs"):
                    lines.append(
                        f"    Epoch {comp['epoch']}: batch count differs "
                        f"({comp['original_count']} vs {comp['rebuilt_count']})"
                    )
                else:
                    lines.append(
                        f"    Epoch {comp['epoch']}: {comp['matches']}/{comp['total']} batches match"
                    )

        return "\n".join(lines)


class BatchPlanShardConfig(CommandConfig):
    """Configuration for sharding a batch plan."""

    plan: Path = Field(..., description="Path to batch plan")
    world_size: int = Field(..., gt=0, description="Number of shards")
    output: Path = Field(..., description="Output directory")

    @classmethod
    def from_args(cls, args: Namespace) -> BatchPlanShardConfig:
        """Create config from argparse namespace."""
        return cls(
            plan=Path(args.plan),
            world_size=args.world_size,
            output=Path(args.output),
        )


class BatchPlanShardResult(CommandResult):
    """Result of sharding a batch plan."""

    source_plan: str = Field(..., description="Source plan path")
    world_size: int = Field(..., description="Number of shards")
    total_batches: int = Field(..., description="Total batches in source")
    shard_details: list[dict[str, Any]] = Field(
        default_factory=list, description="Per-shard details"
    )
    output_dir: Path = Field(..., description="Output directory")

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            "",
            "=" * 60,
            "Batch Plan Sharding",
            "=" * 60,
            f"  Source plan:   {self.source_plan}",
            f"  World size:    {self.world_size}",
            f"  Total batches: {self.total_batches}",
            "",
        ]
        for shard in self.shard_details:
            lines.append(f"  Rank {shard['rank']}: {shard['batches']} batches -> {shard['path']}")
        lines.extend(["", f"  Output:        {self.output_dir}"])
        return "\n".join(lines)
