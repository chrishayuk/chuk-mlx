"""Type definitions for gym CLI commands.

This module contains Pydantic models and enums for gym commands.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from pydantic import Field

from .._base import CommandConfig, CommandResult, OutputMixin


class GymRunConfig(CommandConfig):
    """Configuration for gym run command.

    Attributes:
        tokenizer: Path or name of the tokenizer
        mock: Whether to use mock gym stream
        host: Server host
        port: Server port
        transport: Transport protocol
        output_mode: Output mode
        num_episodes: Number of episodes for mock
        steps_per_episode: Steps per episode for mock
        difficulty_min: Minimum difficulty
        difficulty_max: Maximum difficulty
        success_rate: Target success rate for mock
        buffer_size: Replay buffer size
        max_samples: Maximum samples to collect
        output: Output file path
        timeout: Connection timeout
        retries: Maximum retries
        seed: Random seed
    """

    tokenizer: str = Field(..., description="Tokenizer path or name")
    mock: bool = Field(default=True, description="Use mock gym stream")
    host: str = Field(default="localhost", description="Server host")
    port: int = Field(default=8023, ge=1, le=65535, description="Server port")
    transport: str = Field(default="telnet", description="Transport protocol")
    output_mode: str = Field(default="json", description="Output mode")
    num_episodes: int = Field(default=10, ge=1, description="Number of episodes")
    steps_per_episode: int = Field(default=20, ge=1, description="Steps per episode")
    difficulty_min: float = Field(default=0.0, ge=0.0, le=1.0, description="Min difficulty")
    difficulty_max: float = Field(default=1.0, ge=0.0, le=1.0, description="Max difficulty")
    success_rate: float = Field(default=0.7, ge=0.0, le=1.0, description="Target success rate")
    buffer_size: int = Field(default=10000, ge=1, description="Buffer size")
    max_samples: int | None = Field(default=None, ge=1, description="Max samples")
    output: Path | None = Field(default=None, description="Output file")
    timeout: float = Field(default=10.0, gt=0, description="Connection timeout")
    retries: int = Field(default=3, ge=0, description="Max retries")
    seed: int | None = Field(default=None, description="Random seed")

    @classmethod
    def from_args(cls, args: Namespace) -> GymRunConfig:
        """Create config from argparse Namespace."""
        return cls(
            tokenizer=args.tokenizer,
            mock=getattr(args, "mock", True),
            host=getattr(args, "host", "localhost"),
            port=getattr(args, "port", 8023),
            transport=getattr(args, "transport", "telnet"),
            output_mode=getattr(args, "output_mode", "json"),
            num_episodes=getattr(args, "num_episodes", 10),
            steps_per_episode=getattr(args, "steps_per_episode", 20),
            difficulty_min=getattr(args, "difficulty_min", 0.0),
            difficulty_max=getattr(args, "difficulty_max", 1.0),
            success_rate=getattr(args, "success_rate", 0.7),
            buffer_size=getattr(args, "buffer_size", 10000),
            max_samples=getattr(args, "max_samples", None),
            output=Path(args.output) if getattr(args, "output", None) else None,
            timeout=getattr(args, "timeout", 10.0),
            retries=getattr(args, "retries", 3),
            seed=getattr(args, "seed", None),
        )


class BenchmarkConfig(CommandConfig):
    """Configuration for benchmark command.

    Attributes:
        tokenizer: Tokenizer path or name
        dataset: Dataset path (optional)
        num_samples: Number of synthetic samples
        max_samples: Maximum samples to process
        max_length: Maximum sequence length
        token_budget: Token budget per batch
        bucket_edges: Bucket edge sizes
        seed: Random seed
    """

    tokenizer: str | None = Field(default=None, description="Tokenizer path or name")
    dataset: Path | None = Field(default=None, description="Dataset path")
    num_samples: int = Field(default=10000, ge=1, description="Synthetic samples")
    max_samples: int | None = Field(default=None, ge=1, description="Max samples")
    max_length: int = Field(default=2048, ge=1, description="Max sequence length")
    token_budget: int = Field(default=8192, ge=1, description="Token budget")
    bucket_edges: str = Field(default="128,256,512,1024", description="Bucket edges")
    seed: int = Field(default=42, description="Random seed")

    @classmethod
    def from_args(cls, args: Namespace) -> BenchmarkConfig:
        """Create config from argparse Namespace."""
        return cls(
            tokenizer=getattr(args, "tokenizer", None),
            dataset=Path(args.dataset) if getattr(args, "dataset", None) else None,
            num_samples=getattr(args, "num_samples", 10000),
            max_samples=getattr(args, "max_samples", None),
            max_length=getattr(args, "max_length", 2048),
            token_budget=getattr(args, "token_budget", 8192),
            bucket_edges=getattr(args, "bucket_edges", "128,256,512,1024"),
            seed=getattr(args, "seed", 42),
        )

    def get_bucket_edges(self) -> tuple[int, ...]:
        """Parse bucket edges string into tuple."""
        return tuple(int(x.strip()) for x in self.bucket_edges.split(","))


class GymRunResult(CommandResult, OutputMixin):
    """Result of gym run command.

    Attributes:
        total_samples: Total samples collected
        total_episodes: Total episodes completed
        buffer_size: Final buffer size
        success_rate: Overall success rate
        mean_difficulty: Mean difficulty
        mean_reward: Mean reward
        output_path: Path where buffer was saved
    """

    total_samples: int = Field(default=0, ge=0, description="Total samples")
    total_episodes: int = Field(default=0, ge=0, description="Total episodes")
    buffer_size: int = Field(default=0, ge=0, description="Buffer size")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate")
    mean_difficulty: float = Field(default=0.0, ge=0.0, description="Mean difficulty")
    mean_reward: float = Field(default=0.0, description="Mean reward")
    output_path: Path | None = Field(default=None, description="Output path")

    def to_display(self) -> str:
        """Format result for display."""
        lines = [self.format_header("Gym Run Summary")]
        lines.append(self.format_field("Total samples", self.total_samples))
        lines.append(self.format_field("Total episodes", self.total_episodes))
        lines.append(self.format_field("Buffer size", self.buffer_size))
        lines.append(self.format_field("Success rate", f"{self.success_rate:.1%}"))
        lines.append(self.format_field("Mean difficulty", f"{self.mean_difficulty:.2f}"))
        lines.append(self.format_field("Mean reward", f"{self.mean_reward:.2f}"))
        if self.output_path:
            lines.append(self.format_field("Output", str(self.output_path)))
        return "\n".join(lines)


class BenchmarkResult(CommandResult, OutputMixin):
    """Result of benchmark command.

    Attributes:
        samples: Number of samples processed
        total_tokens: Total tokens
        plan_fingerprint: Plan fingerprint
        bucket_efficiency: Overall bucket efficiency
        packing_ratio: Packing ratio
        packing_efficiency: Packing efficiency
        token_budget_utilization: Token budget utilization
        microbatches: Number of microbatches
    """

    samples: int = Field(default=0, ge=0, description="Samples processed")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens")
    plan_fingerprint: str = Field(default="", description="Plan fingerprint")
    bucket_efficiency: float = Field(default=0.0, ge=0.0, le=1.0, description="Bucket efficiency")
    packing_ratio: float = Field(default=1.0, ge=0.0, description="Packing ratio")
    packing_efficiency: float = Field(default=0.0, ge=0.0, le=1.0, description="Packing efficiency")
    token_budget_utilization: float = Field(default=0.0, ge=0.0, description="Budget utilization")
    microbatches: int = Field(default=0, ge=0, description="Number of microbatches")

    def to_display(self) -> str:
        """Format result for display."""
        lines = [self.format_header("Benchmark Summary")]
        lines.append(self.format_field("Samples", f"{self.samples:,}"))
        lines.append(self.format_field("Total tokens", f"{self.total_tokens:,}"))
        lines.append(self.format_field("Microbatches", f"{self.microbatches:,}"))
        lines.append(self.format_field("Bucket efficiency", f"{self.bucket_efficiency:.1%}"))
        lines.append(self.format_field("Packing ratio", f"{self.packing_ratio:.2f}x"))
        lines.append(self.format_field("Packing efficiency", f"{self.packing_efficiency:.1%}"))
        lines.append(
            self.format_field("Budget utilization", f"{self.token_budget_utilization:.1%}")
        )
        lines.append(self.format_field("Plan fingerprint", self.plan_fingerprint))
        return "\n".join(lines)


__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "GymRunConfig",
    "GymRunResult",
]
