"""Shared types for tokenizer CLI commands."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field

from .._base import CommandConfig, CommandResult


class TokenizerHealthStatus(str, Enum):
    """Health status for tokenizer doctor."""

    HEALTHY = "healthy"
    ISSUES = "issues"
    CRITICAL = "critical"


class InitMethod(str, Enum):
    """Initialization method for soft tokens."""

    RANDOM = "random"
    NORMAL = "normal"
    UNIFORM = "uniform"


class MorphMethod(str, Enum):
    """Morphing method for token embeddings."""

    LINEAR = "linear"
    SLERP = "slerp"
    GEODESIC = "geodesic"


# === Core Command Configs ===


class EncodeConfig(CommandConfig):
    """Config for tokenizer_encode command."""

    tokenizer: str
    text: str | None = None
    file: Path | None = None
    special_tokens: bool = True

    @classmethod
    def from_args(cls, args: Any) -> "EncodeConfig":
        return cls(
            tokenizer=args.tokenizer,
            text=getattr(args, "text", None),
            file=getattr(args, "file", None),
            special_tokens=getattr(args, "special_tokens", True),
        )


class DecodeConfig(CommandConfig):
    """Config for tokenizer_decode command."""

    tokenizer: str
    ids: str

    @classmethod
    def from_args(cls, args: Any) -> "DecodeConfig":
        return cls(tokenizer=args.tokenizer, ids=args.ids)


class DecodeResult(CommandResult):
    """Result for tokenizer_decode command."""

    token_ids: list[int]
    decoded: str

    def to_display(self) -> str:
        return f"Token IDs: {self.token_ids}\nDecoded: {self.decoded}"


class VocabConfig(CommandConfig):
    """Config for tokenizer_vocab command."""

    tokenizer: str
    show_all: bool = False
    search: str | None = None
    limit: int = 20
    chunk_size: int = 100
    pause: bool = False

    @classmethod
    def from_args(cls, args: Any) -> "VocabConfig":
        return cls(
            tokenizer=args.tokenizer,
            show_all=getattr(args, "show_all", False),
            search=getattr(args, "search", None),
            limit=getattr(args, "limit", 20),
            chunk_size=getattr(args, "chunk_size", 100),
            pause=getattr(args, "pause", False),
        )


class CompareConfig(CommandConfig):
    """Config for tokenizer_compare command."""

    tokenizer1: str
    tokenizer2: str
    text: str
    verbose: bool = False

    @classmethod
    def from_args(cls, args: Any) -> "CompareConfig":
        return cls(
            tokenizer1=args.tokenizer1,
            tokenizer2=args.tokenizer2,
            text=args.text,
            verbose=getattr(args, "verbose", False),
        )


class CompareResult(CommandResult):
    """Result for tokenizer_compare command."""

    tokenizer1_count: int
    tokenizer2_count: int
    difference: int
    ratio: float

    def to_display(self) -> str:
        return (
            f"Token count 1: {self.tokenizer1_count}\n"
            f"Token count 2: {self.tokenizer2_count}\n"
            f"Difference: {self.difference:+d} tokens\n"
            f"Ratio: {self.ratio:.2f}x"
        )


# === Health Command Configs ===


class DoctorConfig(CommandConfig):
    """Config for tokenizer_doctor command."""

    tokenizer: str
    verbose: bool = False
    fix: bool = False
    format: str | None = None
    output: Path | None = None

    @classmethod
    def from_args(cls, args: Any) -> "DoctorConfig":
        return cls(
            tokenizer=args.tokenizer,
            verbose=getattr(args, "verbose", False),
            fix=getattr(args, "fix", False),
            format=getattr(args, "format", None),
            output=getattr(args, "output", None),
        )


class DoctorResult(CommandResult):
    """Result for tokenizer_doctor command."""

    status: TokenizerHealthStatus
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    fixes_applied: list[str] = Field(default_factory=list)

    def to_display(self) -> str:
        lines = [f"Status: {self.status.value.upper()}"]
        if self.fixes_applied:
            lines.append(f"Fixes Applied: {len(self.fixes_applied)}")
            for fix in self.fixes_applied:
                lines.append(f"  FIXED: {fix}")
        if self.issues:
            lines.append(f"Issues: {len(self.issues)}")
            for issue in self.issues:
                lines.append(f"  ERROR: {issue}")
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                lines.append(f"  WARN: {warning}")
        return "\n".join(lines)


class FingerprintConfig(CommandConfig):
    """Config for tokenizer_fingerprint command."""

    tokenizer: str
    verify: str | None = None
    save: Path | None = None
    strict: bool = False

    @classmethod
    def from_args(cls, args: Any) -> "FingerprintConfig":
        return cls(
            tokenizer=args.tokenizer,
            verify=getattr(args, "verify", None),
            save=getattr(args, "save", None),
            strict=getattr(args, "strict", False),
        )


class FingerprintResult(CommandResult):
    """Result for tokenizer_fingerprint command."""

    fingerprint: str
    vocab_size: int
    vocab_hash: str
    full_hash: str
    special_tokens_hash: str
    merges_hash: str
    special_tokens: dict[str, int | None]
    verified: bool | None = None
    match: bool | None = None

    def to_display(self) -> str:
        lines = [
            f"Fingerprint:   {self.fingerprint}",
            f"Full hash:     {self.full_hash}",
            f"Vocab size:    {self.vocab_size:,}",
            f"Vocab hash:    {self.vocab_hash}",
            f"Special hash:  {self.special_tokens_hash}",
            f"Merges hash:   {self.merges_hash}",
        ]
        if self.verified is not None:
            result = "MATCH" if self.match else "MISMATCH"
            lines.append(f"\nVerification: {result}")
        return "\n".join(lines)


class BenchmarkConfig(CommandConfig):
    """Config for tokenizer_benchmark command."""

    tokenizer: str
    samples: int = 1000
    avg_length: int = 100
    seed: int | None = None
    workers: int = 1
    file: Path | None = None
    compare: bool = False
    special_tokens: bool = False
    warmup: int = 10

    @classmethod
    def from_args(cls, args: Any) -> "BenchmarkConfig":
        return cls(
            tokenizer=args.tokenizer,
            samples=getattr(args, "samples", 1000),
            avg_length=getattr(args, "avg_length", 100),
            seed=getattr(args, "seed", None),
            workers=getattr(args, "workers", 1),
            file=getattr(args, "file", None),
            compare=getattr(args, "compare", False),
            special_tokens=getattr(args, "special_tokens", False),
            warmup=getattr(args, "warmup", 10),
        )


class BenchmarkResult(CommandResult):
    """Result for tokenizer_benchmark command."""

    backend_type: str
    total_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    samples_per_second: float
    avg_tokens_per_sample: float

    def to_display(self) -> str:
        return (
            f"Backend:      {self.backend_type}\n"
            f"Total tokens: {self.total_tokens:,}\n"
            f"Time:         {self.elapsed_seconds:.2f}s\n"
            f"Throughput:   {self.tokens_per_second:,.0f} tokens/sec\n"
            f"Samples/sec:  {self.samples_per_second:,.1f}\n"
            f"Avg tok/sample: {self.avg_tokens_per_sample:.1f}"
        )


# === Analyze Command Configs ===


class AnalyzeCoverageConfig(CommandConfig):
    """Config for analyze_coverage command."""

    tokenizer: str
    file: Path | None = None
    fragments: bool = False

    @classmethod
    def from_args(cls, args: Any) -> "AnalyzeCoverageConfig":
        return cls(
            tokenizer=args.tokenizer,
            file=getattr(args, "file", None),
            fragments=getattr(args, "fragments", False),
        )


class AnalyzeEntropyConfig(CommandConfig):
    """Config for analyze_entropy command."""

    tokenizer: str
    file: Path | None = None
    top_n: int = 20

    @classmethod
    def from_args(cls, args: Any) -> "AnalyzeEntropyConfig":
        return cls(
            tokenizer=args.tokenizer,
            file=getattr(args, "file", None),
            top_n=getattr(args, "top_n", 20),
        )


class AnalyzeFitScoreConfig(CommandConfig):
    """Config for analyze_fit_score command."""

    tokenizer: str
    file: Path | None = None

    @classmethod
    def from_args(cls, args: Any) -> "AnalyzeFitScoreConfig":
        return cls(
            tokenizer=args.tokenizer,
            file=getattr(args, "file", None),
        )


class AnalyzeEfficiencyConfig(CommandConfig):
    """Config for analyze_efficiency command."""

    tokenizer: str
    file: Path | None = None

    @classmethod
    def from_args(cls, args: Any) -> "AnalyzeEfficiencyConfig":
        return cls(
            tokenizer=args.tokenizer,
            file=getattr(args, "file", None),
        )


class AnalyzeVocabSuggestConfig(CommandConfig):
    """Config for analyze_vocab_suggest command."""

    tokenizer: str
    file: Path | None = None
    min_freq: int = 5
    min_frag: int = 2
    limit: int = 100
    show: int = 20

    @classmethod
    def from_args(cls, args: Any) -> "AnalyzeVocabSuggestConfig":
        return cls(
            tokenizer=args.tokenizer,
            file=getattr(args, "file", None),
            min_freq=getattr(args, "min_freq", 5),
            min_frag=getattr(args, "min_frag", 2),
            limit=getattr(args, "limit", 100),
            show=getattr(args, "show", 20),
        )


class AnalyzeDiffConfig(CommandConfig):
    """Config for analyze_diff command."""

    tokenizer1: str
    tokenizer2: str
    file: Path | None = None

    @classmethod
    def from_args(cls, args: Any) -> "AnalyzeDiffConfig":
        return cls(
            tokenizer1=args.tokenizer1,
            tokenizer2=args.tokenizer2,
            file=getattr(args, "file", None),
        )


# === Curriculum Command Configs ===


class CurriculumLengthBucketsConfig(CommandConfig):
    """Config for curriculum_length_buckets command."""

    tokenizer: str
    file: Path | None = None
    num_buckets: int = 5
    schedule: bool = False

    @classmethod
    def from_args(cls, args: Any) -> "CurriculumLengthBucketsConfig":
        return cls(
            tokenizer=args.tokenizer,
            file=getattr(args, "file", None),
            num_buckets=getattr(args, "num_buckets", 5),
            schedule=getattr(args, "schedule", False),
        )


class CurriculumReasoningConfig(CommandConfig):
    """Config for curriculum_reasoning_density command."""

    tokenizer: str
    file: Path | None = None
    descending: bool = True

    @classmethod
    def from_args(cls, args: Any) -> "CurriculumReasoningConfig":
        return cls(
            tokenizer=args.tokenizer,
            file=getattr(args, "file", None),
            descending=getattr(args, "descending", True),
        )


# === Training Command Configs ===


class TrainingThroughputConfig(CommandConfig):
    """Config for training_throughput command."""

    tokenizer: str
    file: Path | None = None
    batch_size: int = 32
    iterations: int = 10

    @classmethod
    def from_args(cls, args: Any) -> "TrainingThroughputConfig":
        return cls(
            tokenizer=args.tokenizer,
            file=getattr(args, "file", None),
            batch_size=getattr(args, "batch_size", 32),
            iterations=getattr(args, "iterations", 10),
        )


class TrainingPackConfig(CommandConfig):
    """Config for training_pack command."""

    tokenizer: str
    file: Path | None = None
    max_length: int = 2048
    output: Path | None = None

    @classmethod
    def from_args(cls, args: Any) -> "TrainingPackConfig":
        return cls(
            tokenizer=args.tokenizer,
            file=getattr(args, "file", None),
            max_length=getattr(args, "max_length", 2048),
            output=getattr(args, "output", None),
        )


class PackResult(CommandResult):
    """Result for training_pack command."""

    input_sequences: int
    packed_sequences: int
    packing_ratio: float
    efficiency: float
    output_path: Path | None = None

    def to_display(self) -> str:
        lines = [
            f"Input sequences:   {self.input_sequences}",
            f"Packed sequences:  {self.packed_sequences}",
            f"Packing ratio:     {self.packing_ratio:.2f}x",
            f"Efficiency:        {self.efficiency:.2%}",
        ]
        if self.output_path:
            lines.append(f"Saved to: {self.output_path}")
        return "\n".join(lines)


# === Regression Command Configs ===


class RegressionRunConfig(CommandConfig):
    """Config for regression_run command."""

    tokenizer: str
    tests: Path

    @classmethod
    def from_args(cls, args: Any) -> "RegressionRunConfig":
        return cls(
            tokenizer=args.tokenizer,
            tests=Path(args.tests),
        )


class RegressionResult(CommandResult):
    """Result for regression_run command."""

    suite_name: str
    total_tests: int
    passed: int
    failed: int
    failures: list[str] = Field(default_factory=list)

    def to_display(self) -> str:
        lines = [
            f"Suite: {self.suite_name}",
            f"Tests: {self.total_tests}",
            f"Passed: {self.passed}",
            f"Failed: {self.failed}",
        ]
        if self.failures:
            lines.append("\nFailed tests:")
            for failure in self.failures:
                lines.append(f"  - {failure}")
        return "\n".join(lines)


# === Research Command Configs ===


class ResearchSoftTokensConfig(CommandConfig):
    """Config for research_soft_tokens command."""

    num_tokens: int = 10
    embedding_dim: int = 768
    prefix: str = "soft"
    init_method: InitMethod = InitMethod.NORMAL
    init_std: float = 0.02
    output: Path | None = None

    @classmethod
    def from_args(cls, args: Any) -> "ResearchSoftTokensConfig":
        return cls(
            num_tokens=getattr(args, "num_tokens", 10),
            embedding_dim=getattr(args, "embedding_dim", 768),
            prefix=getattr(args, "prefix", "soft"),
            init_method=InitMethod(getattr(args, "init_method", "normal")),
            init_std=getattr(args, "init_std", 0.02),
            output=getattr(args, "output", None),
        )


class ResearchEmbeddingsConfig(CommandConfig):
    """Config for research_analyze_embeddings command."""

    file: Path
    num_clusters: int = 10
    cluster: bool = False
    project: bool = False

    @classmethod
    def from_args(cls, args: Any) -> "ResearchEmbeddingsConfig":
        return cls(
            file=Path(args.file),
            num_clusters=getattr(args, "num_clusters", 10),
            cluster=getattr(args, "cluster", False),
            project=getattr(args, "project", False),
        )


class ResearchMorphConfig(CommandConfig):
    """Config for research_morph command."""

    file: Path
    source: int
    target: int
    method: MorphMethod = MorphMethod.LINEAR
    steps: int = 10
    normalize: bool = False
    output: Path | None = None

    @classmethod
    def from_args(cls, args: Any) -> "ResearchMorphConfig":
        return cls(
            file=Path(args.file),
            source=args.source,
            target=args.target,
            method=MorphMethod(getattr(args, "method", "linear")),
            steps=getattr(args, "steps", 10),
            normalize=getattr(args, "normalize", False),
            output=getattr(args, "output", None),
        )


# === Instrument Command Configs ===


class InstrumentHistogramConfig(CommandConfig):
    """Config for instrument_histogram command."""

    tokenizer: str
    file: Path | None = None
    bins: int = 20
    width: int = 60
    quick: bool = False

    @classmethod
    def from_args(cls, args: Any) -> "InstrumentHistogramConfig":
        return cls(
            tokenizer=args.tokenizer,
            file=getattr(args, "file", None),
            bins=getattr(args, "bins", 20),
            width=getattr(args, "width", 60),
            quick=getattr(args, "quick", False),
        )


class InstrumentOovConfig(CommandConfig):
    """Config for instrument_oov command."""

    tokenizer: str
    file: Path | None = None
    vocab_size: int | None = None
    show_rare: bool = False
    max_freq: int = 5
    top_k: int = 20

    @classmethod
    def from_args(cls, args: Any) -> "InstrumentOovConfig":
        return cls(
            tokenizer=args.tokenizer,
            file=getattr(args, "file", None),
            vocab_size=getattr(args, "vocab_size", None),
            show_rare=getattr(args, "show_rare", False),
            max_freq=getattr(args, "max_freq", 5),
            top_k=getattr(args, "top_k", 20),
        )


class InstrumentWasteConfig(CommandConfig):
    """Config for instrument_waste command."""

    tokenizer: str
    file: Path | None = None
    max_length: int = 2048

    @classmethod
    def from_args(cls, args: Any) -> "InstrumentWasteConfig":
        return cls(
            tokenizer=args.tokenizer,
            file=getattr(args, "file", None),
            max_length=getattr(args, "max_length", 2048),
        )


class InstrumentVocabDiffConfig(CommandConfig):
    """Config for instrument_vocab_diff command."""

    tokenizer1: str
    tokenizer2: str
    file: Path | None = None
    examples: int = 5
    cost: bool = False

    @classmethod
    def from_args(cls, args: Any) -> "InstrumentVocabDiffConfig":
        return cls(
            tokenizer1=args.tokenizer1,
            tokenizer2=args.tokenizer2,
            file=getattr(args, "file", None),
            examples=getattr(args, "examples", 5),
            cost=getattr(args, "cost", False),
        )


# === Runtime Command Configs ===


class RuntimeRegistryConfig(CommandConfig):
    """Config for runtime_registry command."""

    verbose: bool = False

    @classmethod
    def from_args(cls, args: Any) -> "RuntimeRegistryConfig":
        return cls(verbose=getattr(args, "verbose", False))
