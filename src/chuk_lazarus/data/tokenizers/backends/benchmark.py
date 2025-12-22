"""
Tokenizer backend benchmarking utilities.

Provides tools to compare performance between different tokenizer backends,
particularly useful for evaluating HuggingFace vs MLX-Data CharTrie backends.

Usage:
    from chuk_lazarus.data.tokenizers.backends.benchmark import (
        benchmark_tokenizer,
        compare_backends,
        BenchmarkResult,
    )

    # Single tokenizer benchmark
    result = benchmark_tokenizer(tokenizer, corpus, num_workers=4)
    print(f"Throughput: {result.tokens_per_second:,.0f} tok/s")

    # Compare backends
    comparison = compare_backends(hf_tokenizer, corpus, num_workers=4)
    print(comparison.summary())
"""

import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..types import TokenizerProtocol


class BenchmarkResult(BaseModel):
    """Result of a tokenizer benchmark."""

    backend_type: str = Field(description="Backend type used")
    num_samples: int = Field(description="Number of text samples processed")
    total_tokens: int = Field(description="Total tokens generated")
    elapsed_seconds: float = Field(description="Time elapsed in seconds")
    tokens_per_second: float = Field(description="Tokenization throughput")
    samples_per_second: float = Field(description="Samples processed per second")
    avg_tokens_per_sample: float = Field(description="Average tokens per sample")
    num_workers: int = Field(default=1, description="Number of parallel workers used")


class BackendComparison(BaseModel):
    """Comparison between different backends."""

    huggingface_result: BenchmarkResult = Field(description="HuggingFace backend result")
    fast_result: BenchmarkResult | None = Field(
        default=None, description="Fast (MLX) backend result, None if unavailable"
    )
    speedup: float | None = Field(
        default=None, description="Fast backend speedup ratio, None if unavailable"
    )
    fast_error: str | None = Field(default=None, description="Error message if fast backend failed")

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "=" * 60,
            "Backend Comparison Results",
            "=" * 60,
            "",
            "HuggingFace Backend:",
            f"  Throughput: {self.huggingface_result.tokens_per_second:,.0f} tok/s",
            f"  Samples/s:  {self.huggingface_result.samples_per_second:,.1f}",
            f"  Workers:    {self.huggingface_result.num_workers}",
            "",
        ]

        if self.fast_result:
            lines.extend(
                [
                    "Fast (MLX CharTrie) Backend:",
                    f"  Throughput: {self.fast_result.tokens_per_second:,.0f} tok/s",
                    f"  Samples/s:  {self.fast_result.samples_per_second:,.1f}",
                    f"  Workers:    {self.fast_result.num_workers}",
                    "",
                    f"Speedup: {self.speedup:.2f}x" if self.speedup else "Speedup: N/A",
                ]
            )
        elif self.fast_error:
            lines.append(f"Fast Backend: {self.fast_error}")
        else:
            lines.append("Fast Backend: Not available (install mlx-data)")

        lines.append("=" * 60)
        return "\n".join(lines)


def benchmark_tokenizer(
    tokenizer: "TokenizerProtocol",
    corpus: list[str],
    num_workers: int = 1,
    add_special_tokens: bool = False,
    warmup_samples: int = 10,
) -> BenchmarkResult:
    """
    Benchmark a tokenizer on a corpus.

    Args:
        tokenizer: Tokenizer to benchmark
        corpus: List of text samples
        num_workers: Number of parallel workers (for backends that support it)
        add_special_tokens: Whether to add special tokens
        warmup_samples: Number of warmup samples before timing

    Returns:
        BenchmarkResult with performance metrics
    """
    # Determine backend type
    backend_type = "huggingface"
    if hasattr(tokenizer, "backend_type"):
        backend_type = tokenizer.backend_type.value

    # Warmup
    for text in corpus[:warmup_samples]:
        tokenizer.encode(text, add_special_tokens=add_special_tokens)

    # Check if batch encoding with workers is supported
    has_batch_encode = hasattr(tokenizer, "encode_batch")

    # Benchmark
    start_time = time.perf_counter()

    if has_batch_encode and num_workers > 1:
        # Use batch encoding if available
        result = tokenizer.encode_batch(
            corpus, add_special_tokens=add_special_tokens, num_workers=num_workers
        )
        total_tokens = result.total_tokens
    else:
        # Sequential encoding
        total_tokens = 0
        for text in corpus:
            ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
            if hasattr(ids, "token_ids"):
                total_tokens += len(ids.token_ids)
            else:
                total_tokens += len(ids)

    elapsed = time.perf_counter() - start_time

    return BenchmarkResult(
        backend_type=backend_type,
        num_samples=len(corpus),
        total_tokens=total_tokens,
        elapsed_seconds=elapsed,
        tokens_per_second=total_tokens / elapsed if elapsed > 0 else 0,
        samples_per_second=len(corpus) / elapsed if elapsed > 0 else 0,
        avg_tokens_per_sample=total_tokens / len(corpus) if corpus else 0,
        num_workers=num_workers,
    )


def compare_backends(
    tokenizer: "TokenizerProtocol",
    corpus: list[str],
    num_workers: int = 4,
    add_special_tokens: bool = False,
) -> BackendComparison:
    """
    Compare HuggingFace and Fast backends on the same corpus.

    Args:
        tokenizer: HuggingFace tokenizer to use (vocab extracted for fast backend)
        corpus: List of text samples
        num_workers: Number of workers for parallel backends
        add_special_tokens: Whether to add special tokens

    Returns:
        BackendComparison with results for both backends
    """
    from .fast import FastBackend, is_fast_backend_available
    from .huggingface import HuggingFaceBackend

    # Benchmark HuggingFace backend
    hf_backend = HuggingFaceBackend(tokenizer)
    hf_result = benchmark_tokenizer(
        hf_backend, corpus, num_workers=1, add_special_tokens=add_special_tokens
    )

    # Benchmark Fast backend if available
    fast_result = None
    speedup = None

    fast_error: str | None = None
    if is_fast_backend_available():
        try:
            fast_backend = FastBackend.from_tokenizer(tokenizer)
            fast_result = benchmark_tokenizer(
                fast_backend,
                corpus,
                num_workers=num_workers,
                add_special_tokens=add_special_tokens,
            )

            if hf_result.tokens_per_second > 0:
                speedup = fast_result.tokens_per_second / hf_result.tokens_per_second
        except RuntimeError:
            # CharTrie can't handle BPE tokenizers that require merge rules
            fast_error = (
                "Not compatible (BPE tokenizers require merge rules that CharTrie doesn't support)"
            )
        except Exception as e:
            fast_error = str(e)

    return BackendComparison(
        huggingface_result=hf_result,
        fast_result=fast_result,
        speedup=speedup,
        fast_error=fast_error,
    )


def generate_benchmark_corpus(
    num_samples: int = 1000,
    avg_length: int = 100,
    seed: int = 42,
) -> list[str]:
    """
    Generate a synthetic corpus for benchmarking.

    Args:
        num_samples: Number of text samples to generate
        avg_length: Average length in words per sample
        seed: Random seed for reproducibility

    Returns:
        List of text samples
    """
    import random

    random.seed(seed)

    # Common words for synthetic text
    words = [
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "I",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "us",
    ]

    corpus = []
    for _ in range(num_samples):
        # Vary length around average
        length = max(10, int(random.gauss(avg_length, avg_length * 0.3)))
        text = " ".join(random.choice(words) for _ in range(length))
        corpus.append(text)

    return corpus


__all__ = [
    "BenchmarkResult",
    "BackendComparison",
    "benchmark_tokenizer",
    "compare_backends",
    "generate_benchmark_corpus",
]
