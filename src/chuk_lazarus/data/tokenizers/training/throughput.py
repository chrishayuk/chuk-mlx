"""Token throughput profiling utilities."""

import time
from typing import Protocol

from pydantic import BaseModel, Field


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...


class ThroughputMetrics(BaseModel):
    """Token throughput metrics."""

    total_texts: int = Field(ge=0, description="Number of texts processed")
    total_tokens: int = Field(ge=0, description="Total tokens produced")
    total_chars: int = Field(ge=0, description="Total characters processed")
    elapsed_seconds: float = Field(ge=0.0, description="Time elapsed")
    tokens_per_second: float = Field(ge=0.0, description="Tokenization speed")
    chars_per_second: float = Field(ge=0.0, description="Character processing speed")
    avg_tokens_per_text: float = Field(ge=0.0, description="Average tokens per text")
    avg_chars_per_token: float = Field(ge=0.0, description="Compression ratio")


class BatchMetrics(BaseModel):
    """Metrics for a batch of sequences."""

    batch_size: int = Field(ge=0, description="Number of sequences in batch")
    total_tokens: int = Field(ge=0, description="Total tokens in batch")
    max_length: int = Field(ge=0, description="Longest sequence")
    min_length: int = Field(ge=0, description="Shortest sequence")
    padding_tokens: int = Field(ge=0, description="Padding tokens if padded")
    attention_waste: float = Field(ge=0.0, le=1.0, description="Fraction of attention on padding")
    effective_batch_tokens: int = Field(ge=0, description="Real (non-padding) tokens")


class ThroughputProfiler:
    """Profiler for tokenization throughput."""

    def __init__(self, tokenizer: TokenizerProtocol):
        """
        Initialize profiler.

        Args:
            tokenizer: Tokenizer to profile
        """
        self.tokenizer = tokenizer
        self._total_texts = 0
        self._total_tokens = 0
        self._total_chars = 0
        self._total_time = 0.0
        self._batch_metrics: list[BatchMetrics] = []

    def profile_text(self, text: str, add_special_tokens: bool = False) -> int:
        """
        Profile tokenization of a single text.

        Args:
            text: Text to tokenize
            add_special_tokens: Whether to add special tokens

        Returns:
            Number of tokens
        """
        start = time.perf_counter()
        tokens = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        elapsed = time.perf_counter() - start

        self._total_texts += 1
        self._total_tokens += len(tokens)
        self._total_chars += len(text)
        self._total_time += elapsed

        return len(tokens)

    def profile_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = False,
    ) -> BatchMetrics:
        """
        Profile tokenization of a batch.

        Args:
            texts: List of texts
            add_special_tokens: Whether to add special tokens

        Returns:
            BatchMetrics for the batch
        """
        start = time.perf_counter()
        token_counts = []
        total_tokens = 0

        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            token_counts.append(len(tokens))
            total_tokens += len(tokens)
            self._total_chars += len(text)

        elapsed = time.perf_counter() - start
        self._total_time += elapsed
        self._total_texts += len(texts)
        self._total_tokens += total_tokens

        max_len = max(token_counts) if token_counts else 0
        min_len = min(token_counts) if token_counts else 0

        # Calculate padding if we were to pad to max length
        padding = sum(max_len - count for count in token_counts)
        total_with_padding = len(texts) * max_len
        attention_waste = padding / total_with_padding if total_with_padding > 0 else 0.0

        metrics = BatchMetrics(
            batch_size=len(texts),
            total_tokens=total_tokens,
            max_length=max_len,
            min_length=min_len,
            padding_tokens=padding,
            attention_waste=attention_waste,
            effective_batch_tokens=total_tokens,
        )

        self._batch_metrics.append(metrics)
        return metrics

    def get_metrics(self) -> ThroughputMetrics:
        """Get overall throughput metrics."""
        tokens_per_sec = self._total_tokens / self._total_time if self._total_time > 0 else 0.0
        chars_per_sec = self._total_chars / self._total_time if self._total_time > 0 else 0.0
        avg_tokens = self._total_tokens / self._total_texts if self._total_texts > 0 else 0.0
        avg_chars = self._total_chars / self._total_tokens if self._total_tokens > 0 else 0.0

        return ThroughputMetrics(
            total_texts=self._total_texts,
            total_tokens=self._total_tokens,
            total_chars=self._total_chars,
            elapsed_seconds=self._total_time,
            tokens_per_second=tokens_per_sec,
            chars_per_second=chars_per_sec,
            avg_tokens_per_text=avg_tokens,
            avg_chars_per_token=avg_chars,
        )

    def reset(self) -> None:
        """Reset profiler statistics."""
        self._total_texts = 0
        self._total_tokens = 0
        self._total_chars = 0
        self._total_time = 0.0
        self._batch_metrics = []


def profile_tokenization(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    add_special_tokens: bool = False,
) -> ThroughputMetrics:
    """
    Profile tokenization of a list of texts.

    Args:
        texts: List of texts to tokenize
        tokenizer: Tokenizer to use
        add_special_tokens: Whether to add special tokens

    Returns:
        ThroughputMetrics with profiling results
    """
    profiler = ThroughputProfiler(tokenizer)
    profiler.profile_batch(texts, add_special_tokens=add_special_tokens)
    return profiler.get_metrics()


def estimate_training_tokens(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    epochs: int = 1,
    sample_ratio: float = 0.1,
) -> dict:
    """
    Estimate total training tokens for a dataset.

    Args:
        texts: Dataset texts (or sample)
        tokenizer: Tokenizer to use
        epochs: Number of training epochs
        sample_ratio: If < 1.0, texts is a sample of this ratio

    Returns:
        Dict with token estimates
    """
    total_tokens = 0
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(tokens)

    # Scale up if sampling
    if sample_ratio < 1.0:
        estimated_total = int(total_tokens / sample_ratio)
    else:
        estimated_total = total_tokens

    return {
        "sample_tokens": total_tokens,
        "sample_texts": len(texts),
        "sample_ratio": sample_ratio,
        "estimated_dataset_tokens": estimated_total,
        "epochs": epochs,
        "total_training_tokens": estimated_total * epochs,
        "avg_tokens_per_text": total_tokens / len(texts) if texts else 0,
    }
