"""Tests for throughput module."""

from chuk_lazarus.data.tokenizers.training.throughput import (
    BatchMetrics,
    ThroughputMetrics,
    ThroughputProfiler,
    estimate_training_tokens,
    profile_tokenization,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, tokens_per_word: int = 1):
        self._tokens_per_word = tokens_per_word

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        words = text.split()
        return [1] * (len(words) * self._tokens_per_word)


class TestThroughputMetricsModel:
    """Tests for ThroughputMetrics model."""

    def test_valid_metrics(self):
        metrics = ThroughputMetrics(
            total_texts=100,
            total_tokens=1500,
            total_chars=10000,
            elapsed_seconds=1.5,
            tokens_per_second=1000.0,
            chars_per_second=6666.67,
            avg_tokens_per_text=15.0,
            avg_chars_per_token=6.67,
        )
        assert metrics.total_texts == 100
        assert metrics.tokens_per_second == 1000.0

    def test_zero_values(self):
        metrics = ThroughputMetrics(
            total_texts=0,
            total_tokens=0,
            total_chars=0,
            elapsed_seconds=0.0,
            tokens_per_second=0.0,
            chars_per_second=0.0,
            avg_tokens_per_text=0.0,
            avg_chars_per_token=0.0,
        )
        assert metrics.total_texts == 0


class TestBatchMetricsModel:
    """Tests for BatchMetrics model."""

    def test_valid_metrics(self):
        metrics = BatchMetrics(
            batch_size=32,
            total_tokens=512,
            max_length=64,
            min_length=8,
            padding_tokens=100,
            attention_waste=0.2,
            effective_batch_tokens=412,
        )
        assert metrics.batch_size == 32
        assert metrics.attention_waste == 0.2


class TestThroughputProfiler:
    """Tests for ThroughputProfiler class."""

    def test_create_profiler(self):
        tokenizer = MockTokenizer()
        profiler = ThroughputProfiler(tokenizer)
        assert profiler is not None

    def test_profile_single_text(self):
        tokenizer = MockTokenizer()
        profiler = ThroughputProfiler(tokenizer)
        num_tokens = profiler.profile_text("hello world")
        assert num_tokens == 2

    def test_profile_multiple_texts(self):
        tokenizer = MockTokenizer()
        profiler = ThroughputProfiler(tokenizer)
        profiler.profile_text("hello world")
        profiler.profile_text("foo bar baz")
        metrics = profiler.get_metrics()
        assert metrics.total_texts == 2
        assert metrics.total_tokens == 5  # 2 + 3

    def test_profile_batch(self):
        tokenizer = MockTokenizer()
        profiler = ThroughputProfiler(tokenizer)
        texts = ["hello world", "foo bar", "test"]
        batch_metrics = profiler.profile_batch(texts)
        assert isinstance(batch_metrics, BatchMetrics)
        assert batch_metrics.batch_size == 3

    def test_batch_max_min_length(self):
        tokenizer = MockTokenizer()
        profiler = ThroughputProfiler(tokenizer)
        texts = ["a", "a b c", "a b"]
        batch_metrics = profiler.profile_batch(texts)
        assert batch_metrics.min_length == 1
        assert batch_metrics.max_length == 3

    def test_get_metrics(self):
        tokenizer = MockTokenizer()
        profiler = ThroughputProfiler(tokenizer)
        profiler.profile_text("hello world")
        metrics = profiler.get_metrics()
        assert isinstance(metrics, ThroughputMetrics)
        assert metrics.total_texts == 1
        assert metrics.total_tokens == 2
        assert metrics.total_chars == len("hello world")

    def test_reset(self):
        tokenizer = MockTokenizer()
        profiler = ThroughputProfiler(tokenizer)
        profiler.profile_text("hello world")
        profiler.reset()
        metrics = profiler.get_metrics()
        assert metrics.total_texts == 0
        assert metrics.total_tokens == 0

    def test_tokens_per_second_calculation(self):
        tokenizer = MockTokenizer()
        profiler = ThroughputProfiler(tokenizer)
        profiler.profile_text("a b c d e f g h i j")
        metrics = profiler.get_metrics()
        assert metrics.tokens_per_second >= 0.0
        if metrics.elapsed_seconds > 0:
            assert metrics.tokens_per_second == metrics.total_tokens / metrics.elapsed_seconds

    def test_chars_per_token(self):
        tokenizer = MockTokenizer()
        profiler = ThroughputProfiler(tokenizer)
        profiler.profile_text("hello world")
        metrics = profiler.get_metrics()
        if metrics.total_tokens > 0:
            assert metrics.avg_chars_per_token == metrics.total_chars / metrics.total_tokens

    def test_avg_tokens_per_text(self):
        tokenizer = MockTokenizer()
        profiler = ThroughputProfiler(tokenizer)
        profiler.profile_text("hello world")  # 2 tokens
        profiler.profile_text("a b c d")  # 4 tokens
        metrics = profiler.get_metrics()
        assert metrics.avg_tokens_per_text == 3.0  # (2+4)/2

    def test_attention_waste_calculation(self):
        tokenizer = MockTokenizer()
        profiler = ThroughputProfiler(tokenizer)
        texts = ["a", "a b c d e"]  # Very different lengths
        batch_metrics = profiler.profile_batch(texts)
        # Padding waste = (5-1)+(5-5) / (5*2) = 4/10 = 0.4
        assert batch_metrics.attention_waste > 0.0

    def test_special_tokens_parameter(self):
        tokenizer = MockTokenizer()
        profiler = ThroughputProfiler(tokenizer)
        num_tokens = profiler.profile_text("hello", add_special_tokens=False)
        assert num_tokens == 1


class TestProfileTokenization:
    """Tests for profile_tokenization function."""

    def test_basic_profiling(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "foo bar baz"]
        metrics = profile_tokenization(texts, tokenizer)
        assert isinstance(metrics, ThroughputMetrics)
        assert metrics.total_texts == 2
        assert metrics.total_tokens == 5

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        texts: list[str] = []
        metrics = profile_tokenization(texts, tokenizer)
        assert metrics.total_texts == 0
        assert metrics.total_tokens == 0

    def test_single_text(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        metrics = profile_tokenization(texts, tokenizer)
        assert metrics.total_texts == 1

    def test_with_special_tokens(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        metrics = profile_tokenization(texts, tokenizer, add_special_tokens=True)
        assert metrics.total_tokens >= 0


class TestEstimateTrainingTokens:
    """Tests for estimate_training_tokens function."""

    def test_basic_estimate(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "foo bar"]
        estimate = estimate_training_tokens(texts, tokenizer)
        assert "sample_tokens" in estimate
        assert "total_training_tokens" in estimate
        assert estimate["sample_tokens"] == 4  # 2 + 2

    def test_with_epochs(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        estimate = estimate_training_tokens(texts, tokenizer, epochs=3)
        assert estimate["epochs"] == 3
        assert estimate["total_training_tokens"] == estimate["estimated_dataset_tokens"] * 3

    def test_with_sampling(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]  # 2 tokens
        estimate = estimate_training_tokens(texts, tokenizer, sample_ratio=0.1)
        # If this is 10% sample, estimated total is 10x
        assert estimate["estimated_dataset_tokens"] == 20

    def test_full_dataset(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "foo bar"]
        estimate = estimate_training_tokens(texts, tokenizer, sample_ratio=1.0)
        assert estimate["sample_tokens"] == estimate["estimated_dataset_tokens"]

    def test_avg_tokens_per_text(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "foo bar baz"]
        estimate = estimate_training_tokens(texts, tokenizer)
        assert estimate["avg_tokens_per_text"] == 2.5  # (2+3)/2

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        texts: list[str] = []
        estimate = estimate_training_tokens(texts, tokenizer)
        assert estimate["sample_tokens"] == 0
        assert estimate["avg_tokens_per_text"] == 0

    def test_multiple_epochs(self):
        tokenizer = MockTokenizer()
        texts = ["a b c d e"]  # 5 tokens
        estimate = estimate_training_tokens(texts, tokenizer, epochs=10)
        # Total = estimated_dataset_tokens * epochs
        expected = estimate["estimated_dataset_tokens"] * 10
        assert estimate["total_training_tokens"] == expected
