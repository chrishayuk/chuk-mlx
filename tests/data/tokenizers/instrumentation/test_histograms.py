"""Tests for token length histograms."""

from chuk_lazarus.data.tokenizers.instrumentation.histograms import (
    HistogramBin,
    LengthHistogram,
    PercentileStats,
    compute_length_histogram,
    compute_percentiles,
    format_histogram_ascii,
    get_length_stats,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Simple word-level tokenization
        tokens = text.split()
        if add_special_tokens:
            tokens = ["<s>"] + tokens + ["</s>"]
        return list(range(len(tokens)))


class TestPercentileStats:
    """Tests for PercentileStats model."""

    def test_valid_stats(self):
        stats = PercentileStats(p10=10, p25=25, p50=50, p75=75, p90=90, p95=95, p99=99)
        assert stats.p50 == 50
        assert stats.p95 == 95


class TestHistogramBin:
    """Tests for HistogramBin model."""

    def test_valid_bin(self):
        bin = HistogramBin(min_value=0, max_value=10, count=50, percentage=25.0)
        assert bin.min_value == 0
        assert bin.count == 50


class TestLengthHistogram:
    """Tests for LengthHistogram model."""

    def test_valid_histogram(self):
        percentiles = PercentileStats(p10=5, p25=10, p50=20, p75=30, p90=40, p95=45, p99=50)
        histogram = LengthHistogram(
            total_samples=100,
            total_tokens=2000,
            min_length=5,
            max_length=50,
            mean_length=20.0,
            std_length=10.0,
            percentiles=percentiles,
            bins=[],
            bin_width=5,
            recommended_max_length=512,
            samples_over_2048=0,
            samples_over_4096=0,
        )
        assert histogram.total_samples == 100
        assert histogram.mean_length == 20.0


class TestComputePercentiles:
    """Tests for compute_percentiles function."""

    def test_basic_percentiles(self):
        lengths = list(range(1, 101))  # 1 to 100
        stats = compute_percentiles(lengths)
        assert stats.p10 == 10
        assert stats.p50 == 50
        assert stats.p90 == 90

    def test_empty_list(self):
        stats = compute_percentiles([])
        assert stats.p50 == 0

    def test_single_value(self):
        stats = compute_percentiles([42])
        assert stats.p50 == 42


class TestComputeLengthHistogram:
    """Tests for compute_length_histogram function."""

    def test_basic_histogram(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "this is a test", "short", "another longer sentence here"]

        histogram = compute_length_histogram(texts, tokenizer)

        assert histogram.total_samples == 4
        assert histogram.total_tokens > 0
        assert histogram.min_length > 0
        assert histogram.max_length >= histogram.min_length

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        histogram = compute_length_histogram([], tokenizer)
        assert histogram.total_samples == 0
        assert histogram.total_tokens == 0

    def test_histogram_bins(self):
        tokenizer = MockTokenizer()
        texts = ["word " * i for i in range(1, 21)]

        histogram = compute_length_histogram(texts, tokenizer, num_bins=5)

        assert len(histogram.bins) == 5
        assert all(isinstance(b, HistogramBin) for b in histogram.bins)

    def test_percentiles_populated(self):
        tokenizer = MockTokenizer()
        texts = ["word " * i for i in range(1, 101)]

        histogram = compute_length_histogram(texts, tokenizer)

        assert histogram.percentiles.p50 > 0
        assert histogram.percentiles.p95 > histogram.percentiles.p50

    def test_recommendations(self):
        tokenizer = MockTokenizer()
        texts = ["word " * 100 for _ in range(10)]  # Long texts

        histogram = compute_length_histogram(texts, tokenizer)

        # Should recommend a reasonable max length
        assert histogram.recommended_max_length in [128, 256, 512, 1024, 2048, 4096, 8192]


class TestGetLengthStats:
    """Tests for get_length_stats function."""

    def test_basic_stats(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "test", "another sentence"]

        stats = get_length_stats(texts, tokenizer)

        assert stats["total_samples"] == 3
        assert stats["total_tokens"] > 0
        assert stats["min"] > 0
        assert stats["max"] >= stats["min"]

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        stats = get_length_stats([], tokenizer)
        assert stats["total_samples"] == 0


class TestFormatHistogramAscii:
    """Tests for format_histogram_ascii function."""

    def test_basic_formatting(self):
        tokenizer = MockTokenizer()
        texts = ["word " * i for i in range(1, 21)]

        histogram = compute_length_histogram(texts, tokenizer)
        output = format_histogram_ascii(histogram)

        assert "TOKEN LENGTH HISTOGRAM" in output
        assert "Samples:" in output
        assert "Mean:" in output

    def test_percentiles_shown(self):
        tokenizer = MockTokenizer()
        texts = ["word " * 10 for _ in range(50)]

        histogram = compute_length_histogram(texts, tokenizer)
        output = format_histogram_ascii(histogram, show_percentiles=True)

        assert "Percentiles:" in output
        assert "p50=" in output

    def test_recommendations_shown(self):
        tokenizer = MockTokenizer()
        texts = ["word " * 10 for _ in range(20)]

        histogram = compute_length_histogram(texts, tokenizer)
        output = format_histogram_ascii(histogram)

        assert "Recommended max_length:" in output
