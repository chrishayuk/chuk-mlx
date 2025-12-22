"""Tests for token waste metrics."""

from chuk_lazarus.data.tokenizers.instrumentation.waste import (
    PaddingStats,
    TruncationStats,
    WasteReport,
    analyze_padding_waste,
    analyze_truncation_loss,
    analyze_waste,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Word-level tokenization
        tokens = text.split()
        if add_special_tokens:
            tokens = ["<s>"] + tokens + ["</s>"]
        return list(range(len(tokens)))

    @property
    def pad_token_id(self) -> int:
        return 0


class TestPaddingStats:
    """Tests for PaddingStats model."""

    def test_valid_stats(self):
        stats = PaddingStats(
            total_samples=100,
            total_positions=1000,
            total_content_tokens=700,
            total_padding_tokens=300,
            padding_rate=0.3,
            efficiency=0.7,
            mean_padding_per_sample=3.0,
            max_padding=10,
            min_padding=0,
            wasted_compute_factor=0.15,
        )
        assert stats.padding_rate == 0.3
        assert stats.efficiency == 0.7


class TestTruncationStats:
    """Tests for TruncationStats model."""

    def test_valid_stats(self):
        stats = TruncationStats(
            total_samples=100,
            truncated_samples=10,
            truncation_rate=0.1,
            total_tokens_lost=500,
            mean_tokens_lost=50.0,
            max_tokens_lost=200,
            content_loss_rate=0.05,
            minor_truncation=5,
            major_truncation=3,
            severe_truncation=2,
        )
        assert stats.truncation_rate == 0.1
        assert stats.total_tokens_lost == 500


class TestWasteReport:
    """Tests for WasteReport model."""

    def test_valid_report(self):
        padding = PaddingStats(
            total_samples=100,
            total_positions=1000,
            total_content_tokens=700,
            total_padding_tokens=300,
            padding_rate=0.3,
            efficiency=0.7,
            mean_padding_per_sample=3.0,
            max_padding=10,
            min_padding=0,
            wasted_compute_factor=0.15,
        )
        truncation = TruncationStats(
            total_samples=100,
            truncated_samples=0,
            truncation_rate=0,
            total_tokens_lost=0,
            mean_tokens_lost=0,
            max_tokens_lost=0,
            content_loss_rate=0,
            minor_truncation=0,
            major_truncation=0,
            severe_truncation=0,
        )
        report = WasteReport(
            max_length=512,
            total_samples=100,
            padding=padding,
            truncation=truncation,
            overall_efficiency=0.7,
        )
        assert report.max_length == 512


class TestAnalyzePaddingWaste:
    """Tests for analyze_padding_waste function."""

    def test_basic_padding(self):
        tokenizer = MockTokenizer()
        texts = ["short", "also short", "a bit longer text here"]

        stats = analyze_padding_waste(texts, tokenizer, max_length=20)

        assert stats.total_samples == 3
        assert stats.total_positions == 60  # 3 * 20
        assert stats.padding_rate > 0
        assert stats.efficiency < 1

    def test_no_padding(self):
        tokenizer = MockTokenizer()
        texts = ["word " * 10]  # 10 words + 2 special = 12 tokens

        stats = analyze_padding_waste(texts, tokenizer, max_length=12)

        # With max_length = exact token count, minimal padding
        assert stats.padding_rate < 0.1

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        stats = analyze_padding_waste([], tokenizer, max_length=512)

        assert stats.total_samples == 0
        assert stats.efficiency == 1

    def test_high_padding(self):
        tokenizer = MockTokenizer()
        texts = ["hi"]  # Very short

        stats = analyze_padding_waste(texts, tokenizer, max_length=512)

        # Should have high padding rate
        assert stats.padding_rate > 0.9


class TestAnalyzeTruncationLoss:
    """Tests for analyze_truncation_loss function."""

    def test_no_truncation(self):
        tokenizer = MockTokenizer()
        texts = ["short text", "also short"]

        stats = analyze_truncation_loss(texts, tokenizer, max_length=512)

        assert stats.truncated_samples == 0
        assert stats.truncation_rate == 0
        assert stats.total_tokens_lost == 0

    def test_with_truncation(self):
        tokenizer = MockTokenizer()
        texts = ["word " * 100]  # 100 words = 102 tokens with special

        stats = analyze_truncation_loss(texts, tokenizer, max_length=50)

        assert stats.truncated_samples == 1
        assert stats.total_tokens_lost > 0
        assert stats.content_loss_rate > 0

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        stats = analyze_truncation_loss([], tokenizer, max_length=512)

        assert stats.total_samples == 0

    def test_severity_categories(self):
        tokenizer = MockTokenizer()
        texts = [
            "word " * 55,  # ~57 tokens, minor truncation at max=50
            "word " * 100,  # ~102 tokens, major truncation at max=50
            "word " * 200,  # ~202 tokens, severe truncation at max=50
        ]

        stats = analyze_truncation_loss(texts, tokenizer, max_length=50)

        assert stats.truncated_samples == 3
        # Should categorize by severity
        assert stats.minor_truncation + stats.major_truncation + stats.severe_truncation == 3


class TestAnalyzeWaste:
    """Tests for analyze_waste function."""

    def test_basic_analysis(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "test", "another sample text here"]

        report = analyze_waste(texts, tokenizer, max_length=20)

        assert report.total_samples == 3
        assert report.max_length == 20
        assert 0 <= report.overall_efficiency <= 1

    def test_recommendations_generated(self):
        tokenizer = MockTokenizer()
        texts = ["hi" for _ in range(10)]  # Very short, high padding

        report = analyze_waste(texts, tokenizer, max_length=512)

        # Should recommend something about padding
        assert len(report.recommendations) > 0

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        report = analyze_waste([], tokenizer, max_length=512)

        assert report.total_samples == 0

    def test_combined_metrics(self):
        tokenizer = MockTokenizer()
        texts = [
            "short",
            "word " * 100,  # Will be truncated
        ]

        report = analyze_waste(texts, tokenizer, max_length=50)

        # Should have both padding and truncation
        assert report.padding.total_padding_tokens > 0
        assert report.truncation.truncated_samples > 0
