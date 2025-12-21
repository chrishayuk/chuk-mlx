"""Tests for token_stats module."""

from collections import Counter

from chuk_lazarus.data.tokenizers.token_stats import (
    CompressionStats,
    CoverageStats,
    LengthDistribution,
    TokenFrequency,
    calculate_compression_ratio,
    get_rare_tokens,
    get_token_frequencies,
    get_token_length_distribution,
    get_top_tokens,
    get_vocabulary_coverage,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab=None):
        self._vocab = vocab or {"hello": 0, "world": 1, "<unk>": 2, "the": 3}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = text.lower().split()
        return [self._vocab.get(t, self._vocab.get("<unk>", 2)) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        id_to_token = {v: k for k, v in self._vocab.items()}
        return " ".join(id_to_token.get(i, "<unk>") for i in ids)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


class TestCoverageStatsModel:
    """Tests for CoverageStats Pydantic model."""

    def test_valid_coverage_stats(self):
        stats = CoverageStats(
            total_tokens=100,
            known_tokens=95,
            unknown_tokens=5,
            coverage_ratio=0.95,
        )
        assert stats.total_tokens == 100
        assert stats.known_tokens == 95
        assert stats.coverage_ratio == 0.95

    def test_coverage_ratio_bounds(self):
        # Valid bounds
        stats = CoverageStats(
            total_tokens=10, known_tokens=10, unknown_tokens=0, coverage_ratio=1.0
        )
        assert stats.coverage_ratio == 1.0

        stats = CoverageStats(
            total_tokens=10, known_tokens=0, unknown_tokens=10, coverage_ratio=0.0
        )
        assert stats.coverage_ratio == 0.0


class TestCompressionStatsModel:
    """Tests for CompressionStats Pydantic model."""

    def test_valid_compression_stats(self):
        stats = CompressionStats(
            char_count=100,
            byte_count=120,
            token_count=20,
            chars_per_token=5.0,
            bytes_per_token=6.0,
        )
        assert stats.char_count == 100
        assert stats.chars_per_token == 5.0


class TestTokenFrequencyModel:
    """Tests for TokenFrequency Pydantic model."""

    def test_valid_token_frequency(self):
        freq = TokenFrequency(token_id=42, decoded="hello", frequency=100)
        assert freq.token_id == 42
        assert freq.decoded == "hello"
        assert freq.frequency == 100


class TestLengthDistributionModel:
    """Tests for LengthDistribution Pydantic model."""

    def test_valid_length_distribution(self):
        dist = LengthDistribution(
            length_counts={1: 10, 2: 20, 3: 5},
            total_tokens=35,
            avg_length=1.86,
            max_length=3,
            min_length=1,
        )
        assert dist.total_tokens == 35
        assert dist.length_counts[2] == 20


class TestGetTokenFrequencies:
    """Tests for get_token_frequencies function."""

    def test_single_text(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        freqs = get_token_frequencies(texts, tokenizer)
        assert isinstance(freqs, Counter)
        assert freqs[0] == 1  # hello
        assert freqs[1] == 1  # world

    def test_multiple_texts(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "hello hello"]
        freqs = get_token_frequencies(texts, tokenizer)
        assert freqs[0] == 3  # hello appears 3 times

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        freqs = get_token_frequencies([], tokenizer)
        assert len(freqs) == 0

    def test_with_add_special_tokens(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        freqs = get_token_frequencies(texts, tokenizer, add_special_tokens=True)
        assert isinstance(freqs, Counter)


class TestGetVocabularyCoverage:
    """Tests for get_vocabulary_coverage function."""

    def test_full_coverage(self):
        tokenizer = MockTokenizer()
        coverage = get_vocabulary_coverage("hello world", tokenizer)
        assert coverage.total_tokens == 2
        assert coverage.known_tokens == 2
        assert coverage.unknown_tokens == 0
        assert coverage.coverage_ratio == 1.0

    def test_partial_coverage(self):
        tokenizer = MockTokenizer()
        coverage = get_vocabulary_coverage("hello unknown", tokenizer)
        assert coverage.total_tokens == 2
        assert coverage.known_tokens == 1
        assert coverage.unknown_tokens == 1
        assert coverage.coverage_ratio == 0.5

    def test_empty_text(self):
        tokenizer = MockTokenizer()
        coverage = get_vocabulary_coverage("", tokenizer)
        assert coverage.total_tokens == 0
        assert coverage.coverage_ratio == 1.0  # Empty is fully covered

    def test_custom_unk_token_id(self):
        tokenizer = MockTokenizer()
        coverage = get_vocabulary_coverage("hello unknown", tokenizer, unk_token_id=2)
        assert coverage.unknown_tokens == 1


class TestGetTokenLengthDistribution:
    """Tests for get_token_length_distribution function."""

    def test_with_sample_ids(self):
        tokenizer = MockTokenizer({"a": 0, "bb": 1, "ccc": 2})
        dist = get_token_length_distribution(tokenizer, sample_ids=[0, 1, 2])
        assert dist.total_tokens == 3
        assert dist.min_length == 1
        assert dist.max_length == 3
        assert dist.avg_length == 2.0

    def test_full_vocab(self):
        tokenizer = MockTokenizer({"a": 0, "bb": 1})
        dist = get_token_length_distribution(tokenizer)
        assert dist.total_tokens == 2

    def test_empty_sample_ids(self):
        tokenizer = MockTokenizer({"a": 0, "bb": 1})
        dist = get_token_length_distribution(tokenizer, sample_ids=[])
        assert dist.total_tokens == 0
        assert dist.avg_length == 0.0
        assert dist.max_length == 0
        assert dist.min_length == 0


class TestGetTopTokens:
    """Tests for get_top_tokens function."""

    def test_basic_top_tokens(self):
        tokenizer = MockTokenizer()
        texts = ["hello hello hello world world"]
        top = get_top_tokens(texts, tokenizer, n=2)
        assert len(top) == 2
        assert top[0].frequency >= top[1].frequency
        assert isinstance(top[0], TokenFrequency)

    def test_top_tokens_limit(self):
        tokenizer = MockTokenizer()
        texts = ["hello world the"]
        top = get_top_tokens(texts, tokenizer, n=1)
        assert len(top) == 1


class TestGetRareTokens:
    """Tests for get_rare_tokens function."""

    def test_basic_rare_tokens(self):
        tokenizer = MockTokenizer()
        texts = ["hello hello world the the the"]
        rare = get_rare_tokens(texts, tokenizer, max_freq=1)
        assert len(rare) == 1
        assert rare[0].decoded == "world"

    def test_no_rare_tokens(self):
        tokenizer = MockTokenizer()
        texts = ["hello hello world world"]
        rare = get_rare_tokens(texts, tokenizer, max_freq=0)
        assert len(rare) == 0


class TestCalculateCompressionRatio:
    """Tests for calculate_compression_ratio function."""

    def test_basic_compression(self):
        tokenizer = MockTokenizer()
        stats = calculate_compression_ratio("hello world", tokenizer)
        assert stats.char_count == 11
        assert stats.token_count == 2
        assert stats.chars_per_token == 5.5
        assert stats.bytes_per_token == 5.5  # ASCII

    def test_empty_text(self):
        tokenizer = MockTokenizer()
        stats = calculate_compression_ratio("", tokenizer)
        assert stats.char_count == 0
        assert stats.token_count == 0
        assert stats.chars_per_token == 0.0
        assert stats.bytes_per_token == 0.0

    def test_unicode_text(self):
        tokenizer = MockTokenizer({"hello": 0, "世界": 1})
        stats = calculate_compression_ratio("hello 世界", tokenizer)
        assert stats.byte_count > stats.char_count  # UTF-8 multibyte
