"""Tests for entropy module."""

from collections import Counter

from chuk_lazarus.data.tokenizers.analyze.entropy import (
    EntropyReport,
    TokenDistribution,
    analyze_entropy,
    calculate_entropy,
    get_token_distribution,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab: dict[str, int] | None = None):
        self._vocab = vocab or {
            "<pad>": 0,
            "<unk>": 1,
            "hello": 2,
            "world": 3,
            "the": 4,
            "test": 5,
        }
        self._id_to_token = {v: k for k, v in self._vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        words = text.lower().split()
        return [self._vocab.get(w, 1) for w in words]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._id_to_token.get(i, "<unk>") for i in ids)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab.copy()


class TestTokenDistributionModel:
    """Tests for TokenDistribution model."""

    def test_valid_distribution(self):
        dist = TokenDistribution(
            total_tokens=100,
            unique_tokens=20,
            type_token_ratio=0.2,
            top_tokens=[(1, "hello", 50), (2, "world", 30)],
            singleton_count=5,
            singleton_ratio=0.25,
        )
        assert dist.total_tokens == 100
        assert dist.unique_tokens == 20
        assert len(dist.top_tokens) == 2

    def test_default_top_tokens(self):
        dist = TokenDistribution(
            total_tokens=10,
            unique_tokens=5,
            type_token_ratio=0.5,
            singleton_count=2,
            singleton_ratio=0.4,
        )
        assert dist.top_tokens == []

    def test_singleton_fields(self):
        dist = TokenDistribution(
            total_tokens=10,
            unique_tokens=5,
            type_token_ratio=0.5,
            singleton_count=3,
            singleton_ratio=0.6,
        )
        assert dist.singleton_count == 3
        assert dist.singleton_ratio == 0.6


class TestEntropyReportModel:
    """Tests for EntropyReport model."""

    def test_valid_report(self):
        dist = TokenDistribution(
            total_tokens=100,
            unique_tokens=20,
            type_token_ratio=0.2,
            singleton_count=5,
            singleton_ratio=0.25,
        )
        report = EntropyReport(
            entropy=3.5,
            normalized_entropy=0.75,
            perplexity=11.31,
            distribution=dist,
            uniformity_score=0.6,
            concentration_ratio=0.3,
        )
        assert report.entropy == 3.5
        assert report.normalized_entropy == 0.75
        assert report.distribution.total_tokens == 100

    def test_zero_entropy(self):
        dist = TokenDistribution(
            total_tokens=10,
            unique_tokens=1,
            type_token_ratio=0.1,
            singleton_count=0,
            singleton_ratio=0.0,
        )
        report = EntropyReport(
            entropy=0.0,
            normalized_entropy=0.0,
            perplexity=1.0,
            distribution=dist,
            uniformity_score=1.0,
            concentration_ratio=1.0,
        )
        assert report.entropy == 0.0
        assert report.perplexity == 1.0


class TestCalculateEntropy:
    """Tests for calculate_entropy function."""

    def test_uniform_distribution(self):
        # All tokens have same frequency - maximum entropy
        counter = Counter({1: 10, 2: 10, 3: 10, 4: 10})
        entropy = calculate_entropy(counter)
        assert entropy > 0.0
        # With 4 equal items, entropy should be log2(4) = 2.0
        assert abs(entropy - 2.0) < 0.01

    def test_single_token(self):
        # All same token - entropy is 0
        counter = Counter({1: 100})
        entropy = calculate_entropy(counter)
        assert entropy == 0.0

    def test_skewed_distribution(self):
        # One token dominates
        counter = Counter({1: 900, 2: 50, 3: 25, 4: 25})
        entropy = calculate_entropy(counter)
        # Entropy should be low but not zero
        assert 0.0 < entropy < 2.0

    def test_empty_counts(self):
        counter: Counter[int] = Counter()
        entropy = calculate_entropy(counter)
        assert entropy == 0.0

    def test_single_count(self):
        counter = Counter({1: 1})
        entropy = calculate_entropy(counter)
        assert entropy == 0.0


class TestGetTokenDistribution:
    """Tests for get_token_distribution function."""

    def test_basic_distribution(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "hello world hello"]
        dist = get_token_distribution(texts, tokenizer)
        assert isinstance(dist, TokenDistribution)
        assert dist.total_tokens > 0

    def test_unique_tokens(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        dist = get_token_distribution(texts, tokenizer)
        assert dist.unique_tokens == 2  # "hello" and "world"

    def test_type_token_ratio(self):
        tokenizer = MockTokenizer()
        texts = ["hello hello hello hello"]
        dist = get_token_distribution(texts, tokenizer)
        # 1 unique token, 4 total => 0.25
        assert dist.type_token_ratio == 0.25

    def test_top_tokens_sorted(self):
        tokenizer = MockTokenizer()
        texts = ["hello hello hello world world test"]
        dist = get_token_distribution(texts, tokenizer)
        # Should be sorted by frequency
        if len(dist.top_tokens) >= 2:
            assert dist.top_tokens[0][2] >= dist.top_tokens[1][2]

    def test_top_n_limit(self):
        tokenizer = MockTokenizer()
        texts = ["hello world the test"]
        dist = get_token_distribution(texts, tokenizer, top_n=2)
        assert len(dist.top_tokens) <= 2

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        texts: list[str] = []
        dist = get_token_distribution(texts, tokenizer)
        assert dist.total_tokens == 0
        assert dist.unique_tokens == 0

    def test_singleton_count(self):
        tokenizer = MockTokenizer()
        texts = ["hello world the test"]  # Each word appears once
        dist = get_token_distribution(texts, tokenizer)
        # All tokens are singletons
        assert dist.singleton_count == dist.unique_tokens


class TestAnalyzeEntropy:
    """Tests for analyze_entropy function."""

    def test_basic_entropy(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "hello test"]
        report = analyze_entropy(texts, tokenizer)
        assert isinstance(report, EntropyReport)
        assert report.entropy >= 0.0
        assert report.distribution.total_tokens > 0

    def test_perplexity_calculation(self):
        tokenizer = MockTokenizer()
        texts = ["hello world hello world"]
        report = analyze_entropy(texts, tokenizer)
        # Perplexity should be 2^entropy
        assert abs(report.perplexity - 2**report.entropy) < 0.01

    def test_normalized_entropy(self):
        tokenizer = MockTokenizer()
        texts = ["hello world the test"]
        report = analyze_entropy(texts, tokenizer)
        # Normalized should be between 0 and 1
        assert 0.0 <= report.normalized_entropy <= 1.0

    def test_uniformity_score(self):
        tokenizer = MockTokenizer()
        texts = ["hello hello hello hello"]
        report = analyze_entropy(texts, tokenizer)
        # All same token = perfect uniformity within that token
        assert report.uniformity_score >= 0.0

    def test_concentration_ratio(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        report = analyze_entropy(texts, tokenizer)
        # Concentration should be between 0 and 1
        assert 0.0 <= report.concentration_ratio <= 1.0

    def test_single_token_entropy(self):
        tokenizer = MockTokenizer()
        texts = ["hello hello hello"]
        report = analyze_entropy(texts, tokenizer)
        assert report.entropy == 0.0
        assert report.distribution.unique_tokens == 1

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        texts: list[str] = []
        report = analyze_entropy(texts, tokenizer)
        assert report.entropy == 0.0
        assert report.distribution.total_tokens == 0

    def test_distribution_included(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        report = analyze_entropy(texts, tokenizer)
        assert isinstance(report.distribution, TokenDistribution)

    def test_high_entropy_uniform(self):
        tokenizer = MockTokenizer()
        # Each word appears once - high diversity
        texts = ["hello", "world", "the", "test"]
        report = analyze_entropy(texts, tokenizer)
        assert report.entropy > 0.0

    def test_low_entropy_skewed(self):
        tokenizer = MockTokenizer()
        # One token dominates
        texts = ["hello " * 100]
        report = analyze_entropy(texts, tokenizer)
        # Low entropy when one token dominates
        assert report.entropy >= 0.0
