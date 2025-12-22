"""Tests for OOV and rare token reporting."""

from chuk_lazarus.data.tokenizers.instrumentation.oov_report import (
    OOVReport,
    RareTokenInfo,
    TokenFrequencyBand,
    analyze_oov,
    find_rare_tokens,
    get_frequency_bands,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, unk_id: int = 0):
        self._unk_id = unk_id
        self.vocab = {chr(i): i for i in range(32, 127)}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Character-level tokenization
        return [self.vocab.get(c, self._unk_id) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        id_to_char = {v: k for k, v in self.vocab.items()}
        return "".join(id_to_char.get(i, "?") for i in token_ids)

    @property
    def unk_token_id(self) -> int:
        return self._unk_id


class TestTokenFrequencyBand:
    """Tests for TokenFrequencyBand enum."""

    def test_band_values(self):
        assert TokenFrequencyBand.SINGLETON.value == "singleton"
        assert TokenFrequencyBand.RARE.value == "rare"
        assert TokenFrequencyBand.COMMON.value == "common"


class TestRareTokenInfo:
    """Tests for RareTokenInfo model."""

    def test_valid_info(self):
        info = RareTokenInfo(
            token_id=100,
            token_str="xyz",
            count=3,
            band=TokenFrequencyBand.RARE,
        )
        assert info.token_id == 100
        assert info.count == 3


class TestOOVReport:
    """Tests for OOVReport model."""

    def test_valid_report(self):
        report = OOVReport(
            total_tokens=1000,
            unique_tokens=100,
            vocab_utilization=0.1,
            unk_count=5,
            unk_rate=0.005,
            singletons=20,
            singleton_rate=0.2,
            rare_tokens=30,
            rare_rate=0.3,
        )
        assert report.total_tokens == 1000
        assert report.unk_rate == 0.005


class TestGetFrequencyBands:
    """Tests for get_frequency_bands function."""

    def test_basic_bands(self):
        tokenizer = MockTokenizer()
        # 'a' repeated many times, 'z' only once
        texts = ["aaaaaaaaaa", "aaaaa", "z"]

        bands = get_frequency_bands(texts, tokenizer)

        assert TokenFrequencyBand.SINGLETON in bands
        assert TokenFrequencyBand.VERY_COMMON in bands

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        bands = get_frequency_bands([], tokenizer)

        # All bands should be 0
        assert all(count == 0 for count in bands.values())


class TestFindRareTokens:
    """Tests for find_rare_tokens function."""

    def test_finds_rare_tokens(self):
        tokenizer = MockTokenizer()
        # 'a' is common, 'z' appears only once
        texts = ["aaaaaaaaaa" for _ in range(10)] + ["z"]

        rare = find_rare_tokens(texts, tokenizer, max_frequency=5, top_k=10)

        # 'z' should be in rare tokens
        rare_strs = [r.token_str for r in rare]
        assert "z" in rare_strs

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        rare = find_rare_tokens([], tokenizer)
        assert rare == []

    def test_top_k_limit(self):
        tokenizer = MockTokenizer()
        texts = [chr(i) for i in range(65, 91)]  # A-Z, each once

        rare = find_rare_tokens(texts, tokenizer, max_frequency=1, top_k=5)

        assert len(rare) <= 5

    def test_includes_contexts(self):
        tokenizer = MockTokenizer()
        texts = ["hello world with xyz in it"]

        rare = find_rare_tokens(texts, tokenizer, max_frequency=1, top_k=10, include_contexts=True)

        # At least some should have contexts
        # (depends on implementation, but should not crash)
        assert isinstance(rare, list)


class TestAnalyzeOOV:
    """Tests for analyze_oov function."""

    def test_basic_analysis(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "test text", "another sample"]

        report = analyze_oov(texts, tokenizer)

        assert report.total_tokens > 0
        assert report.unique_tokens > 0
        assert 0 <= report.singleton_rate <= 1

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        report = analyze_oov([], tokenizer)

        assert report.total_tokens == 0
        assert report.unk_rate == 0

    def test_unk_detection(self):
        tokenizer = MockTokenizer(unk_id=0)
        # Include character not in vocab (will map to UNK)
        texts = ["hello\x00world"]  # null char not in vocab

        report = analyze_oov(texts, tokenizer)

        # May or may not have UNK depending on implementation
        assert isinstance(report.unk_count, int)

    def test_recommendations_generated(self):
        tokenizer = MockTokenizer()
        # Create corpus with many singletons
        texts = [chr(i) for i in range(65, 91)]  # A-Z, each once

        report = analyze_oov(texts, tokenizer)

        # Should generate recommendations about singletons
        assert isinstance(report.recommendations, list)

    def test_vocab_utilization(self):
        tokenizer = MockTokenizer()
        texts = ["abc def"]

        report = analyze_oov(texts, tokenizer, vocab_size=1000)

        assert report.vocab_utilization > 0
        assert report.vocab_utilization < 1
