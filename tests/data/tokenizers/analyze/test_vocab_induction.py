"""Tests for vocabulary induction analysis."""

from chuk_lazarus.data.tokenizers.analyze.vocab_induction import (
    DomainVocab,
    InductionConfig,
    InductionReport,
    TokenCandidate,
    TokenDomain,
    analyze_vocab_induction,
    find_fragmented_words,
    find_frequent_ngrams,
    get_domain_vocab,
    list_domain_vocabs,
    suggest_domain_tokens,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.vocab = {chr(i): i for i in range(32, 127)}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Character-level tokenization
        return [self.vocab.get(c, 0) for c in text if c in self.vocab]

    def decode(self, token_ids: list[int]) -> str:
        id_to_char = {v: k for k, v in self.vocab.items()}
        return "".join(id_to_char.get(i, "?") for i in token_ids)

    def get_vocab(self) -> dict[str, int]:
        return self.vocab


class TestTokenDomain:
    """Tests for TokenDomain enum."""

    def test_domain_values(self):
        assert TokenDomain.MATH.value == "math"
        assert TokenDomain.CODE.value == "code"
        assert TokenDomain.TOOL.value == "tool"
        assert TokenDomain.GENERAL.value == "general"


class TestInductionConfig:
    """Tests for InductionConfig model."""

    def test_default_values(self):
        config = InductionConfig()
        assert config.min_frequency == 5
        assert config.min_fragmentation == 3
        assert config.max_candidates == 50

    def test_custom_values(self):
        config = InductionConfig(
            min_frequency=10,
            min_fragmentation=4,
            max_candidates=100,
        )
        assert config.min_frequency == 10
        assert config.min_fragmentation == 4


class TestTokenCandidate:
    """Tests for TokenCandidate model."""

    def test_valid_candidate(self):
        candidate = TokenCandidate(
            token_str="preprocessing",
            frequency=50,
            current_tokens=5,
            savings_per_occurrence=4,
            total_savings=200,
            domain=TokenDomain.GENERAL,
            priority_score=150.0,
        )
        assert candidate.token_str == "preprocessing"
        assert candidate.total_savings == 200

    def test_with_examples(self):
        candidate = TokenCandidate(
            token_str="test",
            frequency=10,
            current_tokens=3,
            savings_per_occurrence=2,
            total_savings=20,
            domain=TokenDomain.CODE,
            priority_score=10.0,
            examples=["context 1", "context 2"],
        )
        assert len(candidate.examples) == 2


class TestInductionReport:
    """Tests for InductionReport model."""

    def test_valid_report(self):
        report = InductionReport(
            total_candidates=10,
            total_potential_savings=500,
            savings_percent=5.0,
            candidates=[],
        )
        assert report.total_candidates == 10
        assert report.savings_percent == 5.0

    def test_with_candidates(self):
        candidate = TokenCandidate(
            token_str="test",
            frequency=10,
            current_tokens=3,
            savings_per_occurrence=2,
            total_savings=20,
            domain=TokenDomain.GENERAL,
            priority_score=10.0,
        )
        report = InductionReport(
            total_candidates=1,
            total_potential_savings=20,
            savings_percent=1.0,
            candidates=[candidate],
        )
        assert len(report.candidates) == 1


class TestDomainVocab:
    """Tests for DomainVocab model."""

    def test_valid_vocab(self):
        vocab = DomainVocab(
            domain=TokenDomain.MATH,
            tokens=["π", "∑", "∫"],
            description="Math symbols",
        )
        assert vocab.domain == TokenDomain.MATH
        assert len(vocab.tokens) == 3


class TestFindFragmentedWords:
    """Tests for find_fragmented_words function."""

    def test_basic_fragmentation(self):
        tokenizer = MockTokenizer()
        texts = ["preprocessing " * 10]  # Repeat to meet frequency
        config = InductionConfig(min_frequency=5, min_fragmentation=3)
        candidates = find_fragmented_words(texts, tokenizer, config)
        # Should find fragmented words
        assert isinstance(candidates, list)

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        candidates = find_fragmented_words([], tokenizer)
        assert candidates == []

    def test_short_words_filtered(self):
        tokenizer = MockTokenizer()
        texts = ["a b c d e " * 20]
        config = InductionConfig(min_word_length=4)
        candidates = find_fragmented_words(texts, tokenizer, config)
        # Short words should be filtered out
        for c in candidates:
            assert len(c.token_str) >= 4

    def test_camelcase_detection(self):
        tokenizer = MockTokenizer()
        texts = ["MyClassName " * 10]
        config = InductionConfig(min_frequency=5)
        candidates = find_fragmented_words(texts, tokenizer, config)
        # Should detect camelCase patterns
        assert isinstance(candidates, list)

    def test_snake_case_detection(self):
        tokenizer = MockTokenizer()
        texts = ["my_function_name " * 10]
        config = InductionConfig(min_frequency=5)
        candidates = find_fragmented_words(texts, tokenizer, config)
        assert isinstance(candidates, list)


class TestFindFrequentNgrams:
    """Tests for find_frequent_ngrams function."""

    def test_basic_ngrams(self):
        tokenizer = MockTokenizer()
        texts = ["the quick brown fox " * 20]
        candidates = find_frequent_ngrams(texts, tokenizer, min_frequency=5)
        assert isinstance(candidates, list)

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        candidates = find_frequent_ngrams([], tokenizer)
        assert candidates == []

    def test_ngram_range(self):
        tokenizer = MockTokenizer()
        texts = ["hello world test " * 20]
        candidates = find_frequent_ngrams(texts, tokenizer, n_range=(2, 3), min_frequency=5)
        assert isinstance(candidates, list)


class TestSuggestDomainTokens:
    """Tests for suggest_domain_tokens function."""

    def test_math_domain(self):
        tokenizer = MockTokenizer()
        texts = ["The sum ∑ is used in math π equations"]
        candidates = suggest_domain_tokens(texts, tokenizer, [TokenDomain.MATH])
        # Should find math symbols
        assert isinstance(candidates, list)

    def test_code_domain(self):
        tokenizer = MockTokenizer()
        texts = ["def my_function(): class MyClass: import os"]
        candidates = suggest_domain_tokens(texts, tokenizer, [TokenDomain.CODE])
        assert isinstance(candidates, list)

    def test_tool_domain(self):
        tokenizer = MockTokenizer()
        texts = ["<TOOL_CALL> get_weather </TOOL_CALL>"]
        candidates = suggest_domain_tokens(texts, tokenizer, [TokenDomain.TOOL])
        assert isinstance(candidates, list)

    def test_all_domains(self):
        tokenizer = MockTokenizer()
        texts = ["Test text with ∑ and def function()"]
        candidates = suggest_domain_tokens(texts, tokenizer)
        assert isinstance(candidates, list)

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        candidates = suggest_domain_tokens([], tokenizer)
        assert candidates == []


class TestAnalyzeVocabInduction:
    """Tests for analyze_vocab_induction function."""

    def test_basic_analysis(self):
        tokenizer = MockTokenizer()
        texts = ["preprocessing tokenization " * 10]
        report = analyze_vocab_induction(texts, tokenizer)
        assert isinstance(report, InductionReport)
        assert report.total_candidates >= 0

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        report = analyze_vocab_induction([], tokenizer)
        assert report.total_candidates == 0

    def test_with_config(self):
        tokenizer = MockTokenizer()
        texts = ["hello world " * 20]
        config = InductionConfig(max_candidates=5)
        report = analyze_vocab_induction(texts, tokenizer, config)
        assert len(report.candidates) <= 5

    def test_domain_breakdown(self):
        tokenizer = MockTokenizer()
        texts = ["preprocessing " * 10]
        report = analyze_vocab_induction(texts, tokenizer)
        assert isinstance(report.domain_breakdown, dict)

    def test_recommendations_generated(self):
        tokenizer = MockTokenizer()
        texts = ["test " * 100]
        report = analyze_vocab_induction(texts, tokenizer)
        assert isinstance(report.recommendations, list)

    def test_savings_calculation(self):
        tokenizer = MockTokenizer()
        texts = ["preprocessing " * 50]  # Long word repeated
        report = analyze_vocab_induction(texts, tokenizer)
        # Should calculate some savings for fragmented words
        assert report.total_potential_savings >= 0


class TestGetDomainVocab:
    """Tests for get_domain_vocab function."""

    def test_math_vocab(self):
        vocab = get_domain_vocab(TokenDomain.MATH)
        assert vocab is not None
        assert vocab.domain == TokenDomain.MATH
        assert len(vocab.tokens) > 0

    def test_code_vocab(self):
        vocab = get_domain_vocab(TokenDomain.CODE)
        assert vocab is not None
        assert vocab.domain == TokenDomain.CODE

    def test_tool_vocab(self):
        vocab = get_domain_vocab(TokenDomain.TOOL)
        assert vocab is not None
        assert vocab.domain == TokenDomain.TOOL

    def test_unknown_domain(self):
        vocab = get_domain_vocab(TokenDomain.CUSTOM)
        assert vocab is None


class TestListDomainVocabs:
    """Tests for list_domain_vocabs function."""

    def test_returns_list(self):
        vocabs = list_domain_vocabs()
        assert isinstance(vocabs, list)
        assert len(vocabs) >= 3

    def test_contains_main_domains(self):
        vocabs = list_domain_vocabs()
        domains = {v.domain for v in vocabs}
        assert TokenDomain.MATH in domains
        assert TokenDomain.CODE in domains
        assert TokenDomain.TOOL in domains
