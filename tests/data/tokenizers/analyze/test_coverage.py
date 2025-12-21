"""Tests for coverage module."""

from chuk_lazarus.data.tokenizers.analyze.coverage import (
    CoverageReport,
    FragmentAnalysis,
    analyze_coverage,
    analyze_fragments,
    get_tokens_per_word,
    get_unk_rate,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab: dict[str, int] | None = None, unk_id: int = 1):
        self._vocab = vocab or {
            "<pad>": 0,
            "<unk>": 1,
            "hello": 2,
            "world": 3,
            "the": 4,
            "a": 5,
            "test": 6,
            "##ing": 7,
            "▁word": 8,
            ",": 9,
            ".": 10,
        }
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        self._unk_id = unk_id

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = text.lower().split()
        return [self._vocab.get(t, self._unk_id) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._id_to_token.get(i, "<unk>") for i in ids)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab.copy()


class TestFragmentAnalysisModel:
    """Tests for FragmentAnalysis model."""

    def test_valid_analysis(self):
        analysis = FragmentAnalysis(
            total_tokens=100,
            fragment_tokens=20,
            fragment_ratio=0.2,
            top_fragments=[("##ing", 10), ("##ed", 5)],
            whitespace_tokens=30,
            punctuation_tokens=10,
        )
        assert analysis.total_tokens == 100
        assert analysis.fragment_ratio == 0.2
        assert len(analysis.top_fragments) == 2

    def test_empty_analysis(self):
        analysis = FragmentAnalysis(
            total_tokens=0,
            fragment_tokens=0,
            fragment_ratio=0.0,
            top_fragments=[],
            whitespace_tokens=0,
            punctuation_tokens=0,
        )
        assert analysis.total_tokens == 0
        assert analysis.fragment_ratio == 0.0


class TestCoverageReportModel:
    """Tests for CoverageReport model."""

    def test_valid_report(self):
        report = CoverageReport(
            total_texts=10,
            total_words=100,
            total_tokens=150,
            tokens_per_word=1.5,
            unk_count=5,
            unk_rate=0.033,
            vocab_utilization=0.5,
            unique_tokens_used=500,
            vocab_size=1000,
        )
        assert report.total_texts == 10
        assert report.tokens_per_word == 1.5
        assert report.unk_rate == 0.033

    def test_report_with_warnings(self):
        report = CoverageReport(
            total_texts=5,
            total_words=50,
            total_tokens=100,
            tokens_per_word=2.0,
            unk_count=0,
            unk_rate=0.0,
            vocab_utilization=0.3,
            unique_tokens_used=300,
            vocab_size=1000,
            domain_warnings=["High tokens/word"],
        )
        assert len(report.domain_warnings) == 1

    def test_report_with_fragment_analysis(self):
        fragment = FragmentAnalysis(
            total_tokens=50,
            fragment_tokens=10,
            fragment_ratio=0.2,
            top_fragments=[],
            whitespace_tokens=5,
            punctuation_tokens=3,
        )
        report = CoverageReport(
            total_texts=1,
            total_words=10,
            total_tokens=50,
            tokens_per_word=5.0,
            unk_count=0,
            unk_rate=0.0,
            vocab_utilization=0.1,
            unique_tokens_used=10,
            vocab_size=100,
            fragment_analysis=fragment,
        )
        assert report.fragment_analysis is not None
        assert report.fragment_analysis.fragment_ratio == 0.2


class TestGetUnkRate:
    """Tests for get_unk_rate function."""

    def test_no_unk_tokens(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "the test"]
        rate = get_unk_rate(texts, tokenizer)
        assert rate == 0.0

    def test_with_unk_tokens(self):
        tokenizer = MockTokenizer()
        texts = ["hello unknown_word world"]
        rate = get_unk_rate(texts, tokenizer)
        assert rate > 0.0
        assert rate == 1 / 3  # 1 unk out of 3 tokens

    def test_all_unk_tokens(self):
        tokenizer = MockTokenizer()
        texts = ["foo bar baz"]
        rate = get_unk_rate(texts, tokenizer)
        assert rate == 1.0

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        texts: list[str] = []
        rate = get_unk_rate(texts, tokenizer)
        assert rate == 0.0

    def test_custom_unk_id(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        # Use ID that won't match anything
        rate = get_unk_rate(texts, tokenizer, unk_token_id=999)
        assert rate == 0.0

    def test_auto_detect_unk_id(self):
        tokenizer = MockTokenizer()
        texts = ["hello unknown"]
        rate = get_unk_rate(texts, tokenizer)
        assert rate == 0.5  # "unknown" becomes unk


class TestGetTokensPerWord:
    """Tests for get_tokens_per_word function."""

    def test_one_token_per_word(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "the test"]
        ratio = get_tokens_per_word(texts, tokenizer)
        assert ratio == 1.0

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        texts: list[str] = []
        ratio = get_tokens_per_word(texts, tokenizer)
        assert ratio == 0.0

    def test_empty_text_content(self):
        tokenizer = MockTokenizer()
        texts = [""]
        ratio = get_tokens_per_word(texts, tokenizer)
        assert ratio == 0.0

    def test_multiple_texts(self):
        tokenizer = MockTokenizer()
        texts = ["hello", "world", "test"]
        ratio = get_tokens_per_word(texts, tokenizer)
        assert ratio == 1.0


class TestAnalyzeFragments:
    """Tests for analyze_fragments function."""

    def test_basic_analysis(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        result = analyze_fragments(texts, tokenizer)
        assert isinstance(result, FragmentAnalysis)
        assert result.total_tokens == 2

    def test_fragment_detection(self):
        # Create tokenizer with fragment tokens
        vocab = {
            "<unk>": 0,
            "##ing": 1,
            "test": 2,
            "▁word": 3,
            ",": 4,
        }

        class FragmentTokenizer:
            def __init__(self):
                self._vocab = vocab
                self._id_to_token = {v: k for k, v in vocab.items()}

            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [1, 2, 3, 4]  # Return fragment tokens

            def decode(self, ids: list[int]) -> str:
                return " ".join(self._id_to_token.get(i, "<unk>") for i in ids)

            def get_vocab(self) -> dict[str, int]:
                return self._vocab

        tokenizer = FragmentTokenizer()
        texts = ["test"]
        result = analyze_fragments(texts, tokenizer)
        assert result.fragment_tokens > 0
        assert result.whitespace_tokens > 0
        assert result.punctuation_tokens > 0

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        texts: list[str] = []
        result = analyze_fragments(texts, tokenizer)
        assert result.total_tokens == 0
        assert result.fragment_ratio == 0.0

    def test_top_n_fragments(self):
        tokenizer = MockTokenizer()
        texts = ["hello world test"]
        result = analyze_fragments(texts, tokenizer, top_n=5)
        assert len(result.top_fragments) <= 5

    def test_decode_error_handling(self):
        """Test that decode errors are handled gracefully."""

        class ErrorTokenizer:
            def __init__(self):
                self._vocab = {"<unk>": 0, "test": 1}

            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [1, 2, 3]

            def decode(self, ids: list[int]) -> str:
                raise ValueError("Decode error")

            def get_vocab(self) -> dict[str, int]:
                return self._vocab

        tokenizer = ErrorTokenizer()
        texts = ["test"]
        result = analyze_fragments(texts, tokenizer)
        assert result.total_tokens == 3
        assert result.fragment_tokens == 0  # Errors are skipped


class TestAnalyzeCoverage:
    """Tests for analyze_coverage function."""

    def test_basic_coverage(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "the test"]
        report = analyze_coverage(texts, tokenizer)
        assert isinstance(report, CoverageReport)
        assert report.total_texts == 2
        assert report.total_words == 4
        assert report.total_tokens == 4
        assert report.tokens_per_word == 1.0

    def test_coverage_with_unk(self):
        tokenizer = MockTokenizer()
        texts = ["hello unknown_word world"]
        report = analyze_coverage(texts, tokenizer)
        assert report.unk_count == 1
        assert report.unk_rate > 0.0

    def test_coverage_without_fragments(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        report = analyze_coverage(texts, tokenizer, include_fragments=False)
        assert report.fragment_analysis is None

    def test_coverage_with_fragments(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        report = analyze_coverage(texts, tokenizer, include_fragments=True)
        assert report.fragment_analysis is not None

    def test_vocab_utilization(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        report = analyze_coverage(texts, tokenizer)
        assert report.vocab_utilization > 0.0
        assert report.vocab_utilization <= 1.0
        assert report.unique_tokens_used == 2

    def test_high_tokens_per_word_warning(self):
        """Test warning for high tokens per word."""

        class HighTokenTokenizer:
            def __init__(self):
                self._vocab = {"<unk>": 0, "a": 1, "b": 2, "c": 3}

            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                # Return 3 tokens per word
                words = text.split()
                return [1, 2, 3] * len(words)

            def decode(self, ids: list[int]) -> str:
                return " ".join("a" for _ in ids)

            def get_vocab(self) -> dict[str, int]:
                return self._vocab

        tokenizer = HighTokenTokenizer()
        texts = ["word"]
        report = analyze_coverage(texts, tokenizer, include_fragments=False)
        assert report.tokens_per_word == 3.0
        assert len(report.domain_warnings) > 0
        assert "tokens/word" in report.domain_warnings[0].lower()

    def test_high_unk_rate_warning(self):
        tokenizer = MockTokenizer()
        texts = ["foo bar baz qux quux"]  # All unknown
        report = analyze_coverage(texts, tokenizer, include_fragments=False)
        assert report.unk_rate == 1.0
        assert any("unk" in w.lower() for w in report.domain_warnings)

    def test_low_vocab_utilization_warning(self):
        # Create tokenizer with large vocab
        large_vocab = {f"token_{i}": i for i in range(1000)}
        large_vocab["<unk>"] = 1000
        large_vocab["hello"] = 1001
        large_vocab["world"] = 1002

        class LargeVocabTokenizer:
            def __init__(self):
                self._vocab = large_vocab

            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [1001, 1002]  # Only use 2 tokens

            def decode(self, ids: list[int]) -> str:
                return "hello world"

            def get_vocab(self) -> dict[str, int]:
                return self._vocab

        tokenizer = LargeVocabTokenizer()
        texts = ["hello world"]
        report = analyze_coverage(texts, tokenizer, include_fragments=False)
        assert report.vocab_utilization < 0.1
        assert any("utilization" in w.lower() for w in report.domain_warnings)

    def test_custom_unk_token_id(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        # Use custom unk id that won't match
        report = analyze_coverage(texts, tokenizer, unk_token_id=999)
        assert report.unk_count == 0

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        texts: list[str] = []
        report = analyze_coverage(texts, tokenizer)
        assert report.total_texts == 0
        assert report.total_tokens == 0
        assert report.tokens_per_word == 0.0

    def test_empty_vocab_utilization(self):
        """Test with empty vocabulary."""

        class EmptyVocabTokenizer:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return []

            def decode(self, ids: list[int]) -> str:
                return ""

            def get_vocab(self) -> dict[str, int]:
                return {}

        tokenizer = EmptyVocabTokenizer()
        texts = ["test"]
        report = analyze_coverage(texts, tokenizer, include_fragments=False)
        assert report.vocab_utilization == 0.0
