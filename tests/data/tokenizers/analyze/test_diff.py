"""Tests for diff module."""

from chuk_lazarus.data.tokenizers.analyze.diff import (
    CorpusDiff,
    RetokenizationDiff,
    TokenBoundaryShift,
    compare_tokenizations_detailed,
    diff_corpus,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab: dict[str, int] | None = None, name: str = "mock"):
        self._vocab = vocab or {
            "<pad>": 0,
            "<unk>": 1,
            "hello": 2,
            "world": 3,
            "the": 4,
            "a": 5,
            "test": 6,
        }
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        self.name = name

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = text.lower().split()
        return [self._vocab.get(t, 1) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._id_to_token.get(i, "<unk>") for i in ids)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab.copy()


class TestTokenBoundaryShiftModel:
    """Tests for TokenBoundaryShift model."""

    def test_valid_shift(self):
        shift = TokenBoundaryShift(
            position=5,
            tokenizer1_boundary=True,
            tokenizer2_boundary=False,
            context="hello",
        )
        assert shift.position == 5
        assert shift.tokenizer1_boundary is True
        assert shift.tokenizer2_boundary is False

    def test_context_field(self):
        shift = TokenBoundaryShift(
            position=10,
            tokenizer1_boundary=False,
            tokenizer2_boundary=True,
            context="surrounding text",
        )
        assert shift.context == "surrounding text"


class TestRetokenizationDiffModel:
    """Tests for RetokenizationDiff model."""

    def test_no_changes(self):
        diff = RetokenizationDiff(
            text="hello world",
            tokenizer1_ids=[2, 3],
            tokenizer2_ids=[2, 3],
            tokenizer1_tokens=["hello", "world"],
            tokenizer2_tokens=["hello", "world"],
            length_delta=0,
            length_ratio=1.0,
            boundary_shifts=[],
            common_token_ratio=1.0,
        )
        assert diff.length_delta == 0
        assert diff.common_token_ratio == 1.0

    def test_with_changes(self):
        diff = RetokenizationDiff(
            text="hello world",
            tokenizer1_ids=[1, 2, 3],
            tokenizer2_ids=[4, 5],
            tokenizer1_tokens=["hel", "lo", "world"],
            tokenizer2_tokens=["hello", "world"],
            length_delta=1,  # 3 - 2
            length_ratio=1.5,  # 3 / 2
            boundary_shifts=[],
            common_token_ratio=0.5,
        )
        assert diff.length_delta == 1
        assert len(diff.tokenizer1_ids) == 3
        assert len(diff.tokenizer2_ids) == 2


class TestCorpusDiffModel:
    """Tests for CorpusDiff model."""

    def test_valid_corpus_diff(self):
        corpus = CorpusDiff(
            total_texts=10,
            avg_length_delta=0.5,
            avg_length_ratio=1.1,
            tokenizer1_total_tokens=100,
            tokenizer2_total_tokens=95,
            compression_improvement=5.0,
            texts_with_different_lengths=3,
            worst_cases=[],
        )
        assert corpus.total_texts == 10
        assert corpus.texts_with_different_lengths == 3
        assert corpus.compression_improvement == 5.0


class TestCompareTokenizationsDetailed:
    """Tests for compare_tokenizations_detailed function."""

    def test_identical_tokenization(self):
        tokenizer1 = MockTokenizer(name="tok1")
        tokenizer2 = MockTokenizer(name="tok2")
        text = "hello world"
        diff = compare_tokenizations_detailed(text, tokenizer1, tokenizer2)
        assert isinstance(diff, RetokenizationDiff)
        assert diff.length_delta == 0
        assert diff.length_ratio == 1.0

    def test_different_tokenization(self):
        class Tokenizer1:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [1, 2]  # 2 tokens

            def decode(self, ids: list[int]) -> str:
                return "hello world"

            def get_vocab(self) -> dict[str, int]:
                return {"<unk>": 0, "hello": 1, "world": 2}

        class Tokenizer2:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [1, 2, 3, 4]  # 4 tokens

            def decode(self, ids: list[int]) -> str:
                return "hel lo wor ld"

            def get_vocab(self) -> dict[str, int]:
                return {"<unk>": 0, "hel": 1, "lo": 2, "wor": 3, "ld": 4}

        diff = compare_tokenizations_detailed("hello world", Tokenizer1(), Tokenizer2())
        assert len(diff.tokenizer1_ids) == 2
        assert len(diff.tokenizer2_ids) == 4
        assert diff.length_delta == -2  # 2 - 4 = -2

    def test_text_preserved(self):
        tokenizer1 = MockTokenizer(name="tok1")
        tokenizer2 = MockTokenizer(name="tok2")
        text = "hello world"
        diff = compare_tokenizations_detailed(text, tokenizer1, tokenizer2)
        assert diff.text == text

    def test_common_token_ratio(self):
        tokenizer1 = MockTokenizer(name="tok1")
        tokenizer2 = MockTokenizer(name="tok2")
        text = "hello world"
        diff = compare_tokenizations_detailed(text, tokenizer1, tokenizer2)
        # Same tokenizers = full overlap
        assert diff.common_token_ratio == 1.0

    def test_empty_text(self):
        tokenizer1 = MockTokenizer(name="tok1")
        tokenizer2 = MockTokenizer(name="tok2")
        diff = compare_tokenizations_detailed("", tokenizer1, tokenizer2)
        assert len(diff.tokenizer1_ids) == 0
        assert len(diff.tokenizer2_ids) == 0

    def test_length_ratio_calculation(self):
        class Tokenizer1:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [1, 2, 3]  # 3 tokens

            def decode(self, ids: list[int]) -> str:
                return "a b c"

            def get_vocab(self) -> dict[str, int]:
                return {"a": 1, "b": 2, "c": 3}

        class Tokenizer2:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [1]  # 1 token

            def decode(self, ids: list[int]) -> str:
                return "abc"

            def get_vocab(self) -> dict[str, int]:
                return {"abc": 1}

        diff = compare_tokenizations_detailed("abc", Tokenizer1(), Tokenizer2())
        assert diff.length_ratio == 3.0  # 3 / 1


class TestDiffCorpus:
    """Tests for diff_corpus function."""

    def test_basic_corpus_diff(self):
        tokenizer1 = MockTokenizer(name="tok1")
        tokenizer2 = MockTokenizer(name="tok2")
        texts = ["hello world", "the test", "a hello"]
        corpus_diff = diff_corpus(texts, tokenizer1, tokenizer2)
        assert isinstance(corpus_diff, CorpusDiff)
        assert corpus_diff.total_texts == 3

    def test_empty_corpus(self):
        tokenizer1 = MockTokenizer(name="tok1")
        tokenizer2 = MockTokenizer(name="tok2")
        texts: list[str] = []
        corpus_diff = diff_corpus(texts, tokenizer1, tokenizer2)
        assert corpus_diff.total_texts == 0
        assert corpus_diff.texts_with_different_lengths == 0

    def test_all_texts_identical(self):
        tokenizer1 = MockTokenizer(name="tok1")
        tokenizer2 = MockTokenizer(name="tok2")
        texts = ["hello world", "the test"]
        corpus_diff = diff_corpus(texts, tokenizer1, tokenizer2)
        assert corpus_diff.texts_with_different_lengths == 0

    def test_some_texts_different(self):
        """Test corpus with some different tokenizations."""

        class Tokenizer1:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [1] * len(text.split())  # 1 token per word

            def decode(self, ids: list[int]) -> str:
                return " ".join(["a"] * len(ids))

            def get_vocab(self) -> dict[str, int]:
                return {"a": 1}

        class Tokenizer2:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [1] * (len(text.split()) * 2)  # 2 tokens per word

            def decode(self, ids: list[int]) -> str:
                return " ".join(["a"] * len(ids))

            def get_vocab(self) -> dict[str, int]:
                return {"a": 1}

        texts = ["hello world", "foo bar"]
        corpus_diff = diff_corpus(texts, Tokenizer1(), Tokenizer2())
        assert corpus_diff.total_texts == 2
        assert corpus_diff.texts_with_different_lengths == 2

    def test_avg_length_delta_calculation(self):
        tokenizer1 = MockTokenizer(name="tok1")
        tokenizer2 = MockTokenizer(name="tok2")
        texts = ["hello world", "the test"]
        corpus_diff = diff_corpus(texts, tokenizer1, tokenizer2)
        # Same tokenizers = 0 delta
        assert corpus_diff.avg_length_delta == 0.0

    def test_compression_improvement(self):
        """Test compression improvement calculation."""

        class Tokenizer1:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [1] * 10  # 10 tokens

            def decode(self, ids: list[int]) -> str:
                return "a " * len(ids)

            def get_vocab(self) -> dict[str, int]:
                return {"a": 1}

        class Tokenizer2:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [1] * 5  # 5 tokens (50% compression)

            def decode(self, ids: list[int]) -> str:
                return "a " * len(ids)

            def get_vocab(self) -> dict[str, int]:
                return {"a": 1}

        texts = ["test"]
        corpus_diff = diff_corpus(texts, Tokenizer1(), Tokenizer2())
        # t2 uses 50% fewer tokens
        assert corpus_diff.compression_improvement == 50.0

    def test_worst_cases_included(self):
        """Test that worst cases are included in output."""

        class Tokenizer1:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [1] * len(text.split())

            def decode(self, ids: list[int]) -> str:
                return " ".join(["a"] * len(ids))

            def get_vocab(self) -> dict[str, int]:
                return {"a": 1}

        class Tokenizer2:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                # Variable difference based on text
                if "long" in text:
                    return [1] * 10
                return [1]

            def decode(self, ids: list[int]) -> str:
                return " ".join(["a"] * len(ids))

            def get_vocab(self) -> dict[str, int]:
                return {"a": 1}

        texts = ["short", "this is a long text"]
        corpus_diff = diff_corpus(texts, Tokenizer1(), Tokenizer2(), worst_n=2)
        assert len(corpus_diff.worst_cases) <= 2
        assert all(isinstance(w, RetokenizationDiff) for w in corpus_diff.worst_cases)

    def test_token_totals(self):
        tokenizer1 = MockTokenizer(name="tok1")
        tokenizer2 = MockTokenizer(name="tok2")
        texts = ["hello world"]  # 2 tokens each
        corpus_diff = diff_corpus(texts, tokenizer1, tokenizer2)
        assert corpus_diff.tokenizer1_total_tokens == 2
        assert corpus_diff.tokenizer2_total_tokens == 2
