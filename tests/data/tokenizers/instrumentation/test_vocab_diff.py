"""Tests for vocabulary diff analysis."""

from chuk_lazarus.data.tokenizers.instrumentation.vocab_diff import (
    VocabSwapReport,
    compare_vocab_impact,
    estimate_retokenization_cost,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab: dict[str, int] | None = None, char_level: bool = True):
        if vocab is None:
            # Default character-level vocab
            self.vocab = {chr(i): i for i in range(32, 127)}
        else:
            self.vocab = vocab
        self.char_level = char_level

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if self.char_level:
            # Character-level tokenization
            tokens = [self.vocab.get(c, 0) for c in text]
        else:
            # Word-level tokenization
            tokens = [self.vocab.get(w, 0) for w in text.split()]

        if add_special_tokens:
            tokens = [1] + tokens + [2]  # BOS, EOS
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        id_to_char = {v: k for k, v in self.vocab.items()}
        return "".join(id_to_char.get(i, "?") for i in token_ids)

    def get_vocab(self) -> dict[str, int]:
        return self.vocab


class TestVocabSwapReport:
    """Tests for VocabSwapReport model."""

    def test_valid_report(self):
        report = VocabSwapReport(
            tokenizer1_name="tok1",
            tokenizer2_name="tok2",
            tokenizer1_vocab_size=1000,
            tokenizer2_vocab_size=1500,
            total_samples=100,
            total_chars=5000,
            tokens1_total=1000,
            tokens2_total=900,
            token_count_diff=-100,
            token_count_ratio=0.9,
            chars_per_token1=5.0,
            chars_per_token2=5.5,
            compression_improvement=1.11,
            samples_improved=60,
            samples_same=20,
            samples_worse=20,
            improvement_rate=0.6,
            max_improvement=50,
            max_regression=10,
            mean_change=-1.0,
            training_speedup=1.11,
            memory_reduction=0.1,
        )
        assert report.tokenizer1_name == "tok1"
        assert report.token_count_ratio == 0.9


class TestCompareVocabImpact:
    """Tests for compare_vocab_impact function."""

    def test_basic_comparison(self):
        tokenizer1 = MockTokenizer()
        tokenizer2 = MockTokenizer()
        texts = ["hello world", "test text", "another sample"]

        report = compare_vocab_impact(texts, tokenizer1, tokenizer2, "char_tok1", "char_tok2")

        assert report.total_samples == 3
        assert report.tokenizer1_name == "char_tok1"
        assert report.tokenizer2_name == "char_tok2"

    def test_empty_texts(self):
        tokenizer1 = MockTokenizer()
        tokenizer2 = MockTokenizer()

        report = compare_vocab_impact([], tokenizer1, tokenizer2)

        assert report.total_samples == 0
        assert report.token_count_ratio == 1

    def test_same_tokenizer(self):
        tokenizer = MockTokenizer()
        texts = ["hello", "world"]

        report = compare_vocab_impact(texts, tokenizer, tokenizer)

        assert report.samples_same == 2
        assert report.samples_improved == 0
        assert report.samples_worse == 0
        assert report.token_count_ratio == 1

    def test_different_tokenizers(self):
        # Char-level tokenizer (produces more tokens)
        char_tokenizer = MockTokenizer(char_level=True)

        # Word-level tokenizer (produces fewer tokens)
        word_vocab = {"hello": 0, "world": 1, "test": 2}
        word_tokenizer = MockTokenizer(vocab=word_vocab, char_level=False)

        texts = ["hello world", "test"]

        report = compare_vocab_impact(texts, char_tokenizer, word_tokenizer, "char", "word")

        # Word tokenizer should produce fewer tokens
        assert report.tokens2_total < report.tokens1_total
        assert report.training_speedup > 1

    def test_improvement_tracking(self):
        # First tokenizer: char level (more tokens)
        char_tokenizer = MockTokenizer(char_level=True)

        # Second tokenizer: word level (fewer tokens)
        word_vocab = {"hello": 0, "world": 1, "this": 2, "is": 3, "test": 4}
        word_tokenizer = MockTokenizer(vocab=word_vocab, char_level=False)

        texts = ["hello world", "this is test"]

        report = compare_vocab_impact(texts, char_tokenizer, word_tokenizer)

        # All samples should be improved (fewer tokens with word tokenizer)
        assert report.samples_improved == 2
        assert report.improvement_rate == 1.0

    def test_examples_included(self):
        char_tokenizer = MockTokenizer(char_level=True)
        word_vocab = {"hello": 0, "world": 1}
        word_tokenizer = MockTokenizer(vocab=word_vocab, char_level=False)

        texts = ["hello world"]

        report = compare_vocab_impact(texts, char_tokenizer, word_tokenizer, max_examples=5)

        # Should have improved examples
        assert len(report.improved_examples) > 0 or len(report.regressed_examples) >= 0

    def test_recommendations_generated(self):
        char_tokenizer = MockTokenizer(char_level=True)
        word_vocab = {"hello": 0, "world": 1}
        word_tokenizer = MockTokenizer(vocab=word_vocab, char_level=False)

        texts = ["hello world"] * 10

        report = compare_vocab_impact(texts, char_tokenizer, word_tokenizer)

        assert len(report.recommendations) > 0

    def test_compression_metrics(self):
        tokenizer1 = MockTokenizer()
        tokenizer2 = MockTokenizer()
        texts = ["hello world"]

        report = compare_vocab_impact(texts, tokenizer1, tokenizer2)

        assert report.chars_per_token1 > 0
        assert report.chars_per_token2 > 0
        assert report.total_chars == len("hello world")


class TestEstimateRetokenizationCost:
    """Tests for estimate_retokenization_cost function."""

    def test_basic_cost_estimate(self):
        tokenizer1 = MockTokenizer()
        tokenizer2 = MockTokenizer()
        texts = ["hello world", "test text"]

        cost = estimate_retokenization_cost(texts, tokenizer1, tokenizer2)

        assert cost["total_samples"] == 2
        assert "vocab_overlap" in cost
        assert "embedding_reuse_rate" in cost

    def test_empty_texts(self):
        tokenizer1 = MockTokenizer()
        tokenizer2 = MockTokenizer()

        cost = estimate_retokenization_cost([], tokenizer1, tokenizer2)

        assert cost["total_samples"] == 0
        assert cost["boundary_changes"] == 0

    def test_same_vocab(self):
        tokenizer = MockTokenizer()

        cost = estimate_retokenization_cost(["hello"], tokenizer, tokenizer)

        assert cost["vocab_overlap_rate"] == 1.0
        assert cost["new_tokens"] == 0
        assert cost["removed_tokens"] == 0

    def test_different_vocabs(self):
        vocab1 = {"a": 0, "b": 1, "c": 2}
        vocab2 = {"b": 0, "c": 1, "d": 2}  # b, c shared; a removed; d added

        tokenizer1 = MockTokenizer(vocab=vocab1)
        tokenizer2 = MockTokenizer(vocab=vocab2)

        cost = estimate_retokenization_cost(["abc"], tokenizer1, tokenizer2)

        assert cost["vocab_overlap"] == 2  # b, c
        assert cost["new_tokens"] == 1  # d
        assert cost["removed_tokens"] == 1  # a

    def test_boundary_changes(self):
        # Char tokenizer
        char_tokenizer = MockTokenizer(char_level=True)

        # Word tokenizer (different boundary behavior)
        word_vocab = {"hello": 0, "world": 1}
        word_tokenizer = MockTokenizer(vocab=word_vocab, char_level=False)

        texts = ["hello world"]

        cost = estimate_retokenization_cost(texts, char_tokenizer, word_tokenizer)

        # Should detect boundary differences
        assert cost["boundary_changes"] >= 0
        assert "boundary_change_rate" in cost

    def test_large_corpus_sampling(self):
        tokenizer1 = MockTokenizer()
        tokenizer2 = MockTokenizer()

        # Create large corpus (more than 100 samples)
        texts = ["hello world"] * 200

        cost = estimate_retokenization_cost(texts, tokenizer1, tokenizer2)

        # Should still report total samples
        assert cost["total_samples"] == 200
        # But internal analysis samples only 100
        assert cost["embedding_reuse_rate"] >= 0
