"""Tests for fit_score module."""

from chuk_lazarus.data.tokenizers.analyze.fit_score import (
    FitScore,
    FitScoreConfig,
    TokenizerComparison,
    calculate_fit_score,
    compare_tokenizers_for_dataset,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        compression: float = 1.0,
        name: str = "mock",
    ):
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
        self._compression = compression
        self.name = name

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = text.lower().split()
        # Simulate compression by potentially splitting tokens
        result = []
        for t in tokens:
            if t in self._vocab:
                result.append(self._vocab[t])
            else:
                # Unknown tokens get fragmented based on compression
                result.extend([1] * max(1, int(self._compression)))
        return result

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._id_to_token.get(i, "<unk>") for i in ids)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab.copy()


class TestFitScoreConfigModel:
    """Tests for FitScoreConfig model."""

    def test_default_values(self):
        config = FitScoreConfig()
        assert config.coverage_weight == 0.25
        assert config.compression_weight == 0.25
        assert config.entropy_weight == 0.25
        assert config.vocab_util_weight == 0.25
        assert config.ideal_tokens_per_word == 1.3
        assert config.max_acceptable_unk_rate == 0.01

    def test_custom_values(self):
        config = FitScoreConfig(
            coverage_weight=0.5,
            compression_weight=0.25,
            entropy_weight=0.15,
            vocab_util_weight=0.1,
            ideal_tokens_per_word=1.5,
            max_acceptable_unk_rate=0.05,
        )
        assert config.coverage_weight == 0.5
        assert config.ideal_tokens_per_word == 1.5


class TestFitScoreModel:
    """Tests for FitScore model."""

    def test_valid_score(self):
        score = FitScore(
            overall_score=0.85,
            coverage_score=0.9,
            compression_score=0.8,
            entropy_score=0.85,
            vocab_utilization_score=0.8,
            recommendation="Good fit for domain",
        )
        assert score.overall_score == 0.85
        assert score.recommendation == "Good fit for domain"

    def test_poor_score(self):
        score = FitScore(
            overall_score=0.3,
            coverage_score=0.2,
            compression_score=0.4,
            entropy_score=0.3,
            vocab_utilization_score=0.3,
            recommendation="Poor fit",
        )
        assert score.overall_score == 0.3

    def test_with_details(self):
        score = FitScore(
            overall_score=0.7,
            coverage_score=0.8,
            compression_score=0.7,
            entropy_score=0.6,
            vocab_utilization_score=0.7,
            recommendation="Moderate fit",
            details={"tokens_per_word": 1.5, "unk_rate": 0.02},
        )
        assert score.details["tokens_per_word"] == 1.5


class TestTokenizerComparisonModel:
    """Tests for TokenizerComparison model."""

    def test_valid_comparison(self):
        score1 = FitScore(
            overall_score=0.8,
            coverage_score=0.85,
            compression_score=0.75,
            entropy_score=0.8,
            vocab_utilization_score=0.8,
            recommendation="Good",
        )
        score2 = FitScore(
            overall_score=0.7,
            coverage_score=0.75,
            compression_score=0.65,
            entropy_score=0.7,
            vocab_utilization_score=0.7,
            recommendation="Fair",
        )
        comparison = TokenizerComparison(
            tokenizer1_name="tok1",
            tokenizer2_name="tok2",
            tokenizer1_score=score1,
            tokenizer2_score=score2,
            winner="tok1",
            score_delta=0.1,
            comparison_notes=["tok1 has better coverage"],
        )
        assert comparison.winner == "tok1"
        assert comparison.score_delta == 0.1


class TestCalculateFitScore:
    """Tests for calculate_fit_score function."""

    def test_good_fit(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "the test", "hello a"]
        score = calculate_fit_score(texts, tokenizer)
        assert isinstance(score, FitScore)
        assert 0.0 <= score.overall_score <= 1.0
        assert "tokens_per_word" in score.details

    def test_custom_config(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        config = FitScoreConfig(
            coverage_weight=0.5,
            compression_weight=0.2,
            entropy_weight=0.2,
            vocab_util_weight=0.1,
        )
        score = calculate_fit_score(texts, tokenizer, config)
        assert isinstance(score, FitScore)
        assert 0.0 <= score.overall_score <= 1.0

    def test_high_unk_rate(self):
        tokenizer = MockTokenizer()
        texts = ["foo bar baz qux"]  # All unknown
        score = calculate_fit_score(texts, tokenizer)
        assert score.details["unk_rate"] > 0.0
        assert score.coverage_score < 1.0

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        texts: list[str] = []
        score = calculate_fit_score(texts, tokenizer)
        # Empty texts should have a score (might be low)
        assert 0.0 <= score.overall_score <= 1.0

    def test_recommendation_generation(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        score = calculate_fit_score(texts, tokenizer)
        assert len(score.recommendation) > 0

    def test_poor_compression(self):
        """Test score with poor compression (many tokens per word)."""
        # Tokenizer that fragments everything
        tokenizer = MockTokenizer(compression=3.0)
        texts = ["unknown_word another_unknown"]
        score = calculate_fit_score(texts, tokenizer)
        assert score.compression_score < 1.0

    def test_single_text(self):
        tokenizer = MockTokenizer()
        texts = ["hello world the a test"]
        score = calculate_fit_score(texts, tokenizer)
        assert score.overall_score > 0.0

    def test_score_bounds(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        score = calculate_fit_score(texts, tokenizer)
        assert 0.0 <= score.coverage_score <= 1.0
        assert 0.0 <= score.compression_score <= 1.0
        assert 0.0 <= score.entropy_score <= 1.0
        assert 0.0 <= score.vocab_utilization_score <= 1.0
        assert 0.0 <= score.overall_score <= 1.0


class TestCompareTokenizersForDataset:
    """Tests for compare_tokenizers_for_dataset function."""

    def test_same_tokenizers(self):
        tokenizer1 = MockTokenizer(name="tok1")
        tokenizer2 = MockTokenizer(name="tok2")
        texts = ["hello world", "the test"]
        comparison = compare_tokenizers_for_dataset(
            texts, tokenizer1, tokenizer2, name1="tok1", name2="tok2"
        )
        assert isinstance(comparison, TokenizerComparison)
        # Same tokenizers should have similar scores
        assert comparison.score_delta < 0.1

    def test_different_tokenizers(self):
        tokenizer1 = MockTokenizer(name="good", compression=1.0)
        tokenizer2 = MockTokenizer(name="bad", compression=3.0)
        texts = ["unknown_word test"]
        comparison = compare_tokenizers_for_dataset(
            texts, tokenizer1, tokenizer2, name1="good", name2="bad"
        )
        assert comparison.winner in ("good", "bad")

    def test_winner_determination(self):
        # Create tokenizers with different vocab
        good_vocab = {"<unk>": 0, "hello": 1, "world": 2}
        bad_vocab = {"<unk>": 0}  # Poor coverage

        class GoodTokenizer:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [1, 2]

            def decode(self, ids: list[int]) -> str:
                return "hello world"

            def get_vocab(self) -> dict[str, int]:
                return good_vocab

        class BadTokenizer:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [0, 0, 0, 0, 0]  # All UNK, more tokens

            def decode(self, ids: list[int]) -> str:
                return "<unk> " * 5

            def get_vocab(self) -> dict[str, int]:
                return bad_vocab

        comparison = compare_tokenizers_for_dataset(
            ["hello world"],
            GoodTokenizer(),
            BadTokenizer(),
            name1="good",
            name2="bad",
        )
        assert comparison.winner == "good"  # Good tokenizer should win

    def test_empty_texts(self):
        tokenizer1 = MockTokenizer(name="tok1")
        tokenizer2 = MockTokenizer(name="tok2")
        texts: list[str] = []
        comparison = compare_tokenizers_for_dataset(
            texts, tokenizer1, tokenizer2, name1="tok1", name2="tok2"
        )
        # Empty texts = equal scores
        assert comparison.score_delta == 0.0

    def test_comparison_notes(self):
        tokenizer1 = MockTokenizer(name="tok1")
        tokenizer2 = MockTokenizer(name="tok2")
        texts = ["hello world"]
        comparison = compare_tokenizers_for_dataset(
            texts, tokenizer1, tokenizer2, name1="tok1", name2="tok2"
        )
        assert isinstance(comparison.comparison_notes, list)

    def test_custom_config(self):
        tokenizer1 = MockTokenizer(name="tok1")
        tokenizer2 = MockTokenizer(name="tok2")
        texts = ["hello world"]
        # Weights must sum to 1.0
        config = FitScoreConfig(
            coverage_weight=0.4,
            compression_weight=0.3,
            entropy_weight=0.2,
            vocab_util_weight=0.1,
        )
        comparison = compare_tokenizers_for_dataset(
            texts, tokenizer1, tokenizer2, name1="tok1", name2="tok2", config=config
        )
        assert isinstance(comparison, TokenizerComparison)

    def test_tie_scenario(self):
        """Test when tokenizers have equal scores."""
        tokenizer1 = MockTokenizer(name="tok1")
        tokenizer2 = MockTokenizer(name="tok1")  # Identical
        texts = ["hello world"]
        comparison = compare_tokenizers_for_dataset(
            texts, tokenizer1, tokenizer2, name1="tok1", name2="tok2"
        )
        # Should be a tie or very close
        assert comparison.score_delta < 0.01
