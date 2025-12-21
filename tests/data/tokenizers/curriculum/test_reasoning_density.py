"""Tests for reasoning_density module."""

from chuk_lazarus.data.tokenizers.curriculum.reasoning_density import (
    DifficultyPercentiles,
    ReasoningConfig,
    ReasoningDensityScore,
    get_difficulty_percentiles,
    score_reasoning_density,
    sort_by_reasoning_density,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self._vocab = {"<unk>": 0, "a": 1, "b": 2}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [1] * len(text.split())

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


class TestReasoningConfigModel:
    """Tests for ReasoningConfig model."""

    def test_default_values(self):
        config = ReasoningConfig()
        assert config.math_symbol_weight == 0.2
        assert config.bracket_depth_weight == 0.2
        assert config.variable_weight == 0.15
        assert config.numeric_weight == 0.15
        assert config.operator_weight == 0.15
        assert config.length_weight == 0.15

    def test_custom_values(self):
        config = ReasoningConfig(
            math_symbol_weight=0.4,
            bracket_depth_weight=0.3,
            variable_weight=0.2,
            numeric_weight=0.1,
        )
        assert config.math_symbol_weight == 0.4


class TestReasoningDensityScoreModel:
    """Tests for ReasoningDensityScore model."""

    def test_valid_score(self):
        score = ReasoningDensityScore(
            text_index=0,
            overall_score=0.75,
            math_symbol_score=0.8,
            bracket_depth_score=0.6,
            variable_score=0.7,
            numeric_score=0.9,
            operator_score=0.5,
            length_score=0.3,
            token_count=50,
        )
        assert score.overall_score == 0.75
        assert score.token_count == 50

    def test_zero_score(self):
        score = ReasoningDensityScore(
            text_index=0,
            overall_score=0.0,
            math_symbol_score=0.0,
            bracket_depth_score=0.0,
            variable_score=0.0,
            numeric_score=0.0,
            operator_score=0.0,
            length_score=0.0,
            token_count=10,
        )
        assert score.overall_score == 0.0


class TestDifficultyPercentilesModel:
    """Tests for DifficultyPercentiles model."""

    def test_valid_percentiles(self):
        percentiles = DifficultyPercentiles(
            p10=0.05,
            p25=0.1,
            p50=0.3,
            p75=0.6,
            p90=0.85,
            min_score=0.0,
            max_score=1.0,
            mean_score=0.4,
        )
        assert percentiles.p50 == 0.3
        assert percentiles.min_score == 0.0
        assert percentiles.max_score == 1.0


class TestScoreReasoningDensity:
    """Tests for score_reasoning_density function."""

    def test_simple_text(self):
        tokenizer = MockTokenizer()
        text = "hello world"
        score = score_reasoning_density(text, 0, tokenizer)
        assert isinstance(score, ReasoningDensityScore)
        assert score.overall_score >= 0.0

    def test_math_symbols(self):
        tokenizer = MockTokenizer()
        text = "x + y = z × 2"
        score = score_reasoning_density(text, 0, tokenizer)
        assert score.math_symbol_score >= 0.0

    def test_brackets(self):
        tokenizer = MockTokenizer()
        text = "f(x) = (a + (b × c))"
        score = score_reasoning_density(text, 0, tokenizer)
        assert score.bracket_depth_score > 0.0

    def test_variables(self):
        tokenizer = MockTokenizer()
        text = "let x = 5 and y = 10"
        score = score_reasoning_density(text, 0, tokenizer)
        assert score.variable_score >= 0.0

    def test_numeric_content(self):
        tokenizer = MockTokenizer()
        text = "123 + 456 = 579"
        score = score_reasoning_density(text, 0, tokenizer)
        assert score.numeric_score > 0.0

    def test_complex_expression(self):
        tokenizer = MockTokenizer()
        text = "σ_LT = √(σ² × L) where σ = 3.14"
        score = score_reasoning_density(text, 0, tokenizer)
        assert score.overall_score > 0.0

    def test_empty_text(self):
        tokenizer = MockTokenizer()
        text = ""
        score = score_reasoning_density(text, 0, tokenizer)
        assert score.overall_score == 0.0

    def test_custom_config(self):
        tokenizer = MockTokenizer()
        text = "x + y = z"
        config = ReasoningConfig(
            math_symbol_weight=1.0,
            bracket_depth_weight=0.0,
            variable_weight=0.0,
            numeric_weight=0.0,
            operator_weight=0.0,
            length_weight=0.0,
        )
        score = score_reasoning_density(text, 0, tokenizer, config)
        assert score.overall_score >= 0.0

    def test_no_reasoning_content(self):
        tokenizer = MockTokenizer()
        text = "the quick brown fox jumps"
        score = score_reasoning_density(text, 0, tokenizer)
        assert score.overall_score >= 0.0

    def test_high_reasoning_content(self):
        tokenizer = MockTokenizer()
        text = "∫ f(x)dx = lim(n→∞) Σ f(xi)Δx"
        score = score_reasoning_density(text, 0, tokenizer)
        assert score.overall_score > 0.0

    def test_token_count_recorded(self):
        tokenizer = MockTokenizer()
        text = "a b c d e"
        score = score_reasoning_density(text, 0, tokenizer)
        assert score.token_count == 5

    def test_text_index_recorded(self):
        tokenizer = MockTokenizer()
        text = "hello world"
        score = score_reasoning_density(text, 42, tokenizer)
        assert score.text_index == 42


class TestSortByReasoningDensity:
    """Tests for sort_by_reasoning_density function."""

    def test_basic_sorting(self):
        tokenizer = MockTokenizer()
        texts = [
            "x + y = z",
            "hello world",
            "f(x) = (a + b) × c",
        ]
        scores = sort_by_reasoning_density(texts, tokenizer)
        assert len(scores) == 3
        assert all(isinstance(s, ReasoningDensityScore) for s in scores)

    def test_ascending_order(self):
        tokenizer = MockTokenizer()
        texts = [
            "x + y = z",
            "hello world",
            "f(x) = (a + b) × c",
        ]
        scores = sort_by_reasoning_density(texts, tokenizer)
        # By default, ascending (easiest first)
        for i in range(len(scores) - 1):
            assert scores[i].overall_score <= scores[i + 1].overall_score

    def test_descending_order(self):
        tokenizer = MockTokenizer()
        texts = [
            "hello world",
            "x + y = z",
        ]
        scores = sort_by_reasoning_density(texts, tokenizer, reverse=True)
        for i in range(len(scores) - 1):
            assert scores[i].overall_score >= scores[i + 1].overall_score

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        texts: list[str] = []
        scores = sort_by_reasoning_density(texts, tokenizer)
        assert len(scores) == 0

    def test_preserves_indices(self):
        tokenizer = MockTokenizer()
        texts = ["hello", "x + y", "world"]
        scores = sort_by_reasoning_density(texts, tokenizer)
        indices = [s.text_index for s in scores]
        assert set(indices) == {0, 1, 2}

    def test_single_text(self):
        tokenizer = MockTokenizer()
        texts = ["x + y = z"]
        scores = sort_by_reasoning_density(texts, tokenizer)
        assert len(scores) == 1


class TestGetDifficultyPercentiles:
    """Tests for get_difficulty_percentiles function."""

    def test_basic_percentiles(self):
        tokenizer = MockTokenizer()
        texts = [
            "hello",
            "world",
            "x + y",
            "f(x) = y",
            "∫ f dx",
        ]
        percentiles = get_difficulty_percentiles(texts, tokenizer)
        assert isinstance(percentiles, DifficultyPercentiles)
        assert percentiles.p25 <= percentiles.p50
        assert percentiles.p50 <= percentiles.p75
        assert percentiles.p75 <= percentiles.p90

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        texts: list[str] = []
        percentiles = get_difficulty_percentiles(texts, tokenizer)
        assert percentiles.p50 == 0.0

    def test_single_text(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        percentiles = get_difficulty_percentiles(texts, tokenizer)
        # All percentiles should be the same for single value
        assert percentiles.min_score == percentiles.max_score

    def test_varied_difficulty(self):
        tokenizer = MockTokenizer()
        texts = [
            "hello",
            "the quick fox",
            "x = 5",
            "f(x) = x²",
            "∫₀^∞ e^(-x²) dx = √π/2",
        ]
        percentiles = get_difficulty_percentiles(texts, tokenizer)
        # Max should be >= min
        assert percentiles.max_score >= percentiles.min_score

    def test_all_same_difficulty(self):
        tokenizer = MockTokenizer()
        texts = ["hello", "world", "foo", "bar"]
        percentiles = get_difficulty_percentiles(texts, tokenizer)
        # All simple texts, percentiles should be close
        assert abs(percentiles.p75 - percentiles.p25) < 0.5

    def test_mean_score_calculated(self):
        tokenizer = MockTokenizer()
        texts = ["hello", "x + y", "world"]
        percentiles = get_difficulty_percentiles(texts, tokenizer)
        assert percentiles.mean_score >= percentiles.min_score
        assert percentiles.mean_score <= percentiles.max_score
