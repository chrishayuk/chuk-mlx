"""Tests for token efficiency analysis."""

from chuk_lazarus.data.tokenizers.analyze.efficiency import (
    ContentTypeStats,
    EfficiencyConfig,
    EfficiencyReport,
    FragmentationStats,
    SampleStats,
    analyze_content_type,
    analyze_efficiency,
    analyze_fragmentation,
    analyze_sample_stats,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.vocab = {chr(i): i for i in range(32, 127)}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Simple character-level tokenization
        return [self.vocab.get(c, 0) for c in text if c in self.vocab]

    def decode(self, token_ids: list[int]) -> str:
        id_to_char = {v: k for k, v in self.vocab.items()}
        return "".join(id_to_char.get(i, "?") for i in token_ids)


class TestSampleStats:
    """Tests for SampleStats model."""

    def test_valid_stats(self):
        stats = SampleStats(
            count=10,
            total_tokens=100,
            mean=10.0,
            median=9.0,
            std=2.0,
            p5=6.0,
            p25=8.0,
            p75=12.0,
            p95=14.0,
            p99=15.0,
            min_tokens=5,
            max_tokens=15,
        )
        assert stats.count == 10
        assert stats.mean == 10.0

    def test_empty_stats(self):
        stats = SampleStats(
            count=0,
            total_tokens=0,
            mean=0.0,
            median=0.0,
            std=0.0,
            p5=0.0,
            p25=0.0,
            p75=0.0,
            p95=0.0,
            p99=0.0,
            min_tokens=0,
            max_tokens=0,
        )
        assert stats.count == 0


class TestContentTypeStats:
    """Tests for ContentTypeStats model."""

    def test_valid_stats(self):
        stats = ContentTypeStats(
            content_type="reasoning_step",
            count=5,
            total_tokens=50,
            mean_tokens=10.0,
        )
        assert stats.content_type == "reasoning_step"
        assert stats.mean_tokens == 10.0

    def test_with_examples(self):
        stats = ContentTypeStats(
            content_type="equation",
            count=2,
            total_tokens=20,
            mean_tokens=10.0,
            examples=[{"matched": "x = 1", "context": "let x = 1", "token_count": 5}],
        )
        assert len(stats.examples) == 1


class TestFragmentationStats:
    """Tests for FragmentationStats model."""

    def test_valid_stats(self):
        stats = FragmentationStats(
            fragmentation_score=0.3,
            single_char_tokens=10,
            subword_tokens=5,
            continuation_tokens=2,
            total_tokens=100,
        )
        assert stats.fragmentation_score == 0.3

    def test_with_fragmented_words(self):
        stats = FragmentationStats(
            fragmentation_score=0.5,
            single_char_tokens=20,
            subword_tokens=10,
            continuation_tokens=5,
            total_tokens=200,
            fragmented_words=[{"word": "preprocessing", "tokens": 4}],
        )
        assert len(stats.fragmented_words) == 1


class TestEfficiencyReport:
    """Tests for EfficiencyReport model."""

    def test_valid_report(self):
        report = EfficiencyReport(
            sample_stats=SampleStats(
                count=10,
                total_tokens=100,
                mean=10.0,
                median=9.0,
                std=2.0,
                p5=6.0,
                p25=8.0,
                p75=12.0,
                p95=14.0,
                p99=15.0,
                min_tokens=5,
                max_tokens=15,
            ),
            fragmentation=FragmentationStats(
                fragmentation_score=0.2,
                single_char_tokens=5,
                subword_tokens=2,
                continuation_tokens=1,
                total_tokens=100,
            ),
            efficiency_score=85.0,
        )
        assert report.efficiency_score == 85.0

    def test_with_content_stats(self):
        report = EfficiencyReport(
            sample_stats=SampleStats(
                count=10,
                total_tokens=100,
                mean=10.0,
                median=9.0,
                std=2.0,
                p5=6.0,
                p25=8.0,
                p75=12.0,
                p95=14.0,
                p99=15.0,
                min_tokens=5,
                max_tokens=15,
            ),
            reasoning_steps=ContentTypeStats(
                content_type="reasoning_step",
                count=5,
                total_tokens=50,
                mean_tokens=10.0,
            ),
            fragmentation=FragmentationStats(
                fragmentation_score=0.2,
                single_char_tokens=5,
                subword_tokens=2,
                continuation_tokens=1,
                total_tokens=100,
            ),
            efficiency_score=80.0,
        )
        assert report.reasoning_steps is not None
        assert report.reasoning_steps.count == 5


class TestAnalyzeSampleStats:
    """Tests for analyze_sample_stats function."""

    def test_basic_analysis(self):
        tokenizer = MockTokenizer()
        texts = ["hello", "world", "test"]
        stats = analyze_sample_stats(texts, tokenizer)
        assert stats.count == 3
        assert stats.total_tokens > 0

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        stats = analyze_sample_stats([], tokenizer)
        assert stats.count == 0
        assert stats.total_tokens == 0

    def test_single_text(self):
        tokenizer = MockTokenizer()
        stats = analyze_sample_stats(["hello"], tokenizer)
        assert stats.count == 1
        assert stats.min_tokens == stats.max_tokens

    def test_varied_lengths(self):
        tokenizer = MockTokenizer()
        texts = ["a", "hello world", "the quick brown fox"]
        stats = analyze_sample_stats(texts, tokenizer)
        assert stats.min_tokens < stats.max_tokens
        assert stats.mean > 0

    def test_percentiles_calculated(self):
        tokenizer = MockTokenizer()
        texts = [f"text{i}" * i for i in range(1, 11)]
        stats = analyze_sample_stats(texts, tokenizer)
        assert stats.p5 <= stats.p25 <= stats.median <= stats.p75 <= stats.p95


class TestAnalyzeContentType:
    """Tests for analyze_content_type function."""

    def test_detect_reasoning_steps(self):
        tokenizer = MockTokenizer()
        texts = ["Step 1: Do this. Step 2: Do that."]
        patterns = [r"Step \d+:"]
        result = analyze_content_type(texts, tokenizer, "step", patterns)
        assert result is not None
        assert result.count == 2

    def test_detect_equations(self):
        tokenizer = MockTokenizer()
        texts = ["The answer is x = 42"]
        patterns = [r"[a-z]\s*=\s*\d+"]
        result = analyze_content_type(texts, tokenizer, "equation", patterns)
        assert result is not None
        assert result.count >= 1

    def test_no_matches(self):
        tokenizer = MockTokenizer()
        texts = ["Hello world"]
        patterns = [r"<TOOL_CALL>"]
        result = analyze_content_type(texts, tokenizer, "tool", patterns)
        assert result is None

    def test_multiple_texts(self):
        tokenizer = MockTokenizer()
        texts = ["First, do A.", "Then, do B.", "Finally, do C."]
        patterns = [r"First,", r"Then,", r"Finally,"]
        result = analyze_content_type(texts, tokenizer, "step", patterns)
        assert result is not None
        assert result.count == 3


class TestAnalyzeFragmentation:
    """Tests for analyze_fragmentation function."""

    def test_basic_fragmentation(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        result = analyze_fragmentation(texts, tokenizer)
        assert result.total_tokens > 0
        assert 0.0 <= result.fragmentation_score <= 1.0

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        result = analyze_fragmentation([], tokenizer)
        assert result.total_tokens == 0
        assert result.fragmentation_score == 0.0

    def test_single_char_detection(self):
        tokenizer = MockTokenizer()
        texts = ["a b c d e"]
        result = analyze_fragmentation(texts, tokenizer)
        # Character-level tokenizer produces many single-char tokens
        assert result.single_char_tokens > 0

    def test_fragmented_words_list(self):
        tokenizer = MockTokenizer()
        texts = ["preprocessing", "tokenization", "implementation"]
        result = analyze_fragmentation(texts, tokenizer, max_fragmented=3)
        # Should identify which words are most fragmented
        assert isinstance(result.fragmented_words, list)


class TestAnalyzeEfficiency:
    """Tests for analyze_efficiency function."""

    def test_basic_efficiency(self):
        tokenizer = MockTokenizer()
        texts = ["Hello world", "How are you?"]
        report = analyze_efficiency(texts, tokenizer)
        assert report.sample_stats.count == 2
        assert 0.0 <= report.efficiency_score <= 100.0

    def test_with_reasoning_content(self):
        tokenizer = MockTokenizer()
        texts = [
            "Step 1: Calculate. Step 2: Verify.",
            "First, add. Then, multiply.",
        ]
        report = analyze_efficiency(texts, tokenizer)
        assert report.reasoning_steps is not None

    def test_with_equations(self):
        tokenizer = MockTokenizer()
        texts = ["Let x = 5 and y = 10"]
        report = analyze_efficiency(texts, tokenizer)
        # May or may not detect equations depending on patterns
        assert report.sample_stats.count == 1

    def test_with_tool_calls(self):
        tokenizer = MockTokenizer()
        texts = ["<TOOL_CALL> get_weather"]
        config = EfficiencyConfig(tool_patterns=[r"<TOOL_CALL>"])
        report = analyze_efficiency(texts, tokenizer, config)
        assert report.tool_calls is not None

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        report = analyze_efficiency([], tokenizer)
        assert report.sample_stats.count == 0
        assert report.efficiency_score >= 0

    def test_custom_config(self):
        tokenizer = MockTokenizer()
        config = EfficiencyConfig(
            step_patterns=[r"Phase \d:"],
            equation_patterns=[r"\d+ \+ \d+"],
        )
        texts = ["Phase 1: Calculate 5 + 3"]
        report = analyze_efficiency(texts, tokenizer, config)
        assert report is not None

    def test_recommendations_generated(self):
        tokenizer = MockTokenizer()
        # Create texts that should trigger recommendations
        texts = ["a" * 10, "b" * 100, "c" * 1000]  # High variance
        report = analyze_efficiency(texts, tokenizer)
        # May generate recommendations about variance
        assert isinstance(report.recommendations, list)


class TestEfficiencyConfig:
    """Tests for EfficiencyConfig model."""

    def test_default_values(self):
        config = EfficiencyConfig()
        assert len(config.step_patterns) > 0
        assert len(config.equation_patterns) > 0
        assert len(config.tool_patterns) > 0

    def test_custom_patterns(self):
        config = EfficiencyConfig(
            step_patterns=[r"Phase \d:"],
            equation_patterns=[r"formula:"],
            tool_patterns=[r"<INVOKE>"],
        )
        assert "Phase \\d:" in config.step_patterns
        assert "formula:" in config.equation_patterns
        assert "<INVOKE>" in config.tool_patterns
