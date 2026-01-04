"""Tests for batching types."""

from argparse import Namespace
from pathlib import Path

from chuk_lazarus.cli.commands.data.batching._types import (
    AnalyzeConfig,
    AnalyzeResult,
    GenerateConfig,
    GenerateResult,
    HistogramConfig,
    HistogramResult,
    OptimizationGoalType,
    SuggestConfig,
    SuggestResult,
)


class TestAnalyzeConfig:
    """Tests for AnalyzeConfig."""

    def test_from_args_basic(self):
        """Test creating config from args."""
        args = Namespace(
            cache="/path/to/cache.db",
            bucket_edges="128,256,512",
            overflow_max=1024,
            output=None,
        )
        config = AnalyzeConfig.from_args(args)

        assert config.cache == Path("/path/to/cache.db")
        assert config.overflow_max == 1024
        assert config.output is None

    def test_from_args_with_output(self):
        """Test config with output path."""
        args = Namespace(
            cache="/path/to/cache.db",
            bucket_edges="128,256",
            overflow_max=512,
            output="/path/to/report.json",
        )
        config = AnalyzeConfig.from_args(args)

        assert config.output == Path("/path/to/report.json")

    def test_get_bucket_edges(self):
        """Test parsing bucket edges."""
        args = Namespace(
            cache="/path/to/cache.db",
            bucket_edges="64, 128, 256",
            overflow_max=512,
            output=None,
        )
        config = AnalyzeConfig.from_args(args)

        assert config.get_bucket_edges() == (64, 128, 256)


class TestAnalyzeResult:
    """Tests for AnalyzeResult."""

    def test_to_display_without_output(self):
        """Test display without output path."""
        result = AnalyzeResult(
            report_ascii="Efficiency Report",
            output_path=None,
        )
        display = result.to_display()

        assert "Efficiency Report" in display
        assert "saved to" not in display

    def test_to_display_with_output(self):
        """Test display with output path."""
        result = AnalyzeResult(
            report_ascii="Efficiency Report",
            output_path=Path("/path/to/report.json"),
        )
        display = result.to_display()

        assert "Efficiency Report" in display
        assert "Report saved to:" in display


class TestHistogramConfig:
    """Tests for HistogramConfig."""

    def test_from_args_basic(self):
        """Test creating config from args."""
        args = Namespace(
            cache="/path/to/cache.db",
            bins=20,
            width=80,
        )
        config = HistogramConfig.from_args(args)

        assert config.bins == 20
        assert config.width == 80


class TestHistogramResult:
    """Tests for HistogramResult."""

    def test_to_display(self):
        """Test display formatting."""
        result = HistogramResult(
            histogram_ascii="Histogram ASCII Art",
            p25=10,
            p50=20,
            p75=30,
            p90=40,
            p95=45,
            p99=50,
        )
        display = result.to_display()

        assert "Histogram ASCII Art" in display
        assert "Percentiles" in display
        assert "P25: 10" in display
        assert "P99: 50" in display


class TestSuggestConfig:
    """Tests for SuggestConfig."""

    def test_from_args_waste(self):
        """Test config with waste goal."""
        args = Namespace(
            cache="/path/to/cache.db",
            num_buckets=5,
            goal="waste",
            max_length=2048,
        )
        config = SuggestConfig.from_args(args)

        assert config.goal == OptimizationGoalType.WASTE

    def test_from_args_balance(self):
        """Test config with balance goal."""
        args = Namespace(
            cache="/path/to/cache.db",
            num_buckets=4,
            goal="balance",
            max_length=1024,
        )
        config = SuggestConfig.from_args(args)

        assert config.goal == OptimizationGoalType.BALANCE

    def test_from_args_memory(self):
        """Test config with memory goal."""
        args = Namespace(
            cache="/path/to/cache.db",
            num_buckets=3,
            goal="memory",
            max_length=512,
        )
        config = SuggestConfig.from_args(args)

        assert config.goal == OptimizationGoalType.MEMORY


class TestSuggestResult:
    """Tests for SuggestResult."""

    def test_to_display(self):
        """Test display formatting."""
        result = SuggestResult(
            goal="minimize_waste",
            num_buckets=5,
            edges=[128, 256, 512, 1024, 2048],
            overflow_max=2048,
            estimated_efficiency=0.92,
            rationale="Optimized for minimal padding waste",
        )
        display = result.to_display()

        assert "Bucket Edge Suggestions" in display
        assert "minimize_waste" in display
        assert "92.0%" in display
        assert "Use with:" in display
        assert "128,256,512,1024,2048" in display


class TestGenerateConfig:
    """Tests for GenerateConfig."""

    def test_from_args_basic(self):
        """Test creating config from args."""
        args = Namespace(
            plan="/path/to/plan.msgpack",
            dataset="/path/to/data.jsonl",
            tokenizer="gpt2",
            output="/path/to/batches",
        )
        config = GenerateConfig.from_args(args)

        assert config.plan == Path("/path/to/plan.msgpack")
        assert config.tokenizer == "gpt2"


class TestGenerateResult:
    """Tests for GenerateResult."""

    def test_to_display_without_fingerprint(self):
        """Test display without fingerprint."""
        result = GenerateResult(
            batch_plan="/path/to/plan.msgpack",
            dataset="/path/to/data.jsonl",
            output_dir=Path("/path/to/batches"),
            num_files=10,
            num_epochs=2,
            fingerprint=None,
        )
        display = result.to_display()

        assert "Batch Generation Complete" in display
        assert "Files:        10" in display
        assert "Fingerprint:" not in display

    def test_to_display_with_fingerprint(self):
        """Test display with fingerprint."""
        result = GenerateResult(
            batch_plan="/path/to/plan.msgpack",
            dataset="/path/to/data.jsonl",
            output_dir=Path("/path/to/batches"),
            num_files=10,
            num_epochs=2,
            fingerprint="abc123",
        )
        display = result.to_display()

        assert "Fingerprint:  abc123" in display
