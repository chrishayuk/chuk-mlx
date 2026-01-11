"""Tests for lengths types."""

from argparse import Namespace
from pathlib import Path

import pytest
from pydantic import ValidationError

from chuk_lazarus.cli.commands.data.lengths._types import (
    EmptyStatsResult,
    LengthBuildConfig,
    LengthBuildResult,
    LengthStatsConfig,
    LengthStatsResult,
)


class TestLengthBuildConfig:
    """Tests for LengthBuildConfig."""

    def test_from_args_basic(self):
        """Test creating config from args."""
        args = Namespace(
            tokenizer="gpt2",
            dataset="/path/to/data.jsonl",
            output="/path/to/cache.db",
        )
        config = LengthBuildConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.dataset == Path("/path/to/data.jsonl")
        assert config.output == Path("/path/to/cache.db")

    def test_config_is_frozen(self):
        """Test that config is immutable."""
        args = Namespace(
            tokenizer="gpt2",
            dataset="/path/to/data.jsonl",
            output="/path/to/cache.db",
        )
        config = LengthBuildConfig.from_args(args)

        with pytest.raises(ValidationError):
            config.tokenizer = "other"


class TestLengthBuildResult:
    """Tests for LengthBuildResult."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = LengthBuildResult(
            dataset="/path/to/data.jsonl",
            tokenizer="gpt2",
            samples_processed=1000,
            output_path=Path("/path/to/cache.db"),
            tokenizer_hash="abc123",
        )
        assert result.samples_processed == 1000
        assert result.tokenizer_hash == "abc123"

    def test_to_display(self):
        """Test display formatting."""
        result = LengthBuildResult(
            dataset="/path/to/data.jsonl",
            tokenizer="gpt2",
            samples_processed=1000,
            output_path=Path("/path/to/cache.db"),
            tokenizer_hash="abc123",
        )
        display = result.to_display()

        assert "Length Cache Built" in display
        assert "1,000" in display
        assert "gpt2" in display
        assert "abc123" in display


class TestLengthStatsConfig:
    """Tests for LengthStatsConfig."""

    def test_from_args_basic(self):
        """Test creating config from args."""
        args = Namespace(cache="/path/to/cache.db")
        config = LengthStatsConfig.from_args(args)

        assert config.cache == Path("/path/to/cache.db")


class TestLengthStatsResult:
    """Tests for LengthStatsResult."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = LengthStatsResult(
            cache_path="/path/to/cache.db",
            tokenizer_hash="test_hash",
            total_samples=100,
            total_tokens=5000,
            min_length=10,
            max_length=100,
            mean_length=50.5,
            median_length=50,
            p10=15,
            p25=25,
            p50=50,
            p75=75,
            p90=90,
            p95=95,
            p99=99,
        )
        assert result.total_samples == 100
        assert result.mean_length == 50.5

    def test_to_display(self):
        """Test display formatting."""
        result = LengthStatsResult(
            cache_path="/path/to/cache.db",
            tokenizer_hash="test_hash",
            total_samples=100,
            total_tokens=5000,
            min_length=10,
            max_length=100,
            mean_length=50.5,
            median_length=50,
            p10=15,
            p25=25,
            p50=50,
            p75=75,
            p90=90,
            p95=95,
            p99=99,
        )
        display = result.to_display()

        assert "Length Cache Statistics" in display
        assert "100" in display
        assert "5,000" in display
        assert "P10:" in display
        assert "P99:" in display

    def test_mean_rounding(self):
        """Test mean length is rounded."""
        result = LengthStatsResult(
            cache_path="/path/to/cache.db",
            tokenizer_hash="test_hash",
            total_samples=100,
            total_tokens=5000,
            min_length=10,
            max_length=100,
            mean_length=50.5555,
            median_length=50,
            p10=15,
            p25=25,
            p50=50,
            p75=75,
            p90=90,
            p95=95,
            p99=99,
        )
        assert result.mean_length == 50.6


class TestEmptyStatsResult:
    """Tests for EmptyStatsResult."""

    def test_to_display(self):
        """Test empty result display."""
        result = EmptyStatsResult()
        assert result.to_display() == "Cache is empty"
