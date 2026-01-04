"""Tests for gym CLI type definitions."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from chuk_lazarus.cli.commands.gym._types import (
    BenchmarkConfig,
    BenchmarkResult,
    GymRunConfig,
    GymRunResult,
)


class TestGymRunConfig:
    """Tests for GymRunConfig."""

    def test_from_args_basic(self, gym_run_args):
        """Test creating config from args."""
        config = GymRunConfig.from_args(gym_run_args)

        assert config.tokenizer == "gpt2"
        assert config.mock is True
        assert config.host == "localhost"
        assert config.port == 8023

    def test_from_args_with_output(self, gym_run_args):
        """Test config with output path."""
        gym_run_args.output = "/path/to/output.json"
        config = GymRunConfig.from_args(gym_run_args)

        assert config.output == Path("/path/to/output.json")

    def test_port_validation(self, gym_run_args):
        """Test port must be valid."""
        gym_run_args.port = 70000
        with pytest.raises(ValidationError):
            GymRunConfig.from_args(gym_run_args)

    def test_difficulty_range_validation(self, gym_run_args):
        """Test difficulty must be in range."""
        gym_run_args.difficulty_min = -0.1
        with pytest.raises(ValidationError):
            GymRunConfig.from_args(gym_run_args)

    def test_success_rate_validation(self, gym_run_args):
        """Test success rate must be in range."""
        gym_run_args.success_rate = 1.5
        with pytest.raises(ValidationError):
            GymRunConfig.from_args(gym_run_args)


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_from_args_basic(self, bench_args):
        """Test creating config from args."""
        config = BenchmarkConfig.from_args(bench_args)

        assert config.tokenizer == "gpt2"
        assert config.num_samples == 1000
        assert config.token_budget == 4096

    def test_get_bucket_edges(self, bench_args):
        """Test bucket edges parsing."""
        config = BenchmarkConfig.from_args(bench_args)
        edges = config.get_bucket_edges()

        assert edges == (64, 128, 256, 512)

    def test_from_args_with_dataset(self, bench_args):
        """Test config with dataset."""
        bench_args.dataset = "/path/to/data.jsonl"
        config = BenchmarkConfig.from_args(bench_args)

        assert config.dataset == Path("/path/to/data.jsonl")


class TestGymRunResult:
    """Tests for GymRunResult."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = GymRunResult(
            total_samples=100,
            total_episodes=10,
            buffer_size=100,
            success_rate=0.8,
            mean_difficulty=0.5,
            mean_reward=0.9,
        )
        assert result.total_samples == 100
        assert result.success_rate == 0.8

    def test_to_display(self):
        """Test display formatting."""
        result = GymRunResult(
            total_samples=100,
            total_episodes=10,
            buffer_size=100,
            success_rate=0.8,
            mean_difficulty=0.5,
            mean_reward=0.9,
        )
        display = result.to_display()
        assert "Gym Run Summary" in display
        assert "100" in display
        assert "80.0%" in display


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = BenchmarkResult(
            samples=1000,
            total_tokens=50000,
            plan_fingerprint="abc123",
            bucket_efficiency=0.85,
            packing_ratio=1.5,
            packing_efficiency=0.9,
            token_budget_utilization=0.95,
            microbatches=50,
        )
        assert result.samples == 1000
        assert result.packing_ratio == 1.5

    def test_to_display(self):
        """Test display formatting."""
        result = BenchmarkResult(
            samples=1000,
            total_tokens=50000,
            plan_fingerprint="abc123",
            bucket_efficiency=0.85,
            packing_ratio=1.5,
            packing_efficiency=0.9,
            token_budget_utilization=0.95,
            microbatches=50,
        )
        display = result.to_display()
        assert "Benchmark Summary" in display
        assert "1,000" in display
        assert "1.50x" in display
