"""Tests for gym benchmark command."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.gym._types import BenchmarkConfig, BenchmarkResult


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_from_args(self, bench_args):
        """Test creating config from args."""
        config = BenchmarkConfig.from_args(bench_args)

        assert config.num_samples == 1000
        assert config.max_length == 512
        assert config.token_budget == 4096
        assert config.seed == 42

    def test_from_args_with_dataset(self, bench_args):
        """Test creating config with dataset."""
        bench_args.dataset = "/path/to/data.jsonl"
        config = BenchmarkConfig.from_args(bench_args)

        assert config.dataset == Path("/path/to/data.jsonl")

    def test_get_bucket_edges(self, bench_args):
        """Test parsing bucket edges."""
        config = BenchmarkConfig.from_args(bench_args)
        edges = config.get_bucket_edges()

        assert edges == (64, 128, 256, 512)

    def test_get_bucket_edges_custom(self):
        """Test parsing custom bucket edges."""
        args = Namespace(
            dataset=None,
            tokenizer=None,
            num_samples=100,
            max_samples=None,
            max_length=1024,
            token_budget=2048,
            bucket_edges="32, 64, 128",
            seed=0,
        )
        config = BenchmarkConfig.from_args(args)
        edges = config.get_bucket_edges()

        assert edges == (32, 64, 128)


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_to_display(self):
        """Test result display formatting."""
        result = BenchmarkResult(
            samples=10000,
            total_tokens=500000,
            plan_fingerprint="abc123def456",
            bucket_efficiency=0.85,
            packing_ratio=1.5,
            packing_efficiency=0.90,
            token_budget_utilization=0.95,
            microbatches=100,
        )

        display = result.to_display()

        assert "Benchmark Summary" in display
        assert "10,000" in display
        assert "500,000" in display
        assert "100" in display
        assert "85.0%" in display
        assert "1.50x" in display
        assert "90.0%" in display
        assert "95.0%" in display
        assert "abc123def456" in display

    def test_default_values(self):
        """Test default values."""
        result = BenchmarkResult()

        assert result.samples == 0
        assert result.total_tokens == 0
        assert result.plan_fingerprint == ""
        assert result.bucket_efficiency == 0.0
        assert result.packing_ratio == 1.0
        assert result.packing_efficiency == 0.0
        assert result.token_budget_utilization == 0.0
        assert result.microbatches == 0


class TestBenchPipelineCmd:
    """Tests for bench_pipeline_cmd CLI entry point."""

    @pytest.mark.asyncio
    async def test_bench_pipeline_cmd(self, bench_args, capsys):
        """Test CLI entry point."""
        from chuk_lazarus.cli.commands.gym.benchmark import bench_pipeline_cmd

        mock_result = BenchmarkResult(
            samples=1000,
            total_tokens=50000,
            plan_fingerprint="abc123",
            bucket_efficiency=0.85,
            packing_ratio=1.5,
            packing_efficiency=0.90,
            token_budget_utilization=0.95,
            microbatches=100,
        )

        with patch(
            "chuk_lazarus.cli.commands.gym.benchmark.bench_pipeline",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await bench_pipeline_cmd(bench_args)

            captured = capsys.readouterr()
            assert "Benchmark Summary" in captured.out
            assert "1,000" in captured.out

    @pytest.mark.asyncio
    async def test_bench_pipeline_cmd_creates_config(self, bench_args, capsys):
        """Test that CLI entry point creates config from args."""
        from chuk_lazarus.cli.commands.gym.benchmark import bench_pipeline_cmd

        mock_result = BenchmarkResult(
            samples=500,
            total_tokens=25000,
            plan_fingerprint="xyz789",
            bucket_efficiency=0.80,
            packing_ratio=1.3,
            packing_efficiency=0.85,
            token_budget_utilization=0.90,
            microbatches=50,
        )

        with patch(
            "chuk_lazarus.cli.commands.gym.benchmark.bench_pipeline",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_bench:
            await bench_pipeline_cmd(bench_args)

            mock_bench.assert_called_once()
            call_args = mock_bench.call_args[0]
            config = call_args[0]
            assert isinstance(config, BenchmarkConfig)
            assert config.num_samples == 1000
            assert config.seed == 42

    @pytest.mark.asyncio
    async def test_bench_pipeline_cmd_with_dataset(self, bench_args, capsys):
        """Test CLI entry point with dataset argument."""
        from chuk_lazarus.cli.commands.gym.benchmark import bench_pipeline_cmd

        bench_args.dataset = "/path/to/data.jsonl"

        mock_result = BenchmarkResult(
            samples=100,
            total_tokens=5000,
            plan_fingerprint="data123",
            bucket_efficiency=0.90,
            packing_ratio=1.8,
            packing_efficiency=0.95,
            token_budget_utilization=0.98,
            microbatches=20,
        )

        with patch(
            "chuk_lazarus.cli.commands.gym.benchmark.bench_pipeline",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_bench:
            await bench_pipeline_cmd(bench_args)

            call_args = mock_bench.call_args[0]
            config = call_args[0]
            assert config.dataset == Path("/path/to/data.jsonl")

            captured = capsys.readouterr()
            assert "Benchmark Summary" in captured.out
