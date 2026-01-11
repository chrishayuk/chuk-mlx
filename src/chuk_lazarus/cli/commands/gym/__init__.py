"""Gym CLI commands.

This module provides commands for gym streaming and benchmarking.

Commands:
    gym_run: Run gym episode streaming
    bench_pipeline: Run batching pipeline benchmark
    gym_info: Display gym stream configuration
"""

from ._types import (
    BenchmarkConfig,
    BenchmarkResult,
    GymRunConfig,
    GymRunResult,
)
from .benchmark import bench_pipeline, bench_pipeline_cmd
from .info import gym_info, gym_info_cmd
from .run import gym_run, gym_run_cmd

__all__ = [
    # Types
    "BenchmarkConfig",
    "BenchmarkResult",
    "GymRunConfig",
    "GymRunResult",
    # Run Commands
    "gym_run",
    "gym_run_cmd",
    # Benchmark Commands
    "bench_pipeline",
    "bench_pipeline_cmd",
    # Info Commands
    "gym_info",
    "gym_info_cmd",
]
