"""Batching CLI commands."""

from ._types import (
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
from .analyze import data_batching_analyze, data_batching_analyze_cmd
from .generate import data_batch_generate, data_batch_generate_cmd
from .histogram import data_batching_histogram, data_batching_histogram_cmd
from .suggest import data_batching_suggest, data_batching_suggest_cmd

__all__ = [
    # Types
    "AnalyzeConfig",
    "AnalyzeResult",
    "GenerateConfig",
    "GenerateResult",
    "HistogramConfig",
    "HistogramResult",
    "OptimizationGoalType",
    "SuggestConfig",
    "SuggestResult",
    # Commands
    "data_batching_analyze",
    "data_batching_analyze_cmd",
    "data_batch_generate",
    "data_batch_generate_cmd",
    "data_batching_histogram",
    "data_batching_histogram_cmd",
    "data_batching_suggest",
    "data_batching_suggest_cmd",
]
