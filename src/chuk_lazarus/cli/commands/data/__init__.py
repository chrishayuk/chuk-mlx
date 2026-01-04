"""Data processing CLI commands."""

# Shared types
from ._types import OutputFormat, SampleIdField, SampleTextField

# Batching commands
from .batching import (
    AnalyzeConfig,
    AnalyzeResult,
    GenerateConfig,
    GenerateResult,
    HistogramConfig,
    HistogramResult,
    OptimizationGoalType,
    SuggestConfig,
    SuggestResult,
    data_batch_generate,
    data_batch_generate_cmd,
    data_batching_analyze,
    data_batching_analyze_cmd,
    data_batching_histogram,
    data_batching_histogram_cmd,
    data_batching_suggest,
    data_batching_suggest_cmd,
)

# Batch plan commands
from .batchplan import (
    BatchPlanBuildConfig,
    BatchPlanBuildResult,
    BatchPlanInfoConfig,
    BatchPlanInfoResult,
    BatchPlanMode,
    BatchPlanShardConfig,
    BatchPlanShardResult,
    BatchPlanVerifyConfig,
    BatchPlanVerifyResult,
    InvalidRankError,
    data_batchplan_build,
    data_batchplan_build_cmd,
    data_batchplan_info,
    data_batchplan_info_cmd,
    data_batchplan_shard,
    data_batchplan_shard_cmd,
    data_batchplan_verify,
    data_batchplan_verify_cmd,
)

# Length cache commands
from .lengths import (
    EmptyStatsResult,
    LengthBuildConfig,
    LengthBuildResult,
    LengthStatsConfig,
    LengthStatsResult,
    data_lengths_build,
    data_lengths_build_cmd,
    data_lengths_stats,
    data_lengths_stats_cmd,
)

__all__ = [
    # Shared types
    "OutputFormat",
    "SampleIdField",
    "SampleTextField",
    # Length types
    "EmptyStatsResult",
    "LengthBuildConfig",
    "LengthBuildResult",
    "LengthStatsConfig",
    "LengthStatsResult",
    # Length commands
    "data_lengths_build",
    "data_lengths_build_cmd",
    "data_lengths_stats",
    "data_lengths_stats_cmd",
    # Batch plan types
    "BatchPlanBuildConfig",
    "BatchPlanBuildResult",
    "BatchPlanInfoConfig",
    "BatchPlanInfoResult",
    "BatchPlanMode",
    "BatchPlanShardConfig",
    "BatchPlanShardResult",
    "BatchPlanVerifyConfig",
    "BatchPlanVerifyResult",
    "InvalidRankError",
    # Batch plan commands
    "data_batchplan_build",
    "data_batchplan_build_cmd",
    "data_batchplan_info",
    "data_batchplan_info_cmd",
    "data_batchplan_shard",
    "data_batchplan_shard_cmd",
    "data_batchplan_verify",
    "data_batchplan_verify_cmd",
    # Batching types
    "AnalyzeConfig",
    "AnalyzeResult",
    "GenerateConfig",
    "GenerateResult",
    "HistogramConfig",
    "HistogramResult",
    "OptimizationGoalType",
    "SuggestConfig",
    "SuggestResult",
    # Batching commands
    "data_batch_generate",
    "data_batch_generate_cmd",
    "data_batching_analyze",
    "data_batching_analyze_cmd",
    "data_batching_histogram",
    "data_batching_histogram_cmd",
    "data_batching_suggest",
    "data_batching_suggest_cmd",
]
