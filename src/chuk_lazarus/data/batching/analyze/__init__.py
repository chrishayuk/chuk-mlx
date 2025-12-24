"""
Batching analysis and instrumentation.

This submodule provides:
- LengthHistogram: Length distribution analysis
- BucketAnalysis: Per-bucket efficiency metrics
- BucketSuggestion: Optimal bucket edge suggestions
- BatchingEfficiencyReport: Complete efficiency reports

Mirrors the analysis capabilities in tokenizers/analyze/ and tokenizers/instrumentation/.
"""

from .efficiency import (
    BatchingEfficiencyReport,
    BucketAnalysis,
    BucketEfficiency,
    BucketSuggestion,
    HistogramBin,
    LengthHistogram,
    OptimizationGoal,
    analyze_bucket_efficiency,
    compute_length_histogram,
    create_efficiency_report,
    suggest_bucket_edges,
    visualize_buckets,
)

__all__ = [
    # Histogram
    "LengthHistogram",
    "HistogramBin",
    # Bucket analysis
    "BucketEfficiency",
    "BucketAnalysis",
    # Suggestions
    "BucketSuggestion",
    "OptimizationGoal",
    # Reports
    "BatchingEfficiencyReport",
    # Functions
    "compute_length_histogram",
    "analyze_bucket_efficiency",
    "suggest_bucket_edges",
    "create_efficiency_report",
    "visualize_buckets",
]
