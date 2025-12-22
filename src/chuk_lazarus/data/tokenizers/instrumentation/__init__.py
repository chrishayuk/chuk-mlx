"""
Tokenizer instrumentation for training insights.

Pure observability - no behavior changes:
- Token length histograms per dataset
- OOV / rare-token reporting
- Token waste metrics (padding, truncation loss)
- Before/after stats when swapping vocabularies
"""

from .histograms import (
    HistogramBin,
    LengthHistogram,
    PercentileStats,
    compute_length_histogram,
    compute_percentiles,
    format_histogram_ascii,
    get_length_stats,
)
from .oov_report import (
    OOVReport,
    RareTokenInfo,
    TokenFrequencyBand,
    analyze_oov,
    find_rare_tokens,
    get_frequency_bands,
)
from .vocab_diff import (
    VocabSwapReport,
    compare_vocab_impact,
    estimate_retokenization_cost,
)
from .waste import (
    PaddingStats,
    TruncationStats,
    WasteReport,
    analyze_padding_waste,
    analyze_truncation_loss,
    analyze_waste,
)

__all__ = [
    # Histograms
    "HistogramBin",
    "LengthHistogram",
    "PercentileStats",
    "compute_length_histogram",
    "compute_percentiles",
    "format_histogram_ascii",
    "get_length_stats",
    # OOV Report
    "OOVReport",
    "RareTokenInfo",
    "TokenFrequencyBand",
    "analyze_oov",
    "find_rare_tokens",
    "get_frequency_bands",
    # Waste
    "PaddingStats",
    "TruncationStats",
    "WasteReport",
    "analyze_padding_waste",
    "analyze_truncation_loss",
    "analyze_waste",
    # Vocab Diff
    "VocabSwapReport",
    "compare_vocab_impact",
    "estimate_retokenization_cost",
]
