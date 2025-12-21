"""
Token analysis utilities for deep tokenizer introspection.

Modules:
- coverage: Token coverage and UNK rate analysis
- entropy: Token entropy and distribution analysis
- fit_score: Dataset-tokenizer compatibility scoring
- diff: Retokenization comparison tools
"""

from .coverage import (
    CoverageReport,
    FragmentAnalysis,
    analyze_coverage,
    analyze_fragments,
    get_tokens_per_word,
    get_unk_rate,
)
from .diff import (
    RetokenizationDiff,
    TokenBoundaryShift,
    compare_tokenizations_detailed,
    diff_corpus,
)
from .entropy import (
    EntropyReport,
    TokenDistribution,
    analyze_entropy,
    calculate_entropy,
    get_token_distribution,
)
from .diff import CorpusDiff
from .fit_score import (
    FitScore,
    FitScoreConfig,
    TokenizerComparison,
    calculate_fit_score,
    compare_tokenizers_for_dataset,
)

__all__ = [
    # Coverage
    "CoverageReport",
    "FragmentAnalysis",
    "analyze_coverage",
    "analyze_fragments",
    "get_tokens_per_word",
    "get_unk_rate",
    # Entropy
    "EntropyReport",
    "TokenDistribution",
    "analyze_entropy",
    "calculate_entropy",
    "get_token_distribution",
    # Fit score
    "FitScore",
    "FitScoreConfig",
    "TokenizerComparison",
    "calculate_fit_score",
    "compare_tokenizers_for_dataset",
    # Diff
    "CorpusDiff",
    "RetokenizationDiff",
    "TokenBoundaryShift",
    "compare_tokenizations_detailed",
    "diff_corpus",
]
