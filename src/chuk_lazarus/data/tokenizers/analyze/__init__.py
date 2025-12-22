"""
Token analysis utilities for deep tokenizer introspection.

Modules:
- coverage: Token coverage and UNK rate analysis
- entropy: Token entropy and distribution analysis
- fit_score: Dataset-tokenizer compatibility scoring
- diff: Retokenization comparison tools
- efficiency: Token efficiency metrics
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
    CorpusDiff,
    RetokenizationDiff,
    TokenBoundaryShift,
    compare_tokenizations_detailed,
    diff_corpus,
)
from .efficiency import (
    ContentTypeStats,
    EfficiencyConfig,
    EfficiencyReport,
    FragmentationStats,
    SampleStats,
    analyze_content_type,
    analyze_efficiency,
    analyze_fragmentation,
    analyze_sample_stats,
)
from .entropy import (
    EntropyReport,
    TokenDistribution,
    analyze_entropy,
    calculate_entropy,
    get_token_distribution,
)
from .fit_score import (
    FitScore,
    FitScoreConfig,
    TokenizerComparison,
    calculate_fit_score,
    compare_tokenizers_for_dataset,
)
from .vocab_induction import (
    DomainVocab,
    InductionConfig,
    InductionReport,
    TokenCandidate,
    TokenDomain,
    analyze_vocab_induction,
    find_fragmented_words,
    find_frequent_ngrams,
    get_domain_vocab,
    list_domain_vocabs,
    suggest_domain_tokens,
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
    # Efficiency
    "EfficiencyConfig",
    "EfficiencyReport",
    "SampleStats",
    "ContentTypeStats",
    "FragmentationStats",
    "analyze_efficiency",
    "analyze_sample_stats",
    "analyze_content_type",
    "analyze_fragmentation",
    # Vocab induction
    "InductionConfig",
    "InductionReport",
    "TokenCandidate",
    "TokenDomain",
    "DomainVocab",
    "analyze_vocab_induction",
    "find_fragmented_words",
    "find_frequent_ngrams",
    "suggest_domain_tokens",
    "get_domain_vocab",
    "list_domain_vocabs",
]
