"""Tokenizer CLI commands.

This module provides commands for tokenizer operations including:
- Core operations (encode, decode, vocab, compare)
- Health checks (doctor, fingerprint, benchmark)
- Analysis (coverage, entropy, fit_score, efficiency, vocab_suggest, diff)
- Curriculum (length_buckets, reasoning_density)
- Training (throughput, pack)
- Regression testing
- Research (soft_tokens, embeddings, morph)
- Instrumentation (histogram, oov, waste, vocab_diff)
- Runtime (registry)
"""

# Core operations
# Analysis
from .analyze import (
    analyze_coverage,
    analyze_diff,
    analyze_efficiency,
    analyze_entropy,
    analyze_fit_score,
    analyze_vocab_suggest,
)
from .core import (
    tokenizer_compare,
    tokenizer_decode,
    tokenizer_encode,
    tokenizer_vocab,
)

# Curriculum
from .curriculum import (
    curriculum_length_buckets,
    curriculum_reasoning_density,
)

# Health checks
from .health import (
    tokenizer_benchmark,
    tokenizer_doctor,
    tokenizer_fingerprint,
)

# Instrumentation
from .instrument import (
    instrument_histogram,
    instrument_oov,
    instrument_vocab_diff,
    instrument_waste,
)

# Regression
from .regression import regression_run

# Research
from .research import (
    research_analyze_embeddings,
    research_morph,
    research_soft_tokens,
)

# Runtime
from .runtime import runtime_registry

# Training
from .training import (
    training_pack,
    training_throughput,
)

__all__ = [
    # Core
    "tokenizer_encode",
    "tokenizer_decode",
    "tokenizer_vocab",
    "tokenizer_compare",
    # Health
    "tokenizer_doctor",
    "tokenizer_fingerprint",
    "tokenizer_benchmark",
    # Analysis
    "analyze_coverage",
    "analyze_entropy",
    "analyze_fit_score",
    "analyze_efficiency",
    "analyze_vocab_suggest",
    "analyze_diff",
    # Curriculum
    "curriculum_length_buckets",
    "curriculum_reasoning_density",
    # Training
    "training_throughput",
    "training_pack",
    # Regression
    "regression_run",
    # Research
    "research_soft_tokens",
    "research_analyze_embeddings",
    "research_morph",
    # Instrumentation
    "instrument_histogram",
    "instrument_oov",
    "instrument_waste",
    "instrument_vocab_diff",
    # Runtime
    "runtime_registry",
]
