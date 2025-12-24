"""
Tokenizer utilities and tools.

A comprehensive, Pydantic-native tokenization toolkit for building,
analyzing, and managing tokenizers. Designed as a control surface for
training frameworks - tokenizers are inspectable, mutable, schedulable,
and debuggable.

Core:
- CharacterTokenizer: Character-level tokenizer for classification/experiments
- CustomTokenizer: Whitespace tokenizer with special token handling
- TokenDisplayUtility: Token visualization and debugging

Statistics & Analysis:
- token_stats: Frequency, coverage, compression analysis
- token_debug: Token inspection and comparison

Batch Processing:
- batch_processing: Encoding, padding, chunking utilities

Special Tokens:
- special_tokens: BOS, EOS, PAD token handling

Vocabulary Management:
- vocab_manager: Merge, filter, validate vocabularies
- vocab_utils: Load/save vocabulary files

Validation & Conversion:
- validation: Roundtrip testing and integrity checks
- conversion: Format conversion (JSON, TSV, CSV, HuggingFace)

Submodules:
- analyze: Coverage, entropy, fit scoring, retokenization diff
- curriculum: Token-length buckets, reasoning density scoring
- runtime: Special token registry, dynamic vocab, semantics mapping
- training: Sequence packing, throughput profiling
- regression: Token regression test framework
"""

# Batch processing
# Submodules - import for namespace access
from . import analyze, curriculum, regression, runtime, training
from .batch_processing import (
    BatchResult,
    ChunkConfig,
    PaddingSide,
    SequenceStats,
    chunk_text,
    create_batch,
    decode_batch,
    encode_batch,
    get_sequence_lengths,
    pad_batch,
)
from .bow_tokenizer import BoWCharacterTokenizer, BoWTokenizerConfig
from .character_tokenizer import CharacterTokenizer, CharacterTokenizerConfig

# Conversion
from .conversion import (
    ConversionResult,
    ExportFormat,
    TokenizerConfig,
    TokenizerFormat,
    TokenMapping,
    VocabExport,
    create_huggingface_tokenizer_json,
    create_token_mappings,
    export_vocab_csv,
    export_vocab_json,
    export_vocab_tsv,
    export_vocabulary,
    extract_config_from_tokenizer,
    import_vocab_csv,
    import_vocab_json,
    import_vocab_tsv,
    import_vocabulary,
    load_vocabulary_file,
    save_huggingface_format,
    save_vocabulary_file,
)
from .custom_tokenizer import CustomTokenizer

# Fingerprinting
from .fingerprint import (
    FingerprintMismatch,
    FingerprintRegistry,
    TokenizerFingerprint,
    assert_fingerprint,
    compute_fingerprint,
    get_registry,
    load_fingerprint,
    save_fingerprint,
    verify_fingerprint,
)

# Special tokens
from .special_tokens import (
    SpecialTokenConfig,
    SpecialTokenCount,
    SpecialTokenType,
    add_bos_token,
    add_eos_token,
    add_special_tokens,
    count_special_tokens,
    ensure_special_tokens,
    find_eos_positions,
    get_special_token_ids,
    get_special_token_mask,
    split_on_eos,
    strip_padding,
    strip_special_tokens,
)

# Token debugging
from .token_debug import (
    TokenComparison,
    TokenInfo,
    UnknownTokenAnalysis,
    analyze_unknown_tokens,
    compare_tokenizations,
    find_token_by_string,
    format_token_table,
    get_similar_tokens,
    get_token_info,
    get_tokens_info,
    highlight_tokens,
    token_to_bytes,
)
from .token_display import TokenDisplayUtility

# Token statistics
from .token_stats import (
    CompressionStats,
    CoverageStats,
    LengthDistribution,
    TokenFrequency,
    calculate_compression_ratio,
    get_rare_tokens,
    get_token_frequencies,
    get_token_length_distribution,
    get_top_tokens,
    get_vocabulary_coverage,
)

# Validation
from .validation import (
    BatchRoundtripResult,
    EncodingConsistency,
    RoundtripResult,
    ValidationCategory,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    assert_valid_tokenizer,
    check_batch_roundtrip,
    check_encoding_consistency,
    check_roundtrip,
    create_validation_report,
    validate_encoding_decoding,
    validate_special_tokens,
)
from .validation import (
    validate_vocabulary as validate_tokenizer_vocabulary,
)

# Vocabulary management
from .vocab_manager import (
    ConflictResolution,
    SortOrder,
    VocabularyDiff,
    VocabularyIssues,
    VocabularyStats,
    create_id_to_token,
    extend_vocabulary,
    filter_vocabulary,
    get_vocabulary_diff,
    get_vocabulary_stats,
    merge_vocabularies,
    renumber_vocabulary,
    shrink_vocabulary,
    validate_vocabulary,
)
from .vocab_utils import load_vocabulary, save_vocabulary

__all__ = [
    # Core
    "CharacterTokenizer",
    "CharacterTokenizerConfig",
    "BoWCharacterTokenizer",
    "BoWTokenizerConfig",
    "CustomTokenizer",
    "TokenDisplayUtility",
    "load_vocabulary",
    "save_vocabulary",
    # Token stats
    "CoverageStats",
    "CompressionStats",
    "LengthDistribution",
    "TokenFrequency",
    "calculate_compression_ratio",
    "get_rare_tokens",
    "get_token_frequencies",
    "get_token_length_distribution",
    "get_top_tokens",
    "get_vocabulary_coverage",
    # Batch processing
    "BatchResult",
    "ChunkConfig",
    "PaddingSide",
    "SequenceStats",
    "chunk_text",
    "create_batch",
    "decode_batch",
    "encode_batch",
    "get_sequence_lengths",
    "pad_batch",
    # Special tokens
    "SpecialTokenConfig",
    "SpecialTokenCount",
    "SpecialTokenType",
    "add_bos_token",
    "add_eos_token",
    "add_special_tokens",
    "count_special_tokens",
    "ensure_special_tokens",
    "find_eos_positions",
    "get_special_token_ids",
    "get_special_token_mask",
    "split_on_eos",
    "strip_padding",
    "strip_special_tokens",
    # Vocabulary management
    "ConflictResolution",
    "SortOrder",
    "VocabularyDiff",
    "VocabularyIssues",
    "VocabularyStats",
    "create_id_to_token",
    "extend_vocabulary",
    "filter_vocabulary",
    "get_vocabulary_diff",
    "get_vocabulary_stats",
    "merge_vocabularies",
    "renumber_vocabulary",
    "shrink_vocabulary",
    "validate_vocabulary",
    # Token debugging
    "TokenComparison",
    "TokenInfo",
    "UnknownTokenAnalysis",
    "analyze_unknown_tokens",
    "compare_tokenizations",
    "find_token_by_string",
    "format_token_table",
    "get_similar_tokens",
    "get_token_info",
    "get_tokens_info",
    "highlight_tokens",
    "token_to_bytes",
    # Validation
    "BatchRoundtripResult",
    "EncodingConsistency",
    "RoundtripResult",
    "ValidationCategory",
    "ValidationIssue",
    "ValidationReport",
    "ValidationSeverity",
    "assert_valid_tokenizer",
    "check_batch_roundtrip",
    "check_encoding_consistency",
    "check_roundtrip",
    "create_validation_report",
    "validate_encoding_decoding",
    "validate_special_tokens",
    "validate_tokenizer_vocabulary",
    # Conversion
    "ConversionResult",
    "ExportFormat",
    "TokenizerConfig",
    "TokenizerFormat",
    "TokenMapping",
    "VocabExport",
    "create_huggingface_tokenizer_json",
    "create_token_mappings",
    "export_vocab_csv",
    "export_vocab_json",
    "export_vocab_tsv",
    "export_vocabulary",
    "extract_config_from_tokenizer",
    "import_vocab_csv",
    "import_vocab_json",
    "import_vocab_tsv",
    "import_vocabulary",
    "load_vocabulary_file",
    "save_huggingface_format",
    "save_vocabulary_file",
    # Submodules
    "analyze",
    "curriculum",
    "runtime",
    "training",
    "regression",
    # Fingerprinting
    "TokenizerFingerprint",
    "FingerprintMismatch",
    "FingerprintRegistry",
    "compute_fingerprint",
    "verify_fingerprint",
    "assert_fingerprint",
    "save_fingerprint",
    "load_fingerprint",
    "get_registry",
]
