# Tokenizers Module

A comprehensive, Pydantic-native tokenization toolkit for building, analyzing, and managing tokenizers. **Designed as a control surface for training frameworks** — tokenizers are inspectable, mutable, schedulable, and debuggable.

## Overview

This module provides utilities for:
- **Custom tokenizer creation** - Build character or word-level tokenizers
- **Token statistics & analysis** - Frequency counts, coverage, compression ratios
- **Batch processing** - Efficient encoding/decoding with padding and chunking
- **Special token handling** - BOS, EOS, PAD, UNK token management
- **Vocabulary management** - Merge, filter, validate vocabularies
- **Debugging & visualization** - Token inspection and comparison tools
- **Validation** - Roundtrip testing and integrity checks
- **Format conversion** - Export/import JSON, TSV, CSV, HuggingFace format

### Training Framework Submodules

- **analyze/** - Coverage, entropy, fit scoring, retokenization diff
- **curriculum/** - Token-length buckets, reasoning density scoring
- **preprocessing/** - Pre-tokenization transforms, profiles, byte fallback
- **runtime/** - Special token registry, dynamic vocab, semantics mapping
- **training/** - Sequence packing, throughput profiling
- **regression/** - Token regression test framework

## Design Principles

- **Pydantic-native**: All data structures use Pydantic BaseModel for validation
- **No magic strings**: Enums and constants for type safety
- **Protocol-based**: TokenizerProtocol allows any compatible tokenizer
- **Composable**: Functions work independently or together
- **Training-aware**: Tokenizers as a control surface, not just preprocessing

## Quick Start

```python
from chuk_lazarus.data.tokenizers import CustomTokenizer
from chuk_lazarus.data.tokenizers.batch_processing import create_batch, PaddingSide
from chuk_lazarus.data.tokenizers.token_stats import get_top_tokens

# Create a tokenizer
tokenizer = CustomTokenizer()
tokenizer.build_vocab("The quick brown fox jumps over the lazy dog.", min_freq=1)

# Encode text
tokens = tokenizer.encode("The fox is quick.")
print(f"Tokens: {tokens}")

# Batch processing
batch = create_batch(
    ["Hello world", "How are you?"],
    tokenizer,
    padding=True,
    padding_side=PaddingSide.RIGHT,
)
print(f"Batch shape: {len(batch.input_ids)} x {len(batch.input_ids[0])}")
```

## Modules

### `custom_tokenizer.py`
Custom tokenizer implementation extending HuggingFace's PreTrainedTokenizer.

```python
from chuk_lazarus.data.tokenizers import CustomTokenizer

tokenizer = CustomTokenizer()
tokenizer.build_vocab(text, min_freq=2)
tokenizer.save("./my_tokenizer")
loaded = CustomTokenizer.load("./my_tokenizer")
```

### `token_stats.py`
Statistics and analysis for tokenization.

```python
from chuk_lazarus.data.tokenizers.token_stats import (
    get_token_frequencies,
    get_vocabulary_coverage,
    calculate_compression_ratio,
    get_top_tokens,
    get_rare_tokens,
    get_token_length_distribution,
)

# Analyze token frequency
frequencies = get_token_frequencies(texts, tokenizer)

# Check vocabulary coverage
coverage = get_vocabulary_coverage(text, tokenizer)
print(f"Coverage: {coverage.coverage_ratio:.2%}")

# Compression analysis
stats = calculate_compression_ratio(text, tokenizer)
print(f"Chars per token: {stats.chars_per_token:.2f}")
```

**Models:**
- `CoverageStats` - Known/unknown token counts
- `CompressionStats` - Characters and bytes per token
- `TokenFrequency` - Token ID, string, and count
- `LengthDistribution` - Token length histogram

### `batch_processing.py`
Efficient batch encoding and padding.

```python
from chuk_lazarus.data.tokenizers.batch_processing import (
    create_batch,
    pad_batch,
    chunk_text,
    encode_batch,
    decode_batch,
    get_sequence_lengths,
    PaddingSide,
    ChunkConfig,
)

# Create padded batch
batch = create_batch(
    texts,
    tokenizer,
    max_length=512,
    padding=True,
    truncation=True,
    padding_side=PaddingSide.LEFT,
)

# Chunk long text
config = ChunkConfig(chunk_size=128, overlap=16)
chunks = chunk_text(long_text, tokenizer, config)
```

**Models:**
- `BatchResult` - input_ids and attention_mask
- `SequenceStats` - min/max/mean lengths
- `ChunkConfig` - chunk_size, overlap, add_special_tokens

**Enums:**
- `PaddingSide` - LEFT, RIGHT

### `special_tokens.py`
Special token utilities.

```python
from chuk_lazarus.data.tokenizers.special_tokens import (
    SpecialTokenConfig,
    SpecialTokenType,
    get_special_token_ids,
    strip_special_tokens,
    add_bos_token,
    add_eos_token,
    split_on_eos,
    count_special_tokens,
)

# Get config from tokenizer
config = SpecialTokenConfig.from_tokenizer(tokenizer)
all_special = config.all_special_ids()

# Manipulate sequences
tokens = add_bos_token(tokens, tokenizer.bos_token_id)
tokens = strip_special_tokens(tokens, tokenizer)
segments = split_on_eos(tokens, tokenizer.eos_token_id)
```

**Models:**
- `SpecialTokenConfig` - All special token IDs
- `SpecialTokenCount` - Counts of each special token type

**Enums:**
- `SpecialTokenType` - PAD, UNK, BOS, EOS, SEP, CLS, MASK

### `vocab_manager.py`
Vocabulary manipulation utilities.

```python
from chuk_lazarus.data.tokenizers.vocab_manager import (
    merge_vocabularies,
    filter_vocabulary,
    extend_vocabulary,
    shrink_vocabulary,
    renumber_vocabulary,
    validate_vocabulary,
    get_vocabulary_stats,
    get_vocabulary_diff,
    ConflictResolution,
    SortOrder,
)

# Merge two vocabularies
merged = merge_vocabularies(
    vocab1, vocab2,
    conflict_resolution=ConflictResolution.FIRST
)

# Filter by frequency
filtered = filter_vocabulary(
    vocab, token_counts,
    min_freq=5,
    keep_special={"<pad>", "<unk>"}
)

# Validate integrity
issues = validate_vocabulary(vocab)
if issues.has_issues():
    print(f"Found {len(issues.duplicate_ids)} duplicate IDs")
```

**Models:**
- `VocabularyStats` - Size, ID range, token lengths
- `VocabularyIssues` - Duplicates, gaps, negatives
- `VocabularyDiff` - Differences between vocabularies

**Enums:**
- `ConflictResolution` - FIRST, SECOND, RENUMBER
- `SortOrder` - BY_ID, ALPHABETICAL

### `token_debug.py`
Debugging and visualization tools.

```python
from chuk_lazarus.data.tokenizers.token_debug import (
    get_token_info,
    get_tokens_info,
    compare_tokenizations,
    analyze_unknown_tokens,
    highlight_tokens,
    format_token_table,
    find_token_by_string,
    get_similar_tokens,
)

# Inspect a token
info = get_token_info(token_id, tokenizer)
print(f"Token: {info.token_str}, Bytes: {info.byte_repr}")

# Compare tokenizers
comparison = compare_tokenizations(text, tokenizer1, tokenizer2)
print(f"Tokenizer 1: {comparison.tokenizer1_count} tokens")
print(f"Tokenizer 2: {comparison.tokenizer2_count} tokens")

# Visualize token boundaries
highlighted = highlight_tokens("Hello world", tokenizer, separator="|")
# Output: "Hello|world"

# Format as table
table = format_token_table(token_ids, tokenizer)
print(table)
```

**Models:**
- `TokenInfo` - ID, string, bytes, lengths
- `TokenComparison` - Side-by-side tokenization results
- `UnknownTokenAnalysis` - Unknown token positions and segments

### `validation.py`
Tokenizer validation and testing.

```python
from chuk_lazarus.data.tokenizers.validation import (
    check_roundtrip,
    check_batch_roundtrip,
    check_encoding_consistency,
    validate_special_tokens,
    validate_vocabulary,
    create_validation_report,
    assert_valid_tokenizer,
    ValidationSeverity,
    ValidationCategory,
)

# Test roundtrip encoding
result = check_roundtrip("Hello world", tokenizer)
if not result.is_lossless:
    print(f"Lost characters: {result.diff_chars}")

# Full validation
report = create_validation_report(tokenizer, "MyTokenizer")
print(f"Valid: {report.is_valid}")
print(f"Errors: {report.error_count}, Warnings: {report.warning_count}")

# Assert valid (raises on failure)
report = assert_valid_tokenizer(tokenizer, allow_warnings=True)
```

**Models:**
- `ValidationIssue` - Category, severity, message, details
- `RoundtripResult` - Original, encoded, decoded, is_lossless
- `BatchRoundtripResult` - Aggregated pass/fail statistics
- `EncodingConsistency` - Determinism test results
- `ValidationReport` - Complete validation summary

**Enums:**
- `ValidationSeverity` - INFO, WARNING, ERROR
- `ValidationCategory` - ROUNDTRIP, ENCODING, DECODING, etc.

### `conversion.py`
Format conversion utilities.

```python
from chuk_lazarus.data.tokenizers.conversion import (
    export_vocabulary,
    import_vocabulary,
    save_vocabulary_file,
    load_vocabulary_file,
    save_huggingface_format,
    extract_config_from_tokenizer,
    create_token_mappings,
    ExportFormat,
    TokenizerFormat,
)

# Export vocabulary
export = export_vocabulary(vocab, format=ExportFormat.JSON)
print(export.content)

# Save to file (format inferred from extension)
save_vocabulary_file(vocab, "vocab.tsv")

# Load from file
vocab = load_vocabulary_file("vocab.json")

# Save as HuggingFace format
result = save_huggingface_format(vocab, "./hf_tokenizer")
```

**Models:**
- `TokenizerConfig` - Vocab size, special tokens, max length
- `VocabExport` - Format, size, content
- `ConversionResult` - Success status, paths, errors
- `TokenMapping` - Token ID, string, byte fallback

**Enums:**
- `TokenizerFormat` - HUGGINGFACE, SENTENCEPIECE, TIKTOKEN, CUSTOM_JSON
- `ExportFormat` - JSON, TSV, CSV

## Example: Chess Tokenizer

See `examples/tokenizer/chess_tokenizer.py` for a complete domain-specific tokenizer example that handles chess notation (PGN format).

```python
from examples.tokenizer.chess_tokenizer import ChessTokenizer, ChessGame

tokenizer = ChessTokenizer()
game = ChessGame(
    moves=["e4", "e5", "Nf3", "Nc6", "Bb5"],
    result=ChessResult.ONGOING,
)

tokens = tokenizer.tokenize_game(game)
decoded = tokenizer.decode(tokens)
```

## TokenizerProtocol

All utilities accept any tokenizer implementing this protocol:

```python
class TokenizerProtocol(Protocol):
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def get_vocab(self) -> dict[str, int]: ...
    pad_token_id: int
    unk_token_id: int | None
    bos_token_id: int | None
    eos_token_id: int | None
```

HuggingFace tokenizers, SentencePiece, and custom implementations all work seamlessly.

## Testing

Run tests with:

```bash
pytest tests/data/tokenizers/ -v --cov=src/chuk_lazarus/data/tokenizers
```

Target: 90%+ coverage per file.

## Training Framework Submodules

### `analyze/` - Token Analysis

Coverage, entropy, fit scoring, and retokenization comparison.

```python
from chuk_lazarus.data.tokenizers.analyze import (
    analyze_coverage,
    analyze_entropy,
    calculate_fit_score,
    compare_tokenizers_for_dataset,
    diff_corpus,
)

# Coverage analysis (UNK rate, tokens/word, vocab utilization)
coverage = analyze_coverage(texts, tokenizer)
print(f"UNK rate: {coverage.unk_rate:.2%}")
print(f"Tokens/word: {coverage.tokens_per_word:.2f}")

# Entropy analysis (Shannon entropy, perplexity)
entropy = analyze_entropy(texts, tokenizer)
print(f"Entropy: {entropy.entropy:.2f} bits")
print(f"Perplexity: {entropy.perplexity:.2f}")

# Tokenizer-dataset fit score
fit = calculate_fit_score(texts, tokenizer)
print(f"Fit score: {fit.overall_score:.2f}")
print(f"Recommendation: {fit.recommendation}")

# Compare two tokenizers on the same dataset
comparison = compare_tokenizers_for_dataset(texts, tok1, tok2)
print(f"Winner: {comparison.recommendation}")

# Retokenization diff (compare old vs new tokenizer)
corpus_diff = diff_corpus(texts, old_tokenizer, new_tokenizer)
print(f"Boundary shifts: {corpus_diff.total_boundary_shifts}")
```

**Models:**
- `CoverageReport` - UNK rate, tokens/word, vocab utilization, warnings
- `EntropyReport` - Shannon entropy, normalized entropy, perplexity
- `FitScore` - Coverage, compression, entropy scores with recommendation
- `RetokenizationDiff` - Token boundary shifts and content changes

### `curriculum/` - Curriculum Learning

Token-length buckets and reasoning density scoring for curriculum learning.

```python
from chuk_lazarus.data.tokenizers.curriculum import (
    create_length_buckets,
    get_curriculum_schedule,
    score_reasoning_density,
    sort_by_reasoning_density,
)

# Create length-based curriculum buckets
from chuk_lazarus.data.tokenizers.curriculum.length_buckets import LengthBucketConfig
config = LengthBucketConfig(num_buckets=5)
buckets = create_length_buckets(texts, tokenizer, config)
for bucket in buckets:
    print(f"Bucket {bucket.bucket_id}: {bucket.sample_count} samples, {bucket.min_tokens}-{bucket.max_tokens} tokens")

# Get curriculum schedule (easy → hard)
schedule = get_curriculum_schedule(texts, tokenizer)
print(f"Warmup samples: {schedule.warmup_samples}")
print(f"Schedule: {schedule.schedule_order}")

# Score reasoning density (math symbols, brackets, variables)
for i, text in enumerate(texts):
    score = score_reasoning_density(text, i, tokenizer)
    print(f"Reasoning density: {score.overall_score:.2f}")

# Sort by reasoning density (easiest first)
scores = sort_by_reasoning_density(texts, tokenizer)
```

**Models:**
- `LengthBucket` - Bucket ID, token range, sample count, sample indices
- `CurriculumSchedule` - Ordered buckets with warmup/ramp phases
- `ReasoningDensityScore` - Math, bracket, variable, numeric, operator, length scores
- `DifficultyPercentiles` - p10/p25/p50/p75/p90 percentiles for curriculum planning

### `runtime/` - Runtime Utilities

Special token registry, dynamic vocabulary, and semantics mapping.

```python
from chuk_lazarus.data.tokenizers.runtime import (
    SpecialTokenRegistry,
    TokenCategory,
    create_standard_registry,
    extend_vocab_runtime,
    map_token_to_semantic,
    SemanticDomain,
)

# Special token registry with collision detection
registry = create_standard_registry(vocab_size=50000)
registry.register(
    token_str="<TOOL_CALL>",
    token_id=50001,
    category=TokenCategory.TOOL_CALL,
    description="Start of tool invocation",
)

# Check for collisions
report = registry.check_collisions()
if report.has_collisions:
    print(f"Collisions detected: {report.collisions}")

# Dynamic vocabulary extension
from chuk_lazarus.data.tokenizers.runtime.dynamic_vocab import DynamicVocab
vocab = DynamicVocab.from_tokenizer(tokenizer)
extensions = extend_vocab_runtime(vocab, ["<SOLVER_ADD>", "<SOLVER_MUL>"], tokenizer)
for ext in extensions:
    print(f"New token: {ext.token_str} -> {ext.token_id}")

# Semantics mapping for tools/agents
from chuk_lazarus.data.tokenizers.runtime.semantics import TokenSemantics, create_standard_semantics
semantics = create_standard_semantics()
mapping = semantics.get_by_id(100)  # <LOAD_PAGE> token
print(f"Domain: {mapping.domain}")  # SemanticDomain.MEMORY
print(f"Operation: {mapping.operation}")  # "load"
print(f"Full path: {mapping.full_path}")  # "memory.op.load"
```

**Models:**
- `SpecialTokenRegistry` - Registry with reserved ranges and collision detection
- `SpecialTokenEntry` - Token string, ID, category, description
- `TokenCategory` - PADDING, UNKNOWN, TOOL_CALL, TOOL_RESULT, MEMORY_LOAD, PAGE_IN, SOLVER_OP, THINK_START, THINK_END
- `CollisionReport` - Collision detection results with has_collisions flag
- `DynamicVocab` - Runtime vocab extension tracker
- `VocabExtension` - Token extension with embedding initialization method
- `TokenSemantics` - Semantic domain and operation mapping registry
- `SemanticMapping` - Token to domain/operation mapping with arguments
- `SemanticDomain` - MEMORY, TOOL, SOLVER, CONTROL, DATA, CUSTOM

### `training/` - Training Utilities

Sequence packing and throughput profiling.

```python
from chuk_lazarus.data.tokenizers.training import (
    create_packed_batch,
    calculate_packing_efficiency,
    profile_tokenization,
    estimate_training_tokens,
    ThroughputProfiler,
)

# Smart sequence packing (20-40% throughput improvement)
packed = create_packed_batch(texts, tokenizer, max_seq_length=512)
print(f"Packing ratio: {packed.packing_ratio:.2f}x")

stats = calculate_packing_efficiency(packed)
print(f"Efficiency: {stats.efficiency:.2%}")
print(f"Throughput improvement: {stats.throughput_improvement:.2f}x")

# Throughput profiling
metrics = profile_tokenization(texts, tokenizer)
print(f"Tokens/second: {metrics.tokens_per_second:.0f}")
print(f"Avg tokens/text: {metrics.avg_tokens_per_text:.1f}")

# Estimate training tokens
estimate = estimate_training_tokens(sample_texts, tokenizer, epochs=3, sample_ratio=0.1)
print(f"Total training tokens: {estimate['total_training_tokens']:,}")
```

**Models:**
- `PackedBatch` - Efficiently packed sequences with metadata
- `PackingStats` - Efficiency, padding ratio, throughput improvement
- `ThroughputMetrics` - Tokens/sec, chars/sec, compression stats
- `BatchMetrics` - Per-batch statistics and attention waste

### `preprocessing/` - Pre-Tokenization Transforms

Pre-tokenization hooks, profiles, and byte fallback for robust tokenization.

```python
from chuk_lazarus.data.tokenizers.preprocessing import (
    # Numeric normalization
    NumericConfig,
    detect_numbers,
    normalize_numbers,
    restore_numbers,
    # Structure token injection
    StructureConfig,
    StructureType,
    detect_structures,
    inject_structure_tokens,
    restore_structures,
    # Hooks
    HookPipeline,
    HookedTokenizer,
    create_standard_pipeline,
    create_math_pipeline,
    create_tool_pipeline,
    # Profiles
    TokenizerProfile,
    ProfiledTokenizer,
    create_training_profile,
    create_inference_profile,
    # Byte fallback
    ByteFallbackWrapper,
    wrap_with_fallback,
)

# Numeric normalization - reduce token waste on numbers
text = "Pi is 3.14159 and e is 2.71828"
encoding = normalize_numbers(text)
print(encoding.encoded_text)  # "Pi is <NUM_0> and e is <NUM_1>"
print(encoding.mapping)  # {"<NUM_0>": "3.14159", "<NUM_1>": "2.71828"}
restored = restore_numbers(encoding.encoded_text, encoding.mapping)
assert restored == text

# Structure token injection - atomic tokens for UUIDs, URLs, IPs, etc.
text = "User 550e8400-e29b-41d4-a716-446655440000 at 192.168.1.1"
encoding = inject_structure_tokens(text)
print(encoding.encoded_text)  # "User <UUID_0> at <IP_0>"

# Hook pipeline - composable pre/post transforms
pipeline = create_standard_pipeline(numeric=True, structure=True)
transformed = pipeline.pre_tokenize(text)
restored = pipeline.post_decode(transformed)

# Hooked tokenizer - transparent preprocessing
hooked = HookedTokenizer(tokenizer, pipeline)
tokens = hooked.encode(text)  # Transforms applied automatically
decoded = hooked.decode(tokens)  # Inverse transforms on decode

# Tokenizer profiles - switch between training/inference behavior
training_profile = create_training_profile(
    normalize_numbers=True,
    inject_structures=True,
    max_length=2048,
)
profiled = ProfiledTokenizer(tokenizer, training_profile)
tokens = profiled.encode(text)

# Switch to inference mode
inference_profile = create_inference_profile()
profiled.set_profile(inference_profile)

# Byte fallback - ensure any byte sequence tokenizes without UNK
wrapper = wrap_with_fallback(tokenizer)
tokens = wrapper.encode("Emoji: 🎉🎊 and Chinese: 日本語")
decoded = wrapper.decode(tokens)  # No UNK tokens
```

**Models:**
- `NumericConfig` - Detection settings (integers, floats, scientific, hex, percentages, fractions)
- `NumericSpan` - Detected number with position, format, and parsed value
- `NumericEncoding` - Encoded text with placeholder mapping
- `StructureConfig` - Detection settings (UUIDs, URLs, emails, IPs, dates, paths, JSON)
- `StructureSpan` - Detected structure with position and type
- `StructureEncoding` - Encoded text with structure mapping
- `TokenizerProfile` - Profile configuration (mode, normalization, fallback, truncation)
- `ByteFallbackConfig` - Fallback settings (UNK detection, byte encoding template)
- `ByteFallbackStats` - Fallback statistics (chars encoded, UNK avoided)

**Enums:**
- `NumericFormat` - INTEGER, FLOAT, SCIENTIFIC, HEXADECIMAL, BINARY, PERCENTAGE, FRACTION
- `StructureType` - UUID, URL, EMAIL, IP_ADDRESS, DATE, TIME, DATETIME, PATH, JSON_KEY, VARIABLE
- `ProfileMode` - TRAINING, INFERENCE, EVALUATION

**Classes:**
- `PreTokenizeHook` - Abstract base for pre-tokenization hooks
- `PostDecodeHook` - Abstract base for post-decode hooks
- `HookPipeline` - Chain of hooks with metadata tracking
- `HookedTokenizer` - Tokenizer wrapper with automatic hook application
- `ProfiledTokenizer` - Tokenizer with profile-based behavior switching
- `ProfileManager` - Manage multiple profiles with active switching
- `ByteFallbackWrapper` - Tokenizer wrapper with byte-level fallback

### `regression/` - Regression Testing

Token regression test framework for CI/CD pipelines.

```python
from chuk_lazarus.data.tokenizers.regression import (
    TokenTest,
    TokenTestSuite,
    TestAssertion,
    run_token_tests,
    create_test_suite,
    load_tests_from_yaml,
)

# Create test suite programmatically
suite = create_test_suite(
    name="Math Tokenization",
    tests=[
        {"name": "formula", "text": "σ_LT = √(σ² × L)", "assertion": "max_tokens", "expected": 8},
        {"name": "tool_call", "text": "<TOOL_CALL> solve", "assertion": "contains_token", "expected": 50001},
        {"name": "roundtrip", "text": "Hello world", "assertion": "roundtrip_lossless"},
    ],
)

# Run tests
result = run_token_tests(suite, tokenizer)
print(f"Pass rate: {result.pass_rate:.2%}")
for failure in result.failures:
    print(f"FAIL: {failure.test_name} - {failure.message}")

# Load from YAML (for CI/CD)
suite = load_tests_from_yaml("tests/tokenizer_regression.yaml")
result = run_token_tests(suite, tokenizer)
```

**YAML format:**
```yaml
name: My Tokenizer Tests
tests:
  - name: math_symbols
    text: "σ_LT = √(σ² × L)"
    assertion: max_tokens
    expected: 8
  - name: roundtrip
    text: "Hello world"
    assertion: roundtrip_lossless
  - name: tool_token
    text: "<TOOL_CALL>"
    assertion: exact_tokens
    expected: 1
```

**Models:**
- `TokenTest` - Single test case with assertion
- `TokenTestResult` - Pass/fail with details
- `TokenTestSuite` - Collection of tests
- `TestAssertion` - MAX_TOKENS, EXACT_TOKENS, ROUNDTRIP_LOSSLESS, etc.

## Architecture Alignment

This module implements the CHUK tokenization roadmap:

| Roadmap Phase | Module/Feature |
|--------------|----------------|
| Phase 0: Baseline | `token_stats`, `validation`, `batch_processing` |
| Phase 0: Introspection | `analyze/coverage`, `analyze/entropy` |
| Phase 0: Regression | `regression/tests` |
| Phase 1: Control Tokens | `runtime/special_registry` |
| Phase 1: Token↔Tool Mapping | `runtime/semantics` |
| Phase 1: Robustness | `preprocessing/fallback` (byte fallback) |
| Phase 2: Domain Injection | `runtime/dynamic_vocab` |
| Phase 2: Profiles | `preprocessing/profiles` (training vs inference) |
| Phase 3: Curriculum | `curriculum/length_buckets`, `curriculum/reasoning_density` |
| Phase 3: Structure-Aware | `preprocessing/numeric`, `preprocessing/structure`, `preprocessing/hooks` |
| Phase 4: Soft Extension | `runtime/dynamic_vocab` |
| Phase 6: Observability | `training/throughput`, `analyze/entropy` |

## Control Plane Token Guidelines

Reserve token ID ranges for different purposes:

```
[0-49999]      → language tokens
[50000-50999]  → tool/action tokens
[51000-51999]  → memory/paging tokens
[52000-52999]  → solver/operator tokens
[53000+]       → experimental
```

Use `runtime.SpecialTokenRegistry` to enforce these boundaries and detect collisions.
