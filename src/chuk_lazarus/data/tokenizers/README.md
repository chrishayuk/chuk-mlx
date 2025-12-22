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

- **analyze/** - Coverage, entropy, fit scoring, efficiency metrics, vocab induction
- **curriculum/** - Token-length buckets, reasoning density scoring
- **preprocessing/** - Pre-tokenization transforms, profiles, byte fallback
- **runtime/** - Special token registry, dynamic vocab, semantics mapping, **chat templates**
- **training/** - Sequence packing, throughput profiling
- **regression/** - Token regression test framework
- **research/** - Soft tokens, token morphing, embedding analysis
- **instrumentation/** - Token histograms, OOV analysis, waste metrics, vocab comparison
- **backends/** - Pluggable tokenizer backends (HuggingFace + fast MLX CharTrie) + benchmarking
- **fingerprint.py** - Tokenizer fingerprinting for compatibility verification

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

HuggingFace tokenizers, SentencePiece, tiktoken/OpenAI, and custom implementations all work seamlessly.

## OpenAI Tokenizers (tiktoken)

Support for OpenAI's tokenizers via tiktoken. Install with:

```bash
pip install 'chuk-lazarus[openai]'
# or
uv add 'chuk-lazarus[openai]'
```

### Usage

```python
from chuk_lazarus.data.tokenizers.tiktoken_wrapper import TiktokenWrapper

# Load by model name
tokenizer = TiktokenWrapper.from_model("gpt-4")
tokenizer = TiktokenWrapper.from_model("gpt-4o")
tokenizer = TiktokenWrapper.from_model("gpt-3.5-turbo")
tokenizer = TiktokenWrapper.from_model("o1")

# Load by encoding name
tokenizer = TiktokenWrapper.from_encoding("cl100k_base")  # GPT-4, GPT-3.5
tokenizer = TiktokenWrapper.from_encoding("o200k_base")   # GPT-4o, O1

# Use like any other tokenizer
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)
vocab = tokenizer.get_vocab()

# Works with all analysis tools
from chuk_lazarus.data.tokenizers.analyze import analyze_coverage
report = analyze_coverage(texts, tokenizer)
```

### Auto-Detection via load_tokenizer

The `load_tokenizer` utility automatically detects OpenAI models:

```python
from chuk_lazarus.utils.tokenizer_loader import load_tokenizer

# Automatically uses TiktokenWrapper for OpenAI models
tokenizer = load_tokenizer("gpt-4")
tokenizer = load_tokenizer("gpt-4o")
tokenizer = load_tokenizer("cl100k_base")

# Still works for HuggingFace models
tokenizer = load_tokenizer("gpt2")  # Note: "gpt2" is also a tiktoken encoding
tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

### Supported Models

| Model | Encoding | Vocab Size |
|-------|----------|------------|
| gpt-4, gpt-4-turbo, gpt-3.5-turbo | cl100k_base | 100,277 |
| gpt-4o, gpt-4o-mini, o1, o1-mini, o3-mini | o200k_base | 200,019 |
| text-davinci-003, code-davinci-002 | p50k_base | 50,281 |
| davinci, curie, babbage, ada | r50k_base | 50,257 |

### CLI Usage

```bash
# Encode text with GPT-4's tokenizer
uvx "chuk-lazarus[openai]" tokenizer encode -t "gpt-4" --text "Hello, world!"

# Compare GPT-4 vs GPT-4o tokenization
uvx "chuk-lazarus[openai]" tokenizer compare -t1 "gpt-4" -t2 "gpt-4o" --text "Machine learning"

# Health check
uvx "chuk-lazarus[openai]" tokenizer doctor -t "gpt-3.5-turbo"

# Fingerprint for compatibility tracking
uvx "chuk-lazarus[openai]" tokenizer fingerprint -t "gpt-4" --save gpt4-fingerprint.json
```

## Testing

Run tests with:

```bash
pytest tests/data/tokenizers/ -v --cov=src/chuk_lazarus/data/tokenizers
```

Target: 90%+ coverage per file.

## Training Framework Submodules

### `analyze/` - Token Analysis

Coverage, entropy, fit scoring, efficiency metrics, and retokenization comparison.

```python
from chuk_lazarus.data.tokenizers.analyze import (
    analyze_coverage,
    analyze_entropy,
    analyze_efficiency,
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

# Efficiency analysis (tokens per sample, reasoning steps, equations, tool calls)
efficiency = analyze_efficiency(texts, tokenizer)
print(f"Efficiency score: {efficiency.efficiency_score:.1f}/100")
print(f"Mean tokens/sample: {efficiency.sample_stats.mean:.1f}")
print(f"P95 tokens/sample: {efficiency.sample_stats.p95:.0f}")
print(f"Fragmentation: {efficiency.fragmentation.fragmentation_score:.1%}")
if efficiency.reasoning_steps:
    print(f"Tokens per reasoning step: {efficiency.reasoning_steps.mean_tokens:.1f}")
if efficiency.equations:
    print(f"Tokens per equation: {efficiency.equations.mean_tokens:.1f}")
if efficiency.tool_calls:
    print(f"Tokens per tool call: {efficiency.tool_calls.mean_tokens:.1f}")
```

```python
# Vocabulary induction - find high-impact tokens to add
from chuk_lazarus.data.tokenizers.analyze import (
    analyze_vocab_induction,
    find_fragmented_words,
    suggest_domain_tokens,
    TokenDomain,
)

# Analyze corpus for vocabulary suggestions
report = analyze_vocab_induction(texts, tokenizer)
print(f"Total potential savings: {report.total_potential_savings:,} tokens")
print(f"Savings percent: {report.savings_percent:.1f}%")
for candidate in report.candidates[:5]:
    print(f"  {candidate.token_str}: {candidate.total_savings} tokens saved")
print(f"Recommendations: {report.recommendations}")

# Find fragmented words (strings that split into many tokens)
fragmented = find_fragmented_words(texts, tokenizer)
for word in fragmented[:5]:
    print(f"  {word.token_str}: {word.current_tokens} → 1 token ({word.savings_per_occurrence} saved)")

# Get domain-specific token suggestions
domain_tokens = suggest_domain_tokens(texts, tokenizer, [TokenDomain.MATH, TokenDomain.CODE])
for token in domain_tokens:
    print(f"  {token.domain.value}: {token.token_str}")
```

**Models:**
- `CoverageReport` - UNK rate, tokens/word, vocab utilization, warnings
- `EntropyReport` - Shannon entropy, normalized entropy, perplexity
- `FitScore` - Coverage, compression, entropy scores with recommendation
- `RetokenizationDiff` - Token boundary shifts and content changes
- `EfficiencyReport` - Tokens per sample/step/equation/tool, fragmentation score
- `SampleStats` - Mean, median, percentiles for token counts
- `ContentTypeStats` - Tokens per content type (reasoning, equations, tools)
- `FragmentationStats` - Fragmentation analysis with worst words
- `InductionReport` - Vocabulary induction analysis with candidates and recommendations
- `TokenCandidate` - Token suggestion with frequency, savings, domain, priority
- `DomainVocab` - Pre-defined domain vocabulary (MATH, CODE, TOOL)

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

Special token registry, dynamic vocabulary, semantics mapping, and **chat template management**.

```python
from chuk_lazarus.data.tokenizers.runtime import (
    # Special tokens
    SpecialTokenRegistry,
    TokenCategory,
    create_standard_registry,
    extend_vocab_runtime,
    map_token_to_semantic,
    SemanticDomain,
    # Chat templates
    ChatTemplateRegistry,
    TemplateFormat,
    validate_chat_template,
    patch_chat_template,
    suggest_template_for_model,
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

# ===== Chat Template Management =====

# Registry of known templates
registry = ChatTemplateRegistry()
templates = registry.list_templates()
for t in templates:
    print(f"{t.format.value}: {t.description}")
# Output: chatml, llama, phi, gemma, zephyr, vicuna, alpaca

# Detect format from model name
template = registry.get_for_model_family("Qwen/Qwen2-0.5B")
print(f"Qwen uses: {template.format.value}")  # chatml

# Validate a tokenizer's chat template
result = validate_chat_template(tokenizer)
print(f"Valid: {result.is_valid}")
print(f"Format: {result.format.value}")
print(f"Capabilities: {[c.value for c in result.capabilities]}")
for issue in result.issues:
    print(f"  [{issue.severity}] {issue.message}")

# Patch a missing template
patch_chat_template(tokenizer, "chatml")  # or llama, phi, gemma, etc.

# Auto-detect best template from model name
suggested = suggest_template_for_model("meta-llama/Llama-2-7b")
if suggested:
    patch_chat_template(tokenizer, suggested.format)
    tokenizer.save_pretrained("./patched")
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
- `ChatTemplateRegistry` - Registry of known chat template formats
- `TemplateFormat` - CHATML, LLAMA, PHI, GEMMA, ZEPHYR, VICUNA, ALPACA
- `TemplateDefinition` - Template with format, Jinja2 string, capabilities
- `TemplateValidationResult` - Validation result with issues and capabilities
- `TemplateCapability` - SYSTEM_MESSAGE, MULTI_TURN, TOOL_CALLS, GENERATION_PROMPT

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

### `research/` - Experimental Tokenization

Research playground for soft tokens, embedding manipulation, and embedding analysis.

```python
from chuk_lazarus.data.tokenizers.research import (
    # Soft tokens (learnable embeddings)
    SoftTokenBank,
    SoftTokenConfig,
    InitializationMethod,
    create_soft_token,
    create_prompt_tuning_bank,
    create_control_token,
    interpolate_embeddings,
    # Token morphing
    MorphConfig,
    MorphMethod,
    morph_token,
    blend_tokens,
    BlendMode,
    # Embedding analysis
    find_nearest_neighbors,
    cluster_tokens,
    project_embeddings,
    find_analogies,
    analyze_embeddings,
)

# Create soft prompt tokens for prompt tuning
bank = create_prompt_tuning_bank(
    num_tokens=10,
    embedding_dim=768,
    prefix="task",
)
print(f"Created {len(bank.tokens)} soft tokens")
embeddings_matrix = bank.get_embeddings_matrix()  # (10, 768)

# Create control tokens for style transfer
positive_ctrl = create_control_token("positive_sentiment", embedding_dim=768)
negative_ctrl = create_control_token("negative_sentiment", embedding_dim=768)

# Interpolate between embeddings
import numpy as np
e1 = np.random.randn(768)
e2 = np.random.randn(768)
midpoint = interpolate_embeddings(e1, e2, alpha=0.5, method="spherical")

# Morph between token embeddings (for visualization)
config = MorphConfig(method=MorphMethod.SPHERICAL, num_steps=20)
morph_result = morph_token(e1, e2, "start", "end", config)
print(f"Path length: {morph_result.num_steps} steps")
trajectory = morph_result.get_embeddings_array()  # (20, 768)

# Blend multiple tokens
blended = blend_tokens(
    [e1, e2, np.random.randn(768)],
    ["token1", "token2", "token3"],
    weights=[0.5, 0.3, 0.2],
    mode=BlendMode.WEIGHTED,
)

# Analyze embedding space
embeddings = np.random.randn(1000, 768)
token_ids = list(range(1000))
token_strs = [f"token_{i}" for i in range(1000)]

# Find nearest neighbors
neighbors = find_nearest_neighbors(
    embeddings[0], embeddings, token_ids, token_strs, k=10
)
for n in neighbors[:3]:
    print(f"  {n.token_str}: similarity={n.similarity:.3f}")

# Cluster tokens
clusters = cluster_tokens(embeddings, token_ids, token_strs, num_clusters=10)
for c in clusters[:3]:
    print(f"  Cluster {c.cluster_id}: {c.size} tokens")

# Project to 2D for visualization
projection = project_embeddings(embeddings, token_ids, token_strs, dim=2)
coords = projection.get_coordinates_array()  # (1000, 2)

# Find analogies (a:b :: c:?)
# king - man + woman = queen
analogy_results = find_analogies(embeddings, token_ids, token_strs, a_idx=0, b_idx=1, c_idx=2)

# Comprehensive embedding analysis
analysis = analyze_embeddings(embeddings)
print(f"Isotropy: {analysis.isotropy_score:.2f}")
print(f"Mean similarity: {analysis.mean_pairwise_similarity:.3f}")
```

**Models:**
- `SoftToken` - Soft token metadata (name, ID, config)
- `SoftTokenEmbedding` - Soft token with embedding vector
- `SoftTokenBank` - Collection of soft tokens with management
- `SoftTokenConfig` - Configuration for soft token creation
- `MorphResult` - Token morphing trajectory with embeddings
- `MorphSequence` - Multi-token morphing sequence
- `TokenBlend` - Blended token result
- `NeighborInfo` - Nearest neighbor with distance/similarity
- `ClusterInfo` - Cluster with centroid and members
- `ProjectionResult` - Dimensionality reduction result
- `EmbeddingAnalysis` - Comprehensive embedding space metrics

**Enums:**
- `InitializationMethod` - RANDOM_NORMAL, RANDOM_UNIFORM, FROM_TOKENS, ZEROS, ONES
- `MorphMethod` - LINEAR, SPHERICAL, BEZIER, CUBIC
- `BlendMode` - AVERAGE, WEIGHTED, GEOMETRIC, ATTENTION
- `DistanceMetric` - COSINE, EUCLIDEAN, DOT_PRODUCT
- `ProjectionMethod` - PCA, RANDOM, CENTERED

### `backends/` - Tokenizer Backends

Pluggable backend architecture with multiple implementations for different performance profiles.

```python
from chuk_lazarus.data.tokenizers.backends import (
    # Core types
    TokenizerBackend,
    BackendType,
    TokenizationResult,
    BatchTokenizationResult,
    BackendInfo,
    # Implementations
    HuggingFaceBackend,
    FastBackend,
    # Factory functions
    create_backend,
    get_best_backend,
    is_fast_backend_available,
)

# Create HuggingFace backend (default, portable)
from transformers import AutoTokenizer
hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
backend = HuggingFaceBackend(hf_tokenizer)

result = backend.encode("Hello, world!")
print(f"Token IDs: {result.token_ids}")
print(f"Tokens: {result.tokens}")

# Decode back
text = backend.decode(result.token_ids)

# Access special tokens
print(f"EOS token ID: {backend.eos_token_id}")
print(f"Vocab size: {backend.vocab_size}")

# Apply chat template (if available)
messages = [{"role": "user", "content": "Hello"}]
formatted = backend.apply_chat_template(messages, add_generation_prompt=True)

# Create fast backend (MLX Data CharTrie, optional)
# Install with: pip install chuk-lazarus[fast]
if is_fast_backend_available():
    # From HuggingFace tokenizer
    fast = FastBackend.from_tokenizer(hf_tokenizer)

    # From vocabulary dict
    vocab = {"hello": 0, "world": 1, "<s>": 2, "</s>": 3}
    fast = FastBackend(vocab, bos_token_id=2, eos_token_id=3)

    # From SentencePiece model
    fast = FastBackend.from_sentencepiece("model.spm")

    # From vocabulary file
    fast = FastBackend.from_vocab_file("vocab.txt")

    # Parallel batch encoding (avoids Python GIL)
    texts = ["Hello world", "How are you?", "Nice day!"]
    batch_result = fast.encode_batch(texts, num_workers=4)
    print(f"Total tokens: {batch_result.total_tokens}")
    for result in batch_result.results:
        print(f"  {result.token_ids}")

# Factory function - automatically select best backend
backend = get_best_backend(hf_tokenizer, prefer_fast=True)
print(f"Using backend: {backend.backend_type}")

# Get backend info
info = backend.get_info()
print(f"Supports parallel: {info.supports_parallel}")
print(f"Supports offsets: {info.supports_offsets}")

# ===== Benchmarking =====
from chuk_lazarus.data.tokenizers.backends import (
    benchmark_tokenizer,
    compare_backends,
    generate_benchmark_corpus,
)

# Generate test corpus
corpus = generate_benchmark_corpus(num_samples=1000, avg_length=100)

# Benchmark single tokenizer
result = benchmark_tokenizer(hf_tokenizer, corpus, num_workers=4)
print(f"Throughput: {result.tokens_per_second:,.0f} tok/s")
print(f"Samples/s: {result.samples_per_second:,.1f}")

# Compare HuggingFace vs MLX backends
comparison = compare_backends(hf_tokenizer, corpus, num_workers=4)
print(comparison.summary())
# Output:
# HuggingFace Backend:
#   Throughput: 50,000 tok/s
# Fast (MLX CharTrie) Backend:
#   Throughput: 200,000 tok/s
# Speedup: 4.00x
```

**Models:**
- `TokenizationResult` - Token IDs, token strings, character offsets
- `BatchTokenizationResult` - List of results with total token count
- `BackendInfo` - Backend capabilities (parallel, offsets, availability)
- `BenchmarkResult` - Throughput metrics (tokens/s, samples/s)
- `BackendComparison` - HF vs Fast comparison with speedup ratio

**Enums:**
- `BackendType` - HUGGINGFACE (default), FAST (MLX CharTrie)

**Classes:**
- `TokenizerBackend` - Protocol for tokenizer backends
- `BaseBackend` - Abstract base class with common functionality
- `HuggingFaceBackend` - HuggingFace-compatible backend (correctness + portability)
- `FastBackend` - MLX Data CharTrie backend (parallel, high-throughput)

### `fingerprint.py` - Tokenizer Fingerprinting

Tokenizer fingerprinting for compatibility verification. Generate stable hashes of tokenizer configuration to ensure datasets and models use compatible tokenizers.

```python
from chuk_lazarus.data.tokenizers.fingerprint import (
    # Core functions
    compute_fingerprint,
    verify_fingerprint,
    assert_fingerprint,
    # IO
    save_fingerprint,
    load_fingerprint,
    fingerprint_from_json,
    # Registry
    FingerprintRegistry,
    get_registry,
    # Models
    TokenizerFingerprint,
    FingerprintMismatch,
)

# Compute fingerprint for a tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
fingerprint = compute_fingerprint(tokenizer)

print(f"Fingerprint: {fingerprint.fingerprint}")  # Short 16-char hash
print(f"Full hash: {fingerprint.full_hash}")       # Full SHA-256
print(f"Vocab size: {fingerprint.vocab_size}")
print(f"Vocab hash: {fingerprint.vocab_hash}")
print(f"Special tokens: {fingerprint.special_tokens}")

# Save fingerprint to file
save_fingerprint(fingerprint, "tokenizer_fingerprint.json")

# Load fingerprint from file
loaded = load_fingerprint("tokenizer_fingerprint.json")

# Verify tokenizer matches expected fingerprint
mismatch = verify_fingerprint(tokenizer, fingerprint)
if mismatch is None:
    print("Tokenizer matches!")
else:
    print(f"Mismatch detected!")
    print(f"Compatible: {mismatch.is_compatible}")
    print(f"Warnings: {mismatch.warnings}")
    print(f"Diff: {mismatch.diff}")

# Assert tokenizer matches (raises ValueError if not)
assert_fingerprint(tokenizer, fingerprint, strict=False)

# Compare fingerprints
fp1 = compute_fingerprint(tokenizer1)
fp2 = compute_fingerprint(tokenizer2)

if fp1.matches(fp2):
    print("Tokenizers are identical")
elif fp1.matches_vocab(fp2):
    print("Vocabularies match (special tokens may differ)")

diff = fp1.diff(fp2)
# {'vocab_matches': True, 'special_tokens_match': False, ...}

# Use registry for known tokenizers
registry = get_registry()
registry.register(
    name="gpt2",
    fingerprint=fingerprint,
    aliases=["gpt2-small", "gpt-2"],
)

# Verify against registered fingerprint
mismatch = registry.verify(tokenizer, "gpt2")

# Try to identify unknown tokenizer
matches = registry.identify(tokenizer)
for name, fp in matches:
    print(f"Matches: {name}")

# List all registered fingerprints
print(registry.list_all())
```

**Use Cases:**
- **Dataset metadata**: Store fingerprint to ensure compatible tokenizer is used
- **Model checkpoints**: Record tokenizer fingerprint alongside weights
- **CI/CD**: Detect tokenizer changes that break compatibility
- **Migration**: Verify new tokenizer is compatible with old

**Models:**
- `TokenizerFingerprint` - Stable fingerprint with component hashes
- `FingerprintMismatch` - Mismatch details with compatibility assessment

**Classes:**
- `FingerprintRegistry` - Registry of known tokenizer fingerprints

### `instrumentation/` - Tokenizer Instrumentation

Pure observability tools for analyzing tokenization behavior without modifying the tokenizer.

```python
from chuk_lazarus.data.tokenizers.instrumentation import (
    # Token length histograms
    compute_length_histogram,
    format_histogram_ascii,
    get_length_stats,
    # OOV and rare token analysis
    analyze_oov,
    find_rare_tokens,
    get_frequency_bands,
    # Waste metrics (padding/truncation)
    analyze_waste,
    analyze_padding_waste,
    analyze_truncation_loss,
    # Vocabulary comparison
    compare_vocab_impact,
    estimate_retokenization_cost,
)

# Token length histogram with ASCII visualization
histogram = compute_length_histogram(texts, tokenizer, num_bins=20)
print(format_histogram_ascii(histogram))
# Output:
# ============================================================
# TOKEN LENGTH HISTOGRAM
# ============================================================
# Samples: 1000
# Total tokens: 25000
# Length range: 5 - 512
# Mean: 25.0 (std: 15.3)
# Percentiles: p10=10 p25=15 p50=22 p75=35 p90=50 p95=65 p99=100
#     5-   30 │████████████████████████████████████████  800 ( 80.0%)
#    30-   55 │██████████                               150 ( 15.0%)
#    55-   80 │██                                        40 (  4.0%)
#    80-  105 │                                          10 (  1.0%)

# Quick stats without full histogram
stats = get_length_stats(texts, tokenizer)
print(f"Mean: {stats['mean']:.1f}, P95: {stats['p95']}")

# OOV and rare token analysis
oov_report = analyze_oov(texts, tokenizer, vocab_size=50000)
print(f"UNK rate: {oov_report.unk_rate:.2%}")
print(f"Singleton rate: {oov_report.singleton_rate:.2%}")
print(f"Vocab utilization: {oov_report.vocab_utilization:.2%}")

# Find rare tokens
rare = find_rare_tokens(texts, tokenizer, max_frequency=5, top_k=20)
for token in rare[:5]:
    print(f"  {token.token_str}: {token.count}x ({token.band.value})")

# Token frequency bands
bands = get_frequency_bands(texts, tokenizer)
# {TokenFrequencyBand.SINGLETON: 500, TokenFrequencyBand.RARE: 200, ...}

# Padding and truncation waste analysis
waste = analyze_waste(texts, tokenizer, max_length=512)
print(f"Padding rate: {waste.padding.padding_rate:.1%}")
print(f"Efficiency: {waste.padding.efficiency:.1%}")
print(f"Truncation rate: {waste.truncation.truncation_rate:.1%}")
print(f"Content loss: {waste.truncation.content_loss_rate:.1%}")
print(f"Recommendations: {waste.recommendations}")

# Before/after vocabulary swap analysis
comparison = compare_vocab_impact(
    texts,
    old_tokenizer,
    new_tokenizer,
    tokenizer1_name="old",
    tokenizer2_name="new",
)
print(f"Token ratio: {comparison.token_count_ratio:.2f}x")
print(f"Training speedup: {comparison.training_speedup:.2f}x")
print(f"Samples improved: {comparison.samples_improved}")

# Retokenization cost estimate
cost = estimate_retokenization_cost(texts, old_tokenizer, new_tokenizer)
print(f"Vocab overlap: {cost['vocab_overlap_rate']:.1%}")
print(f"Embedding reuse rate: {cost['embedding_reuse_rate']:.1%}")
```

**Models:**
- `LengthHistogram` - Complete histogram with bins, percentiles, recommendations
- `HistogramBin` - Single histogram bin with count and percentage
- `PercentileStats` - p10, p25, p50, p75, p90, p95, p99 percentiles
- `OOVReport` - UNK rate, singleton rate, vocab utilization, recommendations
- `RareTokenInfo` - Token ID, string, count, frequency band
- `PaddingStats` - Padding tokens, rate, efficiency, compute waste
- `TruncationStats` - Truncated samples, tokens lost, severity categories
- `WasteReport` - Combined padding + truncation analysis
- `VocabSwapReport` - Before/after comparison with training impact

**Enums:**
- `TokenFrequencyBand` - SINGLETON, RARE, UNCOMMON, COMMON, VERY_COMMON

## Architecture Alignment

This module implements the CHUK tokenization roadmap:

| Roadmap Phase | Module/Feature |
|--------------|----------------|
| Phase 0: Baseline | `token_stats`, `validation`, `batch_processing` |
| Phase 0: Introspection | `analyze/coverage`, `analyze/entropy` |
| Phase 0: Regression | `regression/tests` |
| Phase 1: Two Backends | `backends/huggingface`, `backends/fast` (MLX CharTrie) |
| Phase 1: Control Tokens | `runtime/special_registry` |
| Phase 1: Token↔Tool Mapping | `runtime/semantics` |
| Phase 1: Robustness | `preprocessing/fallback` (byte fallback) |
| Phase 2: Domain Injection | `runtime/dynamic_vocab` |
| Phase 2: Profiles | `preprocessing/profiles` (training vs inference) |
| Phase 2: Packing-First | `backends/` (PackedTokens output ready) |
| Phase 3: Curriculum | `curriculum/length_buckets`, `curriculum/reasoning_density` |
| Phase 3: Structure-Aware | `preprocessing/numeric`, `preprocessing/structure`, `preprocessing/hooks` |
| Phase 3: Fingerprinting | `fingerprint.py` (vocab hash, compatibility verification) |
| Phase 4: Soft Extension | `runtime/dynamic_vocab` |
| Phase 4: Efficiency Metrics | `analyze/efficiency` (tokens per sample/step/equation/tool) |
| Phase 5: Vocab Induction | `analyze/vocab_induction` (fragmented words, domain tokens) |
| Phase 6: Research Playground | `research/soft_tokens`, `research/token_morphing`, `research/embedding_analysis` |
| Phase 6: Observability | `training/throughput`, `analyze/entropy` |
| Instrumentation | `instrumentation/histograms`, `instrumentation/oov_report`, `instrumentation/waste`, `instrumentation/vocab_diff` |

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
