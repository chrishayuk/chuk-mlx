# Tokenizers Module

A comprehensive, Pydantic-native tokenization toolkit for building, analyzing, and managing tokenizers. **Designed as a control surface for training frameworks** - tokenizers are inspectable, mutable, schedulable, and debuggable.

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

### Submodules

| Submodule | Purpose |
|-----------|---------|
| `analyze/` | Coverage, entropy, fit scoring, efficiency metrics, vocab induction |
| `curriculum/` | Token-length buckets, reasoning density scoring |
| `preprocessing/` | Pre-tokenization transforms, profiles, byte fallback |
| `runtime/` | Special token registry, dynamic vocab, semantics mapping, chat templates |
| `training/` | Sequence packing, throughput profiling |
| `regression/` | Token regression test framework |
| `research/` | Soft tokens, token morphing, embedding analysis |
| `instrumentation/` | Token histograms, OOV analysis, waste metrics, vocab comparison |
| `backends/` | Pluggable backends (HuggingFace + fast MLX CharTrie) + benchmarking |
| `fingerprint.py` | Tokenizer fingerprinting for compatibility verification |

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

## Core Modules

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

### `batch_processing.py`

Efficient batch encoding and padding.

```python
from chuk_lazarus.data.tokenizers.batch_processing import (
    create_batch,
    pad_batch,
    chunk_text,
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
)

# Get config from tokenizer
config = SpecialTokenConfig.from_tokenizer(tokenizer)
all_special = config.all_special_ids()

# Manipulate sequences
tokens = add_bos_token(tokens, tokenizer.bos_token_id)
tokens = strip_special_tokens(tokens, tokenizer)
```

### `vocab_manager.py`

Vocabulary manipulation utilities.

```python
from chuk_lazarus.data.tokenizers.vocab_manager import (
    merge_vocabularies,
    filter_vocabulary,
    validate_vocabulary,
    get_vocabulary_diff,
    ConflictResolution,
)

# Merge two vocabularies
merged = merge_vocabularies(
    vocab1, vocab2,
    conflict_resolution=ConflictResolution.FIRST
)

# Validate integrity
issues = validate_vocabulary(vocab)
if issues.has_issues():
    print(f"Found {len(issues.duplicate_ids)} duplicate IDs")
```

### `token_debug.py`

Debugging and visualization tools.

```python
from chuk_lazarus.data.tokenizers.token_debug import (
    get_token_info,
    compare_tokenizations,
    highlight_tokens,
    format_token_table,
)

# Inspect a token
info = get_token_info(token_id, tokenizer)
print(f"Token: {info.token_str}, Bytes: {info.byte_repr}")

# Compare tokenizers
comparison = compare_tokenizations(text, tokenizer1, tokenizer2)
print(f"Tokenizer 1: {comparison.tokenizer1_count} tokens")

# Visualize token boundaries
highlighted = highlight_tokens("Hello world", tokenizer, separator="|")
```

### `validation.py`

Tokenizer validation and testing.

```python
from chuk_lazarus.data.tokenizers.validation import (
    check_roundtrip,
    validate_special_tokens,
    create_validation_report,
    assert_valid_tokenizer,
)

# Test roundtrip encoding
result = check_roundtrip("Hello world", tokenizer)
if not result.is_lossless:
    print(f"Lost characters: {result.diff_chars}")

# Full validation
report = create_validation_report(tokenizer, "MyTokenizer")
print(f"Valid: {report.is_valid}")
```

### `conversion.py`

Format conversion utilities.

```python
from chuk_lazarus.data.tokenizers.conversion import (
    export_vocabulary,
    import_vocabulary,
    save_huggingface_format,
    ExportFormat,
)

# Export vocabulary
export = export_vocabulary(vocab, format=ExportFormat.JSON)

# Save as HuggingFace format
result = save_huggingface_format(vocab, "./hf_tokenizer")
```

## Training Framework Submodules

### `analyze/` - Token Analysis

Coverage, entropy, fit scoring, efficiency metrics, and retokenization comparison.

```python
from chuk_lazarus.data.tokenizers.analyze import (
    analyze_coverage,
    analyze_entropy,
    analyze_efficiency,
    calculate_fit_score,
    analyze_vocab_induction,
)

# Coverage analysis
coverage = analyze_coverage(texts, tokenizer)
print(f"UNK rate: {coverage.unk_rate:.2%}")
print(f"Tokens/word: {coverage.tokens_per_word:.2f}")

# Entropy analysis
entropy = analyze_entropy(texts, tokenizer)
print(f"Entropy: {entropy.entropy:.2f} bits")

# Tokenizer-dataset fit score
fit = calculate_fit_score(texts, tokenizer)
print(f"Fit score: {fit.overall_score:.2f}")

# Vocabulary induction - find high-impact tokens to add
report = analyze_vocab_induction(texts, tokenizer)
print(f"Potential savings: {report.total_potential_savings:,} tokens")
```

### `curriculum/` - Curriculum Learning

Token-length buckets and reasoning density scoring.

```python
from chuk_lazarus.data.tokenizers.curriculum import (
    create_length_buckets,
    get_curriculum_schedule,
    score_reasoning_density,
)

# Create length-based curriculum buckets
buckets = create_length_buckets(texts, tokenizer, config)
for bucket in buckets:
    print(f"Bucket {bucket.bucket_id}: {bucket.sample_count} samples")

# Get curriculum schedule (easy -> hard)
schedule = get_curriculum_schedule(texts, tokenizer)

# Score reasoning density
score = score_reasoning_density(text, idx, tokenizer)
print(f"Reasoning density: {score.overall_score:.2f}")
```

### `runtime/` - Runtime Utilities

Special token registry, dynamic vocabulary, semantics mapping, and chat templates.

```python
from chuk_lazarus.data.tokenizers.runtime import (
    SpecialTokenRegistry,
    TokenCategory,
    create_standard_registry,
    ChatTemplateRegistry,
    validate_chat_template,
    patch_chat_template,
)

# Special token registry with collision detection
registry = create_standard_registry(vocab_size=50000)
registry.register(
    token_str="<TOOL_CALL>",
    token_id=50001,
    category=TokenCategory.TOOL_CALL,
)

# Chat template management
result = validate_chat_template(tokenizer)
print(f"Format: {result.format.value}, Valid: {result.is_valid}")

# Patch a missing template
patch_chat_template(tokenizer, "chatml")  # or llama, phi, gemma, etc.
```

### `training/` - Training Utilities

Sequence packing and throughput profiling.

```python
from chuk_lazarus.data.tokenizers.training import (
    create_packed_batch,
    calculate_packing_efficiency,
    profile_tokenization,
)

# Smart sequence packing (20-40% throughput improvement)
packed = create_packed_batch(texts, tokenizer, max_seq_length=512)
print(f"Packing ratio: {packed.packing_ratio:.2f}x")

# Throughput profiling
metrics = profile_tokenization(texts, tokenizer)
print(f"Tokens/second: {metrics.tokens_per_second:.0f}")
```

### `preprocessing/` - Pre-Tokenization Transforms

Pre-tokenization hooks, profiles, and byte fallback.

```python
from chuk_lazarus.data.tokenizers.preprocessing import (
    normalize_numbers,
    inject_structure_tokens,
    HookPipeline,
    HookedTokenizer,
    create_training_profile,
    wrap_with_fallback,
)

# Numeric normalization
text = "Pi is 3.14159 and e is 2.71828"
encoding = normalize_numbers(text)
print(encoding.encoded_text)  # "Pi is <NUM_0> and e is <NUM_1>"

# Hook pipeline
pipeline = create_standard_pipeline(numeric=True, structure=True)
hooked = HookedTokenizer(tokenizer, pipeline)
tokens = hooked.encode(text)  # Transforms applied automatically

# Byte fallback for robustness
wrapper = wrap_with_fallback(tokenizer)
tokens = wrapper.encode("Emoji: 123")  # No UNK tokens
```

### `regression/` - Regression Testing

Token regression test framework for CI/CD pipelines.

```python
from chuk_lazarus.data.tokenizers.regression import (
    TokenTestSuite,
    run_token_tests,
    load_tests_from_yaml,
)

# Load from YAML (for CI/CD)
suite = load_tests_from_yaml("tests/tokenizer_regression.yaml")
result = run_token_tests(suite, tokenizer)
print(f"Pass rate: {result.pass_rate:.2%}")
```

**YAML format:**
```yaml
name: My Tokenizer Tests
tests:
  - name: math_symbols
    text: "x^2 + y^2 = z^2"
    assertion: max_tokens
    expected: 8
  - name: roundtrip
    text: "Hello world"
    assertion: roundtrip_lossless
```

### `research/` - Experimental Tokenization

Research playground for soft tokens, embedding manipulation, and analysis.

```python
from chuk_lazarus.data.tokenizers.research import (
    SoftTokenBank,
    create_prompt_tuning_bank,
    morph_token,
    find_nearest_neighbors,
    cluster_tokens,
    analyze_embeddings,
)

# Create soft prompt tokens for prompt tuning
bank = create_prompt_tuning_bank(
    num_tokens=10,
    embedding_dim=768,
    prefix="task",
)

# Analyze embedding space
neighbors = find_nearest_neighbors(
    embeddings[0], embeddings, token_ids, token_strs, k=10
)

# Comprehensive embedding analysis
analysis = analyze_embeddings(embeddings)
print(f"Isotropy: {analysis.isotropy_score:.2f}")
```

### `backends/` - Tokenizer Backends

Pluggable backend architecture with multiple implementations.

```python
from chuk_lazarus.data.tokenizers.backends import (
    HuggingFaceBackend,
    FastBackend,
    get_best_backend,
    is_fast_backend_available,
    benchmark_tokenizer,
    compare_backends,
)

# Create backends
backend = HuggingFaceBackend(hf_tokenizer)
result = backend.encode("Hello, world!")

# Fast backend (MLX CharTrie, optional)
if is_fast_backend_available():
    fast = FastBackend.from_tokenizer(hf_tokenizer)
    batch_result = fast.encode_batch(texts, num_workers=4)

# Benchmark and compare
comparison = compare_backends(hf_tokenizer, corpus, num_workers=4)
print(comparison.summary())  # Shows speedup ratio
```

### `fingerprint.py` - Tokenizer Fingerprinting

Generate stable hashes for compatibility verification.

```python
from chuk_lazarus.data.tokenizers.fingerprint import (
    compute_fingerprint,
    verify_fingerprint,
    save_fingerprint,
    load_fingerprint,
)

# Compute fingerprint
fingerprint = compute_fingerprint(tokenizer)
print(f"Fingerprint: {fingerprint.fingerprint}")

# Save/load
save_fingerprint(fingerprint, "tokenizer_fingerprint.json")
loaded = load_fingerprint("tokenizer_fingerprint.json")

# Verify tokenizer matches
mismatch = verify_fingerprint(tokenizer, fingerprint)
if mismatch is None:
    print("Tokenizer matches!")
```

### `instrumentation/` - Tokenizer Instrumentation

Pure observability tools for analyzing tokenization behavior.

```python
from chuk_lazarus.data.tokenizers.instrumentation import (
    compute_length_histogram,
    format_histogram_ascii,
    analyze_oov,
    analyze_waste,
    compare_vocab_impact,
)

# Token length histogram
histogram = compute_length_histogram(texts, tokenizer, num_bins=20)
print(format_histogram_ascii(histogram))

# OOV analysis
oov_report = analyze_oov(texts, tokenizer)
print(f"UNK rate: {oov_report.unk_rate:.2%}")

# Waste analysis
waste = analyze_waste(texts, tokenizer, max_length=512)
print(f"Padding rate: {waste.padding.padding_rate:.1%}")
print(f"Truncation rate: {waste.truncation.truncation_rate:.1%}")
```

## OpenAI Tokenizers (tiktoken)

Support for OpenAI's tokenizers via tiktoken.

```bash
pip install 'chuk-lazarus[openai]'
```

```python
from chuk_lazarus.data.tokenizers.tiktoken_wrapper import TiktokenWrapper

# Load by model name
tokenizer = TiktokenWrapper.from_model("gpt-4")
tokenizer = TiktokenWrapper.from_model("gpt-4o")

# Use like any other tokenizer
tokens = tokenizer.encode("Hello, world!")
```

| Model | Encoding | Vocab Size |
|-------|----------|------------|
| gpt-4, gpt-4-turbo, gpt-3.5-turbo | cl100k_base | 100,277 |
| gpt-4o, gpt-4o-mini, o1, o1-mini | o200k_base | 200,019 |

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

## Control Plane Token Guidelines

Reserve token ID ranges for different purposes:

```
[0-49999]      -> language tokens
[50000-50999]  -> tool/action tokens
[51000-51999]  -> memory/paging tokens
[52000-52999]  -> solver/operator tokens
[53000+]       -> experimental
```

Use `runtime.SpecialTokenRegistry` to enforce these boundaries and detect collisions.

## Testing

```bash
pytest tests/data/tokenizers/ -v --cov=src/chuk_lazarus/data/tokenizers
```
