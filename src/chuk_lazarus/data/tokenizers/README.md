# Tokenizers Module

A comprehensive, Pydantic-native tokenization toolkit for building, analyzing, and managing tokenizers.

**Full documentation**: [docs/tokenizers.md](../../../../docs/tokenizers.md)

## Quick Start

```python
from chuk_lazarus.data.tokenizers import CustomTokenizer
from chuk_lazarus.data.tokenizers.analyze import analyze_coverage, calculate_fit_score
from chuk_lazarus.data.tokenizers.fingerprint import compute_fingerprint

# Load any HuggingFace tokenizer
from chuk_lazarus.utils.tokenizer_loader import load_tokenizer
tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Analyze coverage on your corpus
texts = ["Your training data...", "More examples..."]
coverage = analyze_coverage(texts, tokenizer)
print(f"UNK rate: {coverage.unk_rate:.2%}")

# Calculate fit score
fit = calculate_fit_score(texts, tokenizer)
print(f"Fit score: {fit.score}/100 ({fit.grade})")

# Generate fingerprint for compatibility checks
fp = compute_fingerprint(tokenizer)
print(f"Fingerprint: {fp.fingerprint}")
```

## Submodules

| Submodule | Purpose |
|-----------|---------|
| `analyze/` | Coverage, entropy, fit scoring, efficiency metrics, vocab induction |
| `curriculum/` | Token-length buckets, reasoning density scoring |
| `preprocessing/` | Pre-tokenization transforms, profiles, byte fallback |
| `runtime/` | Special token registry, dynamic vocab, chat templates |
| `training/` | Sequence packing, throughput profiling |
| `regression/` | Token regression test framework |
| `research/` | Soft tokens, token morphing, embedding analysis |
| `instrumentation/` | Token histograms, OOV analysis, waste metrics |
| `backends/` | Pluggable backends (HuggingFace + MLX CharTrie) |
| `fingerprint.py` | Tokenizer fingerprinting for compatibility |

## Design Principles

- **Pydantic-native**: All data structures use Pydantic BaseModel
- **No magic strings**: Enums and constants for type safety
- **Protocol-based**: TokenizerProtocol allows any compatible tokenizer
- **Training-aware**: Tokenizers as a control surface, not just preprocessing

## Testing

```bash
pytest tests/data/tokenizers/ -v
```
