# Lazarus

MLX-based LLM training and tokenizer toolkit for Apple Silicon.

## Quick Start with uvx

No installation needed - run directly with `uvx`:

```bash
# Encode text to see how a tokenizer splits it
uvx chuk-lazarus tokenizer encode -t "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --text "Hello, world!"

# Run a health check on any tokenizer
uvx chuk-lazarus tokenizer doctor -t "gpt2"

# Compare how two tokenizers handle the same text
uvx chuk-lazarus tokenizer compare -t1 "gpt2" -t2 "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --text "Machine learning is amazing"
```

## Installation

```bash
# Install with uv (recommended)
uv add chuk-lazarus

# Or with pip
pip install chuk-lazarus

# For faster tokenization (optional MLX backend)
uv add "chuk-lazarus[fast]"
```

After installation, use the `lazarus` command directly:

```bash
lazarus tokenizer encode -t gpt2 --text "Hello"
```

## Tokenizer CLI

The tokenizer CLI is a comprehensive toolkit for inspecting, analyzing, and debugging tokenizers. All commands work with any HuggingFace tokenizer.

### Basic Commands

```bash
# Encode text - see token IDs and boundaries
uvx chuk-lazarus tokenizer encode -t "gpt2" --text "The quick brown fox"

# Decode token IDs back to text
uvx chuk-lazarus tokenizer decode -t "gpt2" --ids "464,2068,7586,21831"

# Search the vocabulary
uvx chuk-lazarus tokenizer vocab -t "gpt2" --search "hello"

# Show vocabulary statistics
uvx chuk-lazarus tokenizer vocab -t "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### Health Check & Fingerprinting

```bash
# Run comprehensive tokenizer health check
uvx chuk-lazarus tokenizer doctor -t "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Generate a fingerprint (for compatibility verification)
uvx chuk-lazarus tokenizer fingerprint -t "gpt2"

# Save fingerprint for CI/CD verification
uvx chuk-lazarus tokenizer fingerprint -t "gpt2" --save gpt2-fingerprint.json

# Verify tokenizer matches expected fingerprint
uvx chuk-lazarus tokenizer fingerprint -t "gpt2" --verify gpt2-fingerprint.json
```

### Corpus Analysis

Analyze how well a tokenizer fits your dataset:

```bash
# Coverage analysis - UNK rate, tokens per word, vocab utilization
uvx chuk-lazarus tokenizer analyze coverage -t "gpt2" --file corpus.txt

# Entropy analysis - token distribution uniformity
uvx chuk-lazarus tokenizer analyze entropy -t "gpt2" --file corpus.txt

# Fit score - overall tokenizer-dataset compatibility (0-100)
uvx chuk-lazarus tokenizer analyze fit-score -t "gpt2" --file corpus.txt

# Efficiency analysis - tokens per sample, fragmentation
uvx chuk-lazarus tokenizer analyze efficiency -t "gpt2" --file corpus.txt

# Vocabulary suggestions - find tokens to add for better compression
uvx chuk-lazarus tokenizer analyze vocab-suggest -t "gpt2" --file corpus.txt

# Compare two tokenizers on your corpus
uvx chuk-lazarus tokenizer analyze diff -t1 "gpt2" -t2 "meta-llama/Llama-2-7b" -f corpus.txt
```

### Instrumentation

Observability tools for understanding tokenization behavior:

```bash
# Token length histogram with ASCII visualization
uvx chuk-lazarus tokenizer instrument histogram -t "gpt2" --file corpus.txt

# OOV and rare token analysis
uvx chuk-lazarus tokenizer instrument oov -t "gpt2" --file corpus.txt --show-rare

# Padding and truncation waste analysis
uvx chuk-lazarus tokenizer instrument waste -t "gpt2" --file corpus.txt --max-length 512

# Compare vocabulary impact (before/after tokenizer swap)
uvx chuk-lazarus tokenizer instrument vocab-diff -t1 "gpt2" -t2 "meta-llama/Llama-2-7b" --file corpus.txt
```

### Training Utilities

Tools for efficient training data preparation:

```bash
# Profile tokenization throughput
uvx chuk-lazarus tokenizer training throughput -t "gpt2" --file corpus.txt

# Pack sequences for efficient training (20-40% speedup)
uvx chuk-lazarus tokenizer training pack -t "gpt2" --file corpus.txt --max-length 512 -o packed.jsonl

# Create curriculum learning buckets by token length
uvx chuk-lazarus tokenizer curriculum length-buckets -t "gpt2" --file corpus.txt

# Score texts by reasoning density for curriculum ordering
uvx chuk-lazarus tokenizer curriculum reasoning-density -t "gpt2" --file corpus.txt
```

### Regression Testing

Ensure tokenization doesn't change unexpectedly:

```bash
# Run regression tests from YAML file
uvx chuk-lazarus tokenizer regression run -t "gpt2" --tests tokenizer_tests.yaml
```

Example `tokenizer_tests.yaml`:
```yaml
name: My Tokenizer Tests
tests:
  - name: basic_text
    text: "Hello, world!"
    assertion: exact_tokens
    expected: 4
  - name: roundtrip
    text: "The quick brown fox"
    assertion: roundtrip_lossless
  - name: math_symbols
    text: "x^2 + y^2 = z^2"
    assertion: max_tokens
    expected: 10
```

## Python API

```python
from chuk_lazarus.utils.tokenizer_loader import load_tokenizer
from chuk_lazarus.data.tokenizers.analyze import (
    analyze_coverage,
    analyze_entropy,
    calculate_fit_score,
)
from chuk_lazarus.data.tokenizers.fingerprint import compute_fingerprint

# Load any HuggingFace tokenizer
tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Analyze coverage on your corpus
texts = ["Your training data...", "More examples..."]
coverage = analyze_coverage(texts, tokenizer)
print(f"UNK rate: {coverage.unk_rate:.2%}")
print(f"Tokens per word: {coverage.tokens_per_word:.2f}")

# Calculate fit score
fit = calculate_fit_score(texts, tokenizer)
print(f"Fit score: {fit.score}/100 ({fit.grade})")

# Generate fingerprint for compatibility checks
fp = compute_fingerprint(tokenizer)
print(f"Fingerprint: {fp.fingerprint}")
```

See the [Tokenizers README](src/chuk_lazarus/data/tokenizers/README.md) for comprehensive documentation of all analysis, preprocessing, and training utilities.

## Training CLI

```bash
# Train with SFT
uvx chuk-lazarus train sft --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --data train.jsonl --use-lora

# Train with DPO
uvx chuk-lazarus train dpo --model ./checkpoints/sft/final --data preferences.jsonl

# Generate synthetic training data
uvx chuk-lazarus generate --type math --output ./data/lazarus

# Run inference
uvx chuk-lazarus infer --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --prompt "What is 2+2?"
```

## Features

- **Tokenizer Toolkit**: Encode, decode, analyze, compare, fingerprint, and debug any tokenizer
- **Training**: SFT, DPO, GRPO, PPO trainers with LoRA support
- **Models**: LLaMA, Mistral, Gemma, Granite, StarCoder2, TinyLlama
- **Analysis**: Coverage, entropy, efficiency, fit scoring, vocabulary induction
- **Instrumentation**: Histograms, OOV analysis, waste metrics, vocab comparison
- **CLI**: Comprehensive command-line interface for all operations

## Project Structure

```
src/chuk_lazarus/
├── cli/                    # Command-line interface
├── data/
│   ├── tokenizers/         # Tokenizer toolkit
│   │   ├── analyze/        # Coverage, entropy, fit scoring
│   │   ├── backends/       # HuggingFace + fast MLX backends
│   │   ├── curriculum/     # Length buckets, reasoning density
│   │   ├── instrumentation/# Histograms, OOV, waste metrics
│   │   ├── preprocessing/  # Hooks, profiles, byte fallback
│   │   ├── regression/     # Token regression testing
│   │   ├── research/       # Soft tokens, embedding analysis
│   │   ├── runtime/        # Special token registry
│   │   └── training/       # Packing, throughput profiling
│   └── generators/         # Synthetic data generation
├── models/                 # Model architectures and loading
├── training/               # Trainers (SFT, DPO, GRPO, PPO)
├── inference/              # Text generation
└── utils/                  # Utilities
```

## License

MIT
