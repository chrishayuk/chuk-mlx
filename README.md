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

# For OpenAI tokenizers (gpt-4, gpt-3.5-turbo, o1, etc.)
uv add "chuk-lazarus[openai]"

# For faster tokenization (optional MLX backend)
uv add "chuk-lazarus[fast]"
```

After installation, use the `chuk-lazarus` command directly:

```bash
chuk-lazarus tokenizer encode -t "gpt2" --text "Hello"
```

## CLI Reference

### Tokenizer Commands

```bash
# Encode text - see token IDs and boundaries
chuk-lazarus tokenizer encode -t "gpt2" --text "The quick brown fox"

# Decode token IDs back to text
chuk-lazarus tokenizer decode -t "gpt2" --ids "464,2068,7586,21831"

# Search the vocabulary
chuk-lazarus tokenizer vocab -t "gpt2" --search "hello"

# Compare two tokenizers
chuk-lazarus tokenizer compare -t1 "gpt2" -t2 "meta-llama/Llama-2-7b" --text "Test"

# Health check with auto-fix
chuk-lazarus tokenizer doctor -t "model-name" --fix

# Generate fingerprint for compatibility
chuk-lazarus tokenizer fingerprint -t "gpt2" --save fingerprint.json
```

### Corpus Analysis

```bash
# Coverage analysis - UNK rate, tokens per word
chuk-lazarus tokenizer analyze coverage -t "gpt2" --file corpus.txt

# Fit score - tokenizer-dataset compatibility (0-100)
chuk-lazarus tokenizer analyze fit-score -t "gpt2" --file corpus.txt

# Efficiency analysis - tokens per sample, fragmentation
chuk-lazarus tokenizer analyze efficiency -t "gpt2" --file corpus.txt
```

### Data Commands

```bash
# Build a length cache (tokenize once, reuse lengths)
chuk-lazarus data lengths build -d train.jsonl -t "gpt2" -o lengths.jsonl

# Build a batch plan for reproducible training
chuk-lazarus data batchplan build -l lengths.jsonl -e 3 -b 4096 -o batch_plan/ --predictable

# Show batch plan info
chuk-lazarus data batchplan info -p batch_plan/ --show-batches 5

# Analyze batching efficiency
chuk-lazarus data batching analyze --cache lengths.jsonl --bucket-edges 128,256,512
```

### Training Commands

```bash
# Train with SFT
chuk-lazarus train sft --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --data train.jsonl --use-lora

# Train with DPO
chuk-lazarus train dpo --model ./checkpoints/sft/final --data preferences.jsonl

# Generate synthetic training data
chuk-lazarus generate --type math --output ./data/lazarus

# Run inference
chuk-lazarus infer --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --prompt "What is 2+2?"
```

### Puzzle Arcade Integration

Stream training data from the puzzle arcade server for online/RL training:

```bash
# Connect to puzzle gym and collect episodes
chuk-lazarus gym run --host localhost --port 8023 --puzzles sudoku,binary --episodes 100
```

```python
from chuk_lazarus.data.batching.streaming import (
    TelnetGymClient, TelnetClientConfig, PuzzleGame, PuzzleDifficulty,
    ReplayBuffer, ReplayBufferConfig, StreamSample, SampleSource,
)

# Connect to puzzle server
config = TelnetClientConfig(host="localhost", port=8023)
async with TelnetGymClient(config) as client:
    # Start a puzzle
    obs = await client.start_puzzle(PuzzleGame.SUDOKU, PuzzleDifficulty.EASY)

    # Get optimal moves as training data
    hint = await client.get_hint()
    print(f"Next move: {hint.message}")

    # Collect into replay buffer
    buffer = ReplayBuffer(ReplayBufferConfig(max_size=10000))
    sample = StreamSample(
        input_ids=tokenize(prompt),
        loss_mask=loss_mask,
        source=SampleSource.GYM,
        difficulty=0.3,
    )
    buffer.add(sample)
```

**Supported puzzles:** Sudoku, KenKen, Kakuro, Binary, Futoshiki, Nonogram, Logic Grid, Killer Sudoku, Lights Out, Mastermind, Slitherlink, Bridges, Hitori, Shikaku, Hidato, Tents, Fillomino, Star Battle, Sokoban, Knapsack, Nurikabe, Minesweeper

## Python API

```python
from chuk_lazarus.utils.tokenizer_loader import load_tokenizer
from chuk_lazarus.data.tokenizers.analyze import analyze_coverage, calculate_fit_score
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

## Architecture

```
src/chuk_lazarus/
├── cli/                    # Command-line interface
├── data/
│   ├── batching/           # Token-budget batching, packing, distributed planning
│   │   ├── core/           # Bucketing, sampling, metrics
│   │   ├── planning/       # Batch plans, predictability, packing
│   │   ├── generation/     # Batch I/O, length caching
│   │   ├── streaming/      # Gym integration, replay buffers
│   │   └── analyze/        # Efficiency analysis
│   ├── samples/            # Sample schema and validation
│   ├── tokenizers/         # Tokenizer toolkit (analysis, preprocessing, runtime)
│   └── generators/         # Synthetic data generation
├── models/                 # Model architectures and loading
├── training/               # Trainers (SFT, DPO, GRPO, PPO)
├── inference/              # Text generation
├── distributed/            # Distributed training utilities
└── utils/                  # Utilities
```

### Key Modules

| Module | Description |
|--------|-------------|
| **Tokenizers** | Comprehensive toolkit for analysis, preprocessing, and runtime management |
| **Batching** | Token-budget batching, sequence packing, distributed batch planning |
| **Streaming** | Puzzle arcade integration, replay buffers, online learning |
| **Training** | SFT, DPO, GRPO, PPO trainers with LoRA support |
| **Models** | LLaMA, Mistral, Gemma, Granite, StarCoder2, TinyLlama |

## Features

- **Tokenizer Toolkit**: Encode, decode, analyze, compare, fingerprint, and debug any tokenizer
- **Tokenizer Doctor**: Health check with auto-fix for missing chat templates
- **Chat Template Registry**: 7 built-in formats (ChatML, Llama, Phi, Gemma, Zephyr, Vicuna, Alpaca)
- **Batching Infrastructure**: Token-budget batching, sequence packing (50-70% token reduction)
- **Puzzle Arcade Integration**: Stream training data from 24 puzzle types for online/RL learning
- **Replay Buffers**: Priority sampling, difficulty tracking, curriculum support
- **Training**: SFT, DPO, GRPO, PPO trainers with LoRA support
- **Analysis**: Coverage, entropy, efficiency, fit scoring, vocabulary induction
- **Instrumentation**: Histograms, OOV analysis, waste metrics, vocab comparison

## Documentation

- [Getting Started](docs/getting-started.md) - Installation and quick reference
- [CLI Reference](docs/cli.md) - Command-line interface documentation
- [Tokenizers Guide](docs/tokenizers.md) - Comprehensive tokenizer toolkit
- [Batching Guide](docs/batching.md) - Token-budget batching, packing, distributed training
- [Training Guide](docs/training.md) - SFT, DPO, GRPO, PPO training
- [API Reference](docs/api-reference.md) - Python API documentation

## Supported Models

- LLaMA / LLaMA 2 / LLaMA 3
- Mistral
- Gemma
- Granite
- StarCoder2
- TinyLlama

## OpenAI Tokenizers

Support for OpenAI's tokenizers via tiktoken:

```bash
uvx "chuk-lazarus[openai]" tokenizer encode -t "gpt-4" --text "Hello, world!"
uvx "chuk-lazarus[openai]" tokenizer compare -t1 "gpt-4" -t2 "gpt-4o" --text "Test"
```

Supported: `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`, `o1`, `o1-mini`, `o3-mini`

## License

MIT
