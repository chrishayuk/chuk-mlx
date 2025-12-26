# Lazarus

**A deterministic data & execution substrate that enables reliable training.**

> Lazarus makes training runs reproducible **the same way lockfiles make builds reproducible**.

Offline batch plans, reproducible batching, measurable efficiency — the stuff you wish every training stack shipped with.

Runs on macOS; designed for Apple Silicon first (MLX backend).

**The core idea:** The BatchPlan is the contract. Trainers enforce it; they don't invent it. Build plans offline, version them, verify them in CI/CD (fingerprints + schema validation), and replay them exactly across distributed workers that share the same plan artifact. BatchPlans are fingerprinted against the tokenizer and length cache, so you can detect drift when data or tokenization changes.

```
Dataset → Tokenizer → Length Cache → BatchPlan Artifact → Trainer (enforces) → Checkpoints
                 fingerprint └─────────── fingerprint ┘
```

Most training pipelines entangle data loading, batching, and execution inside the trainer, making runs hard to reproduce, debug, or scale. Lazarus separates *planning* from *execution*: batching decisions are made once, recorded as artifacts, and enforced consistently across runs and workers.

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

# Run comprehensive pipeline benchmark
chuk-lazarus bench --num-samples 1000
chuk-lazarus bench -d train.jsonl -t gpt2 --bucket-edges 128,256,512

# Benchmark reports are saved as JSON + markdown for tracking regressions:
# length histogram, bucket efficiency, pack vs pad comparison,
# throughput metrics, memory footprint, and actionable recommendations
```

BatchPlans are recommended for production and distributed training; streaming batching (below) is intended for online, exploratory, or RL-style workloads.

**Minimal end-to-end deterministic pipeline:**

```bash
chuk-lazarus data lengths build -d train.jsonl -t gpt2 -o lengths.jsonl
chuk-lazarus data batchplan build -l lengths.jsonl -e 1 -b 4096 -o batch_plan/ --predictable
chuk-lazarus train sft --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data train.jsonl --batch-plan batch_plan/
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

**Supported puzzles:** Sudoku, KenKen, Nonogram, Lights Out, Sokoban, Minesweeper, and [16 others](docs/gym.md).

### BatchPlan-Driven Training

Training in Lazarus is driven entirely by precomputed BatchPlans. The trainer does not decide batching, sequencing, or token budgets — it enforces them.

> **Invariant:** If two runs use the same BatchPlan artifact (including its fingerprints) and seed, Lazarus guarantees identical batch structure and ordering across runs and workers.
>
> *Identical* means: same sample IDs per step, in the same order, with the same packing boundaries and token budgets. (Numerical results may differ slightly across hardware/kernel implementations; the **batch schedule** remains identical.)

```bash
# Canonical deterministic training (always use --batch-plan)
chuk-lazarus train sft \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --data train.jsonl \
  --batch-plan batch_plan/ \
  --use-lora

# Dev convenience (builds plan on the fly; still fingerprints and saves it)
chuk-lazarus train sft \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --data train.jsonl \
  --build-plan --predictable \
  --use-lora

# Train with DPO
chuk-lazarus train dpo --model ./checkpoints/sft/final --data preferences.jsonl --batch-plan batch_plan/

# Generate synthetic training data
chuk-lazarus generate --type math --output ./data/lazarus

# Run inference
chuk-lazarus infer --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --prompt "What is 2+2?"
```

### Inference Pipeline (New!)

The new unified inference pipeline provides a simplified API for running inference with any supported model family. One-liner setup, no boilerplate:

```python
from chuk_lazarus.inference import InferencePipeline, PipelineConfig, DType
from chuk_lazarus.models_v2 import LlamaConfig, LlamaForCausalLM

# One-liner model loading
pipeline = InferencePipeline.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    LlamaForCausalLM,
    LlamaConfig,
)

# Simple chat API
result = pipeline.chat("What is the capital of France?")
print(result.text)
print(result.stats.summary)  # "25 tokens in 0.42s (59.5 tok/s)"
```

**Key features:**
- Typed configuration with Pydantic (`PipelineConfig`, `GenerationConfig`)
- Async support (`InferencePipeline.from_pretrained_async`)
- Chat history management (`ChatHistory`)
- Streaming generation (`generate_stream`)
- No magic strings - uses enums (`DType`, `Role`)

```bash
# Simplified inference examples
uv run python examples/inference/simple_inference.py --prompt "Write a haiku"
uv run python examples/inference/llama_inference.py --model smollm2-360m
uv run python examples/inference/granite_inference.py --model granite-3.1-2b
uv run python examples/inference/gemma_inference.py --chat
```

### Model Family Examples

Run inference with specific model families:

```bash
# Llama family (TinyLlama, SmolLM2, Llama 2/3, Mistral)
uv run python examples/inference/llama_inference.py --model tinyllama
uv run python examples/inference/llama_inference.py --model smollm2-360m
uv run python examples/inference/llama_inference.py --list  # Show all presets

# Gemma 3 (1B, 4B, 12B, 27B with 128K context)
uv run python examples/inference/gemma_inference.py --chat
uv run python examples/inference/gemma_inference.py --model gemma-3-4b

# Granite (IBM, dense and hybrid MoE variants)
uv run python examples/inference/granite_inference.py --model granite-3.1-2b

# Llama 4 Scout (Hybrid Mamba-Transformer MoE)
uv run python examples/inference/llama4_inference.py
```

### FunctionGemma (Function Calling)

Run function calling inference with Google's FunctionGemma model:

```bash
# Run FunctionGemma for tool use / function calling
uv run python examples/models/gemma/01_functiongemma_inference.py
```

FunctionGemma is a 270M parameter model optimized for on-device function calling, supporting:
- Tool use / API calling
- MCP (Model Context Protocol) integration
- Lightweight RAG pipelines
- On-device agents

See [docs/inference.md](docs/inference.md) for detailed inference documentation.

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
├── models_v2/              # Composable model architecture
│   ├── core/               # Backend, config, enums, registry
│   ├── components/         # Attention, FFN, embeddings, SSM, recurrent
│   ├── blocks/             # Transformer, Mamba, recurrent, hybrid blocks
│   ├── backbones/          # Stacks of blocks with embeddings
│   ├── heads/              # LM, classifier, regression heads
│   ├── models/             # CausalLM, classifiers
│   ├── families/           # Llama, Mamba implementations
│   ├── adapters/           # LoRA adapters
│   └── losses/             # Loss functions (pure math)
├── training/               # BatchPlan-driven reference trainers (SFT, DPO, GRPO, PPO)
├── inference/              # Unified inference pipeline
│   ├── pipeline.py         # InferencePipeline high-level API
│   ├── loader.py           # HFLoader, DType, WeightConverter
│   ├── chat.py             # ChatHistory, Role, format_chat_prompt
│   └── generation.py       # GenerationConfig, generate, generate_stream
├── distributed/            # Distributed training utilities
└── utils/                  # Utilities
```

### Key Modules

| Module | Description |
|--------|-------------|
| **Models** | Composable architecture: components, blocks, backbones, heads, families (Llama, Gemma, Granite) |
| **Inference** | Unified pipeline API: `InferencePipeline`, chat history, streaming generation |
| **Tokenizers** | Comprehensive toolkit for analysis, preprocessing, and runtime management |
| **Batching** | Token-budget batching, sequence packing, distributed batch planning |
| **Streaming** | Puzzle arcade integration, replay buffers, online learning |
| **Training** | BatchPlan-driven trainers — enforce, don't decide |

## Features

- **Tokenizer Toolkit**: Encode, decode, analyze, compare, fingerprint, and debug any tokenizer
- **Character Tokenizer**: Built-in character-level tokenizer for classification experiments
- **Tokenizer Doctor**: Health check with auto-fix for missing chat templates
- **Chat Template Registry**: 7 built-in formats (ChatML, Llama, Phi, Gemma, Zephyr, Vicuna, Alpaca)
- **Batching Infrastructure**: Token-budget batching, sequence packing (measurable via `chuk-lazarus bench`)
- **BatchPlan Artifacts**: Versioned, fingerprinted batch schedules for reproducibility and CI/CD
- **Pipeline Benchmark**: Pack vs pad comparison, throughput metrics, memory footprint analysis
- **BatchPlan-Driven Training**: Trainers enforce plans, not build them — deterministic by design
- **Puzzle Arcade Integration**: Stream training data from 24 puzzle types for online/RL learning
- **Replay Buffers**: Priority sampling, difficulty tracking, curriculum support
- **Analysis**: Coverage, entropy, efficiency, fit scoring, vocabulary induction
- **Instrumentation**: Histograms, OOV analysis, waste metrics, vocab comparison

**What Lazarus is NOT:**
- Not a trainer framework competing with Lightning/Accelerate
- Not a new optimizer zoo or model architecture lab
- Not a "magic trainer" that decides things for you

**What Lazarus IS:** A reproducible planning/execution substrate you can plug into anything.

## Artifacts

BatchPlans are the core artifact. When you build a batch plan, Lazarus creates:

```
batch_plan/
├── plan.jsonl          # Batch schedule: sample IDs, packing, token counts per step
├── metadata.json       # Epochs, token budget, strategy, version info
├── fingerprints.json   # Tokenizer + length cache fingerprints for drift detection
└── stats.json          # Efficiency metrics: utilization, waste, packing ratio
```

**Schema promise:** The `plan.jsonl` format is stable. Each line is a JSON object:

```json
{"step":0,"samples":[12,88,104],"tokens":4096,"packing":[[0,128],[128,256]]}
```

Fields: `step` (global index), `samples` (sample IDs), `tokens` (batch total), `packing` (boundaries).

**metadata.json** includes:
- `plan_format_version`: Schema version for forward compatibility
- `tool_version`: Lazarus version that created the plan
- `seed`: Random seed used (if predictable mode)
- `created_at`: Timestamp

**CI/CD validation:**

```bash
# Validate a plan artifact before training (CI-friendly)
chuk-lazarus data batchplan validate -p batch_plan/ --strict
```

If the tokenizer or data changes, fingerprint mismatch is detected before training starts.

## Documentation

- [Getting Started](docs/getting-started.md) - Installation and quick reference
- [CLI Reference](docs/cli.md) - Command-line interface documentation
- [Models Guide](docs/models.md) - Composable model architecture, components, LoRA adapters
- [Inference Guide](docs/inference.md) - Run inference with pretrained HuggingFace models
- [Tokenizers Guide](docs/tokenizers.md) - Comprehensive tokenizer toolkit
- [Batching Guide](docs/batching.md) - Token-budget batching, packing, distributed training
- [Training Guide](docs/training.md) - BatchPlan-driven training
- [API Reference](docs/api-reference.md) - Python API documentation

## Supported Models

| Family | Models | Notes |
|--------|--------|-------|
| **Llama** | TinyLlama, Llama 2 (7B, 13B), Llama 3.1/3.2, Llama 4 Scout | Llama 4 uses Mamba-Transformer hybrid |
| **SmolLM2** | 135M, 360M, 1.7B | No auth required, fast inference |
| **Mistral** | 7B Instruct v0.3 | Sliding window attention |
| **Gemma** | Gemma 3 (270M, 1B, 4B, 12B, 27B), FunctionGemma | 128K context, function calling |
| **Granite** | 3.0/3.1 (2B, 8B), 4.0 Tiny (1B, 1.5B MoE) | IBM, dense and MoE variants |
| **StarCoder2** | 3B, 7B, 15B | Code generation |

## OpenAI Tokenizers

Support for OpenAI's tokenizers via tiktoken:

```bash
uvx "chuk-lazarus[openai]" tokenizer encode -t "gpt-4" --text "Hello, world!"
uvx "chuk-lazarus[openai]" tokenizer compare -t1 "gpt-4" -t2 "gpt-4o" --text "Test"
```

Supported: `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`, `o1`, `o1-mini`, `o3-mini`

## License

MIT
