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

**Supported puzzles:** Sudoku, KenKen, Kakuro, Binary, Futoshiki, Nonogram, Logic Grid, Killer Sudoku, Lights Out, Mastermind, Slitherlink, Bridges, Hitori, Shikaku, Hidato, Tents, Fillomino, Star Battle, Sokoban, Knapsack, Nurikabe, Minesweeper.

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

### UnifiedPipeline

The `UnifiedPipeline` auto-detects model family and provides a simplified API. One-liner setup, no boilerplate:

```python
from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig, DType

# One-liner model loading - auto-detects family!
pipeline = UnifiedPipeline.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Simple chat API
result = pipeline.chat("What is the capital of France?")
print(result.text)
print(result.stats.summary)  # "25 tokens in 0.42s (59.5 tok/s)"
print(f"Model family: {pipeline.family_type}")  # ModelFamilyType.LLAMA
```

**Key features:**
- Auto-detection of model family from HuggingFace config
- Typed configuration with Pydantic (`UnifiedPipelineConfig`, `GenerationConfig`)
- Async support (`UnifiedPipeline.from_pretrained_async`)
- Chat history management (`ChatHistory`)
- Streaming generation (`generate_stream`)
- No magic strings - uses enums (`DType`, `Role`, `ModelFamilyType`)

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

# StarCoder2 (Code generation, 3B/7B/15B)
uv run python examples/inference/starcoder2_inference.py --prompt "def fibonacci(n):"
uv run python examples/inference/starcoder2_inference.py --interactive  # Interactive mode

# Jamba (Hybrid Mamba-Transformer MoE, 256K context)
uv run python examples/inference/jamba_inference.py --test-tiny  # Test without download
uv run python examples/inference/jamba_inference.py --list       # Show models
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

### Introspection (Model Analysis)

Analyze model behavior using logit lens, ablation studies, attention visualization, and MoE expert identification:

```bash
# Run logit lens analysis - see how predictions evolve across layers
chuk-lazarus introspect analyze -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 -p "The capital of France is"

# Track specific tokens through layers
chuk-lazarus introspect analyze -m model -p "Hello" --track "world,there" --layer-strategy all

# Compare two models' predictions
chuk-lazarus introspect compare -m1 google/gemma-3-270m-it -m2 google/functiongemma-270m-it -p "Get the weather" --track "get_"

# Ablation study - find causal circuits
chuk-lazarus introspect ablate -m model -p "What's the weather?" -c function_call --layers 8-15

# Multi-layer ablation - test layers together
chuk-lazarus introspect ablate -m model -p "45 * 45 = " -c "2025" --layers 22,23 --multi

# Test if task type is baked into embeddings (RLVF hypothesis)
chuk-lazarus introspect embedding -m model

# Analyze operand encoding structure (holistic vs compositional)
chuk-lazarus introspect operand-directions -m model

# Test commutativity (lookup table vs algorithm)
chuk-lazarus introspect commutativity -m model

# Activation patching between prompts
chuk-lazarus introspect patch -m model --source "7*8=" --target "7+8="

# Low-level hook demonstration
chuk-lazarus introspect hooks -m model -p "Test" --layers 0,4,8 --capture-attention
```

**MoE Expert Identification** - Discover what each expert specializes in:

```python
from mlx_lm import load
from chuk_lazarus.introspection import ExpertIdentifier, identify_experts

# Load any MoE model
model, tokenizer = load("openai/gpt-oss-20b")

# Identify all experts in a layer
result = identify_experts(model, tokenizer, layer_idx=12)
print(result.summary())

# Results show expert specializations:
# CODE: Experts [1, 14, 22, 23, 27, 28]
# MATH: Experts [6, 7, 19, 24, 30, 31]
# CONTENT_WORDS: Experts [0, 2, 3, 4, 5, 8, 9, ...]
# NAMES: Experts [15, 26]

# Get detailed identity for specific expert
expert_6 = result.expert_identities[6]
print(expert_6.detailed_report())
# Expert 6: math (52% confidence)
# Top tokens: ['+', '2', 'x', '3', ...]
# Semantic clusters: ['numeric_values']
```

**MoE Routing Analysis** - Capture and analyze routing decisions:

```python
from chuk_lazarus.introspection import MoEHooks, MoECaptureConfig

hooks = MoEHooks(model)
hooks.configure(MoECaptureConfig(
    capture_router_logits=True,
    capture_selected_experts=True,
))

logits = hooks.forward(input_ids)

# Analyze routing
utilization = hooks.get_expert_utilization(layer_idx=12)
print(f"Load balance: {utilization.load_balance_score:.2%}")

entropy = hooks.get_router_entropy(layer_idx=12)
print(f"Router confidence: {1 - entropy.normalized_entropy:.2%}")
```

**Logit Lens and Ablation:**

```python
from chuk_lazarus.introspection import ModelAnalyzer, AnalysisConfig, LayerStrategy

# Async API for logit lens analysis
async with ModelAnalyzer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0") as analyzer:
    result = await analyzer.analyze("The capital of France is")
    print(result.predicted_token)  # " Paris"
    for layer in result.layer_predictions:
        print(f"Layer {layer.layer_idx}: {layer.top_token}")

# Track token evolution
config = AnalysisConfig(track_tokens=["Paris", " Paris"])
result = await analyzer.analyze("The capital of France is", config)
for evo in result.token_evolutions:
    print(f"{evo.token} emerges at layer {evo.emergence_layer}")
```

```python
from chuk_lazarus.introspection import AblationStudy, AblationConfig

# Ablation studies - identify causal circuits
study = AblationStudy.from_pretrained("openai/gpt-oss-20b")
config = AblationConfig(max_new_tokens=15)

original = study.ablate_and_generate("45 * 45 = ", layers=[], config=config)
ablated = study.ablate_and_generate("45 * 45 = ", layers=[22, 23], config=config)
print(f"Original: {original}")  # "2025..."
print(f"L22+L23 ablated: {ablated}")  # Broken output
```

See [docs/introspection.md](docs/introspection.md) for detailed introspection documentation.

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
│   ├── unified.py          # UnifiedPipeline with auto-detection
│   ├── loader.py           # HFLoader, DType, WeightConverter
│   ├── chat.py             # ChatHistory, Role, format_chat_prompt
│   └── generation.py       # GenerationConfig, generate, generate_stream
├── introspection/          # Model introspection and analysis
│   ├── analyzer.py         # ModelAnalyzer async API with Pydantic models
│   ├── hooks.py            # ModelHooks for capturing intermediate states
│   ├── logit_lens.py       # Layer-by-layer prediction analysis
│   ├── attention.py        # Attention pattern analysis
│   ├── moe.py              # MoE introspection (routing, expert identification)
│   ├── ablation/           # Ablation studies for causal discovery
│   └── visualizers/        # Heatmaps and evolution plots
├── distributed/            # Distributed training utilities
└── utils/                  # Utilities
```

### Key Modules

| Module | Description |
|--------|-------------|
| **Models** | Composable architecture: components, blocks, backbones, heads, families (Llama, Gemma, Granite) |
| **Inference** | `UnifiedPipeline` with auto-detection, chat history, streaming generation |
| **Introspection** | Model analysis: logit lens, attention visualization, MoE expert identification, ablation studies |
| **Tokenizers** | Comprehensive toolkit for analysis, preprocessing, and runtime management |
| **Batching** | Token-budget batching, sequence packing, distributed batch planning |
| **Streaming** | Puzzle arcade integration, replay buffers, online learning |
| **Training** | BatchPlan-driven trainers — enforce, don't decide |

## Features

- **Introspection**: Logit lens, attention visualization, MoE expert identification, ablation studies, token evolution tracking
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
- [Introspection Guide](docs/introspection.md) - Logit lens, attention visualization, model analysis
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
| **Jamba** | v0.1, 1.5 Mini (52B), 1.5 Large (398B) | AI21 hybrid Mamba-Transformer MoE, 256K context |
| **Mamba** | 130M, 370M, 790M, 1.4B, 2.8B | Pure SSM architecture |

## OpenAI Tokenizers

Support for OpenAI's tokenizers via tiktoken:

```bash
uvx "chuk-lazarus[openai]" tokenizer encode -t "gpt-4" --text "Hello, world!"
uvx "chuk-lazarus[openai]" tokenizer compare -t1 "gpt-4" -t2 "gpt-4o" --text "Test"
```

Supported: `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`, `o1`, `o1-mini`, `o3-mini`

## License

MIT
