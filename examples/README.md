# Lazarus Examples

Examples organized by functionality area.

## Directory Structure

```
examples/
├── batching/           # Batching and distributed training
│   ├── 01_basic_batching.py    # Token-budget batching
│   ├── 02_sequence_packing.py  # Sequence packing algorithms
│   ├── 03_batch_plan.py        # BatchPlan for reproducibility
│   ├── 04_distributed.py       # Distributed training
│   ├── 05_e2e_pipeline.py      # Complete end-to-end pipeline
│   └── 06_analyze.py           # Batching analysis & optimization
├── tokenizer/          # Tokenization examples
│   ├── basic_tokenization.py
│   ├── custom_tokenizer.py
│   ├── tokenizer_analysis.py
│   ├── curriculum_learning.py
│   ├── runtime_vocab.py
│   ├── training_utils.py
│   ├── preprocessing_demo.py
│   ├── regression_tests.py
│   ├── research_playground.py
│   ├── instrumentation_demo.py
│   ├── backends_demo.py
│   └── hero_doctor_demo.py     # ⭐ Hero Demo 1
├── inference/          # Text generation examples
│   ├── basic_inference.py
│   └── chat_inference.py
├── training/           # Model training examples
│   ├── sft_training.py
│   ├── dpo_training.py
│   ├── hero_sft_math_demo.py   # ⭐ Hero Demo 2
│   └── hero_dpo_demo.py        # ⭐ Hero Demo 3
├── data/               # Data handling examples
│   ├── generate_math_data.py
│   └── create_sft_dataset.py
└── models/             # Model inference examples
    ├── gemma/              # Gemma family examples
    │   ├── 01_functiongemma_inference.py  # FunctionGemma tool calling
    │   ├── 02_load_pretrained.py          # Load pretrained weights
    │   ├── 03_gemma3_inference.py         # Gemma 3 text generation
    │   └── 04_gemma3_vision_inference.py  # Gemma 3 vision (multimodal)
    ├── granite/            # IBM Granite examples
    │   └── 01_granite_inference.py        # Granite inference
    ├── llama/              # Llama family examples
    │   ├── 01_causal_lm.py                # Basic causal LM
    │   ├── 02_tinyllama_inference.py      # TinyLlama inference
    │   └── 03_llama_family_inference.py   # Multi-model inference
    ├── llama4/             # Llama 4 examples
    │   └── 01_llama4_inference.py         # Llama 4 Scout inference
    ├── lora/               # LoRA examples
    └── mlp/                # MLP classifier examples
```

## Quick Start

### Load and Generate

```python
from chuk_lazarus.models import load_model, generate_response

model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
response = generate_response(model, tokenizer, "Hello, ", max_tokens=50)
```

### Fine-tune with SFT

```python
from chuk_lazarus.models import load_model
from chuk_lazarus.training import SFTTrainer, SFTConfig
from chuk_lazarus.data import SFTDataset

model, tokenizer = load_model("model-name", use_lora=True)
dataset = SFTDataset("./data/train.jsonl", tokenizer)
trainer = SFTTrainer(model, tokenizer, SFTConfig())
trainer.train(dataset)
```

### Generate Training Data

```python
from chuk_lazarus.data.generators import generate_lazarus_dataset

paths = generate_lazarus_dataset(
    output_dir="./data",
    sft_samples=1000,
    dpo_samples=500,
)
```

## Hero Demos

Three comprehensive demos showcasing the "small tired model resurrected" workflow:

### Hero Demo 1: Tokenizer Doctor + Chat Template

Health check tokenizers and auto-fix missing chat templates:

```bash
uv run python examples/tokenizer/hero_doctor_demo.py
```

Features demonstrated:
- Comprehensive tokenizer health check
- Chat template format detection (ChatML, Llama, Phi, Gemma, etc.)
- Auto-patching missing templates
- Validation and capability detection

### Hero Demo 2: SFT on Synthetic Math

Train TinyLlama on synthetic math with step-by-step reasoning:

```bash
uv run python examples/training/hero_sft_math_demo.py
```

Features demonstrated:
- Synthetic math data generation
- SFT dataset creation with chat formatting
- LoRA training configuration
- Inference after training

### Hero Demo 3: DPO/GRPO Preference Tuning

Preference tuning on puzzle outcomes:

```bash
uv run python examples/training/hero_dpo_demo.py
```

Features demonstrated:
- Preference pair generation (chosen vs rejected)
- DPO training configuration
- GRPO variant for group preferences
- Complete SFT → DPO pipeline

## Tokenizer Examples

### Preprocessing Demo

Demonstrates the preprocessing module features for reducing token waste:

```bash
uv run python examples/tokenizer/preprocessing_demo.py
```

Features demonstrated:
- **Numeric normalization** - Detect and encode numbers (integers, floats, scientific, hex, percentages, fractions)
- **Structure token injection** - Replace UUIDs, URLs, emails, IPs with atomic tokens
- **Hook pipeline** - Composable pre/post tokenization transforms
- **Tokenizer profiles** - Switch between training and inference modes
- **Byte fallback** - Ensure any byte sequence tokenizes without UNK

### Regression Tests

Run tokenizer regression tests:

```bash
uv run python examples/tokenizer/regression_tests.py
```

### Research Playground

Demonstrates experimental tokenization techniques:

```bash
uv run python examples/tokenizer/research_playground.py
```

Features demonstrated:
- **Soft tokens** - Learnable embeddings for prompt tuning and controllable generation
- **Token morphing** - Linear, spherical, bezier, cubic interpolation between embeddings
- **Token blending** - Average, weighted, geometric, attention-based blending
- **Embedding analysis** - Nearest neighbors, clustering, PCA projection, analogies
- **Quality metrics** - Isotropy, silhouette score, pairwise similarity

### Instrumentation Demo

Analyze tokenization behavior without modifying the tokenizer:

```bash
uv run python examples/tokenizer/instrumentation_demo.py
```

Features demonstrated:
- **Token length histograms** - Visualize length distributions with ASCII charts
- **OOV/rare token analysis** - Identify vocabulary coverage issues
- **Waste metrics** - Measure padding and truncation inefficiency
- **Vocabulary comparison** - Before/after analysis for tokenizer swaps

## Batching & Distributed Training

### Basic Batching

Token-budget batching for efficient GPU utilization:

```bash
uv run python examples/batching/01_basic_batching.py
```

Features demonstrated:
- Length-based bucketing with `BucketSpec`
- Token-budget batch formation
- Padding waste metrics

### Sequence Packing

Pack multiple sequences to reduce padding waste:

```bash
uv run python examples/batching/02_sequence_packing.py
```

Features demonstrated:
- FIRST_FIT, BEST_FIT, GREEDY algorithms
- Segment-aware attention masks
- Packing efficiency metrics

### Batch Plans

Precomputed schedules for reproducible training:

```bash
uv run python examples/batching/03_batch_plan.py
```

Features demonstrated:
- Building `BatchPlan` with `BatchPlanBuilder`
- Saving/loading plans to disk
- Fingerprint verification

### Distributed Training

Sharding and checkpointing for multi-worker training:

```bash
uv run python examples/batching/04_distributed.py
```

Features demonstrated:
- `DistributedConfig` from environment variables
- Batch plan sharding across workers
- Checkpoint resume with `CheckpointPosition`
- CLI: `lazarus data batchplan shard`

### End-to-End Pipeline

Complete data pipeline from raw dataset to training-ready batches:

```bash
uv run python examples/batching/05_e2e_pipeline.py
```

Features demonstrated:
- Dataset preparation (JSONL)
- Tokenization with instruction/response formatting
- Length caching for efficient batching
- Batch plan creation with bucketing
- Two training paths: runtime streaming or offline NPZ files
- Distributed training with plan sharding

### Batching Analysis

Analyze and optimize batching configuration:

```bash
uv run python examples/batching/06_analyze.py
```

Features demonstrated:
- Length histograms with percentiles (P25, P50, P75, P90, P99)
- Bucket efficiency analysis
- Optimal bucket edge suggestions with three goals:
  - `minimize_waste` - Maximize token utilization
  - `balance_buckets` - Even sample distribution
  - `minimize_memory` - Reduce peak memory usage
- Complete efficiency reports with recommendations
- CLI equivalents: `lazarus data batching histogram`, `analyze`, `suggest`

## Model Inference Examples

Run inference with various model families using pretrained weights from HuggingFace.

### Gemma 3

```bash
# Basic inference
uv run python examples/models/gemma/03_gemma3_inference.py --prompt "What is the capital of France?"

# Interactive chat
uv run python examples/models/gemma/03_gemma3_inference.py --chat

# Use larger model
uv run python examples/models/gemma/03_gemma3_inference.py \
  --model mlx-community/gemma-3-4b-it-bf16 \
  --prompt "Explain machine learning"
```

**Available models:** `gemma-3-1b-it-bf16`, `gemma-3-4b-it-bf16`, `gemma-3-12b-it-bf16`, `gemma-3-27b-it-bf16`

### Gemma 3 Vision (Multimodal)

```bash
# Image understanding
uv run python examples/models/gemma/04_gemma3_vision_inference.py \
  --image /path/to/image.jpg \
  --prompt "What is in this image?"

# Detailed description
uv run python examples/models/gemma/04_gemma3_vision_inference.py \
  --image photo.jpg \
  --prompt "Describe this image in detail" \
  --max-tokens 200
```

**Available models:** `gemma-3-4b-it-bf16` (4B), `gemma-3-12b-it-bf16` (12B), `gemma-3-27b-it-bf16` (27B)

### FunctionGemma (Tool Calling)

```bash
# Function calling example
uv run python examples/models/gemma/01_functiongemma_inference.py
```

### Llama Family

```bash
# List available models
uv run python examples/models/llama/03_llama_family_inference.py --list-models

# Run with different models
uv run python examples/models/llama/03_llama_family_inference.py --model tinyllama
uv run python examples/models/llama/03_llama_family_inference.py --model smollm2-360m
```

**Available presets:** `tinyllama`, `smollm2-135m`, `smollm2-360m`, `smollm2-1.7b`, `llama2-7b`, `llama3.2-1b`, `mistral-7b`

### Granite

```bash
# Basic inference
uv run python examples/models/granite/01_granite_inference.py --prompt "What is machine learning?"
```

### Llama 4

```bash
# Llama 4 Scout inference (requires HF auth)
uv run python examples/models/llama4/01_llama4_inference.py --prompt "Explain quantum computing"
```

**Note:** Llama 4 requires HuggingFace authentication. Run `huggingface-cli login` first.

## Running Examples

```bash
# From project root
uv run python examples/inference/basic_inference.py
```
