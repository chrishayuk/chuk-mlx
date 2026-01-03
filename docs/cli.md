# CLI Reference

Lazarus provides a unified command-line interface for training, inference, data generation, and tokenizer utilities.

## Installation

After installing the package, the `lazarus` command is available:

```bash
lazarus --help
```

## Commands

### train

Train models using SFT or DPO.

#### train sft

Supervised Fine-Tuning on instruction data.

```bash
lazarus train sft --model MODEL --data DATA [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | required | Model name or path |
| `--data` | required | Training data path (JSONL) |
| `--eval-data` | - | Evaluation data path |
| `--output` | `./checkpoints/sft` | Output directory |
| `--epochs` | 3 | Number of epochs |
| `--batch-size` | 4 | Batch size |
| `--learning-rate` | 1e-5 | Learning rate |
| `--max-length` | 512 | Max sequence length |
| `--use-lora` | false | Enable LoRA |
| `--lora-rank` | 8 | LoRA rank |
| `--mask-prompt` | false | Mask prompt in loss |
| `--log-interval` | 10 | Log every N steps |

Example:
```bash
lazarus train sft \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data ./data/train.jsonl \
  --use-lora \
  --epochs 3
```

#### train dpo

Direct Preference Optimization training.

```bash
lazarus train dpo --model MODEL --data DATA [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | required | Policy model name or path |
| `--ref-model` | same as model | Reference model |
| `--data` | required | Preference data path (JSONL) |
| `--eval-data` | - | Evaluation data path |
| `--output` | `./checkpoints/dpo` | Output directory |
| `--epochs` | 3 | Number of epochs |
| `--batch-size` | 4 | Batch size |
| `--learning-rate` | 1e-6 | Learning rate |
| `--beta` | 0.1 | DPO beta parameter |
| `--max-length` | 512 | Max sequence length |
| `--use-lora` | false | Enable LoRA |
| `--lora-rank` | 8 | LoRA rank |

Example:
```bash
lazarus train dpo \
  --model ./checkpoints/sft/final \
  --data ./data/preferences.jsonl \
  --beta 0.1
```

### generate

Generate synthetic training data.

```bash
lazarus generate --type TYPE [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--type` | required | Data type (`math`) |
| `--output` | `./data/generated` | Output directory |
| `--sft-samples` | 10000 | Number of SFT samples |
| `--dpo-samples` | 5000 | Number of DPO samples |
| `--seed` | 42 | Random seed |

Example:
```bash
lazarus generate --type math --output ./data/lazarus --sft-samples 5000
```

### infer

Run inference on a model.

```bash
lazarus infer --model MODEL [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | required | Model name or path |
| `--adapter` | - | LoRA adapter path |
| `--prompt` | - | Single prompt |
| `--prompt-file` | - | File with prompts |
| `--max-tokens` | 256 | Max tokens to generate |
| `--temperature` | 0.7 | Sampling temperature |

Examples:
```bash
# Single prompt
lazarus infer --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Hello!"

# With adapter
lazarus infer --model model-name --adapter ./checkpoints/lora --prompt "Test"

# Interactive mode (no --prompt)
lazarus infer --model model-name
```

### tokenizer

Tokenizer utilities for inspecting and debugging tokenization.

#### tokenizer encode

Encode text to tokens and display in a table.

```bash
lazarus tokenizer encode -t TOKENIZER [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-t, --tokenizer` | required | Tokenizer name or path |
| `--text` | - | Text to encode |
| `-f, --file` | - | File to encode |
| `--special-tokens` | false | Add special tokens |

Examples:
```bash
# Encode text
lazarus tokenizer encode -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text "Hello world"

# Encode file
lazarus tokenizer encode -t model-name --file input.txt --special-tokens
```

#### tokenizer decode

Decode token IDs back to text.

```bash
lazarus tokenizer decode -t TOKENIZER --ids IDS
```

| Option | Default | Description |
|--------|---------|-------------|
| `-t, --tokenizer` | required | Tokenizer name or path |
| `--ids` | required | Token IDs (comma or space separated) |

Example:
```bash
lazarus tokenizer decode -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --ids "1,2,3,4,5"
```

#### tokenizer vocab

Display vocabulary information and search tokens.

```bash
lazarus tokenizer vocab -t TOKENIZER [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-t, --tokenizer` | required | Tokenizer name or path |
| `--show-all` | false | Show full vocabulary |
| `-s, --search` | - | Search for tokens containing string |
| `--limit` | 50 | Max search results |
| `--chunk-size` | 1000 | Chunk size for full display |
| `--pause` | false | Pause between chunks |

Examples:
```bash
# Show vocab stats
lazarus tokenizer vocab -t TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Search for tokens
lazarus tokenizer vocab -t model-name --search "hello" --limit 20

# Show full vocabulary
lazarus tokenizer vocab -t model-name --show-all --pause
```

#### tokenizer compare

Compare tokenization between two tokenizers.

```bash
lazarus tokenizer compare -t1 TOKENIZER1 -t2 TOKENIZER2 --text TEXT
```

| Option | Default | Description |
|--------|---------|-------------|
| `-t1, --tokenizer1` | required | First tokenizer |
| `-t2, --tokenizer2` | required | Second tokenizer |
| `--text` | required | Text to compare |

Example:
```bash
lazarus tokenizer compare \
  -t1 TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  -t2 meta-llama/Llama-2-7b-hf \
  --text "The quick brown fox jumps over the lazy dog."
```

### introspect

Mechanistic interpretability tools for understanding model internals. See [introspection.md](introspection.md) for full documentation.

**Quick Examples:**

```bash
# Logit lens analysis
lazarus introspect analyze -m model -p "The capital of France is"

# Activation steering
lazarus introspect steer -m model --extract --positive "good" --negative "bad" -o direction.npz

# Ablation study
lazarus introspect ablate -m model -p "45 * 45 =" -c "2025" --layers 20-23

# Linear probe
lazarus introspect probe -m model --class-a "hard problems" --class-b "easy problems"

# Systematic arithmetic testing
lazarus introspect arithmetic -m model --hard-only

# Uncertainty detection
lazarus introspect uncertainty -m model --prompts "test prompts"
```

**All introspect subcommands:** analyze, compare, generate, hooks, probe, neurons, directions, operand-directions, embedding, early-layers, activation-cluster, steer, ablate, patch, weight-diff, activation-diff, layer, format-sensitivity, arithmetic, commutativity, metacognitive, uncertainty, memory, memory-inject, circuit (capture, invoke, test, view, compare, decode).

## Data Formats

### SFT Data (JSONL)

```json
{"prompt": "What is 2+2?", "response": "2+2 equals 4."}
{"prompt": "Explain gravity.", "response": "Gravity is a force..."}
```

### DPO Preference Data (JSONL)

```json
{"prompt": "Question?", "chosen": "Good answer", "rejected": "Bad answer"}
```
