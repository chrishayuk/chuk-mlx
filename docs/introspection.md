# Introspection Tools

Mechanistic interpretability toolkit for understanding LLM behavior on MLX.

## Overview

The introspection module provides tools for:
- **Logit Lens** - See what the model "thinks" at each layer
- **Token Evolution** - Track specific tokens through the model
- **Entropy & Divergence** - Measure model confidence and layer-to-layer changes
- **Ablation Studies** - Identify causal circuits by zeroing components
- **Activation Steering** - Modify behavior via activation addition
- **Circuit Analysis** - Full pipeline for interpretability research

## CLI Tools

| Command | Description | Documentation |
|---------|-------------|---------------|
| `lazarus introspect analyze` | Logit lens analysis | [introspect-analyze.md](tools/introspect-analyze.md) |
| `lazarus introspect compare` | Compare two models | [introspect-compare.md](tools/introspect-compare.md) |
| `lazarus introspect ablate` | Ablation studies | [introspect-ablate.md](tools/introspect-ablate.md) |
| `lazarus introspect probe` | Linear probes & direction extraction | [introspect-probe.md](tools/introspect-probe.md) |
| `lazarus introspect neurons` | Analyze individual neuron activations | [introspect-neurons.md](tools/introspect-neurons.md) |
| `lazarus introspect directions` | Compare directions for orthogonality | [introspect-directions.md](tools/introspect-directions.md) |
| `lazarus introspect steer` | Activation steering | [introspect-steer.md](tools/introspect-steer.md) |
| `lazarus introspect hooks` | Low-level hook demo | [introspect-hooks.md](tools/introspect-hooks.md) |
| `lazarus introspect weight-diff` | Weight comparison | [introspect-weight-diff.md](tools/introspect-weight-diff.md) |
| `lazarus introspect activation-diff` | Activation comparison | [introspect-activation-diff.md](tools/introspect-activation-diff.md) |
| `lazarus introspect layer` | Layer representation analysis | [introspect-layer.md](tools/introspect-layer.md) |
| `lazarus introspect format-sensitivity` | Format sensitivity check | [introspect-format-sensitivity.md](tools/introspect-format-sensitivity.md) |
| `lazarus introspect generate` | Multi-token generation test | [introspect-generate.md](tools/introspect-generate.md) |

### Circuit CLI

For batch operations and mechanistic interpretability research, use the standalone `circuit` CLI:

| Command | Description |
|---------|-------------|
| `circuit dataset` | Create/manage labeled prompt datasets |
| `circuit collect` | Collect activations from models |
| `circuit analyze` | Analyze activation geometry (PCA, UMAP) |
| `circuit directions` | Extract steering directions |
| `circuit visualize` | Create visualizations |
| `circuit probes` | Run probe batteries for stratigraphy |

See [circuit-cli.md](tools/circuit-cli.md) for full documentation.

## Quick Start

```bash
# Basic logit lens analysis
lazarus introspect analyze -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 -p "The capital of France is"

# Capture all layers
lazarus introspect analyze -m model -p "Hello world" --all-layers

# Track specific tokens across layers
lazarus introspect analyze -m model -p "2 + 2 =" -t " 4" -t "4"

# Compare two models
lazarus introspect compare -m1 base-model -m2 finetuned-model -p "prompt"

# Run ablation study - sweep layers individually
lazarus introspect ablate -m model -p "What's the weather?" -c function_call --layers 8-15

# Multi-layer ablation - test layers together
lazarus introspect ablate -m model -p "45 * 45 = " -c "2025" --layers 22,23 --multi

# Multi-prompt difficulty gradient - find differential causality
lazarus introspect ablate -m model -p x -c x \
    --prompts "10*10=:100|45*45=:2025|47*47=:2209" --layers 20-23

# Layer representation analysis
lazarus introspect layer -m model -p "100 - 37 = |100 - 37 =" -l working,broken --layers 2,4,6,8

# Format sensitivity check
lazarus introspect format-sensitivity -m model -p "156 + 287 =|50 + 25 ="

# Activation steering - extract direction
lazarus introspect steer -m model --extract --positive "100 - 37 = " --negative "100 - 37 =" -o direction.npz

# Activation steering - apply direction
lazarus introspect steer -m model -d direction.npz -p "50 + 25 =" --compare "-2,-1,0,1,2"

# Multi-token generation with/without chat template
lazarus introspect generate -m model -p "2 + 2 =" --compare-format --show-tokens
lazarus introspect generate -m model -p "2 + 2 =" --raw  # Skip chat template
```

### Python API

```python
from chuk_lazarus.introspection import ModelAnalyzer, AnalysisConfig, LayerStrategy

async with ModelAnalyzer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0") as analyzer:
    result = await analyzer.analyze("The capital of France is")

    print(f"Predicted: {result.predicted_token}")
    print(f"Probability: {result.predicted_probability:.4f}")

    for layer in result.layer_predictions:
        print(f"Layer {layer.layer_idx}: {layer.top_token} ({layer.top_probability:.4f})")
```

## CLI Commands

### `lazarus introspect analyze`

Run logit lens analysis to see how predictions evolve through the model.

```bash
lazarus introspect analyze -m MODEL -p PROMPT [OPTIONS]
```

**Required:**
- `-m, --model MODEL` - HuggingFace model ID or local path
- `-p, --prompt PROMPT` - Prompt to analyze

**Options:**
- `--all-layers` - Capture all layers (default: evenly spaced)
- `-s, --layer-step N` - Capture every Nth layer (default: 4)
- `--layer-strategy {all,evenly_spaced,first_last,custom}` - Layer selection
- `-t, --track TOKEN` - Track token probability across layers (repeatable)
- `--top-k N` - Show top N predictions (default: 5)
- `--embedding-scale FLOAT` - Manual embedding scale (auto-detected for Gemma)
- `-o, --output FILE` - Save results to JSON

**Example Output:**
```
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  Family: llama
  Layers: 22
  Hidden size: 2048
  Vocab size: 32000
  Tied embeddings: True

Analyzing: 'The capital of France is'

Tokens (7): ['<s>', 'The', 'capital', 'of', 'France', 'is']
Captured layers: [0, 4, 8, 12, 16, 20, 21]

=== Final Prediction ===
  0.8234 ######################################### ' Paris'
  0.0312 # ' the'

=== Logit Lens (top prediction at each layer) ===
  Layer  0: 'the' (0.0523)
  Layer  4: 'France' (0.0891)
  Layer  8: 'Paris' (0.2341)
  Layer 12: ' Paris' (0.5672)
  Layer 16: ' Paris' (0.7891)
  Layer 20: ' Paris' (0.8123)
  Layer 21: ' Paris' (0.8234)

=== Token Evolution ===
Token ' Paris':
  Layer  0: 0.0012  (rank 847)
  Layer  4: 0.0234 ## (rank 23)
  Layer  8: 0.2341 ####################### (rank 2)
  Layer 12: 0.5672 ######################################################## (rank 1)
  --> Becomes top-1 at layer 12
```

### `lazarus introspect compare`

Compare how two models process the same prompt.

```bash
lazarus introspect compare -m1 MODEL1 -m2 MODEL2 -p PROMPT [OPTIONS]
```

### `lazarus introspect ablate`

Run ablation studies to identify causal circuits. Supports single-layer sweeps and multi-layer ablation.

```bash
lazarus introspect ablate -m MODEL -p PROMPT -c CRITERION [OPTIONS]
```

**Built-in criteria:** `function_call`, `sorry`, `positive`, `negative`, `refusal`, or any substring.

**Options:**
- `--component {mlp,attention,both}` - Component to ablate
- `--layers L1,L2,...` - Specific layers to test (supports ranges like `20-23`)
- `--multi` - Ablate all specified layers together (default: sweep individually)
- `--prompts` - Multiple prompts with expected outputs (e.g., `"10*10=:100|45*45=:2025"`)
- `--raw` - Skip chat template
- `--max-tokens N` - Max tokens to generate

**Example: Arithmetic Circuit Discovery**

```bash
# Find which layer gates "hard" arithmetic
lazarus introspect ablate -m openai/gpt-oss-20b \
    -p x -c x \
    --prompts "10*10=:100|45*45=:2025|47*47=:2209" \
    --layers 20-23
```

### `lazarus introspect hooks`

Low-level demonstration of activation hooks.

```bash
lazarus introspect hooks -m MODEL -p PROMPT [OPTIONS]
```

### `lazarus introspect weight-diff`

Compare weight differences between base and fine-tuned models.

```bash
lazarus introspect weight-diff -b BASE -f FINETUNED [-o OUTPUT]
```

### `lazarus introspect activation-diff`

Compare activation divergence between models.

```bash
lazarus introspect activation-diff -b BASE -f FINETUNED -p PROMPTS [-o OUTPUT]
```

## Logit Lens

The "logit lens" technique projects intermediate hidden states to vocabulary logits, revealing how the model's predictions develop through its depth.

### Key Insight

For function-calling models, if the correct tool token appears early (layers 2-4), a shallow model might suffice. If it only emerges in final layers, depth is necessary.

### Layer Selection Strategies

```python
from chuk_lazarus.introspection import AnalysisConfig, LayerStrategy

# Capture all layers (most detailed, highest memory)
config = AnalysisConfig(layer_strategy=LayerStrategy.ALL)

# Every Nth layer (balanced)
config = AnalysisConfig(layer_strategy=LayerStrategy.EVENLY_SPACED, layer_step=4)

# Only first and last (minimal)
config = AnalysisConfig(layer_strategy=LayerStrategy.FIRST_LAST)

# Custom layer selection
config = AnalysisConfig(
    layer_strategy=LayerStrategy.CUSTOM,
    custom_layers=[0, 4, 8, 12, 16, 20, 25]
)
```

## Token Evolution

Track how a specific token's probability changes across layers:

```python
from chuk_lazarus.introspection import AnalysisConfig

config = AnalysisConfig(track_tokens=["Paris", " Paris", "London"])

async with ModelAnalyzer.from_pretrained("model-id") as analyzer:
    result = await analyzer.analyze("The capital of France is", config)

    for evo in result.token_evolutions:
        print(f"\nToken '{evo.token}':")
        for layer, prob in evo.layer_probabilities.items():
            rank = evo.layer_ranks.get(layer)
            print(f"  Layer {layer}: prob={prob:.4f}, rank={rank}")

        if evo.emergence_layer:
            print(f"  -> Becomes top-1 at layer {evo.emergence_layer}")
```

## Entropy & Divergence Analysis

### Layer Entropy

```python
async with ModelAnalyzer.from_pretrained("model-id") as analyzer:
    result = await analyzer.analyze("The answer is")

    for pred in result.layer_predictions:
        print(f"Layer {pred.layer_idx}:")
        print(f"  Entropy: {pred.entropy:.3f}")
        print(f"  Is confident: {pred.is_confident}")  # normalized < 0.3
```

### Layer Transitions (KL & JS Divergence)

```python
result = await analyzer.analyze("The answer is")

# Find where the most computation happens
if result.max_kl_transition:
    t = result.max_kl_transition
    print(f"Max KL divergence: {t.kl_divergence:.3f}")
    print(f"  Between layers {t.from_layer} -> {t.to_layer}")

# Find the decision layer (where model becomes confident)
if result.decision_layer is not None:
    print(f"Decision made at layer: {result.decision_layer}")
```

## Low-Level Hooks API

For fine-grained control:

```python
from chuk_lazarus.introspection import ModelHooks, CaptureConfig, LogitLens, PositionSelection
from mlx_lm import load
import mlx.core as mx

model, tokenizer = load("model-id")

hooks = ModelHooks(model)
hooks.configure(CaptureConfig(
    layers=[0, 4, 8, 12, 16],
    capture_hidden_states=True,
    capture_attention_weights=True,
    positions=PositionSelection.LAST,
))

input_ids = mx.array(tokenizer.encode("Hello world"))[None, :]
logits = hooks.forward(input_ids)

# Access captured states
for layer_idx, hidden in hooks.state.hidden_states.items():
    print(f"Layer {layer_idx}: shape {hidden.shape}")

# Use logit lens
lens = LogitLens(hooks, tokenizer)
lens.print_evolution(position=-1, top_k=5)
```

## Ablation Studies

Identify causal circuits by zeroing model components. Ablation is the key **causal** test - if ablating layer N breaks behavior X, then layer N is causally involved in X.

### CLI Usage

```bash
# Sweep layers individually
lazarus introspect ablate -m model -p "What's the weather?" -c function_call --layers 8-15

# Ablate multiple layers together
lazarus introspect ablate -m model -p "45 * 45 = " -c "2025" --layers 22,23 --multi

# Multi-prompt difficulty gradient
lazarus introspect ablate -m openai/gpt-oss-20b \
    -p x -c x \
    --prompts "10*10=:100|45*45=:2025|47*47=:2209" \
    --layers 20-23
```

### Python API

```python
from chuk_lazarus.introspection import AblationStudy, AblationConfig, ComponentType

study = AblationStudy.from_pretrained(model_id)

# Single-layer sweep
def criterion(output: str) -> bool:
    return "<function_call>" in output

result = study.run_layer_sweep(
    prompt="What's the weather?",
    criterion=criterion,
    component=ComponentType.MLP,
    layers=range(8, 15),
)
study.print_sweep_summary(result)

# Multi-layer ablation
config = AblationConfig(max_new_tokens=15)
original = study.ablate_and_generate("45 * 45 = ", layers=[], config=config)
ablated = study.ablate_and_generate("45 * 45 = ", layers=[22, 23], config=config)
print(f"Original: {original}")
print(f"Ablated L22+L23: {ablated}")
```

### Differential Causality

Test prompts of varying difficulty to find layers that gate specific capabilities:

| Result | Interpretation |
|--------|----------------|
| Trivial works, hard breaks | Layer gates access to complex computation |
| All break | Layer is load-bearing for the entire capability |
| None break | Layer is epiphenomenal (correlate, not cause) |

### Example: Arithmetic Circuit in GPT-OSS 20B

| Ablation | 10×10 (trivial) | 45×45 (factorizable) | 47×47 (prime) |
|----------|-----------------|----------------------|---------------|
| None | ✓ 100 | ✓ 2025 | ✓ 2209 |
| L22 only | ✓ 100 | ✓ 2025 | ✗ 2,209 |
| L22,23 | ✓ 100 | ✗ 2.025e6 | ✗ 2,209 |
| L21,22,23 | ✓ 1000 (wrong!) | ✗ 1.5×45 | ✗ 2.47×47 |

**Finding:** L22 is the "hard problem gate" - only breaks prime multiplication. L22-23 together are the full arithmetic gate.

## Circuit Analysis CLI

For deeper mechanistic interpretability research:

```bash
# Create labeled dataset
circuit dataset create -o prompts.json --per-tool 50

# Collect activations
circuit collect -m model -d prompts.json -o activations

# Analyze geometry (PCA, probes)
circuit analyze -a activations.safetensors --layer 11

# Extract directions
circuit directions -a activations.safetensors --layer 11 -o directions

# Visualize
circuit visualize -a activations.safetensors --all -o plots/

# Run probe battery
circuit probes run -m model
```

See `src/chuk_lazarus/introspection/circuit/README.md` for full documentation.

## Model-Specific Notes

### Gemma Models

Gemma models scale embeddings by `sqrt(hidden_size)`. The analyzer handles this automatically, but you can override:

```python
async with ModelAnalyzer.from_pretrained(
    "gemma-model",
    embedding_scale=33.94  # sqrt(1152) for 270M model
) as analyzer:
    result = await analyzer.analyze("Hello")
```

### Quantized Models

Quantized models (4-bit, 8-bit) are loaded via mlx_lm. Logit lens still works but hidden states are in the quantized representation.

## Architecture

```
src/chuk_lazarus/introspection/
├── __init__.py          # Public API exports
├── hooks.py             # Capture intermediate activations
├── logit_lens.py        # Layer-by-layer prediction analysis
├── analyzer.py          # Async-native ModelAnalyzer API
├── ablation.py          # Ablation studies
├── attention.py         # Attention pattern analysis
├── steering.py          # Activation steering
├── circuit/             # Circuit analysis toolkit
│   ├── dataset.py       # Labeled prompt datasets
│   ├── collector.py     # Activation collection
│   ├── geometry.py      # PCA, UMAP, linear probes
│   ├── directions.py    # Direction extraction
│   ├── probes.py        # Stratigraphy probe batteries
│   └── cli.py           # Standalone circuit CLI
└── visualizers/
    ├── attention_heatmap.py
    └── logit_evolution.py
```

## Examples

Examples are organized in `examples/introspection/`:

```
examples/introspection/
├── demos/              # Tool usage examples
│   ├── logit_lens.py          # Async analyzer demo
│   ├── low_level_hooks.py     # Direct hook usage
│   └── circuit_analysis.py    # Circuit pipeline walkthrough
└── experiments/        # Research experiments
    ├── circuits/       # Circuit discovery
    ├── ablation/       # Ablation studies
    ├── comparison/     # Model comparisons
    ├── steering/       # Activation steering
    ├── probing/        # Linear probes
    ├── layers/         # Layer analysis
    └── model_specific/ # Model-specific experiments
```

## API Reference

### ModelAnalyzer

| Method | Description |
|--------|-------------|
| `from_pretrained(model_id)` | Load model and create analyzer (async context manager) |
| `analyze(prompt, config)` | Run logit lens analysis |
| `model_info` | Get model information |
| `config` | Get model config (includes family, embedding_scale) |

### AnalysisConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `layer_strategy` | `LayerStrategy` | `EVENLY_SPACED` | How to select layers |
| `layer_step` | `int` | `4` | Step for evenly spaced |
| `top_k` | `int` | `5` | Number of top predictions |
| `track_tokens` | `list[str]` | `[]` | Tokens to track |
| `compute_entropy` | `bool` | `True` | Compute entropy for each layer |
| `compute_transitions` | `bool` | `True` | Compute KL/JS divergence |

### AnalysisResult

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | `str` | The analyzed prompt |
| `tokens` | `list[str]` | Tokenized prompt |
| `num_layers` | `int` | Total model layers |
| `captured_layers` | `list[int]` | Captured layer indices |
| `final_prediction` | `list[TokenPrediction]` | Top-k final predictions |
| `layer_predictions` | `list[LayerPredictionResult]` | Predictions at each layer |
| `token_evolutions` | `list[TokenEvolutionResult]` | Tracked token evolution |
| `predicted_token` | `str` | Top predicted token (property) |
| `decision_layer` | `int \| None` | Layer where model becomes confident |

## See Also

- `src/chuk_lazarus/introspection/README.md` - Module documentation
- `src/chuk_lazarus/introspection/circuit/README.md` - Circuit CLI documentation
- `docs/gemma_alignment_circuits.md` - Gemma circuit findings
- `docs/tool_calling_circuit.md` - Tool-calling circuit analysis
