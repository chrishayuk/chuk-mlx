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
- **MoE Introspection** - Expert identification, routing analysis, and expert ablation for MoE models

## CLI Tools

### Core Analysis

| Command | Description | Documentation |
|---------|-------------|---------------|
| `lazarus introspect analyze` | Logit lens analysis | [introspect-analyze.md](tools/introspect-analyze.md) |
| `lazarus introspect compare` | Compare two models | [introspect-compare.md](tools/introspect-compare.md) |
| `lazarus introspect generate` | Multi-token generation test | [introspect-generate.md](tools/introspect-generate.md) |
| `lazarus introspect hooks` | Low-level hook demo | [introspect-hooks.md](tools/introspect-hooks.md) |

### Probing & Directions

| Command | Description | Documentation |
|---------|-------------|---------------|
| `lazarus introspect probe` | Linear probes & direction extraction | [introspect-probe.md](tools/introspect-probe.md) |
| `lazarus introspect neurons` | Analyze individual neuron activations | [introspect-neurons.md](tools/introspect-neurons.md) |
| `lazarus introspect directions` | Compare directions for orthogonality | [introspect-directions.md](tools/introspect-directions.md) |
| `lazarus introspect operand-directions` | Analyze operand A/B encoding structure | [introspect-operand-directions.md](tools/introspect-operand-directions.md) |
| `lazarus introspect embedding` | Test what's encoded at embedding level | [introspect-embedding.md](tools/introspect-embedding.md) |
| `lazarus introspect early-layers` | Analyze early layer information encoding | [introspect-early-layers.md](tools/introspect-early-layers.md) |
| `lazarus introspect activation-cluster` | Visualize activation clusters using PCA | [introspect-activation-cluster.md](tools/introspect-activation-cluster.md) |

### Steering & Intervention

| Command | Description | Documentation |
|---------|-------------|---------------|
| `lazarus introspect steer` | Activation steering | [introspect-steer.md](tools/introspect-steer.md) |
| `lazarus introspect ablate` | Ablation studies | [introspect-ablate.md](tools/introspect-ablate.md) |
| `lazarus introspect patch` | Activation patching between prompts | [introspect-patch.md](tools/introspect-patch.md) |

### Model Comparison

| Command | Description | Documentation |
|---------|-------------|---------------|
| `lazarus introspect weight-diff` | Weight comparison | [introspect-weight-diff.md](tools/introspect-weight-diff.md) |
| `lazarus introspect activation-diff` | Activation comparison | [introspect-activation-diff.md](tools/introspect-activation-diff.md) |
| `lazarus introspect layer` | Layer representation analysis | [introspect-layer.md](tools/introspect-layer.md) |
| `lazarus introspect format-sensitivity` | Format sensitivity check | [introspect-format-sensitivity.md](tools/introspect-format-sensitivity.md) |

### Classifier Emergence

| Command | Description | Documentation |
|---------|-------------|---------------|
| `lazarus introspect classifier` | Multi-class linear probes for operation classification | [introspect-classifier.md](tools/introspect-classifier.md) |
| `lazarus introspect logit-lens` | Check if classifiers project to vocabulary tokens | [introspect-logit-lens.md](tools/introspect-logit-lens.md) |
| `lazarus introspect dual-reward` | Train V/O projections for classifier + answer | [introspect-dual-reward.md](tools/introspect-dual-reward.md) |

### Arithmetic & Lookup Table Analysis

| Command | Description | Documentation |
|---------|-------------|---------------|
| `lazarus introspect arithmetic` | Systematic arithmetic study to find emergence layers | [introspect-arithmetic.md](tools/introspect-arithmetic.md) |
| `lazarus introspect commutativity` | Test if A*B and B*A have identical representations | [introspect-commutativity.md](tools/introspect-commutativity.md) |
| `lazarus introspect metacognitive` | Detect strategy switch (direct vs chain-of-thought) | [introspect-metacognitive.md](tools/introspect-metacognitive.md) |
| `lazarus introspect uncertainty` | Predict model confidence using hidden state geometry | [introspect-uncertainty.md](tools/introspect-uncertainty.md) |

### Memory Structure

| Command | Description | Documentation |
|---------|-------------|---------------|
| `lazarus introspect memory` | Extract memory organization structure for facts | [introspect-memory.md](tools/introspect-memory.md) |
| `lazarus introspect memory-inject` | External memory injection for fact retrieval | [introspect-memory.md](tools/introspect-memory.md) |

### MoE Expert Commands

#### Interactive & Exploration

| Command | Description |
|---------|-------------|
| `lazarus introspect moe-expert explore` | Interactive REPL for exploring expert routing in real-time |
| `lazarus introspect moe-expert chat` | Force routing to a specific expert |
| `lazarus introspect moe-expert interactive` | Interactive expert explorer REPL (legacy) |

#### Analysis & Hypothesis Testing

| Command | Description |
|---------|-------------|
| `lazarus introspect moe-expert domain-test` | Test if domain experts exist (demonstrates they don't) |
| `lazarus introspect moe-expert token-routing` | Test if single tokens have stable routing (demonstrates context-dependence) |
| `lazarus introspect moe-expert full-taxonomy` | Semantic trigram pattern analysis across categories |
| `lazarus introspect moe-expert analyze` | Identify expert specializations across categories |

#### Routing Visualization

| Command | Description |
|---------|-------------|
| `lazarus introspect moe-expert trace` | Trace expert routing across ALL layers |
| `lazarus introspect moe-expert weights` | Show router weights for a prompt |
| `lazarus introspect moe-expert heatmap` | Generate routing heatmap visualization |
| `lazarus introspect moe-expert entropy` | Analyze routing entropy (confidence) by layer |

#### Expert Comparison & Ablation

| Command | Description |
|---------|-------------|
| `lazarus introspect moe-expert compare` | Compare multiple experts on the same prompt |
| `lazarus introspect moe-expert ablate` | Remove an expert and see what breaks |
| `lazarus introspect moe-expert topk` | Experiment with different top-k values |
| `lazarus introspect moe-expert collab` | Analyze expert co-activation patterns |
| `lazarus introspect moe-expert pairs` | Test specific expert pairs/groups together |

#### Advanced Analysis

| Command | Description |
|---------|-------------|
| `lazarus introspect moe-expert layer-sweep` | Sweep all layers, analyze expert patterns |
| `lazarus introspect moe-expert pipeline` | Track expert pipelines across layers |
| `lazarus introspect moe-expert vocab-contrib` | Analyze expert vocabulary contributions |
| `lazarus introspect moe-expert compression` | Analyze compression opportunities |

#### Quick Start: Video Demo Workflow

```bash
# 1. Show that "domain experts" don't exist
lazarus introspect moe-expert domain-test -m openai/gpt-oss-20b

# 2. Show that single token routing is context-dependent
lazarus introspect moe-expert token-routing -m openai/gpt-oss-20b --token 127

# 3. Show the semantic trigram breakthrough
lazarus introspect moe-expert full-taxonomy -m openai/gpt-oss-20b --categories arithmetic,analogy

# 4. Interactive exploration
lazarus introspect moe-expert explore -m openai/gpt-oss-20b
# Then type: King is to queen as man is to woman
# Compare: c "2 + 3 = 5"
```

### Circuit Commands

| Command | Description |
|---------|-------------|
| `lazarus introspect circuit capture` | Capture circuit activations for known computations |
| `lazarus introspect circuit invoke` | Invoke circuit with new operands (interpolate/steer) |
| `lazarus introspect circuit test` | Test circuit generalization on novel inputs |
| `lazarus introspect circuit view` | View captured circuit contents |
| `lazarus introspect circuit compare` | Compare multiple circuits for similarity |
| `lazarus introspect circuit decode` | Decode circuit activations by injection |
| `lazarus introspect circuit export` | Export circuit graph to DOT/JSON/Mermaid/HTML |

### Standalone Circuit CLI

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

### `lazarus introspect arithmetic`

Run systematic arithmetic study to find where correct answers emerge across layers.

```bash
lazarus introspect arithmetic -m MODEL [OPTIONS]
```

**Options:**
- `--quick` - Run subset of tests
- `--easy-only` - Only 1-digit problems
- `--hard-only` - Only 3-digit problems
- `--raw` - Skip chat template
- `-o, --output` - Save results to JSON

### `lazarus introspect metacognitive`

Detect strategy switch at the "decision layer" (~70% depth). Shows whether the model will output an answer directly or use chain-of-thought reasoning.

```bash
lazarus introspect metacognitive -m MODEL -p PROBLEMS [OPTIONS]
```

**Options:**
- `-p, --problems` - Arithmetic problems (pipe-separated or @file.txt)
- `-g, --generate` - Auto-generate random problems
- `-n, --num-problems` - Number of problems to generate (default: 20)
- `-l, --decision-layer` - Layer to probe (default: ~70% of model depth)
- `--raw` - Skip chat template

### `lazarus introspect uncertainty`

Predict model confidence using hidden state geometry before generation.

```bash
lazarus introspect uncertainty -m MODEL -p PROMPTS [OPTIONS]
```

Uses distance to "compute center" vs "refusal center" to predict output behavior.

**Options:**
- `-l, --layer` - Detection layer (default: ~70% depth)
- `-w, --working` - Working examples for calibration
- `-b, --broken` - Broken examples for calibration

### `lazarus introspect cluster`

Visualize how prompts cluster in activation space using PCA.

```bash
lazarus introspect cluster -m MODEL [OPTIONS]
```

Supports multi-class syntax:
```bash
lazarus introspect cluster -m model \
    --prompts "45*45=|25*25=" --label mult \
    --prompts "123+456=|100+37=" --label add \
    --prompts "Capital of France is" --label language \
    --layer 19 --save-plot cluster.png
```

### `lazarus introspect operand-directions`

Analyze how operands A and B are encoded in activation space. Tests whether the model uses compositional encoding (separate orthogonal subspaces for A and B, like GPT-OSS) or holistic encoding (entire expression encoded together, like Gemma).

```bash
lazarus introspect operand-directions -m MODEL [OPTIONS]
```

**Options:**
- `--digits` - Digits to use (comma-separated, default: 2,3,4,5,6,7,8,9)
- `--operation` - Operation to test (default: `*`)
- `--layers` - Layers to analyze (comma-separated, default: auto key layers)
- `-o, --output` - Save results (.json or .npz)

**Example:**
```bash
# Analyze multiplication operand encoding
lazarus introspect operand-directions -m mlx-community/gemma-3-4b-it-bf16 \
    --digits 2,3,4,5,6,7,8,9 --operation "*" --layers 8,16,20,24

# Quick check
lazarus introspect operand-directions -m model
```

**Key output metrics:**
- A_i vs A_j: If low (<0.5), distinct operand directions (compositional)
- A_i vs B_j: If low (<0.3), orthogonal subspaces
- A_i vs B_i: If high (>0.8), digit identity dominates position

### `lazarus introspect embedding`

Test what information is encoded at the embedding level vs after layer computation. This tests the RLVF backprop hypothesis: if task type is 100% detectable from raw embeddings, RLVF gradients backpropagate to the embedding layer.

```bash
lazarus introspect embedding -m MODEL [OPTIONS]
```

**Options:**
- `--operation` - Operation type: `mult`, `add`, `all`, `*`, `+` (default: all)
- `--layers` - Layers to compare against embeddings (comma-separated, default: 0,1,2)
- `-o, --output` - Save results to JSON

**Example:**
```bash
# Full embedding analysis
lazarus introspect embedding -m mlx-community/gemma-3-4b-it-bf16

# Test specific operation
lazarus introspect embedding -m model --operation mult
```

**Key output:**
- Task type from embeddings: If 100%, RLVF backprop confirmed
- Answer R² from embeddings: Should be low (computation required)

### `lazarus introspect commutativity`

Test if the model's internal representations respect commutativity (A*B = B*A). High similarity (>0.99) between commutative pairs suggests a lookup table structure rather than an algorithm.

```bash
lazarus introspect commutativity -m MODEL [OPTIONS]
```

**Options:**
- `--pairs` - Explicit pairs to test (e.g., `"2*3,3*2|7*8,8*7"`)
- `-l, --layer` - Layer to analyze (default: ~60% of model depth)
- `-o, --output` - Save results to JSON

**Example:**
```bash
# Test all commutative pairs (2-9)
lazarus introspect commutativity -m model

# Test specific pairs
lazarus introspect commutativity -m model \
    --pairs "2*3,3*2|7*8,8*7|4*5,5*4" --layer 20
```

**Interpretation:**
- Mean similarity >0.999: Strong evidence for lookup table (memorization)
- Mean similarity <0.9: Model may use different algorithms for A*B vs B*A

### `lazarus introspect patch`

Perform activation patching: transfer activations from a source prompt to a target prompt. This is a causal intervention technique to test whether specific layers encode "the answer" vs "the operands".

```bash
lazarus introspect patch -m MODEL --source SOURCE --target TARGET [OPTIONS]
```

**Options:**
- `-s, --source` - Source prompt to patch FROM (required)
- `-t, --target` - Target prompt to patch INTO (required)
- `-l, --layer` - Single layer to patch at
- `--layers` - Multiple layers to sweep (comma-separated)
- `--blend` - Blend factor: 0=no change, 1=full replacement (default: 1.0)
- `-n, --max-tokens` - Max tokens to generate (default: 10)
- `-o, --output` - Save results to JSON

**Example:**
```bash
# Patch multiplication into addition
lazarus introspect patch -m model --source "7*8=" --target "7+8="

# Patch at specific layers
lazarus introspect patch -m model \
    --source "7*8=" --target "7+8=" \
    --layers 0,8,16,20,24,28
```

**Key output:**
- "TRANSFERRED!": Source answer produced at this layer
- "no change": Patching had no effect at this layer

### `lazarus introspect memory`

Extract how facts are organized in model memory by analyzing neighborhood activation patterns.

```bash
lazarus introspect memory -m MODEL --facts FACT_TYPE --layer LAYER [OPTIONS]
```

**Built-in fact types:** `multiplication`, `addition`, `capitals`, `elements`, or `@file.json`

**Options:**
- `--top-k N` - Number of top predictions per query (default: 30)
- `--save-plot FILE` - Save visualization
- `--classify` - Show memorization classification

**Example:**
```bash
lazarus introspect memory -m model --facts multiplication --layer 20
```

### `lazarus introspect memory-inject`

External memory injection for fact retrieval. Injects correct answers from an external store based on query similarity.

```bash
lazarus introspect memory-inject -m MODEL --facts FACT_TYPE --query QUERY [OPTIONS]
```

**Options:**
- `--query-layer` - Layer for query matching (default: ~92% depth)
- `--inject-layer` - Layer to inject values (default: ~88% depth)
- `--blend FLOAT` - Blend factor: 0=no injection, 1=full replacement
- `--threshold FLOAT` - Minimum similarity for injection (default: 0.7)
- `--force` - Force injection even if below threshold
- `--evaluate` - Evaluate baseline vs injected accuracy

### `lazarus introspect circuit`

Direct circuit capture, interpolation, and invocation.

**Subcommands:**

#### `circuit capture`
Capture circuit activations for known computations.

```bash
lazarus introspect circuit capture -m MODEL \
    --prompts "7*4=|6*8=|9*3=" --results "28|48|27" \
    --layer 19 --save circuit.npz
```

**Options:**
- `--extract-direction` - Extract linear direction for steering
- `--generate` - Generate multiplication table automatically

#### `circuit invoke`
Invoke circuit with new operands using interpolation or steering.

```bash
# Using steering (most accurate, requires --extract-direction during capture)
lazarus introspect circuit invoke -m model \
    -c mult_circuit.npz --prompts "5*6=|8*9=" --method steer

# Using linear interpolation (no model needed)
lazarus introspect circuit invoke \
    -c mult_circuit.npz --operands "5,6|8,9" --method linear
```

#### `circuit test`
Test circuit generalization on novel inputs.

```bash
lazarus introspect circuit test \
    -c mult_circuit.npz -m model \
    -p "12*13=|25*4=|11*11=" -r "156|100|121"
```

#### `circuit view`
View captured circuit contents.

```bash
lazarus introspect circuit view -c circuit.npz --stats --table
```

#### `circuit compare`
Compare multiple circuits for similarity.

```bash
lazarus introspect circuit compare \
    -c mult_circuit.npz add_circuit.npz sub_circuit.npz
```

Shows cosine similarity matrix, angles between circuits, and shared neurons.

#### `circuit decode`
Decode circuit activations by injecting them during generation.

```bash
lazarus introspect circuit decode -m model \
    --inject circuit.npz --prompt "5 * 6 =" --blend 1.0
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

## MoE Introspection

Mixture of Experts (MoE) models route tokens to different expert networks. Understanding which experts specialize in what content is crucial for interpretability.

### Supported Models

| Model | Experts | Active | Notes |
|-------|---------|--------|-------|
| GPT-OSS 20B | 32 | 4 | OpenAI's first open MoE |
| Mixtral 8x7B | 8 | 2 | Mistral's MoE |
| Llama 4 Scout | 16 | 1 + shared | Hybrid Mamba-Transformer |
| Granite 4.0 MoE | varies | varies | IBM's MoE variants |

### Expert Identification

Discover what each expert specializes in:

```python
from mlx_lm import load
from chuk_lazarus.introspection import (
    MoEHooks, identify_all_experts, print_expert_summary
)

# Load model
model, tokenizer = load("openai/gpt-oss-20b")

# Create hooks and identify all experts in layer 12
hooks = MoEHooks(model)
identities = identify_all_experts(hooks, layer_idx=12, tokenizer=tokenizer)
print_expert_summary(identities)

# Output:
# Expert Identification: gpt_oss
# Layer 12 (32 experts)
# ============================================================
# CODE: Experts [1, 14, 22, 23, 27, 28]
# MATH: Experts [6, 7, 19, 24, 30, 31]
# CONTENT_WORDS: Experts [0, 2, 3, 4, 5, 8, 9, ...]
# NAMES: Experts [15, 26]
#
# Specialists (focused): [6, 19]
# Generalists (diverse): [21, 5]
#
# Redundant pairs (high similarity):
#   Experts 0 & 5: 72.61% similar
#   Experts 6 & 19: 70.30% similar
```

### Expert Identity Details

Get detailed information about a specific expert:

```python
# Get identity for expert 6 (math specialist)
expert_6 = result.expert_identities[6]
print(expert_6.detailed_report())

# Output:
# Expert 6 Identity Report (Layer 12)
# ==================================================
# Primary Category: math (52% confidence)
#
# Category Breakdown:
#   math                 ██████████░░░░░░░░░░ 52%
#   code                 ████░░░░░░░░░░░░░░░░ 20%
#   content_words        ███░░░░░░░░░░░░░░░░░ 15%
#   ...
#
# Activations: 108 total, 53 unique tokens
# Token Entropy: 3.60 (lower = more specialized)
# Positional Bias: uniform
# Context Sensitivity: 0.45
#
# Top Tokens:
#   ' +': 8
#   '2': 7
#   'x': 6
#   '3': 5
#
# Semantic Clusters:
#   - numeric_values
```

### ExpertIdentity Fields

| Field | Description |
|-------|-------------|
| `expert_idx` | Expert index in the layer |
| `layer_idx` | Layer index |
| `primary_category` | Main specialization (CODE, MATH, LANGUAGE, etc.) |
| `category_confidence` | Confidence in primary category (0-1) |
| `category_scores` | Breakdown by all categories |
| `total_activations` | Number of tokens routed to this expert |
| `unique_tokens` | Number of unique tokens |
| `top_tokens` | Most common tokens (with counts) |
| `token_entropy` | Diversity of tokens (higher = more diverse) |
| `positional_bias` | "early", "middle", "late", or "uniform" |
| `context_sensitivity` | How much routing depends on context (0-1) |
| `semantic_clusters` | Detected clusters: "python_keywords", "json_syntax", etc. |

### Expert Categories

| Category | Description | Example Tokens |
|----------|-------------|----------------|
| `CODE` | Programming keywords/symbols | `def`, `class`, `=`, `{}` |
| `MATH` | Mathematical expressions | `+`, `-`, `²`, digits |
| `LANGUAGE` | Natural language | General prose |
| `PUNCTUATION` | Punctuation marks | `.`, `,`, `!`, `?` |
| `WHITESPACE` | Formatting | `\n`, `\t`, spaces |
| `NAMES` | Proper nouns/entities | "Paris", "Microsoft" |
| `FUNCTION_WORDS` | Articles/prepositions | "the", "of", "in" |
| `CONTENT_WORDS` | Nouns/verbs/adjectives | General content |

### MoE Routing Analysis

Capture and analyze routing decisions in real-time:

```python
from chuk_lazarus.introspection import MoEHooks, MoECaptureConfig

# Create hooks
hooks = MoEHooks(model)
hooks.configure(MoECaptureConfig(
    capture_router_logits=True,
    capture_router_weights=True,
    capture_selected_experts=True,
))

# Run forward pass
input_ids = mx.array([tokenizer.encode("def fibonacci(n):")])
logits = hooks.forward(input_ids)
mx.eval(logits)

# Analyze utilization
utilization = hooks.get_expert_utilization(layer_idx=12)
print(f"Load balance: {utilization.load_balance_score:.2%}")
print(f"Most used: Expert {utilization.most_used_expert}")
print(f"Least used: Expert {utilization.least_used_expert}")

# Analyze router confidence
entropy = hooks.get_router_entropy(layer_idx=12)
confidence = 1 - entropy.normalized_entropy
print(f"Router confidence: {confidence:.2%}")

# Get routing pattern for last token
pattern = hooks.get_routing_pattern(layer_idx=12, position=-1)
print(f"Selected experts: {pattern['selected_experts']}")
print(f"Routing weights: {pattern['routing_weights']}")
```

### MoECaptureConfig Options

| Option | Default | Description |
|--------|---------|-------------|
| `capture_router_logits` | `True` | Raw logits before softmax |
| `capture_router_weights` | `True` | Normalized routing weights |
| `capture_selected_experts` | `True` | Which experts were selected |
| `capture_expert_contributions` | `False` | Per-expert outputs (memory intensive) |
| `capture_shared_expert` | `True` | Shared expert output (Llama4-style) |
| `layers` | `None` | Which layers to capture (None = all MoE) |
| `detach` | `True` | Detach from computation graph |

### Expert Ablation

Test what happens when specific experts are disabled:

```python
from chuk_lazarus.introspection import ablate_expert, find_causal_experts
import mlx.core as mx

# Ablate expert 6 (math specialist) at layer 12
input_ids = mx.array(tokenizer.encode("What is 45 * 45?"))[None, :]
result = ablate_expert(
    model=model,
    layer_idx=12,
    expert_idx=6,
    input_ids=input_ids,
    tokenizer=tokenizer,
)
print(f"Baseline: {result.baseline_output}")
print(f"Ablated:  {result.ablated_output}")
print(f"Output changed: {result.output_changed}")
print(f"Would have activated: {result.would_have_activated}")

# Find all experts whose ablation changes output
causal_experts = find_causal_experts(
    model=model,
    layer_idx=12,
    input_ids=input_ids,
    tokenizer=tokenizer,
)
for r in causal_experts:
    print(f"Expert {r.expert_idx}: causal (activations={r.activation_count})")
```

### MoE Logit Lens

See how routing evolves across MoE layers:

```python
from chuk_lazarus.introspection import MoEHooks, MoECaptureConfig, MoELogitLens
import mlx.core as mx

# Setup hooks
hooks = MoEHooks(model)
hooks.configure(MoECaptureConfig(
    capture_router_logits=True,
    capture_selected_experts=True,
))

# Run forward pass
input_ids = mx.array(tokenizer.encode("def fibonacci(n):"))[None, :]
hooks.forward(input_ids)

# Create logit lens and analyze
lens = MoELogitLens(hooks, tokenizer)
snapshots = lens.get_routing_evolution(position=-1)

for snap in snapshots:
    experts = ", ".join(f"E{e}" for e in snap.selected_experts)
    print(f"Layer {snap.layer_idx}: [{experts}] entropy={snap.router_entropy:.3f}")

# Print in human-readable format
lens.print_routing_evolution()
```

### Example: GPT-OSS Expert Analysis

Real results from GPT-OSS 20B layer 12:

```
Expert Identification Summary:
==============================

CODE EXPERTS (6):
  Expert 14: spaces, operators (=, +, -)
  Expert 23: Python keywords, newlines
  Expert 27: JSON syntax (":", return)
  Expert 22: General code structure
  Expert 28: Function definitions
  Expert  1: Variable names

MATH EXPERTS (6):
  Expert  6: Arithmetic (+ 2 x 3) - 52% math confidence
  Expert 19: Algebraic expressions (x², +) - 55% math confidence
  Expert  7: Numbers and operators
  Expert 24: Mathematical notation
  Expert 30: Equations
  Expert 31: Numeric computation

LANGUAGE EXPERTS (18):
  Expert 21: Most active - punctuation, articles
  Expert  5: General content
  Expert  8: Prepositions, context words
  ...

NAME EXPERTS (2):
  Expert 26: Proper nouns ("Eiffel", "headquarters")
  Expert 15: Entity names

REDUNDANT PAIRS:
  Experts 0 & 5: 72.61% similar (both content_words)
  Experts 6 & 19: 70.30% similar (both math specialists)
```

### API Reference

#### Identification Functions

| Function | Description |
|----------|-------------|
| `identify_expert(hooks, layer_idx, expert_idx, tokenizer)` | Identify single expert |
| `identify_all_experts(hooks, layer_idx, tokenizer)` | Identify all experts in layer |
| `find_specialists(identities, category=None)` | Find specialist experts |
| `find_generalists(identities)` | Find generalist experts |
| `cluster_experts_by_specialization(identities)` | Group by primary category |
| `print_expert_summary(identities)` | Print summary report |

#### MoEHooks

| Method | Description |
|--------|-------------|
| `configure(config)` | Set capture configuration |
| `forward(input_ids)` | Run forward with capture |
| `get_expert_utilization(layer_idx)` | Get utilization stats |
| `get_router_entropy(layer_idx)` | Get entropy/confidence |
| `get_routing_pattern(layer_idx, position)` | Get routing for position |
| `compare_routing_across_layers()` | Compare routing evolution |

#### Convenience Functions

```python
from chuk_lazarus.introspection import (
    # Detection
    detect_moe_architecture,   # Detect MoE type (GPT_OSS, MIXTRAL, LLAMA4, etc.)
    get_moe_layer_info,        # Get layer info
    get_moe_layers,            # Get indices of MoE layers
    is_moe_model,              # Check if model has MoE
    # Identification
    identify_expert,           # Identify single expert
    identify_all_experts,      # Identify all experts in layer
    find_specialists,          # Find specialist experts
    find_generalists,          # Find generalist experts
    print_expert_summary,      # Print summary report
    # Ablation
    ablate_expert,             # Ablate single expert
    find_causal_experts,       # Find experts that affect output
    # Compression
    create_compression_plan,   # Plan expert merging
    analyze_compression_opportunities,  # Analyze all layers
    # Datasets (for custom analysis)
    PromptCategory,            # 27 prompt categories
    get_category_prompts,      # Get prompts for a category
    get_grouped_prompts,       # Get all prompts by category name
)
```

#### Direct Module Imports

For more control, import directly from the moe subpackage:

```python
from chuk_lazarus.introspection.moe import (
    # Hooks
    MoEHooks, MoECaptureConfig,
    # Models
    ExpertUtilization, RouterEntropy, ExpertIdentity,
    # Enums
    MoEArchitecture, ExpertCategory, ExpertRole,
)
```

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
├── ablation/            # Ablation studies
├── attention.py         # Attention pattern analysis
├── steering/            # Activation steering
├── moe/                 # MoE introspection (modular subpackage)
│   ├── __init__.py      # Clean exports
│   ├── enums.py         # MoEArchitecture, ExpertCategory, ExpertRole
│   ├── config.py        # MoECaptureConfig, MoEAblationConfig
│   ├── models.py        # Pydantic models (frozen, validated)
│   ├── detector.py      # Architecture detection
│   ├── hooks.py         # MoEHooks (composes ModelHooks)
│   ├── router.py        # Router analysis utilities
│   ├── ablation.py      # Expert ablation studies
│   ├── logit_lens.py    # MoE-specific logit lens
│   ├── identification.py # Expert specialization detection
│   ├── compression.py   # Expert merging/pruning analysis
│   └── datasets/        # JSON prompt datasets
│       ├── prompts.json # Categorized prompts (27 categories)
│       ├── prompts.py   # Dataset loader
│       └── categories.json # Token category keywords
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
    ├── moe/            # MoE introspection experiments
    │   └── moe_routing_analysis.py  # Routing analysis demo
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
