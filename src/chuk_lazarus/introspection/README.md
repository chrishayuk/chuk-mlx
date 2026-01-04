# Introspection Tools

A mechanistic interpretability toolkit for understanding LLM behavior on MLX.

## Overview

This module provides reusable tools for:
- **Activation capture** - Hook into model internals during forward pass
- **Logit lens** - See what the model "thinks" at each layer
- **Attention analysis** - Extract and analyze attention patterns
- **Ablation studies** - Identify causal circuits by zeroing components
- **Activation steering** - Modify behavior by adding directions to activations
- **Circuit analysis** - Full pipeline for mechanistic interpretability research
- **MoE introspection** - Expert identification, routing analysis, compression planning (25 CLI commands)
- **Circuit export** - Export circuit graphs to DOT, JSON, Mermaid, HTML formats

## Quick Start

### Recommended: Async Analyzer

```python
from chuk_lazarus.introspection import ModelAnalyzer, AnalysisConfig

async with ModelAnalyzer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0") as analyzer:
    result = await analyzer.analyze("The capital of France is")
    print(result.predicted_token)  # " Paris"

    # See layer-by-layer evolution
    for layer in result.layer_predictions:
        print(f"Layer {layer.layer_idx}: {layer.top_token}")
```

### CLI Usage

```bash
# Logit lens analysis
lazarus introspect analyze -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 -p "The capital of France is"

# Compare two models
lazarus introspect compare --model1 base-model --model2 finetuned-model -p "prompt"

# Run ablation study
lazarus introspect ablate -m model-id -p "prompt" --component mlp --criterion function_call

# Weight divergence between base and fine-tuned
lazarus introspect weight-diff --base base-model --finetuned finetuned-model
```

## Core Modules

### `analyzer.py` - High-Level Analyzer (Recommended)

Async-native analyzer with pydantic models for type safety.

```python
from chuk_lazarus.introspection import (
    ModelAnalyzer,
    AnalysisConfig,
    LayerStrategy,
)

config = AnalysisConfig(
    layer_strategy=LayerStrategy.EVENLY_SPACED,
    layer_step=4,
    top_k=5,
    track_tokens=["Paris", " Paris"],  # Track specific tokens
)

async with ModelAnalyzer.from_pretrained(model_id) as analyzer:
    result = await analyzer.analyze(prompt, config)

    # Access results
    result.predicted_token       # Final prediction
    result.layer_predictions     # Per-layer predictions
    result.token_evolutions      # How tracked tokens evolve
    result.layer_transitions     # Where predictions change
```

### `hooks.py` - Low-Level Activation Capture

For when you need direct access to activations.

```python
from chuk_lazarus.introspection import ModelHooks, CaptureConfig, LayerSelection

hooks = ModelHooks(model)
hooks.configure(CaptureConfig(
    layers=[0, 4, 8, 12],
    capture_hidden_states=True,
    capture_attention_weights=True,
))

logits = hooks.forward(input_ids)

# Access captured state
hooks.state.hidden_states[4]      # Hidden state at layer 4
hooks.state.attention_weights[4]  # Attention weights at layer 4
```

### `logit_lens.py` - Layer-by-Layer Predictions

See how predictions evolve through the model.

```python
from chuk_lazarus.introspection import LogitLens

lens = LogitLens(hooks, tokenizer)
lens.print_evolution(position=-1, top_k=3)

# Or get structured data
predictions = lens.get_layer_predictions(top_k=5)
evolution = lens.track_token_emergence("Paris")
```

### `attention.py` - Attention Analysis

Extract and analyze attention patterns.

```python
from chuk_lazarus.introspection import AttentionAnalyzer, AggregationStrategy

analyzer = AttentionAnalyzer(hooks)
pattern = analyzer.get_pattern(layer=8, head=0)

# Find what tokens attend to
focus = analyzer.get_attention_focus(layer=8, threshold=0.1)
entropy = analyzer.compute_entropy(layer=8)
```

### `ablation.py` - Causal Circuit Discovery

Identify which layers matter for specific behaviors.

```python
from chuk_lazarus.introspection import AblationStudy, ComponentType

study = AblationStudy.from_pretrained(model_id)

# Define criterion function
def produces_tool_call(output: str) -> bool:
    return "<function_call>" in output

# Run layer sweep
result = study.run_layer_sweep(
    prompt="What's the weather?",
    criterion=produces_tool_call,
    component=ComponentType.MLP,
)

study.print_sweep_summary(result)
```

### `steering.py` - Activation Steering

Modify model behavior by adding directions to activations.

```python
from chuk_lazarus.introspection import ActivationSteering

steerer = ActivationSteering.from_pretrained(model_id)

# Load or compute a direction
direction = load_direction("tool_calling_direction.safetensors")

# Generate with steering
output = steerer.generate_with_steering(
    prompt="The weather is",
    direction=direction,
    layer=11,
    coefficient=1.5,
)
```

## Circuit Analysis Submodule

For full mechanistic interpretability research, use the circuit analysis pipeline.

### CLI

```bash
# Create dataset
circuit dataset create -o prompts.json --per-tool 25

# Collect activations
circuit collect -m functiongemma-270m -d prompts.json -o activations

# Analyze geometry
circuit analyze -a activations.safetensors --layer 11

# Extract directions
circuit directions -a activations.safetensors --layer 11 -o directions

# Visualize
circuit visualize -a activations.safetensors --layer 11 --all

# Run probe battery
circuit probes run -m functiongemma-270m
```

### Python API

```python
from chuk_lazarus.introspection.circuit import (
    ToolPromptDataset,
    ActivationCollector,
    GeometryAnalyzer,
    DirectionExtractor,
    ProbeBattery,
)

# 1. Create dataset
dataset = create_tool_calling_dataset(prompts_per_tool=50)

# 2. Collect activations
collector = ActivationCollector.from_pretrained(model_id)
activations = collector.collect(dataset)

# 3. Analyze geometry
analyzer = GeometryAnalyzer(activations)
pca = analyzer.compute_pca(layer=11)
probes = analyzer.train_probe(layer=11)

# 4. Extract directions
extractor = DirectionExtractor(activations)
tool_direction = extractor.extract_tool_mode_direction(layer=11)

# 5. Run probe battery
battery = ProbeBattery.from_pretrained(model_id)
results = battery.run_all_probes()
battery.print_stratigraphy(results)
```

## Architecture

```
introspection/
├── __init__.py          # Public API exports
├── hooks.py             # Low-level activation capture
├── logit_lens.py        # Layer prediction analysis
├── attention.py         # Attention pattern analysis
├── analyzer.py          # High-level async analyzer
├── ablation.py          # Ablation studies
├── steering.py          # Activation steering
├── circuit/             # Full circuit analysis pipeline
│   ├── dataset.py       # Labeled prompt datasets
│   ├── collector.py     # Activation collection
│   ├── geometry.py      # PCA, UMAP, probes
│   ├── directions.py    # Direction extraction
│   ├── probes.py        # Linear probe battery
│   └── cli.py           # Standalone circuit CLI
└── visualizers/         # HTML/chart visualizations
    ├── logit_evolution.py
    └── attention_heatmap.py
```

## MoE Introspection

Analyze Mixture of Experts models (GPT-OSS, Mixtral, Llama 4, Granite MoE):

### CLI Commands

```bash
# Expert analysis - identify what each expert specializes in
lazarus introspect moe-expert analyze -m openai/gpt-oss-20b

# Generate routing heatmap visualization
lazarus introspect moe-expert heatmap -m openai/gpt-oss-20b -p "def fib(n):"

# Track expert pipelines across layers
lazarus introspect moe-expert pipeline -m openai/gpt-oss-20b --num-prompts 20

# Analyze expert vocabulary contributions
lazarus introspect moe-expert vocab-contrib -m openai/gpt-oss-20b --top-k 30

# Analyze compression opportunities
lazarus introspect moe-expert compression -m openai/gpt-oss-20b --threshold 0.8

# Export circuit graph
lazarus introspect circuit export -i ablation_results.json -o circuit.html -f html
```

### Python API

```python
from chuk_lazarus.introspection.moe import ExpertRouter

async with await ExpertRouter.from_pretrained("openai/gpt-oss-20b") as router:
    # Get model info
    info = router.info
    print(f"Experts: {info.num_experts}, Active: {info.num_active_experts}")

    # Analyze compression opportunities
    analysis = await router.analyze_compression(prompts, layer_idx=12)
    print(f"Merge candidates: {len(analysis.merge_candidates)}")
```

## Circuit Export

Export ablation results or extracted directions as circuit graphs:

```bash
# Export to interactive HTML
lazarus introspect circuit export -i ablation.json -o circuit.html -f html

# Export to DOT (Graphviz)
lazarus introspect circuit export -i ablation.json -o circuit.dot -f dot
# Then render: dot -Tpng circuit.dot -o circuit.png

# Export to Mermaid (for markdown docs)
lazarus introspect circuit export -i ablation.json -o circuit.md -f mermaid
```

## See Also

- `examples/introspection/` - Research experiments and demos
- `docs/introspection.md` - Conceptual documentation
- `docs/tools/circuit-cli.md` - Circuit CLI documentation
- `docs/roadmap-introspection-moe.md` - MoE roadmap and features
