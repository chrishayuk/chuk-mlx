# Circuit Analysis Toolkit

A complete pipeline for mechanistic interpretability research on MLX.

## Overview

This module provides tools for:
1. **Dataset creation** - Labeled prompts for circuit analysis
2. **Activation collection** - Collect hidden states from models
3. **Geometry analysis** - PCA, UMAP, probes on activation space
4. **Direction extraction** - Find interpretable directions (diff-of-means, LDA)
5. **Probe battery** - Linear probes for computational stratigraphy
6. **Visualization** - Charts and plots for analysis

## CLI Usage

The `circuit` command is a standalone CLI for circuit analysis.

### Quick Start

```bash
# 1. Create a dataset of labeled prompts
circuit dataset create -o prompts.json --per-tool 50

# 2. Collect activations from a model
circuit collect -m google/gemma-3-1b-it -d prompts.json -o activations

# 3. Analyze the geometry of activations
circuit analyze -a activations.safetensors --layer 11

# 4. Extract interpretable directions
circuit directions -a activations.safetensors --layer 11 -o directions

# 5. Visualize with PCA/UMAP
circuit visualize -a activations.safetensors --layer 11 --all -o plots/

# 6. Run probe battery for stratigraphy
circuit probes run -m google/gemma-3-1b-it
```

### Commands

#### `circuit dataset`
Manage prompt datasets for analysis.

```bash
# Create a tool-calling dataset
circuit dataset create -o prompts.json --per-tool 50 --no-tool 100

# Show dataset contents
circuit dataset show prompts.json --samples 20
```

#### `circuit collect`
Collect activations from a model.

```bash
# Collect at decision layers (auto-detected)
circuit collect -m model-id -d prompts.json -o activations --layers decision

# Collect all layers
circuit collect -m model-id -d prompts.json -o activations --layers all

# Collect specific layers
circuit collect -m model-id -d prompts.json -o activations --layers 8,10,11,12

# Also capture attention weights
circuit collect -m model-id -d prompts.json -o activations --attention

# Generate tokens to evaluate criterion
circuit collect -m model-id -d prompts.json -o activations --generate 50
```

#### `circuit analyze`
Analyze activation geometry.

```bash
# Analyze all captured layers
circuit analyze -a activations.safetensors

# Analyze specific layer
circuit analyze -a activations.safetensors --layer 11

# Include UMAP
circuit analyze -a activations.safetensors --layer 11 --umap

# Save results
circuit analyze -a activations.safetensors -o results.json
```

#### `circuit directions`
Extract interpretable directions from activations.

```bash
# Extract tool-mode direction using diff-of-means
circuit directions -a activations.safetensors --layer 11 --method diff_means

# Extract using LDA
circuit directions -a activations.safetensors --layer 11 --method lda

# Extract per-tool directions
circuit directions -a activations.safetensors --layer 11 --per-tool

# Save direction bundle
circuit directions -a activations.safetensors --layer 11 -o directions.safetensors
```

#### `circuit visualize`
Create visualizations.

```bash
# PCA variance plot
circuit visualize -a activations.safetensors --layer 11 --pca

# UMAP scatter plot
circuit visualize -a activations.safetensors --layer 11 --umap

# Probe accuracy across layers
circuit visualize -a activations.safetensors --probes

# All visualizations
circuit visualize -a activations.safetensors --all -o plots/
```

#### `circuit probes`
Run linear probe battery for computational stratigraphy.

```bash
# Run probes on default layers
circuit probes run -m model-id

# Run on specific layers
circuit probes run -m model-id --layers 0,4,8,11,12,14

# Filter by category
circuit probes run -m model-id --category syntactic,semantic

# Save results
circuit probes run -m model-id -o probe_results.json

# Initialize custom probe datasets
circuit probes init -o my_probes/
```

## Python API

### Dataset Creation

```python
from chuk_lazarus.introspection.circuit import (
    create_tool_calling_dataset,
    ToolPromptDataset,
    PromptCategory,
)

# Create dataset
dataset = create_tool_calling_dataset(
    prompts_per_tool=50,
    no_tool_prompts=100,
    include_edge_cases=True,
)

# Save/load
dataset.save("prompts.json")
dataset = ToolPromptDataset.load("prompts.json")

# Iterate
for prompt in dataset:
    print(f"[{prompt.category}] {prompt.text}")
```

### Activation Collection

```python
from chuk_lazarus.introspection.circuit import (
    ActivationCollector,
    CollectorConfig,
)

collector = ActivationCollector.from_pretrained("model-id")

config = CollectorConfig(
    layers="decision",  # or "all" or [8, 10, 11, 12]
    capture_hidden_states=True,
    capture_attention_weights=False,
)

activations = collector.collect(dataset, config, progress=True)
activations.save("activations")
```

### Geometry Analysis

```python
from chuk_lazarus.introspection.circuit import (
    CollectedActivations,
    GeometryAnalyzer,
    ProbeType,
)

activations = CollectedActivations.load("activations.safetensors")
analyzer = GeometryAnalyzer(activations)

# PCA
pca = analyzer.compute_pca(layer=11, n_components=50)
print(f"Intrinsic dim (90%): {pca.intrinsic_dimensionality_90}")

# UMAP
umap_result = analyzer.compute_umap(layer=11)

# Linear probes
binary_probe = analyzer.train_probe(layer=11, ProbeType.BINARY)
print(f"Binary accuracy: {binary_probe.accuracy:.2%}")

category_probe = analyzer.train_probe(layer=11, ProbeType.MULTICLASS)
print(f"Category accuracy: {category_probe.accuracy:.2%}")
```

### Direction Extraction

```python
from chuk_lazarus.introspection.circuit import (
    DirectionExtractor,
    DirectionMethod,
)

extractor = DirectionExtractor(activations)

# Tool-mode direction
tool_direction = extractor.extract_tool_mode_direction(
    layer=11,
    method=DirectionMethod.DIFF_MEANS,
)
print(f"Separation: {tool_direction.separation_score:.3f}")

# Per-tool directions
per_tool = extractor.extract_per_tool_directions(layer=11)
for name, direction in per_tool.items():
    print(f"{name}: {direction.separation_score:.3f}")

# Save bundle
bundle = extractor.create_bundle(layer=11, include_per_tool=True)
bundle.save("directions.safetensors")
```

### Probe Battery

```python
from chuk_lazarus.introspection.circuit import ProbeBattery

battery = ProbeBattery.from_pretrained("model-id")

# Run all probes
results = battery.run_all_probes(
    layers=[0, 4, 8, 11, 12, 14],
    progress=True,
)

# Print results
battery.print_results_table(results)
battery.print_stratigraphy(results, threshold=0.75)
```

## Architecture

```
circuit/
├── __init__.py      # Public API
├── cli.py           # Standalone CLI
├── dataset.py       # Prompt datasets
├── collector.py     # Activation collection
├── geometry.py      # PCA, UMAP, probes
├── directions.py    # Direction extraction
└── probes.py        # Linear probe battery
```

## Concepts

### Decision Layers
Layers where the model makes key decisions (e.g., tool vs no-tool). These scale with model size:
- 270M: ~L11
- 1B: ~L16
- 2B: ~L20-22

### Computational Stratigraphy
Linear probes reveal what each layer "knows":
- Early layers: syntactic features
- Middle layers: semantic features
- Late layers: task-specific decisions

### Interpretable Directions
Vectors in activation space that correspond to behaviors:
- **Diff-of-means**: Simple difference between class centroids
- **LDA**: Linear discriminant analysis
- **Probe weights**: Weights from trained linear probe

These can be used for:
- Understanding what the model represents
- Steering behavior via activation addition
