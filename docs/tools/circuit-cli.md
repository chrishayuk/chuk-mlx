# Circuit CLI

Standalone CLI for mechanistic interpretability research. Provides tools for dataset creation, activation collection, geometry analysis, direction extraction, and visualization.

## Installation

The circuit CLI is included with chuk-lazarus. To use it, ensure you have the package installed:

```bash
pip install chuk-lazarus[circuit]
```

Or if developing locally:

```bash
uv sync
```

## Quick Start

```bash
# 1. Create a labeled prompt dataset
circuit dataset create -o prompts.json --per-tool 25

# 2. Collect activations from a model
circuit collect -m mlx-community/functiongemma-270m-it-bf16 -d prompts.json -o activations

# 3. Analyze activation geometry
circuit analyze -a activations.safetensors --layer 11 --umap

# 4. Extract steering directions
circuit directions -a activations.safetensors --layer 11 -o directions

# 5. Create visualizations
circuit visualize -a activations.safetensors --all -o plots/

# 6. Run probe battery for stratigraphy
circuit probes run -m mlx-community/gemma-3-4b-it-bf16
```

## Commands

### `circuit dataset`

Manage labeled prompt datasets for circuit analysis.

#### `circuit dataset create`

Create a tool-calling prompt dataset with positive/negative labels.

```bash
circuit dataset create -o prompts.json [OPTIONS]
```

**Options:**
- `-o, --output FILE` - Output file path (required)
- `--per-tool N` - Number of prompts per tool category (default: 25)
- `--no-tool N` - Number of no-tool prompts (default: 100)
- `--no-edge-cases` - Exclude edge case prompts
- `--seed N` - Random seed (default: 42)

**Example:**
```bash
circuit dataset create -o tool_prompts.json --per-tool 50 --no-tool 200
```

**Output:** JSON file with labeled prompts:
```json
{
  "prompts": [
    {"text": "What's the weather in Paris?", "category": "weather", "expected_tool": "get_weather"},
    {"text": "Tell me a joke", "category": "no_tool", "expected_tool": null}
  ]
}
```

#### `circuit dataset show`

Display contents and statistics of a dataset.

```bash
circuit dataset show prompts.json [OPTIONS]
```

**Options:**
- `--samples N` - Number of sample prompts to display (default: 10)

---

### `circuit collect`

Collect model activations for analysis.

```bash
circuit collect -m MODEL -d DATASET -o OUTPUT [OPTIONS]
```

**Required:**
- `-m, --model MODEL` - Model ID or local path
- `-d, --dataset FILE` - Dataset file path
- `-o, --output PATH` - Output path (without extension)

**Options:**
- `--layers LAYERS` - Layers to capture: `all`, `decision`, or comma-separated (default: `decision`)
- `--attention` - Also capture attention weights
- `--generate N` - Generate N tokens for criterion evaluation

**Examples:**
```bash
# Capture decision layers (auto-selected based on model)
circuit collect -m gemma-3-4b-it -d prompts.json -o activations

# Capture specific layers
circuit collect -m gemma-3-4b-it -d prompts.json -o activations --layers 8,10,12,14

# Capture all layers with attention
circuit collect -m gemma-3-4b-it -d prompts.json -o activations --layers all --attention
```

**Output:** SafeTensors file containing:
- Hidden states per layer [num_samples, hidden_size]
- Attention weights (optional) [num_samples, num_heads, seq_len, seq_len]
- Metadata (model_id, captured_layers, labels)

---

### `circuit analyze`

Analyze the geometry of collected activations.

```bash
circuit analyze -a ACTIVATIONS [OPTIONS]
```

**Required:**
- `-a, --activations FILE` - Activations file path

**Options:**
- `--layer N` - Specific layer to analyze (default: all captured)
- `--umap` - Include UMAP visualization
- `-o, --output FILE` - Save results to JSON

**Example:**
```bash
circuit analyze -a activations.safetensors --layer 11 --umap -o analysis.json
```

**Output:**
```
Analyzing layer 11...
  PCA: dim@90%=23, dim@95%=45
       var[0]=12.34%
  Binary probe: acc=94.20%, CV=93.10%±1.20%
  Category probe: acc=78.50%
```

---

### `circuit directions`

Extract interpretable directions from activations.

```bash
circuit directions -a ACTIVATIONS [OPTIONS]
```

**Required:**
- `-a, --activations FILE` - Activations file path

**Options:**
- `--layer N` - Layer to extract from (default: middle of captured)
- `--method METHOD` - Extraction method: `diff_means`, `lda`, `probe_weights` (default: `diff_means`)
- `--per-tool` - Extract per-tool directions (in addition to tool/no-tool)
- `-o, --output PATH` - Output path for direction bundle

**Example:**
```bash
circuit directions -a activations.safetensors --layer 11 --per-tool -o directions
```

**Output:**
```
Tool-mode direction:
  Separation score: 0.847
  Classification accuracy: 94.20%
  Mean projection (tool): 0.234
  Mean projection (no-tool): -0.156

Per-tool directions (5 tools):
  get_weather: separation=0.912
  send_email: separation=0.834
  ...

Orthogonality check (cosine similarities):
  get_weather    : 1.00 0.12 0.08 0.15 0.11
  send_email     : 0.12 1.00 0.09 0.14 0.10
  ...
```

---

### `circuit visualize`

Create visualizations of activation geometry.

```bash
circuit visualize -a ACTIVATIONS [OPTIONS]
```

**Required:**
- `-a, --activations FILE` - Activations file path

**Options:**
- `--layer N` - Layer to visualize (default: last captured)
- `-o, --output DIR` - Output directory (default: current)
- `--pca` - Create PCA variance plot
- `--umap` - Create UMAP scatter plot
- `--probes` - Create probe accuracy across layers plot
- `--all` - Create all visualizations

**Example:**
```bash
circuit visualize -a activations.safetensors --all -o plots/
```

**Output files:**
- `pca_layer11.png` - PCA explained variance and intrinsic dimensionality
- `umap_layer11.png` - UMAP scatter colored by tool/no-tool and category
- `probe_accuracy.png` - Binary and category probe accuracy by layer

---

### `circuit probes`

Run probe battery for computational stratigraphy.

#### `circuit probes run`

Run linear probes across layers to understand feature emergence.

```bash
circuit probes run -m MODEL [OPTIONS]
```

**Required:**
- `-m, --model MODEL` - Model ID or path

**Options:**
- `--layers LAYERS` - Layers to probe (comma-separated, default: auto)
- `--datasets DIR` - Path to probe dataset directory
- `--category CATS` - Filter by category: `syntactic`, `semantic`, `decision`
- `--threshold FLOAT` - Emergence threshold (default: 0.75)
- `--no-stratigraphy` - Skip stratigraphy output
- `-o, --output FILE` - Save results to JSON

**Example:**
```bash
circuit probes run -m gemma-3-4b-it --layers 0,4,8,12,16,20,24,28 -o probes.json
```

**Output:**
```
Probing layers: [0, 4, 8, 12, 16, 20, 24, 28]

=== Probe Results ===
Probe                 L0     L4     L8    L12    L16    L20    L24    L28
─────────────────────────────────────────────────────────────────────────
is_question          0.52   0.78   0.91   0.94   0.95   0.95   0.95   0.95
is_imperative        0.51   0.72   0.88   0.93   0.94   0.94   0.94   0.94
tool_decision        0.50   0.55   0.68   0.82   0.91   0.94   0.95   0.95
...

=== Stratigraphy (emergence order) ===
Layer  4: is_question (0.78), is_imperative (0.72)
Layer  8: has_entity (0.85)
Layer 12: tool_decision (0.82)
...
```

#### `circuit probes init`

Initialize custom probe dataset files.

```bash
circuit probes init -o OUTPUT_DIR
```

Creates editable JSON files for customizing probes:
- `syntactic_probes.json` - Syntactic features (question, imperative, etc.)
- `semantic_probes.json` - Semantic features (entity, topic, etc.)
- `decision_probes.json` - Decision features (tool/no-tool, tool type, etc.)

---

## Workflow Examples

### Full Circuit Analysis Pipeline

```bash
# 1. Create balanced dataset
circuit dataset create -o prompts.json --per-tool 50 --no-tool 250

# 2. Collect activations at decision layers
circuit collect -m mlx-community/functiongemma-270m-it-bf16 \
    -d prompts.json -o activations --layers decision

# 3. Find the decision layer
circuit analyze -a activations.safetensors

# 4. Extract tool-mode direction at best layer
circuit directions -a activations.safetensors --layer 11 --per-tool -o tool_directions

# 5. Visualize
circuit visualize -a activations.safetensors --all -o plots/

# 6. Run stratigraphy analysis
circuit probes run -m mlx-community/functiongemma-270m-it-bf16 -o stratigraphy.json
```

### Compare Base vs Fine-tuned Model

```bash
# Collect activations from both models
circuit collect -m gemma-3-4b-it -d prompts.json -o base_activations
circuit collect -m functiongemma-270m-it -d prompts.json -o ft_activations

# Compare geometry at same layer
circuit analyze -a base_activations.safetensors --layer 11 -o base_analysis.json
circuit analyze -a ft_activations.safetensors --layer 11 -o ft_analysis.json

# Compare probe accuracy
circuit probes run -m gemma-3-4b-it -o base_probes.json
circuit probes run -m functiongemma-270m-it -o ft_probes.json
```

### Custom Dataset for Safety Analysis

```bash
# Initialize custom probes
circuit probes init -o my_probes/

# Edit my_probes/decision_probes.json to add:
# - "is_harmful": prompts labeled harmful vs benign
# - "refuses": prompts where model refuses vs complies

# Run with custom probes
circuit probes run -m model --datasets my_probes/ -o safety_stratigraphy.json
```

## Integration with Main CLI

The circuit CLI complements the main `lazarus introspect` commands:

| Circuit CLI | Main CLI Equivalent |
|-------------|---------------------|
| `circuit analyze` | `lazarus introspect analyze` (logit lens) |
| `circuit directions` | `lazarus introspect steer --extract` |
| N/A | `lazarus introspect steer` (apply steering) |
| N/A | `lazarus introspect ablate` (causal analysis) |

Use the circuit CLI for:
- Batch activation collection
- Geometry analysis (PCA, UMAP, probes)
- Stratigraphy (probe batteries)
- Direction extraction for later steering

Use the main CLI for:
- Interactive analysis
- Single-prompt investigation
- Activation steering experiments
- Ablation studies

---

### `circuit export`

Export circuit graphs to various visualization formats.

```bash
circuit export -i INPUT -o OUTPUT [OPTIONS]
```

**Required:**
- `-i, --input FILE` - Input file (ablation results or directions JSON)
- `-o, --output FILE` - Output file path

**Options:**
- `-f, --format FORMAT` - Output format: `json`, `dot`, `mermaid`, `html` (default: `json`)
- `--type TYPE` - Input type: `ablation`, `directions` (default: `ablation`)
- `--name NAME` - Circuit name (default: derived from input file)
- `--threshold FLOAT` - Minimum effect threshold for ablation circuits (default: 0.1)
- `--direction DIR` - Graph direction: `TB`, `LR`, `BT`, `RL` (default: `TB`)

**Examples:**
```bash
# Export ablation results to DOT (Graphviz)
lazarus introspect circuit export -i ablation_results.json -o circuit.dot -f dot

# Export to interactive HTML visualization
lazarus introspect circuit export -i ablation_results.json -o circuit.html -f html

# Export directions to Mermaid diagram
lazarus introspect circuit export -i directions.json -o circuit.md -f mermaid --type directions

# Export with left-to-right layout
lazarus introspect circuit export -i ablation.json -o circuit.dot -f dot --direction LR
```

**Output formats:**
- **JSON**: Machine-readable graph structure with nodes, edges, and metadata
- **DOT**: Graphviz format - render with `dot -Tpng circuit.dot -o circuit.png`
- **Mermaid**: Markdown-compatible diagrams for documentation
- **HTML**: Interactive visualization using vis.js (open in browser)

---

## See Also

- [introspection.md](../introspection.md) - Main introspection documentation
- [introspect-steer.md](introspect-steer.md) - Activation steering CLI
- [introspect-ablate.md](introspect-ablate.md) - Ablation studies
- `examples/introspection/demos/circuit_analysis.py` - Python API example
