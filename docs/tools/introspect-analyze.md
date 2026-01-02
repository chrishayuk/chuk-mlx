# lazarus introspect analyze

Logit lens analysis to see how predictions evolve through the model's layers.

## Synopsis

```bash
lazarus introspect analyze -m MODEL -p PROMPT [OPTIONS]
```

## Description

The `analyze` command runs logit lens analysis on a prompt, showing what the model would predict at each layer. This reveals:

- When predictions stabilize
- Where specific tokens "emerge" as the top prediction
- How the model refines its understanding through depth

## Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | HuggingFace model ID or local path (required) |
| `-p, --prompt PROMPT` | Prompt to analyze (required) |
| `--all-layers` | Capture all layers (overrides `--layer-strategy`) |
| `-s, --layer-step N` | Capture every Nth layer (default: 4) |
| `--layer-strategy {all,evenly_spaced,first_last,custom}` | Layer selection strategy |
| `-t, --track TOKEN` | Track token probability across layers (repeatable) |
| `--top-k N` | Show top N predictions (default: 5) |
| `--embedding-scale FLOAT` | Manual embedding scale (auto-detected for Gemma) |
| `--raw` | Use raw prompt without chat template |
| `-o, --output FILE` | Save results to JSON |

## Chat Template Mode

By default, if the model has a chat template (e.g., Gemma-IT, Llama-Instruct), it will be applied to wrap your prompt. Use `--raw` to analyze the prompt directly without chat formatting:

```bash
# Default: Uses chat template for instruct models
lazarus introspect analyze -m mlx-community/gemma-3-4b-it-bf16 -p "2 + 2 ="

# Raw mode: Analyzes the exact prompt you provide
lazarus introspect analyze -m mlx-community/gemma-3-4b-it-bf16 -p "2 + 2 =" --raw

# Base models: Always raw (no chat template available)
lazarus introspect analyze -m mlx-community/gemma-3-4b-pt-bf16 -p "2 + 2 ="
```

## Examples

### Basic Analysis

```bash
lazarus introspect analyze \
    -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    -p "The capital of France is"
```

### All Layers

```bash
lazarus introspect analyze \
    -m model \
    -p "Hello world" \
    --all-layers
```

### Track Specific Tokens

Track when specific tokens become the top prediction:

```bash
lazarus introspect analyze \
    -m mlx-community/gemma-3-4b-it-bf16 \
    -p "156 + 287 =" \
    -t " 443" \
    -t "443" \
    --all-layers
```

### Save to JSON

```bash
lazarus introspect analyze \
    -m model \
    -p "The answer is" \
    -o analysis.json
```

## Output

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
  0.0156  ' a'

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

## How It Works

The logit lens technique:

1. **Captures hidden states** at selected layers during the forward pass
2. **Projects each hidden state** through the model's output layer (unembedding matrix)
3. **Converts to probabilities** via softmax
4. **Shows the top prediction** at each layer

This reveals how the model's "thinking" evolves from early syntactic processing to final semantic understanding.

## Layer Selection Strategies

| Strategy | Description |
|----------|-------------|
| `evenly_spaced` | Every Nth layer (default, N=4) |
| `all` | Capture all layers |
| `first_last` | Only first and last layer |
| `custom` | Specific layers (use with `--custom-layers`) |

## Token Evolution

When you use `-t/--track`, the tool shows how that token's probability changes across layers:

- **Probability**: Raw probability of the token
- **Rank**: Position in the top-k predictions
- **Emergence layer**: First layer where it becomes the #1 prediction

This is useful for understanding when the model "decides" on a particular token.

## Model Info

The output shows:
- **Family**: Detected model architecture (llama, gemma, etc.)
- **Layers**: Total number of transformer layers
- **Hidden size**: Dimension of hidden states
- **Vocab size**: Number of tokens in vocabulary
- **Tied embeddings**: Whether input/output embeddings are shared
- **Embedding scale**: Scale factor (Gemma uses sqrt(hidden_size))

## Python API

```python
from chuk_lazarus.introspection import ModelAnalyzer, AnalysisConfig, LayerStrategy

async with ModelAnalyzer.from_pretrained("model-id") as analyzer:
    config = AnalysisConfig(
        layer_strategy=LayerStrategy.ALL,
        track_tokens=["Paris", " Paris"],
        top_k=10,
    )
    result = await analyzer.analyze("The capital of France is", config)

    for layer in result.layer_predictions:
        print(f"Layer {layer.layer_idx}: {layer.top_token}")
```

## See Also

- [introspect compare](introspect-compare.md) - Compare two models
- [introspect ablate](introspect-ablate.md) - Ablation studies
- [Introspection Overview](../introspection.md) - Full module documentation
