# lazarus introspect hooks

Low-level demonstration of activation hooks.

## Synopsis

```bash
lazarus introspect hooks -m MODEL -p PROMPT [OPTIONS]
```

## Description

The `hooks` command demonstrates the low-level hooks API for capturing intermediate model states. Useful for debugging and understanding the hook system.

## Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | Model to load (required) |
| `-p, --prompt PROMPT` | Prompt to process (required) |
| `--layers L1,L2,...` | Layers to capture (default: 0,4,8,...) |
| `--capture-attention` | Also capture attention weights |
| `--last-only` | Only capture last position (memory efficient) |
| `--no-logit-lens` | Skip logit lens output |

## Examples

### Basic Usage

```bash
lazarus introspect hooks \
    -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    -p "Hello world"
```

### Specific Layers with Attention

```bash
lazarus introspect hooks \
    -m model \
    -p "Test prompt" \
    --layers 0,4,8,12 \
    --capture-attention
```

### Memory Efficient (Last Position Only)

```bash
lazarus introspect hooks \
    -m large-model \
    -p "Long prompt..." \
    --last-only
```

## Output

```
Loading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Tokens (3): ['<s>', 'Hello', 'world']

Capturing layers: [0, 4, 8, 12, 16, 20]
Running forward pass...

=== Captured States ===
Layers captured: [0, 4, 8, 12, 16, 20]
  Layer 0: hidden shape (1, 3, 2048)
  Layer 4: hidden shape (1, 3, 2048)
  Layer 8: hidden shape (1, 3, 2048)
  ...

=== Logit Lens ===
Layer  0: 'the' (0.0523)
Layer  4: 'France' (0.0891)
...
```

## Use Cases

### Debugging Hook Configuration

Verify that hooks capture the expected layers and shapes:

```bash
lazarus introspect hooks -m model -p "test" --layers 0,11,21
```

### Memory Profiling

Test memory usage with different configurations:

```bash
# Full sequence
lazarus introspect hooks -m model -p "long prompt..."

# Last position only (much less memory)
lazarus introspect hooks -m model -p "long prompt..." --last-only
```

## Python API

```python
from chuk_lazarus.introspection import ModelHooks, CaptureConfig, PositionSelection

hooks = ModelHooks(model)
hooks.configure(CaptureConfig(
    layers=[0, 4, 8, 12],
    capture_hidden_states=True,
    capture_attention_weights=True,
    positions=PositionSelection.LAST,
))

logits = hooks.forward(input_ids)

# Access captured states
for layer, hidden in hooks.state.hidden_states.items():
    print(f"Layer {layer}: {hidden.shape}")
```

## See Also

- [introspect analyze](introspect-analyze.md) - High-level logit lens
- [Introspection Overview](../introspection.md) - Full API documentation
