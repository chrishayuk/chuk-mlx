# lazarus introspect dual-reward

Train V/O projections with dual reward: classification + answer correctness.

## Synopsis

```bash
lazarus introspect dual-reward -m MODEL [OPTIONS]
```

## Description

The `dual-reward` command trains the model's V (value) and O (output) projections using LoRA to create vocabulary-mappable classifiers at intermediate layers while preserving answer generation ability.

This is useful when you want to:
1. Make existing internal classifiers **readable** via logit lens
2. Study how classification emerges during training
3. Create interpretable intermediate representations

## Training Objective

The training uses a combined loss function:

```
total_loss = cls_weight * classification_loss + (1 - cls_weight) * answer_loss
```

- **Classification loss**: Cross-entropy at intermediate layer for emitting operation tokens (multiply, add, etc.)
- **Answer loss**: Cross-entropy at final layer for correct answer generation

## Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | HuggingFace model ID or local path (required) |
| `--steps N` | Number of training steps (default: 500) |
| `--classifier-layer N` | Layer for classification loss (default: 55% depth) |
| `--cls-weight FLOAT` | Weight for classification loss (default: 0.4) |
| `--learning-rate FLOAT` | Learning rate (default: 5e-4) |
| `--num-samples N` | Number of training samples to generate (default: 800) |
| `--lora-rank N` | LoRA rank (default: 16) |
| `-o, --output DIR` | Save checkpoint to directory |

## Examples

### Basic Training

```bash
lazarus introspect dual-reward -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --steps 500
```

### Custom Configuration

```bash
lazarus introspect dual-reward -m meta-llama/Llama-3.2-1B \
  --steps 1000 \
  --classifier-layer 10 \
  --cls-weight 0.4 \
  --learning-rate 1e-4 \
  --output checkpoint/llama_dual_reward
```

### Save Checkpoint

```bash
lazarus introspect dual-reward -m model \
  --steps 500 \
  --output checkpoint/my_experiment
```

This saves:
- `lora_weights.npz` - LoRA adapter weights
- `config.json` - Training configuration

## Output

```
Loading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  Layers: 22
  Classifier layer: L12 (55% depth)
  Class tokens: {'multiply': 13647, 'add': 788, 'subtract': 1014, 'divide': 29876}
  LoRA rank: 16
  Trainable params: 1,048,576

  Training samples: 800
  Classification weight: 0.4
  Learning rate: 0.0005
  Steps: 500

Training...
  Step   Total       Cls Loss   Ans Loss
----------------------------------------
   100     2.3456     1.8234     2.6892
   200     1.8912     1.2341     2.3456
   300     1.4567     0.8912     1.8234
   400     1.2341     0.6789     1.6123
   500     1.0123     0.5678     1.3456
----------------------------------------

Evaluating classifier after training...

Prompt               Predicted        Expected         Status
------------------------------------------------------------
  7 * 8 =            multiply         multiply         [OK]
  12 * 5 =           multiply         multiply         [OK]
  23 + 45 =          add              add              [OK]
  17 + 38 =          add              add              [OK]
  50 - 23 =          subtract         subtract         [OK]
  89 - 34 =          subtract         subtract         [OK]
  48 / 6 =           divide           divide           [OK]
  81 / 9 =           divide           divide           [OK]
------------------------------------------------------------

Accuracy: 8/8 (100%)

Checkpoint saved to: checkpoint/my_experiment
```

## Workflow

### Complete Classifier Emergence Study

1. **Baseline**: Check if classifiers exist in hidden space
   ```bash
   lazarus introspect classifier -m model \
     --classes "multiply:7 * 8 = |12 * 5 = " \
     --classes "add:23 + 45 = |17 + 38 = "
   ```

2. **Baseline Logit Lens**: Verify they don't map to vocabulary
   ```bash
   lazarus introspect logit-lens -m model \
     --prompts "7 * 8 = |23 + 45 = " \
     --targets "multiply" --targets "add"
   ```

3. **Train**: Add vocabulary projection
   ```bash
   lazarus introspect dual-reward -m model \
     --steps 500 \
     --output checkpoint/trained
   ```

4. **Verify**: Check logit lens after training
   ```bash
   lazarus introspect logit-lens -m model \
     --adapter checkpoint/trained \
     --prompts "7 * 8 = |23 + 45 = " \
     --targets "multiply" --targets "add"
   ```

## Training Data

The command automatically generates arithmetic training data:
- **Multiply**: `a * b` where a, b in [1, 50]
- **Add**: `a + b` where a, b in [1, 50]
- **Subtract**: `a - b` where a > b
- **Divide**: `a / b` where a is divisible by b

Each sample includes:
- `prompt`: e.g., "7 * 8 = "
- `answer`: e.g., "56"
- `class`: e.g., "multiply"
- `class_token`: Token ID for the class name

## Why V/O Projections?

The training only updates V (value) and O (output) projections because:

1. **V projection** determines what information enters the attention output
2. **O projection** maps the attention output to the residual stream
3. **Q/K projections** control attention routing, which is already fixed

By training V/O, we teach the model to project classification information into vocabulary space without changing attention patterns.

## Saved Checkpoint Format

### lora_weights.npz
Contains LoRA adapter matrices for each layer:
```
layer_0_v_A: (hidden_dim, lora_rank)
layer_0_v_B: (lora_rank, hidden_dim)
layer_0_o_A: (hidden_dim, lora_rank)
layer_0_o_B: (lora_rank, hidden_dim)
...
```

### config.json
```json
{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "classifier_layer": 12,
  "lora_rank": 16,
  "cls_weight": 0.4,
  "steps": 500,
  "final_accuracy": 1.0
}
```

## See Also

- [introspect classifier](introspect-classifier.md) - Detect classifiers via linear probes
- [introspect logit-lens](introspect-logit-lens.md) - Check vocabulary projection
- [introspect probe](introspect-probe.md) - Binary classification probes
- [Introspection Overview](../introspection.md) - Full module documentation
