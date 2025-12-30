# lazarus introspect activation-diff

Compare activation divergence between two models.

## Synopsis

```bash
lazarus introspect activation-diff -b BASE -f FINETUNED -p PROMPTS [-o OUTPUT]
```

## Description

The `activation-diff` command runs the same prompts through two models and compares their internal hidden state representations using cosine similarity. This reveals where the models' internal processing diverges.

## Options

| Option | Description |
|--------|-------------|
| `-b, --base MODEL` | Base model (required) |
| `-f, --finetuned MODEL` | Fine-tuned model (required) |
| `-p, --prompts PROMPTS` | Prompts to test (required) |
| `-o, --output FILE` | Save results to JSON |

## Prompts Format

- Comma-separated: `"prompt1,prompt2,prompt3"`
- From file: `@prompts.txt` (one prompt per line)

## Examples

### Compare on Multiple Prompts

```bash
lazarus introspect activation-diff \
    -b google/gemma-3-1b \
    -f google/gemma-3-1b-it \
    -p "Hello,How are you,What is AI"
```

### From File

```bash
lazarus introspect activation-diff \
    -b base-model \
    -f finetuned-model \
    -p @test_prompts.txt
```

### Save Results

```bash
lazarus introspect activation-diff \
    -b base \
    -f finetuned \
    -p "test prompt" \
    -o activation_diff.json
```

## Output

```
Loading base model: google/gemma-3-1b
Loading fine-tuned model: google/gemma-3-1b-it
Testing 3 prompts

Prompt: Hello...
Prompt: How are you...
Prompt: What is AI...

Layer    Avg Cos Sim    Divergence
-----------------------------------
0           0.9998        0.0002
1           0.9995        0.0005
2           0.9991        0.0009
...
10          0.9234        0.0766
11          0.8567        0.1433 ***
12          0.8123        0.1877 ***
...
```

## Interpreting Results

- **Cosine Similarity**: 1.0 = identical, 0.0 = orthogonal
- **Divergence**: 1 - cosine_similarity
- **`***` marker**: Indicates divergence > 0.1 (10%)

Higher divergence at a layer means the models' representations differ more at that point.

## Use Cases

### Finding Divergence Points

Identify at which layer base and fine-tuned models start to differ:

```bash
lazarus introspect activation-diff \
    -b base \
    -f instruction-tuned \
    -p "Explain AI,Write code,Tell a joke"
```

### Testing on Domain-Specific Prompts

```bash
# Create file with domain prompts
echo "Calculate 2+2
What is the capital of France
Write a Python function" > prompts.txt

lazarus introspect activation-diff \
    -b base \
    -f domain-tuned \
    -p @prompts.txt
```

## See Also

- [introspect weight-diff](introspect-weight-diff.md) - Compare weights
- [introspect compare](introspect-compare.md) - Compare predictions
