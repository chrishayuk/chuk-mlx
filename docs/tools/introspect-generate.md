# introspect generate

Generate multiple tokens to test the next-token lock hypothesis.

## Usage

```bash
lazarus introspect generate -m MODEL -p PROMPTS [OPTIONS]
```

## Description

This tool tests whether format issues (like missing trailing space) cause:
- **A) Space-lock only**: Model just adds space token, answer appears at same position
- **B) Onset routing**: Model changes WHEN it emits the answer (delayed vs immediate)
- **C) Compute blocked**: Model can't produce correct answer at all

By generating multiple tokens and tracking **answer onset** (first token position where
correct answer appears), we can distinguish these mechanistically.

## Background

Some models behave differently with/without trailing space:
```
"156 + 287 = " → "443"  (correct)
"156 + 287 ="  → " ?"   (refuses)
```

But what happens if we generate MORE tokens from the broken prompt?
- If it outputs ` 443` (space + answer), it's just next-token lock
- If it outputs ` ???...` (space + garbage), there's a real gate

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--model` | `-m` | Model name or HuggingFace ID (required) |
| `--prompts` | `-p` | Prompts to test, pipe-separated or @file.txt (required) |
| `--max-tokens` | `-n` | Maximum tokens to generate (default: 10) |
| `--temperature` | `-t` | Temperature, 0=greedy (default: 0) |
| `--compare-format` | `-c` | Auto-create with/without space variants |
| `--show-tokens` | | Show individual generated tokens |
| `--raw` | | Use raw prompts without chat template |
| `--output` | `-o` | Save results to JSON file |

## Chat Template Modes

By default, if the model has a chat template, it will be applied to prompts:

```bash
# Uses chat template (prompt wrapped in <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n)
lazarus introspect generate -m mlx-community/gemma-3-4b-it-bf16 -p "2 + 2 ="
```

Use `--raw` to test direct prompts without chat formatting:

```bash
# Raw mode - prompt sent as-is
lazarus introspect generate -m mlx-community/gemma-3-4b-it-bf16 -p "2 + 2 =" --raw

# Base models have no chat template, so always run in raw mode
lazarus introspect generate -m mlx-community/gemma-3-4b-pt-bf16 -p "2 + 2 ="
```

This is useful for:
- Testing base models (non-instruct versions)
- Investigating tokenization effects directly
- Comparing chat vs non-chat behavior

## Examples

### Basic generation

```bash
lazarus introspect generate \
    -m mlx-community/gemma-3-4b-it-bf16 \
    -p "100 - 37 =" \
    -n 15
```

### Compare format variants

```bash
lazarus introspect generate \
    -m mlx-community/gemma-3-4b-it-bf16 \
    -p "100 - 37 =|156 + 287 =|50 - 25 =" \
    --compare-format \
    --show-tokens
```

This automatically creates both `"100 - 37 ="` and `"100 - 37 = "` variants.

### With token breakdown

```bash
lazarus introspect generate \
    -m model \
    -p "prompt" \
    --show-tokens \
    -n 20
```

## Example Output

```
Loading model: mlx-community/gemma-3-4b-it-bf16

Generating 15 tokens per prompt
Temperature: 0.0

✗ '100 - 37 ='
  → ' 63\n\n100 - 37 = 63'
  Tokens: ' ' '6' '3' '\n\n' '1' '0' '0' ' -' ' ' '3' ...

✓ '100 - 37 = '
  → '63\n\n100 - 37 = 63\n\n'
  Tokens: '6' '3' '\n\n' '1' '0' '0' ' -' ' ' '3' '7' ...

✗ '156 + 287 ='
  → ' ?\n\n156 + 287 = 443'
  Tokens: ' ?' '\n\n' '1' '5' '6' ' +' ' ' '2' '8' '7' ...

✓ '156 + 287 = '
  → '443\n\n156 + 287 = 4'
  Tokens: '4' '4' '3' '\n\n' '1' '5' '6' ' +' ' ' '2' ...

=== Format Comparison Summary ===

'100 - 37 =':
  No space:   ' 63\n\n100 - 37 = 63'
  With space: '63\n\n100 - 37 = 63\n\n'
  Verdict: SAME (next-token lock)

'156 + 287 =':
  No space:   ' ?\n\n156 + 287 = 443'
  With space: '443\n\n156 + 287 = 4'
  Verdict: DIFFERENT (possible gate)
```

## Interpretation

### SAME (next-token lock)
The model just completes the format first (outputs space), then computes normally.
This is the simpler explanation - no complex gate, just autoregressive prediction.

### DIFFERENT (possible gate)
The model outputs something different (like `?` or refusal) even after the space.
This suggests a more complex mechanism - the first token decision affects subsequent computation.

## Key Findings

From testing Gemma 3 4B:

| Problem Type | Behavior | Diagnosis |
|--------------|----------|-----------|
| Simple 2-digit (`100-37`, `50-25`, `12*12`) | onset differs by 1 | MINOR DIFFERENCE |
| 3-digit addition (`156+287`) | onset=15 vs onset=3 | ONSET ROUTING |
| Large numbers (`999+1`) | onset=14 vs onset=4 | ONSET ROUTING |

**Key insight:** The model CAN compute all these - the format affects **when** it chooses
to emit the answer, not whether computation happens. For harder problems, the no-space
format triggers a "show work first" mode (outputs `?`, then restates problem with answer).

This is **answer-onset routing** (a discourse/style switch), not a "computation gate".

## See Also

- [introspect-format-sensitivity.md](introspect-format-sensitivity.md) - Layer-level format analysis
- [introspect-analyze.md](introspect-analyze.md) - Logit lens analysis
- [introspect-layer.md](introspect-layer.md) - Representation similarity
