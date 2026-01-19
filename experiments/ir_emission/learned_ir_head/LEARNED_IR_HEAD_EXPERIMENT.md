# Experiment: Learned IR Head

## Goal

Replace external regex parsing with a **learned projection head** that reads IR structure directly from hidden states.

```
Current Architecture:
  NL → LLM → text → regex → (op, a, b) → WASM → result
                      ↑
               external, brittle

Proposed Architecture:
  NL → LLM → hidden states → IR Head → (op, a, b) → WASM → result
                               ↑
                        learned, integrated
```

## The Insight

We already proved with dual-reward LoRA that layer 12 hidden states encode operation intent (add/subtract/multiply/divide with ~100% accuracy).

But we're still:
1. Generating text ("50 - 15 = ")
2. Regex parsing to extract operands
3. Building IR from parsed values

**The regex is the bottleneck.** If the model "knows" the operation at L12, why can't it also "know" the operands?

## Hypothesis

A small learned head can extract the complete IR structure (operation + operands) directly from hidden states, eliminating text generation and regex parsing entirely.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Input: "Subtract 15 from 50"                               │
│                         ↓                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Frozen Backbone (TinyLlama)                          │  │
│  │  Layers 0-11 → Layer 12 hidden states                 │  │
│  └───────────────────────────────────────────────────────┘  │
│                         ↓                                   │
│           h ∈ ℝ^(seq_len × hidden_dim)                      │
│                         ↓                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  IR Head (Learned)                                    │  │
│  │                                                       │  │
│  │  Pool: h_pooled = mean(h, dim=seq)  # or last token   │  │
│  │                                                       │  │
│  │  Branch 1: op_logits = W_op @ h_pooled + b_op         │  │
│  │            op = argmax(op_logits)  # 4 classes        │  │
│  │                                                       │  │
│  │  Branch 2: operand_1 = W_a @ h_pooled + b_a           │  │
│  │            (regression or binned classification)      │  │
│  │                                                       │  │
│  │  Branch 3: operand_2 = W_b @ h_pooled + b_b           │  │
│  │            (regression or binned classification)      │  │
│  └───────────────────────────────────────────────────────┘  │
│                         ↓                                   │
│  Output: IRInstruction(op=SUB, a=50, b=15)                  │
│                         ↓                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  WASM Backend (Deterministic)                         │  │
│  │  [i32.const 50, i32.const 15, i32.sub] → 35           │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## IR Head Design Options

### Option A: Regression Head
```python
class IRHeadRegression(nn.Module):
    def __init__(self, hidden_dim=2048):
        self.op_head = nn.Linear(hidden_dim, 4)      # 4 operations
        self.operand_a = nn.Linear(hidden_dim, 1)    # scalar
        self.operand_b = nn.Linear(hidden_dim, 1)    # scalar

    def forward(self, h_pooled):
        op_logits = self.op_head(h_pooled)
        a = self.operand_a(h_pooled).squeeze()
        b = self.operand_b(h_pooled).squeeze()
        return op_logits, a, b
```

**Pros**: Simple, continuous output
**Cons**: Regression for integers is awkward, may need rounding

### Option B: Binned Classification Head
```python
class IRHeadBinned(nn.Module):
    def __init__(self, hidden_dim=2048, num_bins=256):
        self.op_head = nn.Linear(hidden_dim, 4)
        self.operand_a = nn.Linear(hidden_dim, num_bins)  # classify into bins
        self.operand_b = nn.Linear(hidden_dim, num_bins)

    def forward(self, h_pooled):
        op_logits = self.op_head(h_pooled)
        a_logits = self.operand_a(h_pooled)
        b_logits = self.operand_b(h_pooled)
        return op_logits, a_logits, b_logits
```

**Pros**: Classification is what transformers do well
**Cons**: Limited range (0-255 with 256 bins)

### Option C: Digit-by-Digit Autoregressive
```python
class IRHeadDigits(nn.Module):
    def __init__(self, hidden_dim=2048, max_digits=4):
        self.op_head = nn.Linear(hidden_dim, 4)
        self.digit_head = nn.Linear(hidden_dim, 10)  # 0-9
        self.sign_head = nn.Linear(hidden_dim, 2)    # +/-
        # Decode digits autoregressively
```

**Pros**: Handles arbitrary integers
**Cons**: More complex, sequential decoding

## Training

### Dataset
```python
training_examples = [
    {"input": "Add 11 and 94", "op": "add", "a": 11, "b": 94},
    {"input": "Subtract 15 from 50", "op": "subtract", "a": 50, "b": 15},
    {"input": "What is 7 times 8?", "op": "multiply", "a": 7, "b": 8},
    {"input": "Divide 48 by 6", "op": "divide", "a": 48, "b": 6},
    # ... more examples with varied NL
]
```

### Loss Function
```python
def ir_head_loss(pred_op, pred_a, pred_b, true_op, true_a, true_b):
    # Structural loss: operation classification
    op_loss = cross_entropy(pred_op, true_op)

    # Semantic loss: operand accuracy
    if using_regression:
        a_loss = mse(pred_a, true_a)
        b_loss = mse(pred_b, true_b)
    else:  # binned classification
        a_loss = cross_entropy(pred_a, true_a)
        b_loss = cross_entropy(pred_b, true_b)

    # Combined with structural weight
    return 0.4 * op_loss + 0.3 * a_loss + 0.3 * b_loss
```

### Reward Signal (for RL fine-tuning)
```python
def structural_reward(pred_op, pred_a, pred_b):
    """Does this form a valid IR instruction?"""
    if pred_op not in [ADD, SUB, MUL, DIV]:
        return 0.0
    if not (0 <= pred_a <= 1000 and 0 <= pred_b <= 1000):
        return 0.0
    if pred_op == DIV and pred_b == 0:
        return 0.0  # Division by zero
    return 1.0  # Valid structure

def semantic_reward(pred_op, pred_a, pred_b, expected_result):
    """Is the result correct?"""
    result = execute_ir(pred_op, pred_a, pred_b)
    return 1.0 if result == expected_result else 0.0

def combined_reward(pred, expected):
    struct = structural_reward(*pred)
    if struct == 0:
        return 0.0  # Invalid structure
    semantic = semantic_reward(*pred, expected)
    return 0.5 + 0.5 * semantic  # Partial credit for valid structure
```

## Experiment Plan

### Phase 1: Baseline (Current Architecture)
- Run existing pipeline: NL → text → regex → WASM
- Measure: accuracy, failure modes, latency
- Document regex parse failures

### Phase 2: Train IR Head
```python
# Freeze backbone
model.requires_grad_(False)

# Add IR head after L12
ir_head = IRHeadBinned(hidden_dim=2048, num_bins=256)

# Training loop
for batch in dataloader:
    h = get_layer_12_hidden(model, batch.input_ids)
    h_pooled = h[:, -1, :]  # Last token

    op_logits, a_logits, b_logits = ir_head(h_pooled)
    loss = ir_head_loss(op_logits, a_logits, b_logits,
                        batch.op, batch.a, batch.b)
    loss.backward()
    optimizer.step()
```

### Phase 3: Evaluate
- **Structural accuracy**: % of outputs that form valid IR
- **Semantic accuracy**: % of outputs that produce correct result
- **Compare to regex**: Same inputs, which fails more?
- **Latency**: IR head vs text generation + regex

### Phase 4: Extend to Multi-Op
If single-op works, extend to sequences:
```python
class MultiOpIRHead(nn.Module):
    def __init__(self, hidden_dim, max_ops=3):
        self.num_ops_head = nn.Linear(hidden_dim, max_ops)
        self.op_heads = nn.ModuleList([...])
        self.operand_heads = nn.ModuleList([...])
```

## Success Criteria

| Metric | Target | Why |
|--------|--------|-----|
| Structural accuracy | >99% | Head should always output valid IR |
| Semantic accuracy | >95% | Match or exceed regex pipeline |
| Regex failures eliminated | 100% | No more parse errors |
| Latency reduction | >2x | Skip text generation |

## Key Questions to Answer

1. **Does L12 encode operands?** We know it encodes operation. Do operands emerge too?

2. **Binned vs regression?** Which works better for integer operands?

3. **Pooling strategy?** Last token vs mean pooling vs attention pooling?

4. **How much data?** Minimum examples needed for >95% accuracy?

5. **Generalization?** Does it handle unseen NL phrasings?

## Expected Outcome

If successful, this proves:
- The model's hidden states contain complete IR structure (not just operation)
- A small learned head can extract this structure
- No text generation or regex parsing required
- The "parser" is now part of the model weights

**The pipeline collapses from:**
```
NL → generate tokens → parse text → build IR → execute
```
**To:**
```
NL → project hidden states → execute
```

## Files to Create

```
experiments/ir_emission/
├── learned_ir_head/
│   ├── __init__.py
│   ├── heads.py          # IRHeadRegression, IRHeadBinned
│   ├── dataset.py        # Training data loader
│   ├── train.py          # Training loop
│   ├── evaluate.py       # Metrics and comparison
│   └── demo.py           # End-to-end demonstration
```
