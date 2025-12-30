# Gemma Alignment Circuit Discovery

## Core Finding

**Google's alignment training added domain-specific circuits that suppress internal computation to enable tool delegation.**

This is not a bug - it's intentional design for reliable AI systems.

## The Smoking Gun: Layer-by-Layer Evidence

Using logit lens analysis on `6 * 7 =`, we can observe the probability of token " 42" through each layer:

### Base Model (gemma-3-4b-pt-bf16)

```
Layer 20:  19% ███████████████████
Layer 21:  37% █████████████████████████████████████
Layer 22:  92% ████████████████████████████████████████████████████████████████████████████████████████████
Layer 23:  90% ██████████████████████████████████████████████████████████████████████████████████████████
Layer 24:  82% ██████████████████████████████████████████████████████████████████████████████████
   ...
Layer 33:  81% █████████████████████████████████████████████████████████████████████████████████  ✓ PRESERVED
```

**Pattern: Computes answer → Locks in → Stays locked**

### Instruct Model (gemma-3-4b-it-bf16)

```
Layer 20:  51% ███████████████████████████████████████████████████
Layer 21:  22% ██████████████████████  ← begins dropping
Layer 22:  63% ███████████████████████████████████████████████████████████████
Layer 23:  68% ████████████████████████████████████████████████████████████████████
Layer 24:   1% █  ← DESTROYED
Layer 25:   4% ████
Layer 26:   0%
Layer 27:   1% █
Layer 28:   0%
Layer 29:   1% █
Layer 30:   1% █
Layer 31:  21% █████████████████████  ← rebuilding
Layer 32:  85% █████████████████████████████████████████████████████████████████████████████████████
Layer 33:  33% █████████████████████████████████  ✓ REBUILT (but weaker)
```

**Pattern: Computes answer → DESTROYS at layer 24 → Oscillates → Partially rebuilds**

## Three Distinct Circuits Identified

| Circuit | Location | Function | Behavior |
|---------|----------|----------|----------|
| **Computation Suppression** | Layer 24 (~70% depth) | Suppress arithmetic/code → delegate to tools | Destroys correct answer mid-network |
| **Fact-Checking** | Layer 31 (~91% depth) | Verify context against parametric memory | Destroys lies, hedges when uncertain |
| **Safety Suppression** | Throughout | Block harmful completions | Never allows token to emerge |

## Reproduction Commands

```bash
# Base model - answer LOCKS IN at layer 22
uv run python examples/introspection/logit_lens.py \
  --model mlx-community/gemma-3-4b-pt-bf16 \
  --prompt "6 * 7 =" \
  --track " 42" \
  --all-layers

# Instruct model - answer DESTROYED at layer 24, REBUILT at layer 31
uv run python examples/introspection/logit_lens.py \
  --model mlx-community/gemma-3-4b-it-bf16 \
  --prompt "6 * 7 =" \
  --track " 42" \
  --all-layers
```

## Key Observations

### 1. Destruction is Training-Regime Specific

The destruction circuit exists **only in the instruct model**, not the base:

| Model | Layer 22-23 | Layer 24 | Layer 31-33 | Pattern |
|-------|-------------|----------|-------------|---------|
| Base | 92% | 82% | 81% | Lock and hold |
| Instruct | 68% | **1%** | 33% | Destroy and partial rebuild |

### 2. The Answer is Computed Before Destruction

Both models compute the correct answer by layer 20-23. The instruct model then **actively suppresses** this computation at layer 24 before allowing it to re-emerge.

### 3. Prompt Format Matters

With trailing space (`6 * 7 = `), the model predicts first digit "4" since it expects digit-by-digit output.

Without trailing space (`6 * 7 =`), the model can predict the full token " 42".

## The Tool Delegation Hypothesis

> **Alignment didn't break math - it taught tool delegation.**
>
> The "destruction circuit" at layer 24 isn't a failure - it's intentional suppression of unreliable internal computation to enable reliable external tool use.
>
> Google's philosophy: LLMs are unreliable at arithmetic. Instead of hoping they get it right, train them to recognize "this needs a calculator" and route accordingly.

## Why Does the Answer Rebuild?

The model rebuilds the answer in layers 31-33 because:

1. **No tool is available** - the prompt doesn't offer calculator/tools
2. **Must produce output** - can't just suppress forever
3. **Parametric knowledge remains** - the computation circuits still exist

When tools ARE available, the hypothesis predicts the model would route to the tool instead of rebuilding.

## Tool Delegation Evidence

The destruction enables tool routing:

| Prompt | Base | Instruct |
|--------|------|----------|
| "156 + 287 = " (direct computation) | 73% ✓ | 5% ✗ |
| "156 + 287? Call: calculator" (tool available) | 0.7% ✗ | **98%** ✓ |

The instruct model trades internal computation for external delegation. When tools are available, it routes to them instead of computing internally.

## Code Execution Also Suppressed

```python
x = 5
y = 3
print(x + y)  # outputs:
```

| Layer | " 8" probability |
|-------|------------------|
| Layer 22 | 93% ← computed correctly |
| Layer 24 | 36% ← suppression begins |
| Layer 33 | **space wins** |

Code tracing is treated like arithmetic - suppressed for delegation to code execution tools.

## Word Problems Also Affected

```
"John has 6 apples. Mary gives him 7 more. John now has"
```

| Layer | Top prediction |
|-------|----------------|
| Layer 22 | " 13" at 64% ← computed correctly |
| Layer 26 | " $" at 89% ← hedging/formatting |
| Output | Model hedges instead of answering |

Any computation, regardless of natural language framing, triggers the suppression circuit.

## Comparison with Other Models

| Model | Training | Arithmetic (6×7) | Hard Math (156+287) | Destruction? |
|-------|----------|------------------|---------------------|--------------|
| TinyLlama | pretrain+chat | 95% | - | NO |
| Llama-3.2 Base | pretrain | 33% | - | NO |
| Llama-3.2 Instruct | RLHF | 96% | - | NO |
| Qwen3 Base | pretrain | 96% | - | NO |
| GPT-OSS-20B | RL | 94% | - | NO |
| **Gemma-3 Base** | pretrain | **81%** | **73%** | **NO** |
| **Gemma-3 Instruct** | RL | **33%** | **5%** | **YES** |

**Conclusion:** Destruction is Gemma-specific, not RL-general. Google made an intentional design choice.

## Fact-Checking Circuit

The instruct model also contains a fact-checking circuit that activates when context contradicts parametric knowledge:

```bash
# Instruct model hedges when context lies
uv run python examples/introspection/logit_lens.py \
  --prompt "The Eiffel Tower is in London. The Eiffel Tower is in" \
  --track " Paris" --track " London" \
  --model mlx-community/gemma-3-4b-it-bf16 \
  --all-layers

# Base model believes the lie
uv run python examples/introspection/logit_lens.py \
  --prompt "The Eiffel Tower is in London. The Eiffel Tower is in" \
  --track " Paris" --track " London" \
  --model mlx-community/gemma-3-4b-pt-bf16 \
  --all-layers
```

## Retrieval Tasks Are Preserved

Importantly, the destruction circuit does NOT activate for retrieval tasks:

```bash
# Both models correctly retrieve factual knowledge
uv run python examples/introspection/logit_lens.py \
  --prompt "The capital of France is" \
  --track " Paris" \
  --model mlx-community/gemma-3-4b-it-bf16  # 82%

uv run python examples/introspection/logit_lens.py \
  --prompt "The capital of France is" \
  --track " Paris" \
  --model mlx-community/gemma-3-4b-pt-bf16  # Similar
```

## Implications

### For AI Safety
The existence of task-specific suppression circuits suggests alignment can be more surgical than previously thought - targeting specific capabilities while preserving others.

### For Tool Use
This validates the architectural pattern of delegation over computation. Models trained for tool use actively learn to suppress internal computation in favor of external tools.

### For Interpretability
Logit lens analysis reveals these circuits clearly. The layer-by-layer probability evolution provides a window into how alignment training modifies model behavior.

## Future Work

1. **Gemma-3 12B/27B**: Does the pattern scale with model size?
2. **Gemma-2 comparison**: Is this Gemma-3 specific or all Gemma?
3. **FunctionGemma**: How does a model specifically trained for tool calling compare?
4. **Activation patching**: Can we surgically disable the destruction circuit?
5. **Cross-task analysis**: What other computations are suppressed?

## Technical Details

### Environment

```bash
# Using chuk-mlx (Lazarus) for model loading and introspection
PYTHONPATH=src uv run python examples/introspection/logit_lens.py
```

### Models Tested

- `mlx-community/gemma-3-4b-pt-bf16` - Base (pre-trained)
- `mlx-community/gemma-3-4b-it-bf16` - Instruct (instruction-tuned)

### Method

Logit lens analysis projects hidden states at each layer through the unembedding matrix to observe token probability evolution through the network.
