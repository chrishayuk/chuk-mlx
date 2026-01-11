# Virtual Math Expert: Teaching MoE Routers to Use Tools

## Video Demo

Run the full narrative demo:

```bash
# Full demo with all sections (press Enter between sections)
uv run python examples/introspection/experiments/moe/virtual_expert_video_demo.py

# Individual sections
uv run python ... --section multi-use          # Expert hijacking breaks other capabilities
uv run python ... --section layer-specificity  # Which layer to intercept?
uv run python ... --section routing-ambiguity  # Pattern matching can't understand intent
uv run python ... --section calibration-viz    # Learned direction separates math from non-math
uv run python ... --section solution           # Virtual expert slot in action
```

---

## The Problem

Large language models are notoriously bad at arithmetic. A 20B parameter model might confidently answer "127 × 89 = 11263" when the correct answer is 11303. The model isn't uncertain—it's *confidently wrong*.

Traditional approaches use post-hoc verification: generate an answer, check if it looks like math, maybe run it through a calculator. But this is wasteful—we've already done the forward pass through billions of parameters.

**What if the model could learn to route math queries to an external calculator, just like it routes tokens to different experts?**

## The Insight: MoE Routing as Tool Use

Mixture of Experts (MoE) models already have a mechanism for conditional computation: a router that decides which expert(s) should handle each token. GPT-OSS, for example, has 32 experts per layer but only activates 4 for each token.

The router learns patterns like:
- "This looks like code → route to experts 3, 7, 12, 28"
- "This is natural language → route to experts 1, 5, 9, 15"

**Our insight**: We can extend this to include *virtual experts*—experts that don't have neural network weights, but instead execute Python code.

```
Token: "127 * 89 = "
     ↓
  Router Decision
     ↓
┌────────────────────────────────────────┐
│  Expert 0: Neural weights (language)   │
│  Expert 1: Neural weights (code)       │
│  Expert 2: Neural weights (reasoning)  │
│  ...                                   │
│  Expert 31: Neural weights             │
│  ─────────────────────────────────────│
│  Virtual Expert: Python eval()    ← NEW│
└────────────────────────────────────────┘
     ↓
  Output: "11303"
```

## Three Approaches

We implemented three different strategies for creating virtual math experts:

### Approach 1: Expert Hijacking

**Strategy**: Identify which existing expert handles math-related tokens, then intercept its forward pass.

```python
from chuk_lazarus.introspection import ExpertHijacker

hijacker = ExpertHijacker(model, tokenizer)
result = hijacker.solve("127 * 89 = ")
# Answer: 11303 (via expert 6 @ layer 12)
```

**How it works**:
1. Run math prompts through the model, observe which experts activate
2. Identify the "math expert" (e.g., expert 6 gets selected most often for arithmetic)
3. When math is detected in input, intercept the forward pass
4. Replace the expert's contribution with the computed result

**Pros**: Simple, uses existing routing decisions
**Cons**: Requires identifying which expert to hijack; may interfere with non-math uses of that expert

> **Why This Breaks**: See [Failure Case 1](#failure-case-1-the-multi-use-expert-problem) below.

### Approach 2: Virtual Expert Slot

**Strategy**: Add a new "virtual" expert that the router can select. Train the router to recognize when to use it.

```python
from chuk_lazarus.introspection import VirtualExpertSlot

slot = VirtualExpertSlot(model, tokenizer)
result = slot.solve("127 * 89 = ")
# Answer: 11303 (routing score: 0.659)
```

**How it works**:
1. Calibrate on math vs non-math prompts to learn a "math direction" in activation space
2. For each input, project onto this direction to get a routing score
3. If score exceeds threshold, route to the virtual expert (Python)
4. Otherwise, use the model's normal generation

**Pros**: Learnable threshold, explicit routing decision
**Cons**: Requires calibration data; binary decision may not capture nuance

### Approach 3: Hybrid Embedding Injection

**Strategy**: Use introspection to detect model confidence at key layers. Only intervene when the model is uncertain.

```python
from chuk_lazarus.introspection import HybridEmbeddingInjector

hybrid = HybridEmbeddingInjector(model, tokenizer)
result = hybrid.solve("127 * 89 = ")
# Answer depends on model confidence
```

**How it works**:
1. Run a forward pass, capturing hidden states at key layers (e.g., layers 20, 22, 24, 28)
2. At each layer, project to vocabulary and check probability of correct first digit
3. If max confidence is below threshold, delegate to Python
4. If model is confident, trust its output

**Pros**: Most surgical—respects model confidence, only intervenes when needed
**Cons**: Model can be confidently wrong; more complex implementation

## Why Hijacking Fails: Three Lessons

Before celebrating the virtual expert slot solution, it's worth understanding *why* the simpler hijacking approach breaks. These failure cases illuminate what a principled solution needs to address.

### Failure Case 1: The Multi-Use Expert Problem

The naive approach: "Find the expert that handles math, hijack it."

But when we analyze expert activations across different prompt types:

```
Expert     MATH       CODE       LOGIC      LANGUAGE
--------------------------------------------------
Expert 6   4          0          1          1          ← 'math expert'
Expert 7   3          0          0          0
Expert 14  3          0          4          1
Expert 21  2          1          1          4
```

Expert 6 lights up for math—but also for logic. Expert 14 handles both math AND logic even more. These aren't specialists; they're generalists with preferences.

**The problem**: If we hijack Expert 6 for math, we might break logic or symbolic reasoning. The same expert that computes "127 * 89" also helps with "If A implies B, then..."

**Visual for video**: Show a prompt like "If A implies B, then" routing to Expert 6. Then show what happens when we hijack it—suddenly logic completion degrades.

### Failure Case 2: The Layer Specificity Issue

Which layer should we intercept? Let's trace the probability of the correct first digit ("1" for 11303) through all 24 layers:

```
Layer    P("1")     Interpretation
──────────────────────────────────────────
L0-L15   ~0%        Building representation
L16      7%         Starting to emerge
L17      19%        Computation happening
L18      54%        ← PEAK computation
L19      80%        ← Model "knows" the answer here
L20      71%        Starting to forget
L21-L23  ~0%        Committed to wrong path (outputs "112")
```

**The problem**:
- Hijack before L18 → computation hasn't happened yet
- Hijack after L20 → model already committed to wrong answer
- The "sweet spot" is narrow and problem-dependent

**Visual for video**: Animate the probability flowing through layers. Show the peak at L18-19, then the catastrophic drop at L21 where the model "forgets" the right answer and commits to the wrong one.

### Failure Case 3: Routing Ambiguity

Pattern matching (regex) can't understand intent:

```
Prompt                                  Intent       Should Compute?
────────────────────────────────────────────────────────────────────
"127 * 89 = "                          exact        YES ✓
"127 * 89 is approximately"            approximate  NO  ✗
"Is 127 * 89 greater than 10000?"      comparison   NO  ✗
"How would you compute 127 * 89?"      explanation  NO  ✗
```

But simple regex `\d+ \* \d+` matches ALL of these.

**The problem**: Hijacking is binary—all or nothing. It can't distinguish:
- "Compute this exactly" vs "Give me a rough estimate"
- "What's the answer?" vs "What's the method?"
- "Calculate" vs "Compare"

**Visual for video**: Show the regex pattern lighting up on all prompts. Then show the disastrous results when "127 * 89 is approximately..." returns "11303" instead of "about 11,000" or "around ten thousand".

### The Insight: What We Actually Need

These failure cases tell us what a solution must provide:

1. **Additive, not substitutive**: Don't replace existing experts; add a new virtual one
2. **Learned, not pattern-matched**: Use the model's own activation space to decide
3. **Granular routing scores**: Not binary—a continuous signal we can threshold
4. **Layer-agnostic**: Operate at the routing level, not by intercepting specific layers

This is exactly what the Virtual Expert Slot provides.

---

## Benchmark Results

Tested on GPT-OSS 20B (32 experts, 4 active per token):

| Approach | Model Only | With Virtual Expert | Improvement |
|----------|------------|---------------------|-------------|
| Expert Hijacking | 73.3% | **100%** | +26.7% |
| Virtual Expert Slot | 73.3% | **100%** | +26.7% |
| Hybrid Injection | 73.3% | 86.7% | +13.3% |

### Per-Problem Breakdown

```
Problem             Model Answer    Correct    Virtual Expert
─────────────────────────────────────────────────────────────
2 + 2 =             4              ✓
5 * 5 =             25             ✓
10 - 3 =            7              ✓
23 * 17 =           391            ✓
127 * 89 =          11263          ✗          → 11303 ✓
456 * 78 =          35712          ✗          → 35568 ✓
999 * 888 =         887112         ✓
1234 + 5678 =       7000           ✗          → 6912 ✓
999 * 999 =         998001         ✓
12345 + 67890 =     80235          ✓
```

The model handles simple arithmetic but fails on harder multi-digit multiplication. The virtual expert catches these failures.

## Why Hybrid Sometimes Fails

The hybrid approach trusts the model when confidence is high. But confidence ≠ correctness:

```
Prompt: 127 * 89 =
Model confidence: 71.1%  (HIGH - trusts model)
Model answer: 11263
Correct answer: 11303
Result: WRONG (model was confidently incorrect)
```

This reveals an important insight: **for arithmetic, model confidence doesn't correlate well with correctness**. The model has memorized some multiplication facts but applies faulty heuristics to novel problems—and it doesn't know the difference.

## Implementation Details

### Safe Math Evaluation

All approaches use `SafeMathEvaluator`, which parses expressions via Python's AST to prevent code injection:

```python
from chuk_lazarus.introspection import SafeMathEvaluator

evaluator = SafeMathEvaluator()
evaluator.evaluate("127 * 89")      # → 11303
evaluator.evaluate("sqrt(144)")     # → 12.0
evaluator.evaluate("2 ** 10")       # → 1024
evaluator.evaluate("__import__")    # → None (blocked)
```

Supported operations:
- Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- Functions: `sqrt`, `sin`, `cos`, `tan`, `log`, `exp`, `abs`, `round`, `min`, `max`
- Constants: `pi`, `e`, `inf`

### Expert Identification

The `ExpertHijacker` finds which expert handles math by running test prompts and counting activations:

```python
math_prompts = ["127 * 89 = ", "456 + 789 = ", "100 - 37 = "]

# For each prompt, track which experts are selected at the target layer
# The expert with highest activation count becomes our target
# Result: Expert 6 @ Layer 12 for GPT-OSS 20B
```

### Routing Calibration

The `VirtualExpertSlot` learns a math direction via contrastive examples:

```python
math_prompts = ["127 * 89 = ", "456 + 789 = ", ...]
non_math_prompts = ["The capital of France is ", "Hello, how are you?", ...]

# Get hidden states before MoE layer
math_activations = [get_hidden(p) for p in math_prompts]
non_math_activations = [get_hidden(p) for p in non_math_prompts]

# Compute difference of means
math_direction = mean(math_activations) - mean(non_math_activations)
math_direction = normalize(math_direction)

# For new input: score = dot(hidden_state, math_direction)
# High score → route to virtual expert
```

## Usage

### Quick Start

```python
from chuk_lazarus.introspection import create_virtual_expert

# Load your MoE model
model, tokenizer = load_model("openai/gpt-oss-20b")

# Create virtual expert (choose approach)
expert = create_virtual_expert(model, tokenizer, approach="hijack")

# Solve math problems
result = expert.solve("127 * 89 = ")
print(result.answer)  # "11303"
print(result.is_correct)  # True
```

### Compare All Approaches

```python
from chuk_lazarus.introspection import demo_all_approaches

results = demo_all_approaches(model, tokenizer, problems=[
    "127 * 89 = ",
    "456 * 78 = ",
    "999 * 888 = ",
])

for name, analysis in results.items():
    print(f"{name}: {analysis.virtual_accuracy:.1%}")
```

### Command Line

```bash
# Compare all approaches on one problem
uv run python examples/introspection/experiments/moe/virtual_math_expert.py \
    --model openai/gpt-oss-20b \
    --prompt "127 * 89 = "

# Full benchmark
uv run python examples/introspection/experiments/moe/virtual_math_expert.py \
    --model openai/gpt-oss-20b \
    --benchmark

# Interactive mode
uv run python examples/introspection/experiments/moe/virtual_math_expert.py \
    --model openai/gpt-oss-20b \
    --interactive
```

## Architectural Implications

### MoE as a Tool-Use Framework

This work suggests MoE architectures are naturally suited for tool use:

1. **Routing is already learned**: The model already decides which expert handles which tokens
2. **Sparse activation**: Only a few experts fire per token, so adding a "tool expert" has minimal overhead
3. **Differentiable selection**: The router's softmax over experts could be extended to include tool probabilities

### Future Directions

**1. Learned Routing to Tools**

Instead of heuristic detection, train the router end-to-end to select the virtual expert:

```python
# Add virtual expert logit to router output
router_logits = [expert_0, expert_1, ..., expert_31, virtual_math]
# Train with reward signal based on answer correctness
```

**2. Multiple Virtual Experts**

Extend beyond math to other tools:

```python
virtual_experts = {
    "math": PythonCalculator(),
    "search": WebSearchTool(),
    "code": CodeInterpreter(),
    "memory": VectorDatabase(),
}
```

**3. Embedding Injection**

Instead of replacing model output, inject the computed answer as an embedding and let the model "read" it:

```python
# Compute answer
answer = "11303"
# Embed as if it were in context
answer_embedding = embed(answer)
# Inject into residual stream at key layer
hidden_states[layer] += answer_embedding
# Continue forward pass - model now "knows" the answer
```

**4. Confidence Calibration**

Train a separate head to predict when the model will be wrong:

```python
# Instead of using model confidence (unreliable for math)
# Train: P(model_wrong | hidden_state)
# Route to tool when P(wrong) > threshold
```

## Comparison to Other Approaches

| Approach | When to Delegate | Integration | Training Required |
|----------|------------------|-------------|-------------------|
| Tool-use prompting | Model decides via text | External API call | Prompt engineering |
| ReAct/Chain-of-thought | Model reasons step-by-step | Text-based | Few-shot examples |
| **Virtual Expert (ours)** | Router decides | Integrated in forward pass | Optional calibration |
| Toolformer | Model learns tool tokens | Fine-tuned embeddings | Full fine-tuning |

Our approach is unique in leveraging the MoE architecture's existing routing mechanism, requiring no fine-tuning and minimal overhead.

## Conclusion

MoE models already have the machinery for conditional computation—we're just extending it to include "virtual experts" that execute code instead of neural network weights.

The key insight: **routing to tools is just another form of expert selection**. By framing tool use this way, we get:

- Seamless integration with existing MoE inference
- No fine-tuning required
- Configurable confidence thresholds
- 100% accuracy on arithmetic (when routing is aggressive)

The model doesn't need to "understand" math. It just needs to recognize "this looks like math" and route to something that does.

---

## API Reference

### Classes

- `VirtualMathExpert` - Abstract base class
- `ExpertHijacker` - Approach 1: Hijack existing expert
- `VirtualExpertSlot` - Approach 2: Virtual expert routing
- `HybridEmbeddingInjector` - Approach 3: Confidence-based injection
- `SafeMathEvaluator` - Secure math expression evaluation

### Functions

- `create_virtual_expert(model, tokenizer, approach)` - Factory function
- `demo_all_approaches(model, tokenizer, problems)` - Benchmark all approaches

### Result Classes

- `VirtualExpertResult` - Single problem result
- `VirtualExpertAnalysis` - Benchmark analysis
- `VirtualExpertApproach` - Enum of approach types
