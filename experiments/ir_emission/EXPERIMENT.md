# Neural Compiler: NL → WASM IR → Execute

## Research Question

**Can transformers serve as a semantic frontend that emits executable IR, achieving Turing completeness through composition with a deterministic runtime?**

This experiment tests the hypothesis that "Chain-of-Thought is format normalization, not reasoning" - the model's job is to translate varied natural language into canonical forms that downstream circuits (or runtimes) can process deterministically.

## Background

### The Core Insight

```
Varied NL:     "The difference of 69 and 49 is"  →  ~60% classifier accuracy
Canonical:     "69 - 49 = "                       →  100% classifier accuracy
```

The model doesn't need to "think" - it needs to normalize. Once in canonical form, a deterministic runtime (WASM) handles the computation.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NEURAL COMPILER                          │
├─────────────────────────────────────────────────────────────┤
│  STAGE 1: FRONTEND (NL → Canonical)                         │
│    "Janet has 50 apples. She gives away 15."               │
│                     ↓ (few-shot prompting)                  │
│    "50 - 15 = "                                             │
├─────────────────────────────────────────────────────────────┤
│  STAGE 2: MIDDLE-END (Canonical → IR)                       │
│    L12 logit lens → operation classifier                    │
│                     ↓ (dual-reward LoRA)                    │
│    Operation: "subtract"                                    │
├─────────────────────────────────────────────────────────────┤
│  STAGE 3: BACKEND (IR → Execute)                            │
│    [i32.const 50, i32.const 15, i32.sub]                   │
│                     ↓ (WASM runtime)                        │
│    Result: 35                                               │
└─────────────────────────────────────────────────────────────┘
```

## Running the Experiment

```bash
# Run via framework
lazarus experiment run ir_emission

# View results
lazarus experiment status ir_emission

# Or run directly
python -m chuk_lazarus.cli.main experiment run ir_emission
```

### Running Individual Pipelines

```bash
# Single-op pipeline (100% accuracy)
python experiments/ir_emission/full_pipeline_v2.py

# Multi-op chains (75% accuracy)
python experiments/ir_emission/multiop_pipeline.py

# Loop constructs (100% accuracy)
python experiments/ir_emission/loop_pipeline.py
```

## Configuration

See `config.yaml` for parameters:

```yaml
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

pipelines:
  - single_op    # Single arithmetic operations
  - multi_op     # Multi-operation chains
  - loop         # Loop constructs (Turing completeness demo)

parameters:
  decision_layer_pct: 0.55    # Layer for logit lens classification
  classifier_checkpoint: checkpoints/dual_reward/final/adapters.safetensors
```

## Results

### Summary

| Pipeline | Accuracy | Test Cases | Description |
|----------|----------|------------|-------------|
| single_op | **100%** | 12/12 | Single arithmetic operations |
| multi_op | **75%** | 6/8 | Multi-operation chains |
| loop | **100%** | 9/9 | Loop constructs (sum/product/count) |

### Pipeline Details

#### Single-Op (full_pipeline_v2.py)

Converts varied natural language to single arithmetic operations:

```
Input:  "Janet has 50 apples. She gives away 15. How many remain?"
Canon:  "50 - 15 = "
Op:     subtract
IR:     [0x41 0x32 0x41 0x0f 0x6b]  (i32.const 50, i32.const 15, i32.sub)
Result: 35
```

Test cases:
- Imperative: "Add 11 and 94", "Subtract 49 from 69"
- Declarative: "The sum of 25 and 17 is", "The difference of 100 and 37 is"
- Questions: "What is 12 times 9?", "What is 144 divided by 12?"
- Word problems: "Janet has 50 apples. She gives away 15."

#### Multi-Op (multiop_pipeline.py)

Handles sequential operations with stack-based execution:

```
Input:  "16 - 3, then multiply by 5"
Steps:  Step 1: 16 - 3 = 13
        Step 2: 13 * 5 = 65
IR:     [i32.const 16, i32.const 3, i32.sub, i32.const 5, i32.mul]
Result: 65
```

**Note**: Parenthesized expressions like "(8 + 4) * 3" need improved parsing.

#### Loop (loop_pipeline.py)

Demonstrates Turing completeness - the transformer emits loop intent, WASM executes:

```
Input:  "Sum 1 to 100"
Parsed: type=sum, start=1, end=100
WASM:   loop { acc += counter; counter++; if counter <= 100: branch }
Result: 5050 (computed via 100 iterations in WASM)
```

Test cases:
- Sum: "Sum 1 to 10" → 55
- Product: "Multiply 1 to 5" → 120 (5!)
- Count: "Count down from 10" → 0

## Key Findings

### 1. Few-Shot > Fine-Tuning for Normalization

We tried LoRA fine-tuning for the normalizer but achieved only ~50% accuracy. Switching to few-shot prompting on the base model achieved **100%** without any training.

### 2. Logit Lens Classification Works

At layer 12 (~55% depth), the dual-reward trained model produces high-confidence operation tokens:
- "add", "subtract", "multiply", "divide"
- 100% classification accuracy on canonical inputs

### 3. Turing Completeness via Composition

Transformers alone are bounded computation. By emitting intent that WASM executes, we achieve unbounded computation:
- "Sum 1 to 1000000" → 1 forward pass → 1,000,000 loop iterations

### 4. CoT Demystified

Chain-of-thought isn't "reasoning" - it's **format normalization**. The model rewrites varied inputs into a canonical form that downstream circuits process deterministically.

## Files

```
ir_emission/
├── EXPERIMENT.md              # This file
├── README.md                  # Detailed technical documentation
├── experiment.py              # ExperimentBase implementation
├── config.yaml                # Configuration
│
├── pipelines/                 # Pipeline implementations
│   ├── __init__.py
│   ├── base.py                # NeuralCompilerBase
│   ├── single_op.py           # Single operation pipeline
│   ├── multi_op.py            # Multi-operation pipeline
│   └── loop.py                # Loop pipeline
│
├── codebook.py                # IR opcode vocabulary, WASM encoding
├── wasm_runtime.py            # WASM module builder and executor
│
├── full_pipeline_v2.py        # Standalone single-op script
├── multiop_pipeline.py        # Standalone multi-op script
├── loop_pipeline.py           # Standalone loop script
│
├── data/                      # Training data
├── checkpoints/               # Trained LoRA weights
│   └── dual_reward/final/     # Classifier checkpoint
└── results/                   # Experiment results
```

## Training the Classifier

The dual-reward classifier is trained separately:

```bash
# Generate training data
python experiments/ir_emission/generate_normalizer_data_v2.py

# Train dual-reward classifier
python experiments/ir_emission/train_phase1.py
```

Training uses:
- Classifier loss at L12 (weight=0.7): Cross-entropy for operation token
- Answer loss at final layer (weight=0.3): Standard LM loss
- LoRA on v_proj, o_proj only

## Why This Matters

1. **Clean separation of concerns**: Frontend (learnable) → Middle-end (trained) → Backend (deterministic)
2. **Debuggability**: Every stage is inspectable
3. **Composability**: The architecture extends to conditionals, memory, functions
4. **Insight into LLMs**: "Reasoning" may largely be format normalization

## Future Directions

1. **Conditional IR**: `if/else` constructs for branching logic
2. **Memory operations**: `i32.load`, `i32.store` for array access
3. **Function calls**: Multi-function WASM modules
4. **Recursive patterns**: Factorial, Fibonacci via recursion
5. **Float operations**: `f32.add`, `f32.mul` for floating-point math
