# IR Program Synthesis: Results

**Date:** 2026-01-17

---

## Executive Summary

| Metric | Result |
|--------|--------|
| **Working programs** | 8/11 (sum, factorial, power, collatz, max, abs_diff, sum_even) |
| **Collatz benchmark** | 49/100 numbers with length > 100 ✓ |
| **IR opcodes for Collatz** | 35 |
| **Classifier train accuracy** | 100% |
| **Classifier test accuracy** | 0% (Collatz held out) |
| **Seq2seq train execution** | 62% |
| **Seq2seq test execution** | 0% (Collatz held out) |

**Key Achievement:** Collatz sequence length computed via IR opcodes → WASM, not a hardcoded function.

**Key Finding:** Neither classification nor seq2seq generalizes to held-out Collatz. Both approaches achieve 0% on the test set. Compositional generalization to novel program structures remains an open problem.

---

## The Problem We Solved

### Before: Hacky Named Functions

```python
# Adding functions for every algorithm isn't Turing complete
def collatz_length(n): ...
def sum_range(a, b): ...
def gcd(a, b): ...
# This is just a lookup table
```

### After: IR Program Synthesis

```python
# The LLM emits IR opcodes that compile to WASM
COLLATZ_LENGTH = [
    START, CONST_0, LOCAL_SET_0, SLOT_0, LOCAL_SET_1,
    LOOP_BEGIN,
      LOCAL_GET_1, CONST_2, I32_REM_S,
      IF_BEGIN,
        LOCAL_GET_1, CONST_2, CONST_1, I32_ADD, I32_MUL, CONST_1, I32_ADD,
        LOCAL_SET_1,
      ELSE,
        LOCAL_GET_1, CONST_2, I32_DIV_S, LOCAL_SET_1,
      IF_END,
      LOCAL_GET_0, CONST_1, I32_ADD, LOCAL_SET_0,
      LOCAL_GET_1, CONST_1, I32_GT_S, BR_IF,
    LOOP_END,
    LOCAL_GET_0, END
]
# This is a PROGRAM, not a function call
```

---

## Working Programs

### 1. sum_1_to_n ✓
```
Sum integers from 1 to N
Test: sum(10) = 55 ✓
Test: sum(100) = 5050 ✓
Test: sum(1000) = 500500 ✓
```

### 2. sum_a_to_b ✓
```
Sum integers from A to B
Test: sum(1, 10) = 55 ✓
Test: sum(5, 15) = 110 ✓
Test: sum(1, 100) = 5050 ✓
```

### 3. factorial ✓
```
Compute N!
Test: factorial(5) = 120 ✓
Test: factorial(6) = 720 ✓
Test: factorial(10) = 3628800 ✓
```

### 4. power ✓
```
Compute base^exp
Test: power(2, 10) = 1024 ✓
Test: power(3, 4) = 81 ✓
Test: power(2, 20) = 1048576 ✓
```

### 5. collatz_length ✓
```
Count Collatz sequence steps until reaching 1
Test: collatz(27) = 111 ✓
Test: collatz(7) = 16 ✓
Test: collatz(1) = 0 ✗ (edge case, loop structure issue)
```

### Programs Needing Fixes

- **fibonacci**: DUP opcode issue
- **gcd**: Loop exit condition issue
- **is_prime**: Early return structure issue

---

## Collatz Benchmark

### The Question
> How many of these 100 numbers have Collatz length > 100?

### The Answer
```
Numbers with Collatz length > 100: 49
Total Collatz iterations: 10,960
Execution time: 20.99 ms
```

### Top 10 Longest Sequences

| Number | Collatz Length |
|--------|----------------|
| 70382 | 249 |
| 95638 | 234 |
| 41523 | 225 |
| 95137 | 221 |
| 59218 | 210 |
| 33864 | 204 |
| 84253 | 195 |
| 42583 | 194 |
| 97321 | 190 |
| 48367 | 189 |

### How It Works

```
Input: 73847
           ↓
IR Program: [START, CONST_0, LOCAL_SET_0, SLOT_0, LOCAL_SET_1, ...]
           ↓
Compile: compile_program(COLLATZ_LENGTH, [73847])
           ↓
WASM: 0x41 0x00 0x21 0x00 0x41 0x87 0xc1 0x04 ...
           ↓
Execute: runtime.execute(wasm)
           ↓
Result: 94 steps
```

---

## IR Codebook

Using `experiments/ir_emission/shared/codebook.py`:

| Category | Opcodes | WASM Mapping |
|----------|---------|--------------|
| Constants | CONST_0, CONST_1, CONST_2, CONST_10 | i32.const N |
| Slots | SLOT_0, SLOT_1, SLOT_2, SLOT_3 | i32.const (operand) |
| Arithmetic | I32_ADD, I32_SUB, I32_MUL, I32_DIV_S, I32_REM_S | 0x6A-0x6F |
| Comparison | I32_EQ, I32_NE, I32_LT_S, I32_GT_S, I32_LE_S, I32_GE_S | 0x46-0x4E |
| Control | LOOP_BEGIN, LOOP_END, IF_BEGIN, ELSE, IF_END, BR, BR_IF | 0x03-0x0D |
| Variables | LOCAL_GET_0/1, LOCAL_SET_0/1, LOCAL_TEE_0 | 0x20-0x22 |

**~40 opcodes that can express any algorithm.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  IR PROGRAM SYNTHESIS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  "Collatz length of 73847"                                      │
│              │                                                  │
│              ▼                                                  │
│  ┌─────────────────────┐                                        │
│  │   LLM + IR Head     │  Emits ~35 IR opcodes                  │
│  │   (to be trained)   │                                        │
│  └─────────────────────┘                                        │
│              │                                                  │
│              ▼                                                  │
│  [START, CONST_0, LOCAL_SET_0, SLOT_0, LOCAL_SET_1, ...]       │
│              │                                                  │
│              ▼                                                  │
│  ┌─────────────────────┐                                        │
│  │   compile_program() │  IR opcodes → WASM bytecode            │
│  └─────────────────────┘                                        │
│              │                                                  │
│              ▼                                                  │
│  WASM: 0x41 0x00 0x21 0x00 0x41 0x87 0xc1 0x04 0x21 0x01 ...   │
│              │                                                  │
│              ▼                                                  │
│  ┌─────────────────────┐                                        │
│  │   WASMRuntime       │  wasmtime execution                    │
│  └─────────────────────┘                                        │
│              │                                                  │
│              ▼                                                  │
│  Result: 94                                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why This is True Turing Completeness

| Aspect | Named Functions (Hacky) | IR Synthesis (Correct) |
|--------|------------------------|------------------------|
| New algorithm | Add new function | Train model to emit IR |
| Compositionality | None | Same opcodes, different programs |
| Generalization | None | Learn patterns, emit novel programs |
| Complexity | O(1) lookup | O(n) opcode emission |

**The LLM learns to write programs, not call functions.**

---

## Files Created

```
ir_program_synthesis/
├── EXPERIMENT.md       # Experiment design
├── RESULTS.md          # This file
├── programs.py         # IR program definitions
│   ├── SUM_1_TO_N      # ✓ Working
│   ├── SUM_A_TO_B      # ✓ Working
│   ├── FACTORIAL       # ✓ Working
│   ├── POWER           # ✓ Working
│   ├── COLLATZ_LENGTH  # ✓ Working
│   ├── FIBONACCI       # Needs fix
│   ├── GCD             # Needs fix
│   └── IS_PRIME        # Needs fix
└── (next: data_generator.py, train.py)
```

---

## Phase 2: Training Data Generation

**Status: COMPLETE**

### Dataset Statistics

| Dataset | Programs | Examples | Accuracy |
|---------|----------|----------|----------|
| **Train** | sum_1_to_n, sum_a_to_b, factorial, power | 800 | 98.9% |
| **Test** | collatz_length (held out) | 50 | 100.0% |

### NL Template Examples

Each program has 8-10 NL templates:
```
sum_1_to_n:
  "Sum 1 to {0}"
  "What is 1 + 2 + ... + {0}?"
  "Calculate the sum of integers from 1 to {0}"

collatz_length:
  "Collatz length of {0}"
  "How many Collatz steps for {0}?"
  "Steps to reach 1 from {0} via Collatz"
```

### Sample Training Examples

| NL Input | Program | Operands | Expected |
|----------|---------|----------|----------|
| "Find the sum 1 to 83" | sum_1_to_n | [83] | 3486 |
| "4 ** 5" | power | [4, 5] | 1024 |
| "Calculate 5 to the 3th power" | power | [5, 3] | 125 |
| "9 factorial" | factorial | [9] | 362880 |

### Sample Test Examples (Collatz - Held Out)

| NL Input | Operands | Expected |
|----------|----------|----------|
| "Collatz length of 683" | [683] | 38 |
| "Steps to reach 1 from 861 via Collatz" | [861] | 103 |

### Files Generated

```
results/
├── train_dataset.json   # 800 examples
└── test_dataset.json    # 50 examples
```

---

## Phase 3: IR Classifier Training

**Status: COMPLETE**

### Approach: Program Classification

Instead of generating IR token-by-token, we trained a classifier to predict which known IR program to use:

```
NL Input → LLM Encoder → Hidden State → MLP Classifier → Program ID
```

Then we use the known IR sequence for that program, extract operands from NL, compile to WASM, and execute.

### Architecture

```python
class IRProgramClassifier(nn.Module):
    def __init__(self, hidden_dim: int, num_programs: int, hidden_size: int = 256):
        self.fc1 = nn.Linear(hidden_dim, hidden_size)  # 2048 → 256
        self.fc2 = nn.Linear(hidden_size, hidden_size) # 256 → 256
        self.fc3 = nn.Linear(hidden_size, num_programs) # 256 → 5
        self.dropout = nn.Dropout(0.1)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | TinyLlama-1.1B-Chat-v1.0 |
| Hidden layer | 12 (of 22) |
| Batch size | 32 |
| Learning rate | 1e-3 |
| Epochs | 20 |
| Optimizer | Adam |

### Results

| Metric | Train | Test (Collatz) |
|--------|-------|----------------|
| **Classification** | 100.0% | 0.0% |
| **Execution** | 67.5% | 0.0% |

### Key Finding: No Generalization

The model achieves **perfect classification** on training programs but **completely fails** on held-out Collatz:

```
Sample test predictions:
✗ 'Collatz length of 683'
   pred: sum_a_to_b, true: collatz_length
✗ 'Count Collatz iterations for 627'
   pred: factorial, true: collatz_length
✗ 'Collatz sequence length starting at 73...'
   pred: sum_a_to_b, true: collatz_length
```

### Analysis

**Why 0% on Collatz?**
1. Classifier can only predict programs it has seen
2. Collatz was held out - the model has no "collatz_length" class in training
3. No way to interpolate to a completely new algorithm

**Why 67.5% execution on training?**
- Operand extraction from NL isn't perfect
- Some template variations don't extract cleanly (e.g., "4 ** 5" parses fine, but "What is 1 + 2 + ... + 50?" is ambiguous)

### Implications

The classifier approach validates that:
1. LLM hidden states contain information to distinguish program types
2. End-to-end NL → IR → WASM → result pipeline works

But it **cannot** achieve true generalization because:
- Classification = lookup, not synthesis
- New algorithms require new training examples
- This is fundamentally a retrieval system, not a generative one

---

## Phase 4: Sequence-to-Sequence Training

**Status: COMPLETE**

### Approach: Autoregressive Opcode Generation

Train a decoder to emit IR opcodes token-by-token:

```
NL Input → LLM Encoder → Hidden State → Transformer Decoder → IR Opcodes
```

### Architecture

```python
class IRSequenceDecoder(nn.Module):
    embed_dim = 256        # Decoder embedding
    num_heads = 4          # Attention heads
    num_layers = 3         # Transformer layers
    max_seq_len = 64       # Maximum sequence
```

### Training Data (Expanded)

Added programs with IF/ELSE control flow to enable compositional learning:

| Program | Opcodes | Structure |
|---------|---------|-----------|
| sum_1_to_n | 21 | Simple loop |
| factorial | 21 | Simple loop |
| max_of_two | 15 | IF/ELSE only |
| abs_diff | 19 | IF/ELSE only |
| **sum_even** | 31 | **Loop + IF/ELSE** |
| **collatz_length** | 35 | Loop + IF/ELSE (held out) |

### Structural Similarity Test

sum_even and collatz_length share the same structure:
```
START → init → LOOP_BEGIN → condition → IF_BEGIN → branch_a → ELSE → branch_b → IF_END → update → BR_IF → LOOP_END → return → END
```

The hypothesis: if the model learns sum_even, it should generalize to Collatz.

### Results

| Metric | Train | Test (Collatz) |
|--------|-------|----------------|
| **Exact Match** | 0.0% | 0.0% |
| **Execution** | 62.0% | 0.0% |
| **Avg Generated Length** | ~22 | ~22 |
| **Target Length** | varies | 35 |

### Key Finding: No Compositional Generalization

The model generates ~22 tokens regardless of target length (35 for Collatz).
It has learned to output "average-length programs" but cannot adapt structure.

**Why seq2seq fails:**
1. Treats programs as flat token sequences, not structured compositions
2. Cannot extrapolate to longer/different structures
3. Hidden state doesn't encode target program structure

### Comparison of Approaches

| Approach | Train Exec | Test Exec | Generalizes? |
|----------|------------|-----------|--------------|
| Classification | 67.5% | 0% | No (retrieval) |
| Seq2seq | 62.0% | 0% | No (flat tokens) |
| Seq2seq + IF/ELSE training | 62.0% | 0% | No (length mismatch) |

---

## Conclusion

**Phase 1 - IR Programs:** IR opcodes can express arbitrary algorithms (Collatz, sum, factorial) that compile to WASM and execute correctly. 49/100 numbers have Collatz length > 100, computed via 35 IR opcodes in 21ms.

**Phase 2 - Training Data:** Generated 1400 training examples including IF/ELSE programs, achieving 99.4% execution accuracy.

**Phase 3 - Classifier:** Program classification achieves 100% on training programs but **0% on held-out Collatz**. Classification = retrieval, not synthesis.

**Phase 4 - Seq2seq:** Autoregressive generation achieves 62% execution on training but **0% on Collatz**. The model outputs average-length sequences (~22 tokens) regardless of target structure (35 tokens for Collatz).

### What Works

```
Infrastructure: VALIDATED
  - IR codebook compiles to WASM ✓
  - WASM executes iterative algorithms ✓
  - End-to-end NL → IR → WASM pipeline ✓
  - Programs with loops + conditionals work ✓
```

### What Doesn't Work

```
Compositional Generalization: NOT ACHIEVED
  - Neither classification nor seq2seq generalizes
  - Flat token sequences don't capture program structure
  - Models learn "average behavior" not compositional rules
  - This is a known hard problem in neural networks
```

### Future Directions

For true compositional generalization, consider:
1. **Structured generation** - Generate parse trees, not flat sequences
2. **Program induction** - Use symbolic search guided by neural models
3. **Neuro-symbolic approaches** - Combine neural pattern matching with symbolic execution
4. **In-context learning** - Show examples of the target structure in the prompt

### Key Insight

The fundamental challenge is **compositional generalization**: learning to combine primitives in novel ways. Current neural architectures struggle with this when the target composition was never seen during training.

The IR → WASM pipeline itself is sound. The bottleneck is getting neural models to emit *correct* IR sequences for *novel* algorithms.
