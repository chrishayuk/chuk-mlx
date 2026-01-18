# IR Program Synthesis: True Turing Completeness

## Research Question

**Can an LLM learn to emit arbitrary IR programs (not just invoke named functions), achieving true Turing completeness?**

## The Problem with Named Functions

Previous approach (hacky):
```
"Sum 1 to 1000000" → sum(1, 1000000) = → WASM sum function → result
"Collatz of 27"    → ??? need to add collatz() function
```

Adding a function for every algorithm isn't Turing complete - it's a lookup table.

## The Solution: Program Synthesis

The LLM emits a **sequence of IR opcodes** that compile to WASM:

```
"Collatz length of 27" →
  [START, CONST_0, LOCAL_SET_0, SLOT_0, LOCAL_SET_1,
   LOOP_BEGIN,
     LOCAL_GET_1, CONST_1, I32_NE, BR_IF,
     LOCAL_GET_1, CONST_2, I32_REM_S,
     IF_BEGIN,
       LOCAL_GET_1, CONST_2, I32_DIV_S,
     ELSE,
       LOCAL_GET_1, CONST_3, I32_MUL, CONST_1, I32_ADD,
     IF_END,
     LOCAL_SET_1,
     LOCAL_GET_0, CONST_1, I32_ADD, LOCAL_SET_0,
     BR,
   LOOP_END,
   LOCAL_GET_0, END]
→ compile to WASM → execute → 111
```

The LLM learns to **write programs**, not call functions.

## IR Codebook

Using the existing `experiments/ir_emission/shared/codebook.py`:

| Category | Opcodes |
|----------|---------|
| Stack | SLOT_0-3, CONST_0/1/2/10 |
| Arithmetic | I32_ADD, I32_SUB, I32_MUL, I32_DIV_S, I32_REM_S |
| Comparison | I32_EQ, I32_NE, I32_LT_S, I32_GT_S, I32_LE_S, I32_GE_S |
| Control | LOOP_BEGIN, LOOP_END, IF_BEGIN, ELSE, IF_END, BR, BR_IF |
| Variables | LOCAL_GET_0/1, LOCAL_SET_0/1, LOCAL_TEE_0 |
| Meta | START, END, PAD |

~40 opcodes that can express any algorithm.

## Training Data

Generate (NL description, IR sequence) pairs:

```python
TRAINING_EXAMPLES = [
    # Sum range
    ("Sum 1 to N", [START, CONST_0, LOCAL_SET_0, CONST_1, LOCAL_SET_1,
                   LOOP_BEGIN, LOCAL_GET_0, LOCAL_GET_1, I32_ADD, LOCAL_SET_0,
                   LOCAL_GET_1, CONST_1, I32_ADD, LOCAL_TEE_1, SLOT_0, I32_LE_S,
                   BR_IF, LOOP_END, LOCAL_GET_0, END]),

    # Collatz length
    ("Collatz length of N", [START, CONST_0, LOCAL_SET_0, SLOT_0, LOCAL_SET_1,
                             LOOP_BEGIN, LOCAL_GET_1, CONST_1, I32_NE, BR_IF,
                             LOCAL_GET_1, CONST_2, I32_REM_S, IF_BEGIN,
                             LOCAL_GET_1, CONST_2, I32_DIV_S, ELSE,
                             LOCAL_GET_1, CONST_3, I32_MUL, CONST_1, I32_ADD,
                             IF_END, LOCAL_SET_1, LOCAL_GET_0, CONST_1, I32_ADD,
                             LOCAL_SET_0, BR, LOOP_END, LOCAL_GET_0, END]),

    # Factorial
    ("Factorial of N", [...]),

    # Fibonacci
    ("Nth Fibonacci number", [...]),

    # GCD
    ("GCD of A and B", [...]),
]
```

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
│  │   LLM + IR Head     │  Emits sequence of ~30 IR opcodes      │
│  │   (trained)         │                                        │
│  └─────────────────────┘                                        │
│              │                                                  │
│              ▼                                                  │
│  [START, CONST_0, LOCAL_SET_0, SLOT_0, LOCAL_SET_1, ...]       │
│              │                                                  │
│              ▼                                                  │
│  ┌─────────────────────┐                                        │
│  │   IRCodebook        │  indices_to_wasm(opcodes, [73847])     │
│  │   (compiler)        │                                        │
│  └─────────────────────┘                                        │
│              │                                                  │
│              ▼                                                  │
│  WASM bytecode: 0x41 0x00 0x21 0x00 0x41 0x87 0xc1 0x04 ...    │
│              │                                                  │
│              ▼                                                  │
│  ┌─────────────────────┐                                        │
│  │   WASMRuntime       │  Execute loop: 116 iterations          │
│  │   (interpreter)     │                                        │
│  └─────────────────────┘                                        │
│              │                                                  │
│              ▼                                                  │
│  Result: 116                                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Why This is True Turing Completeness

1. **No function lookup** - LLM emits arbitrary programs
2. **Compositional** - Same opcodes combine differently for different algorithms
3. **Generalizable** - Train on sum/factorial, test on Collatz/GCD
4. **Unbounded** - Loops execute O(n) from O(1) generation

## Experiments

### Experiment 1: IR Program Library
Define correct IR sequences for 10+ algorithms.

### Experiment 2: Training Data Generation
Generate diverse (NL, IR) pairs with variations.

### Experiment 3: IR Head Training
Train IRSequenceDecoder on the dataset.

### Experiment 4: Generalization Test
Train on {sum, product, factorial, power}.
Test on {collatz, gcd, fib, is_prime}.

### Experiment 5: Collatz Benchmark
Run on the 100-number Collatz test from the conversation.

## Success Criteria

| Metric | Target |
|--------|--------|
| IR sequence accuracy (train set) | 100% |
| IR sequence accuracy (test set) | ≥80% |
| Novel algorithm generalization | ≥50% |
| Collatz benchmark | 100% (once IR is correct) |

## Files

```
ir_program_synthesis/
├── EXPERIMENT.md           # This file
├── RESULTS.md              # Results documentation
├── programs.py             # IR program definitions
├── data_generator.py       # Training data generation
├── train.py                # IR head training
├── evaluate.py             # End-to-end evaluation
└── results/                # Output
```

---

## Results (Phase 1: IR Program Definitions)

**Status: COMPLETE**

| Program | Status | Test Results |
|---------|--------|--------------|
| sum_1_to_n | ✓ | 55, 5050, 500500 |
| sum_a_to_b | ✓ | 55, 110, 5050 |
| factorial | ✓ | 120, 720, 3628800 |
| power | ✓ | 1024, 81, 1048576 |
| collatz_length | ✓ | 111, 16 (edge case n=1 needs fix) |
| fibonacci | ✗ | DUP opcode issue |
| gcd | ✗ | Loop exit issue |
| is_prime | ✗ | Early return issue |

**Collatz Benchmark:**
```
Question: How many of 100 numbers have Collatz length > 100?
Answer: 49
Total iterations: 10,960
Execution time: 20.99 ms
```

See `RESULTS.md` for full details.
