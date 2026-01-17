# IR Emission Experiments

**Core Thesis**: "Chain-of-Thought is format normalization, not reasoning."

This directory contains a family of experiments exploring neural compilers, CoT as parsing, and verifiable rewards.

---

## Directory Structure

```
ir_emission/
├── neural_compiler/        # NL → WASM → Execute (100% accuracy)
├── suffix_routing/         # Training-dependent routing analysis
├── cot_learned_parser/     # CoT as format normalization (100% accuracy)
├── learned_ir_head/        # Hidden state structure extraction
├── rl_verifiable_rewards/  # WASM execution as reward signal (WIP)
├── shared/                 # Shared utilities (codebook, wasm_runtime)
├── pipelines/              # Pipeline implementations
└── archive/                # Historical experiments
```

---

## Experiment Index

| Experiment | Status | Description |
|------------|--------|-------------|
| [`neural_compiler/`](./neural_compiler/) | **100%** | NL → WASM → Execute pipeline |
| [`suffix_routing/`](./suffix_routing/) | **Complete** | Training-dependent routing analysis |
| [`cot_learned_parser/`](./cot_learned_parser/) | **100%** | CoT as format normalization |
| [`learned_ir_head/`](./learned_ir_head/) | **Validated** | Hidden state structure extraction |
| [`rl_verifiable_rewards/`](./rl_verifiable_rewards/) | **WIP** | WASM execution as reward signal |

---

## Quick Start

### Neural Compiler (100% accuracy)
NL → WASM → Execute pipeline:
```bash
python experiments/ir_emission/neural_compiler/experiment.py
```

### Suffix Routing
Proves suffix routing is learned from training data:
```bash
python experiments/ir_emission/suffix_routing/multi_model_comparison.py
```

### CoT Learned Parser (100% accuracy)
CoT normalization IS the learned parser:
```bash
python experiments/ir_emission/cot_learned_parser/word_problems.py
```

### Learned IR Head
Extract IR structure from hidden states:
```bash
python experiments/ir_emission/learned_ir_head/quick_test.py
```

### RL with Verifiable Rewards
WASM execution as ground truth reward:
```bash
python experiments/ir_emission/rl_verifiable_rewards/rl_train.py
```

---

## Shared Utilities

All experiments use shared code:

```python
from experiments.ir_emission.shared import IROpcode, WASMRuntime, OPCODE_TO_WASM
```

**Files:**
- `shared/codebook.py` - IR opcode definitions
- `shared/wasm_runtime.py` - WASM execution engine
- `pipelines/` - Pipeline implementations (NeuralCompilerBase, SingleOpPipeline, etc.)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: FRONTEND (NL → Canonical)                              │
│   "Janet has 50 apples. She gives 15" → "50 - 15 ="             │
│   Method: Few-shot prompting (100% accuracy)                    │
├─────────────────────────────────────────────────────────────────┤
│ Stage 2: MIDDLE-END (Canonical → IR Operation)                  │
│   Logit lens at L12 → classifier token probabilities            │
│   Method: Dual-reward trained LoRA (v_proj, o_proj)             │
├─────────────────────────────────────────────────────────────────┤
│ Stage 3: BACKEND (IR → Execute)                                 │
│   WASM bytecode execution (deterministic, always 100%)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Insights

1. **Few-shot > Fine-tuning for Normalization** - Few-shot prompting achieves 100% accuracy without training.

2. **Layer 12 Encodes Operation Intent** - At ~55% depth, the model has high confidence in classifier tokens.

3. **Turing Completeness via Composition** - Transformer emits loop *intent* that WASM executes.

4. **Suffix Routing is Learned** - Instruction-tuned models route by suffix; base models don't.

---

## Documentation

Each experiment has its own `EXPERIMENT.md` with research question, methodology, results, and implications.
