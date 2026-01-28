# GSM-8K YAML Trace Training

> **LLMs should route and structure, not compute.**

A 1B model can learn to emit symbolic YAML traces that an external solver executes deterministically. The model never sees the answer — it wires the computation graph, the expert produces the result.

## Quick Start

```bash
# Train with TinyLlama (best format learning)
python train_gsm8k_yaml.py --n-train 1500 --sft-epochs 1 --rl-iters 10

# Train with Llama-3.2-3B-Instruct (best GSM-8K accuracy)
python train_gsm8k_yaml.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --n-train 3000 \
    --sft-epochs 1 \
    --rl-iters 20

# Evaluate checkpoint
python train_gsm8k_yaml.py --load-checkpoint checkpoints/llama32_3b_instruct_best --eval-only
```

## Architecture

```
Math Problem (natural language)
       ↓
┌─────────────────────────────┐
│  Model (1-3B parameters)    │
│  - Classifies expert type   │
│  - Extracts quantities      │
│  - Wires computation graph  │
└─────────────────────────────┘
       ↓  (YAML trace)
┌─────────────────────────────┐
│  TraceSolverExpert          │
│  - Parses via Pydantic      │
│  - Executes trace steps     │
│  - Returns verified answer  │
└─────────────────────────────┘
       ↓
Answer + Verification Status
```

## Trace Format

```yaml
expert: arithmetic
trace:
- {op: init, var: eggs, value: 16}
- {op: init, var: breakfast, value: 3}
- {op: compute, compute_op: sub, args: [eggs, breakfast], var: step1}
- {op: init, var: price, value: 2}
- {op: compute, compute_op: mul, args: [step1, price], var: result}
- {op: query, var: result}
```

## Expert Types

| Expert | Purpose | Example |
|--------|---------|---------|
| `arithmetic` | Multi-step computation chains | price + tax + shipping |
| `entity_track` | Quantities moving between entities | Alice gives 5 to Bob |
| `rate_equation` | Rate × time = quantity | 60 mph × 5 hours |
| `comparison` | Compare quantities, compute differences | A has 2× as many as B |
| `percentage` | Percent off/increase/of operations | 20% off $80 |

## Multi-Expert Composition

For problems crossing expert boundaries:

```yaml
- expert: percentage
  trace:
  - {op: init, var: price, value: 80}
  - {op: init, var: rate, value: 20}
  - {op: percent_off, base: price, rate: rate, var: result}
  - {op: query, var: result}
- expert: arithmetic
  trace:
  - {op: init, var: prev, source: prev.result}
  - {op: init, var: shipping, value: 10}
  - {op: compute, compute_op: add, args: [prev, shipping], var: result}
  - {op: query, var: result}
```

## Results Summary

| Model | SFT | GSM-8K | Notes |
|-------|-----|--------|-------|
| TinyLlama 1.1B | 100% | ~2% | Best format learning |
| Llama-3.2-1B-Instruct | 95% | 17% | Best 1B efficiency |
| **Llama-3.2-3B-Instruct** | 100% | **27%** | **Best overall** |

**Key Finding**: Format mastery ≠ Reasoning. 100% parse rate with ~2% accuracy reveals the model learns trace FORMAT, not REASONING.

## Project Structure

```
experiments/csp_cot_gsm8k/
├── README.md                 # This file
├── train_gsm8k_yaml.py       # Main training script
├── __init__.py
│
├── docs/                     # Documentation
│   ├── REVIEW.md             # Critical experiment review
│   ├── NEXT_EXPERIMENTS.md   # Proposed next experiments
│   ├── ARITHMETIC_EXPERT_REVIEW.md  # New generator analysis
│   ├── GENERATOR_VS_GSM8K_COMPARISON.md  # Side-by-side comparison
│   ├── RESULTS.md            # Full run history (25 runs)
│   ├── PATTERNS.md           # 17 training patterns discovered
│   ├── DESIGN.md             # Architecture and design details
│   ├── FORMAT_SPEC.md        # Trace format specification
│   ├── GSM8K_GAPS.md         # Gap analysis
│   ├── GSM8K_PATTERNS.md     # Pattern catalog
│   └── models/               # Model-specific analysis
│       ├── TINYLLAMA_1.1B.md
│       ├── SMOLLM2_1.7B.md
│       └── LLAMA32_1B.md
│
├── scripts/                  # Utility scripts
│   ├── diagnose_gsm8k.py     # Diagnostic evaluation
│   ├── diversity_analysis.py # Training data analysis
│   ├── template_expander.py  # Template expansion
│   ├── spec_generator.py     # Spec generation
│   ├── llm_sample_gen.py     # LLM-based sample generation
│   └── audit_vocab_system.py # Vocabulary auditing
│
├── data/                     # Training data specs
│   └── *.json, *.jsonl
│
├── evaluation/               # Evaluation utilities
│   └── gsm8k_loader.py
│
└── checkpoints/              # Model checkpoints
    ├── tinyllama_1.1b_best/     # Best TinyLlama
    ├── llama32_1b_instruct_best/ # Best 1B (17%)
    ├── llama32_3b_instruct_best/ # Best overall (27%)
    └── archive/                  # Old runs
```

## Key Training Patterns

1. **Anti-Short-Circuit** — Query must target computed vars, not init vars
2. **Structural Consistency** — All patterns in an expert have same shape
3. **One Template Per Pattern** — Max repetition signal for 1B models
4. **Hybrid Variable Naming** — Semantic inits + fixed scaffolding (`step1`, `result`)
5. **Expert Composition** — Multi-expert chains with `prev.result` wiring

See [docs/PATTERNS.md](docs/PATTERNS.md) for all 17 patterns.

## Dependencies

- `chuk_virtual_expert` — TraceSolverExpert, TraceVerifier, ExpertRegistry
- `chuk_virtual_expert_arithmetic` — Expert implementations + schema-based generators
- `chuk_lazarus` — Model loading and training

## Key Conclusions

1. **Model size helps (sublinearly)** — 3B achieves 27% vs 1B's 17%
2. **Layer unfreezing doesn't help** — 8 layers = 6 layers performance
3. **Full fine-tune is catastrophic** — Drops from 17% to 7% (forgetting)
4. **The bottleneck is DATA DIVERSITY** — Not capacity or layers

The fix is more diverse training patterns, not bigger models.
