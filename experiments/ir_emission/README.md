# IR Emission Experiment

**Goal**: Train a model to emit executable WASM IR directly from L13 hidden states.

## The Key Insight

**CoT is format normalization, not reasoning.**

```
Varied NL:     "The difference of 69 and 49 is"  →  ~60% classifier accuracy
Canonical:     "69 - 49 = "                       →  100% classifier accuracy
```

The architecture writes itself:

```
┌─────────────────────────────────────────┐
│  "Janet's ducks lay 16 eggs daily,      │
│   she eats 3 for breakfast..."          │
└────────────────┬────────────────────────┘
                 │
         NL → NL (Normalizer)
                 │
┌────────────────▼────────────────────────┐
│  "16 - 3 = "                            │
│  "13 - 4 = "                            │
│  "9 * 7 = "                             │
└────────────────┬────────────────────────┘
                 │
         L13 classifier (100%)
                 │
┌────────────────▼────────────────────────┐
│  [i32.const 16, i32.const 3, i32.sub]   │
│  [i32.const 13, i32.const 4, i32.sub]   │
│  [i32.const 9, i32.const 7, i32.mul]    │
└────────────────┬────────────────────────┘
                 │
         WASM execute (100%)
                 │
┌────────────────▼────────────────────────┐
│  63                                     │
└─────────────────────────────────────────┘
```

## Results

| Component | Accuracy | Notes |
|-----------|----------|-------|
| NL → Canonical | ~50-80% | Trainable with more data |
| Canonical → IR | **100%** | L13 logit lens after dual-reward |
| IR → Execute | **100%** | Deterministic WASM |

The execution layer is solved. The only learnable gap is NL normalization.

## Training

### Stage 1: Dual-Reward Classifier Training
```bash
python experiments/ir_emission/train_phase1.py --steps 1000
```
This trains LoRA (v_proj, o_proj) to make classifier tokens emerge at L12.

### Stage 2: Normalizer Training
```bash
python experiments/ir_emission/generate_normalizer_data.py --num-samples 3000
python experiments/ir_emission/train_normalizer.py --steps 800
```
This trains LoRA (q_proj, v_proj) to rewrite varied NL to canonical form.

### Full Pipeline Test
```bash
python experiments/ir_emission/full_pipeline.py
```

## Files

- `codebook.py` - IR opcode vocabulary and WASM encoding
- `wasm_runtime.py` - WASM module builder and executor
- `train_phase1.py` - Dual-reward classifier training
- `train_normalizer.py` - NL→Canonical normalizer training
- `full_pipeline.py` - Complete neural compiler test

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NEURAL COMPILER                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Stage 1: FRONTEND (NL → Canonical)                        │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Transformer layers 1-12 (semantic parsing)         │   │
│   │  LoRA fine-tuned for format normalization           │   │
│   └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│   Stage 2: MIDDLE-END (Canonical → IR)      ← SOLVED        │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  L13 logit lens → classifier token probabilities    │   │
│   │  Dual-reward trained to 100% accuracy               │   │
│   └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│   Stage 3: BACKEND (IR → Execute)           ← SOLVED        │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  WASM runtime (deterministic)                       │   │
│   │  [i32.const a, i32.const b, i32.op] → result        │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Why This Matters

1. **CoT demystified**: It's not "thinking step by step." It's format normalization for downstream circuits.

2. **Clean separation**: Frontend is learnable NL processing, backend is deterministic execution.

3. **Composable**: Chain multiple canonical operations for multi-step problems.

4. **Debuggable**: IR is inspectable, WASM is traceable.

## Checkpoints

- `checkpoints/dual_reward/final/` - Classifier LoRA weights (v_proj, o_proj)
- `checkpoints/normalizer/` - Normalizer LoRA weights (q_proj, v_proj)

## Next Steps

1. **Improve normalizer**: More diverse training data, larger model
2. **Multi-op chains**: Parse "16 - 3 then * 5" into sequence
3. **Variable binding**: Track intermediate results
4. **Control flow**: Loops and conditionals in IR
