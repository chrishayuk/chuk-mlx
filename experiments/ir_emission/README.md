# IR Emission Experiment

**Goal**: Train a model to emit executable WASM IR directly from L13 hidden states, rather than routing to external tools.

## The Paradigm Shift

```
Current:   L13 → classify("multiply") → route to MathExpertPlugin → Python eval
Proposed:  L13 → emit([i32.const, X, i32.const, Y, i32.mul]) → WASM execute
```

## Architecture

```
Layers 1-12:  NL embedding space (understanding)
Layer 13:     IR emission (compilation) via learned codebook
WASM:         Execution (loops, arithmetic, search)
Layers 14+:   Back to NL (interpretation of result)
```

## Training Curriculum

1. **Phase 1: Single-op arithmetic** - "3 + 4" → `[const, const, add]`
2. **Phase 2: Multi-op chains** - "3 + 4 - 2" → `[const, const, add, const, sub]`
3. **Phase 3: Word problems** - NL → full program
4. **Phase 4: Loops** - "Sum 1 to 10" → loop structure

## Files

- `codebook.py` - Learned IR codebook (VQ-VAE style)
- `decoder.py` - h13 → IR sequence decoder
- `wasm_runtime.py` - WASM compilation/execution wrapper
- `generate_data.py` - Training data generation
- `train_phase1.py` - Phase 1 training script

## Why WASM?

| Property | Benefit |
|----------|---------|
| Stack-based | Linear sequences, no tree parsing |
| ~200 opcodes | Smaller vocab than any language |
| Typed | Constraints guide generation |
| Sandboxed | Safe to let model "write" code |
| Deterministic | Perfect gradients - right or wrong |
| Near-native speed | Millions of training iterations |

## The Key Insight

Transformers are loop-free computation graphs. WASM gives you Turing completeness back.
The model becomes the specification layer, WASM becomes the execution layer.
