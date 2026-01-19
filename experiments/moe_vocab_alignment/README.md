# MoE Routing Correlation Experiment

**Question:** Does MoE architecture create pressure for vocabulary-aligned task representations?

## Quick Start

```bash
# Run via framework
lazarus experiment run moe_routing_correlation

# Direct execution
python experiments/moe_routing_correlation/experiment.py
```

## Background

Your experiments found:
- **Probe classifier**: 100% task extraction at L4 (25% depth) - task info EXISTS early
- **Logit lens on Llama-3.2-1B**: ~0% vocab alignment at intermediate layers
- **GPT-OSS reportedly**: 50-80% operation token probability at L13 (~54% depth)

The mystery: **Why does GPT-OSS have vocabulary-aligned classifiers when dense models don't?**

## Hypothesis

MoE routing requires **discrete decisions**. Unlike dense models where task info can exist in arbitrary subspaces, MoE models must make explicit routing choices. This architectural pressure may force vocabulary-aligned representations to emerge naturally.

## What This Experiment Does

1. **OLMoE-1B-7B Analysis**
   - Run logit lens at layers 4, 6, 8, 10, 12, 14
   - Track operation token probabilities ("add", "multiply", etc.)
   - Capture expert routing decisions
   - Compute correlation: when expert X activates, what's P(task token)?

2. **Llama-3.2-1B Baseline**
   - Same logit lens analysis
   - No routing (dense model)
   - Establishes the "no MoE" baseline

3. **Comparison**
   - Delta in vocab alignment between MoE and dense
   - If MoE >> dense → hypothesis supported
   - If MoE ≈ dense → architecture alone doesn't explain it

## Expected Results

| Outcome | MoE Vocab | Dense Vocab | Interpretation |
|---------|-----------|-------------|----------------|
| **Supported** | >10% | ~0% | MoE creates vocab alignment |
| **Partial** | 5-10% | ~0% | Weak effect, may need scale |
| **Rejected** | ~0% | ~0% | MoE alone doesn't explain GPT-OSS |

## Model Requirements

- **OLMoE-1B-7B**: ~14GB download (7B params, 1B active)
- **Llama-3.2-1B**: ~2.5GB download

## Configuration

Edit `config.yaml` to modify:
- Target layers for analysis
- Task tokens to track
- Test prompts
- Probability thresholds

## Output

Results saved to `results/run_YYYYMMDD_HHMMSS.json` with:
- Per-layer vocab alignment scores
- Expert-task correlation (for MoE)
- Comparison metrics
- Conclusion and interpretation
