# Introspection Examples & Experiments

Research experiments and usage examples for the introspection toolkit.

## Organization

```
examples/introspection/
├── README.md           # This file
├── _loader.py          # Shared utilities for examples
├── demos/              # Usage examples for the tools
│   ├── logit_lens.py          # Async analyzer demo
│   ├── low_level_hooks.py     # Direct hook usage
│   └── circuit_analysis.py    # Circuit pipeline walkthrough
└── experiments/        # Research experiments
    ├── circuits/       # Circuit discovery
    ├── ablation/       # Ablation studies
    ├── comparison/     # Model comparisons
    ├── distillation/   # Layer distillation
    ├── steering/       # Activation steering
    ├── probing/        # Linear probes
    ├── layers/         # Layer analysis
    └── model_specific/ # Model-specific experiments
```

## Demos

Start here to learn how to use the tools:

| File | Description |
|------|-------------|
| `demos/logit_lens.py` | Async analyzer with pydantic models |
| `demos/low_level_hooks.py` | Direct hook usage for activation capture |
| `demos/circuit_analysis.py` | Full circuit analysis pipeline walkthrough |

## Experiments

### circuits/ - Circuit Discovery
Research into how models implement specific behaviors.

| File | Description |
|------|-------------|
| `gemma_alignment_circuits.py` | Arithmetic suppression in Gemma 3 |
| `arithmetic_circuit_study.py` | Deep dive into arithmetic circuits |
| `arithmetic_readout.py` | Arithmetic readout mechanisms |
| `virtual_calculator.py` | Synthetic calculator task |
| `tool_router.py` | Tool selection routing |
| `computation_locator.py` | Find where computation happens |
| `computation_flow.py` | Trace computation flow |
| `feature_emergence_neurons.py` | Find feature neurons |
| `gate_neuron_finder.py` | Find gating neurons |
| `circuit_deep_dive.py` | Deep circuit exploration |
| `circuit_summary.py` | Summarize circuit results |

### ablation/ - Ablation Studies
Identifying causal components by zeroing them.

| File | Description |
|------|-------------|
| `mlp_ablation.py` | MLP ablation at each layer |
| `head_ablation.py` | Attention head ablation |
| `gemma_scale_ablation.py` | Gemma at different scales |
| `qwen3_scale_ablation.py` | Qwen3 at different scales |
| `qwen3_ablation_test.py` | Qwen3 specific tests |

### comparison/ - Model Comparison
Comparing base vs fine-tuned models.

| File | Description |
|------|-------------|
| `activation_divergence.py` | Where hidden states diverge |
| `weight_divergence.py` | Weight differences by layer |
| `attention_divergence.py` | Attention pattern differences |
| `gemma_base_vs_function.py` | Gemma 3 vs FunctionGemma |
| `gemma3_vs_functiongemma.py` | Alternative comparison |
| `all_families.py` | Test across model families |

### steering/ - Activation Steering
Modifying model behavior via activations.

| File | Description |
|------|-------------|
| `steering_demo.py` | FunctionGemma steering demo |
| `steering_validation.py` | Validate steering effects |
| `generic_steering.py` | Model-agnostic steering |
| `activation_steering.py` | Various steering approaches |
| `activation_patching.py` | Causal intervention via patching |
| `refusal_direction.py` | Extract refusal direction |

### distillation/ - Layer Distillation
Experiments on layer distillation and feature injection.

| File | Description |
|------|-------------|
| `layer_distillation.py` | Basic layer distillation |
| `normalized_distillation.py` | With normalization |
| `adapted_distillation.py` | Adapted approach |
| `find_injection_point.py` | Find injection layers |
| `trained_injection.py` | Trained injection |
| `injection_upper_bound.py` | Upper bounds |
| `collapsed_inference.py` | Collapsed layer inference |
| `INJECTION_FINDINGS.md` | Summary of injection findings |

### probing/ - Linear Probes
Linear probes for feature detection.

| File | Description |
|------|-------------|
| `linear_probe.py` | Basic linear probing |
| `gemma_tool_probe.py` | Tool-calling probes for Gemma |
| `uncertainty_detector.py` | Detect model uncertainty |
| `neuron_causal_analysis.py` | Neuron-level causality |
| `neuron_causal_v2.py` | V2 of neuron analysis |

### layers/ - Layer Analysis
Understanding what each layer does.

| File | Description |
|------|-------------|
| `early_layer_analysis.py` | Early layer behavior |
| `embedding_analysis.py` | Embedding space |
| `layer_decode.py` | Decode layer outputs |
| `layer_deep_dive.py` | Deep dive into specific layers |
| `layer_structure_experiments.py` | Layer structure variations |
| `decision_layer_universality.py` | Are decision layers universal? |
| `attention_analysis.py` | Attention pattern analysis |

### model_specific/ - Model-Specific Experiments

| File | Description |
|------|-------------|
| `functiongemma.py` | FunctionGemma analysis |
| `qwen3_circuit_analysis.py` | Qwen3 circuit study |
| `qwen3_detailed_analysis.py` | Qwen3 detailed analysis |
| `ghost_hunter.py` | Find "ghost" computations |

## Running Experiments

```bash
# Run a demo
uv run python examples/introspection/demos/logit_lens.py

# Run an experiment
uv run python examples/introspection/experiments/ablation/mlp_ablation.py

# With custom model
uv run python examples/introspection/experiments/circuits/gemma_alignment_circuits.py --model my-model
```

## Key Findings

### Gemma 3 Alignment Circuits
- **L24** destroys arithmetic capability (MLP ablation restores it)
- **L31-33** rebuild arithmetic in "safe" form
- This is a computation suppression -> safe rebuild pattern

### Tool-Calling Circuits
- Tool/no-tool decision encoded by **layer 11** in 270M models
- Linear probe achieves 95%+ accuracy at decision layers
- Direction vectors can steer tool-calling behavior

### Decision Layers Scale with Model Size
- 270M: L11
- 1B: ~L16
- 2B: ~L20-22

## See Also

- `src/chuk_lazarus/introspection/README.md` - Tool documentation
- `docs/gemma_alignment_circuits.md` - Gemma findings
- `docs/tool_calling_circuit.md` - Tool-calling circuits
