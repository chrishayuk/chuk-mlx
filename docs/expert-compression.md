# MoE Expert Compression

This guide explains how to use the expert compression system to reduce the memory footprint of Mixture-of-Experts (MoE) models while maintaining output quality.

## Overview

MoE models like GPT-OSS, Mixtral, and Llama-4 use multiple "expert" networks, but only a subset are active for each token. Our analysis shows that:

- Many experts have **high similarity** (>70%) and can be merged
- Some experts have **low utilization** and can be pruned
- Compression can achieve **40%+ memory reduction** with minimal quality loss

## Quick Start

```python
from mlx_lm import load
from chuk_lazarus.introspection.moe import (
    ExpertCompressor,
    MoEHooks,
    get_moe_layer_info,
)

# Load model
model, tokenizer = load("path/to/moe-model")

# Create compressor
compressor = ExpertCompressor(model, tokenizer)

# Analyze compression potential for a layer
analysis = compressor.analyze_compression_potential(layer_idx=12)
print(f"Potential reduction: {analysis['potential_reduction']} experts")
print(f"Recommended target: {analysis['recommended_target']} experts")

# Create compression plan
plan = compressor.plan_compression(layer_idx=12, strategy="balanced")
print(f"Original: {plan.original_num_experts} -> Target: {plan.target_num_experts}")
print(f"Memory reduction: {plan.estimated_memory_reduction:.1%}")

# Apply compression (modifies model in-place)
config = compressor.apply_compression(plan, layer_idx=12, inplace=True)
```

## Compression Strategies

### Balanced (Recommended)
Mix of merging similar experts and pruning low-utilization ones.

```python
plan = compressor.plan_compression(layer_idx, strategy="balanced")
```

### Conservative
Minimal compression, preserves specialist experts.

```python
plan = compressor.plan_compression(layer_idx, strategy="conservative")
```

### Aggressive
Maximum compression, may impact quality on specialized tasks.

```python
plan = compressor.plan_compression(layer_idx, strategy="aggressive")
```

### Target-Based
Specify exact number of experts you want.

```python
plan = compressor.plan_compression(layer_idx, target_experts=16)
```

## Analyzing Compression Potential

Before compressing, analyze what's possible:

```python
analysis = compressor.analyze_compression_potential(
    layer_idx=12,
    test_prompts=[
        "def fibonacci(n):",
        "The capital of France is",
        "SELECT * FROM users WHERE",
    ]
)

# View results
print(f"Number of experts: {analysis['num_experts']}")
print(f"Merge candidates: {analysis['merge_candidates'][:5]}")  # Most similar pairs
print(f"Prune candidates: {analysis['prune_candidates']}")       # Low utilization
print(f"Specialist experts: {analysis['specialist_experts']}")   # Keep these!
print(f"Mergeable groups: {analysis['mergeable_groups']}")       # Can combine
print(f"Recommended target: {analysis['recommended_target']}")
```

### Example Output (GPT-OSS 20B, Layer 12)

```
Number of experts: 32
Merge candidates (most similar pairs):
  Experts 23 & 26: 76.67% similar
  Experts 1 & 6: 75.00% similar
  Experts 20 & 27: 70.83% similar

Prune candidates (low utilization): [2, 6, 7, 10, 11, 12, 13, 16, 18, 19, 22, 24, 28, 29, 30, 31]
Specialist experts: []
Generalist experts: [15, 21, 23]
Mergeable groups: [[9, 15, 23, 26], [1, 6], [20, 27], [5, 8, 14]]

Compression potential:
  Potential reduction: 23 experts
  Max compression ratio: 28.1%
  Recommended target: 16 experts
```

## Understanding the Compression Plan

```python
plan = compressor.plan_compression(layer_idx=12, strategy="balanced")

print(f"Original experts: {plan.original_num_experts}")
print(f"Target experts: {plan.target_num_experts}")
print(f"Memory reduction: {plan.estimated_memory_reduction:.1%}")
print(f"Quality impact: {plan.estimated_quality_impact}")

# Merges to perform
for merge in plan.merges:
    print(f"  Merge {merge.source_experts} -> Expert {merge.target_expert}")
    print(f"    Similarity: {merge.similarity:.1%}")
    print(f"    Blend method: {merge.weight_blend}")

# Experts to prune
print(f"Pruned experts: {plan.pruned_experts}")

# Experts kept unchanged
print(f"Kept experts: {plan.kept_experts}")
```

## Applying Compression

### In-Place (Modifies Original Model)

```python
config = compressor.apply_compression(plan, layer_idx=12, inplace=True)

# Model is now compressed
print(f"Compressed from {config.original_num_experts} to {config.compressed_num_experts}")
```

### Creating a New Model (Preserves Original)

```python
config = compressor.apply_compression(plan, layer_idx=12, inplace=False)

# Use config.router_remap and config.expert_mapping to build new model
```

## Verifying Quality

Always verify the compressed model produces acceptable outputs:

```python
from mlx_lm import generate

test_prompts = [
    "The capital of France is",
    "def fibonacci(n):",
    "Hello, how are you?",
]

# Get baseline outputs BEFORE compression
baseline = [generate(model, tokenizer, p, max_tokens=30) for p in test_prompts]

# Apply compression
config = compressor.apply_compression(plan, layer_idx, inplace=True)

# Get post-compression outputs
compressed = [generate(model, tokenizer, p, max_tokens=30) for p in test_prompts]

# Compare
for i, prompt in enumerate(test_prompts):
    print(f"Prompt: {prompt}")
    print(f"  Before: {baseline[i][:60]}...")
    print(f"  After:  {compressed[i][:60]}...")
```

## Full Example: Compress GPT-OSS 20B

```python
#!/usr/bin/env python3
"""Compress GPT-OSS 20B experts for reduced memory usage."""

from pathlib import Path
from mlx_lm import load, generate
import mlx.core as mx

from chuk_lazarus.introspection.moe import (
    ExpertCompressor,
    MoEHooks,
)

# Load model
model_path = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
model, tokenizer = load(str(model_path))

# Get MoE layer indices
hooks = MoEHooks(model)
print(f"Found {len(hooks.moe_layer_indices)} MoE layers")

# Create compressor
compressor = ExpertCompressor(model, tokenizer)

# Compress each layer
for layer_idx in hooks.moe_layer_indices:
    print(f"\nProcessing layer {layer_idx}...")

    # Analyze
    analysis = compressor.analyze_compression_potential(layer_idx)

    if analysis['potential_reduction'] > 0:
        # Plan compression
        plan = compressor.plan_compression(layer_idx, strategy="balanced")

        print(f"  {plan.original_num_experts} -> {plan.target_num_experts} experts")
        print(f"  Memory reduction: {plan.estimated_memory_reduction:.1%}")

        # Apply
        compressor.apply_compression(plan, layer_idx, inplace=True)

# Verify
mx.eval(model.parameters())
output = generate(model, tokenizer, "The capital of France is", max_tokens=20)
print(f"\nTest output: {output}")
```

## API Reference

### ExpertCompressor

```python
class ExpertCompressor:
    def __init__(self, model, tokenizer):
        """Create compressor for an MoE model."""

    def analyze_compression_potential(
        self,
        layer_idx: int,
        test_prompts: list[str] | None = None,
    ) -> dict:
        """Analyze how compressible a layer is."""

    def plan_compression(
        self,
        layer_idx: int,
        target_experts: int | None = None,
        strategy: str = "balanced",  # "balanced", "aggressive", "conservative"
    ) -> CompressionPlan:
        """Create a compression plan."""

    def apply_compression(
        self,
        plan: CompressionPlan,
        layer_idx: int,
        inplace: bool = False,
    ) -> CompressedMoEConfig:
        """Apply compression to the model."""
```

### CompressionPlan

```python
@dataclass
class CompressionPlan:
    original_num_experts: int
    target_num_experts: int
    merges: list[ExpertMergeResult]
    pruned_experts: list[int]
    kept_experts: list[int]
    estimated_memory_reduction: float
    estimated_quality_impact: str  # "none", "minimal", "moderate", "significant"
    expert_params: int | None = None  # Params per expert for size estimation

    @property
    def params_removed(self) -> int | None:
        """Number of parameters removed by this compression."""

    @property
    def compression_ratio(self) -> float:
        """Ratio of target to original experts."""
```

### CompressedMoEConfig

```python
@dataclass
class CompressedMoEConfig:
    layer_idx: int
    original_num_experts: int
    compressed_num_experts: int
    expert_mapping: dict[int, int]      # old_idx -> new_idx (-1 = pruned)
    merged_from: dict[int, list[int]]   # new_idx -> [old_idx, ...]
    router_remap: mx.array | None       # Remapped router weights
```

### Size Estimation Functions

```python
def estimate_model_size(model) -> dict[str, int]:
    """
    Estimate model size breakdown by component type.

    Returns dict with: total, expert, attention, embeddings, other
    """

def estimate_compressed_size(model, compression_plans: list[CompressionPlan]) -> dict:
    """
    Estimate model size after applying compression plans.

    Returns dict with:
    - original_params, compressed_params, params_removed
    - reduction_ratio (0.0-1.0)
    - expert_params_original, expert_params_compressed
    """

def print_compression_summary(model, compression_plans, model_name="Model"):
    """Print a formatted summary of model compression."""
```

## Validated Results

### GPT-OSS Full Model Compression (All Layers)

With balanced compression across all 24 MoE layers:

| Metric | Before | After |
|--------|--------|-------|
| Total Parameters | 4.79B | 3.94B |
| Expert Parameters | 2.99B | 2.15B |
| Parameters Removed | - | 845M (17.7%) |
| Quality | Baseline | 100% token overlap |

Layer-by-layer compression (32 experts each → varies):
- Early layers: Minor reduction (29, 25, 28 experts)
- Middle layers: Aggressive reduction (17-21 experts)
- Layer 16: Most compressed (14 experts, 56% reduction)

### Single Layer Test (Layer 12)

| Metric | Before | After |
|--------|--------|-------|
| Experts | 32 | 19 |
| Memory | 100% | 59.4% |
| Quality | Baseline | 100% token overlap |

Test outputs with balanced compression:

```
Prompt: "The capital of France is"
Before: Paris." Sure! Here's a simple example...
After:  Paris." Sure! Here's a simple example...
Token overlap: 100%

Prompt: "def fibonacci(n):"
Before: if n==0: return 0 if n==1: return 1...
After:  if n==0: return 0 if n==1: return 1...
Token overlap: 100%
```

## Best Practices

1. **Always verify quality** after compression with diverse test prompts
2. **Start with balanced strategy** before trying aggressive
3. **Compress one layer at a time** to isolate issues
4. **Keep specialist experts** - they handle specific token types
5. **Test on your specific use case** - code, math, languages differ
6. **Save compressed model** for reuse without re-compressing

## Troubleshooting

### No merge candidates found
- Try more diverse test prompts
- Lower the similarity threshold (default 0.6)
- Some models have well-differentiated experts

### Quality degradation
- Use conservative strategy
- Reduce target expert count less aggressively
- Check if specialist experts were accidentally pruned

### Memory not reduced
- Ensure `inplace=True` when applying
- Call `mx.eval(model.parameters())` after compression
- Check that compression was actually applied to the layer
