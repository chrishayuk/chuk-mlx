"""
LoRA Fine-tuning Example

Demonstrates LoRA (Low-Rank Adaptation) for efficient fine-tuning
using the models_v2 architecture.

This example shows:
- Creating a base model
- Applying LoRA adapters
- Parameter counting (base vs trainable)
- Forward pass with LoRA
- Merging LoRA weights

Run with:
    uv run python examples/models/03_lora_finetuning.py
"""

import mlx.core as mx

from chuk_lazarus.models_v2 import (
    LlamaConfig,
    LlamaForCausalLM,
    LoRAConfig,
    apply_lora,
    count_lora_parameters,
    merge_lora_weights,
)


def count_params(params):
    """Recursively count parameters in nested dict."""
    total = 0
    for v in params.values():
        if isinstance(v, dict):
            total += count_params(v)
        elif hasattr(v, "size"):
            total += v.size
    return total


def main():
    print("=" * 60)
    print("LoRA Fine-tuning Example")
    print("=" * 60)

    # Create base model
    print("\n1. Creating base model...")
    config = LlamaConfig.tiny()
    model = LlamaForCausalLM(config)

    base_params = count_params(model.parameters())
    print(f"   Base model parameters: {base_params:,}")

    # Configure LoRA
    print("\n2. Configuring LoRA (Pydantic)...")
    lora_config = LoRAConfig(
        rank=4,  # Low-rank dimension
        alpha=8.0,  # Scaling factor (scaling = alpha / rank)
        dropout=0.0,  # No dropout for demo
        target_modules=[  # Which layers to adapt
            "q_proj",
            "v_proj",
        ],
    )

    print(f"   LoRA rank: {lora_config.rank}")
    print(f"   LoRA alpha: {lora_config.alpha}")
    print(f"   LoRA scaling: {lora_config.alpha / lora_config.rank}")
    print(f"   Target modules: {lora_config.target_modules}")

    # Apply LoRA
    print("\n3. Applying LoRA to model...")
    lora_layers = apply_lora(model, lora_config)

    print(f"   LoRA layers created: {len(lora_layers)}")
    for name in list(lora_layers.keys())[:4]:  # Show first 4
        print(f"     - {name}")
    if len(lora_layers) > 4:
        print(f"     ... and {len(lora_layers) - 4} more")

    # Count LoRA parameters
    lora_params = count_lora_parameters(lora_layers)
    print(f"\n   LoRA parameters: {lora_params:,}")
    print(f"   Base parameters: {base_params:,}")
    print(f"   Trainable ratio: {100 * lora_params / base_params:.2f}%")

    # Forward pass with LoRA
    print("\n4. Forward pass with LoRA...")
    batch_size = 2
    seq_len = 16
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

    output = model(input_ids, labels=labels)
    mx.eval(output.loss)

    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output logits shape: {output.logits.shape}")
    print(f"   Loss: {float(output.loss):.4f}")

    # Show LoRA weight shapes
    print("\n5. LoRA weight inspection...")
    first_layer = list(lora_layers.values())[0]
    print(f"   LoRA A shape: {first_layer.lora_A.shape}")
    print(f"   LoRA B shape: {first_layer.lora_B.shape}")
    print(f"   Base weight shape: {first_layer.base_layer.weight.shape}")

    # Demonstrate weight merging (for inference)
    print("\n6. Merging LoRA weights (for efficient inference)...")
    print("   Before merge: Model uses LoRA adapters")

    # Get output before merge
    output_before = model(input_ids)
    mx.eval(output_before.logits)

    # Merge weights
    merge_lora_weights(model, lora_layers)
    print("   After merge: LoRA weights merged into base model")

    # Get output after merge (should be identical)
    output_after = model(input_ids)
    mx.eval(output_after.logits)

    # Verify outputs match
    diff = mx.abs(output_before.logits - output_after.logits)
    max_diff = float(mx.max(diff))
    print(f"   Max difference after merge: {max_diff:.6f}")
    print(f"   Merge successful: {max_diff < 1e-5}")

    print("\n" + "=" * 60)
    print("SUCCESS! LoRA fine-tuning works correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
