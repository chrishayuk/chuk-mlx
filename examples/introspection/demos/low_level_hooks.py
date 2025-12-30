#!/usr/bin/env python3
"""
Low-level hooks demonstration.

This example shows how to use the lower-level ModelHooks API directly,
which gives more control over what's captured during inference.

Use this when you need:
- Direct access to raw hidden states
- Custom position selection
- Integration with your own inference pipeline

For most use cases, prefer the high-level ModelAnalyzer API instead.

Usage:
    uv run python examples/introspection/low_level_hooks.py
"""

import mlx.core as mx

from chuk_lazarus.introspection import (
    CaptureConfig,
    CapturedState,
    LayerSelection,
    LogitLens,
    ModelHooks,
    PositionSelection,
)
from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM


def create_tiny_llama() -> tuple[LlamaForCausalLM, LlamaConfig]:
    """Create a tiny Llama model for testing."""
    config = LlamaConfig.tiny()
    model = LlamaForCausalLM(config)
    return model, config


def main() -> None:
    print("=" * 60)
    print("Low-Level Hooks Demo")
    print("=" * 60)

    # Create model
    print("\n1. Creating tiny Llama model...")
    model, config = create_tiny_llama()
    print(f"   Model: {config.num_hidden_layers} layers, {config.hidden_size} hidden dim")
    print(f"   Vocab size: {config.vocab_size}")

    # Create fake input
    input_ids = mx.array([[1, 42, 100, 7, 99]])
    print(f"\n2. Input tokens: {input_ids.tolist()[0]}")

    # Setup hooks with proper enums - no magic strings
    print("\n3. Setting up hooks...")
    hooks = ModelHooks(model)
    hooks.configure(
        CaptureConfig(
            layers=LayerSelection.ALL,  # Use enum, not "all"
            capture_hidden_states=True,
            positions=PositionSelection.ALL,  # Use enum, not "all"
        )
    )

    # Run forward pass
    print("\n4. Running forward pass with hooks...")
    logits = hooks.forward(input_ids)
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Layers captured: {hooks.state.captured_layers}")

    # Check hidden state shapes
    print("\n5. Captured hidden states:")
    for layer_idx, hidden in hooks.state.hidden_states.items():
        print(f"   Layer {layer_idx}: {hidden.shape}")

    # Logit lens analysis
    print("\n6. Logit Lens Analysis:")
    print("-" * 40)

    # Create a simple mock tokenizer for demo
    class SimpleTokenizer:
        def __init__(self, vocab_size: int):
            self.vocab_size = vocab_size

        def decode(self, ids: list[int]) -> str:
            return "".join(f"<{i}>" for i in ids)

        def encode(self, text: str) -> list[int]:
            if text.startswith("<") and text.endswith(">"):
                try:
                    return [int(text[1:-1])]
                except ValueError:
                    pass
            return [0]

    tokenizer = SimpleTokenizer(config.vocab_size)
    lens = LogitLens(hooks, tokenizer)

    # Get predictions at each layer for last position
    predictions = lens.get_layer_predictions(position=-1, top_k=5)
    for pred in predictions:
        top_3 = list(zip(pred.top_ids[:3], [f"{p:.4f}" for p in pred.top_probs[:3]]))
        print(f"   Layer {pred.layer_idx}: top tokens = {top_3}")

    # Track a specific token
    print("\n7. Tracking token ID 50 across layers:")
    evolution = lens.track_token(token=50, position=-1)
    for layer, prob in zip(evolution.layers, evolution.probabilities):
        bar = "#" * int(prob * 50)
        print(f"   Layer {layer}: {prob:.4f} {bar}")

    if evolution.emergence_layer is not None:
        print(f"   Token 50 becomes top-1 at layer {evolution.emergence_layer}")
    else:
        print("   Token 50 never becomes top-1")

    # Get layer logits directly
    print("\n8. Layer-by-layer logit projection:")
    for layer_idx in [0, config.num_hidden_layers // 2, config.num_hidden_layers - 1]:
        layer_logits = hooks.get_layer_logits(layer_idx, normalize=True)
        if layer_logits is not None:
            # Get top prediction at last position
            last_pos_logits = layer_logits[0, -1, :]
            probs = mx.softmax(last_pos_logits)
            top_idx = mx.argmax(probs).item()
            top_prob = probs[top_idx].item()
            print(f"   Layer {layer_idx}: top token = {top_idx} (prob={top_prob:.4f})")

    # Forward to specific layer
    print("\n9. Forward to layer 1 only:")
    hidden_at_1 = hooks.forward_to_layer(input_ids, target_layer=1)
    print(f"   Hidden state shape at layer 1: {hidden_at_1.shape}")

    # Demonstrate using specific layer indices instead of ALL
    print("\n10. Capturing specific layers only:")
    hooks2 = ModelHooks(model)
    hooks2.configure(
        CaptureConfig(
            layers=[0, 2, 4],  # Specific layer indices
            capture_hidden_states=True,
            positions=PositionSelection.LAST,
        )
    )
    hooks2.forward(input_ids)
    print(f"    Captured layers: {hooks2.state.captured_layers}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
