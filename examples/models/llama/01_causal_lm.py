"""
Llama Causal LM Example

Demonstrates a full LlamaForCausalLM model using Pydantic configs
and the models_v2 architecture.

This example shows:
- LlamaConfig Pydantic configuration
- LlamaForCausalLM model creation
- Forward pass with loss computation
- Config introspection

Run with:
    uv run python examples/models/02_llama_causal_lm.py
"""

import mlx.core as mx

from chuk_lazarus.models_v2 import LlamaConfig, LlamaForCausalLM


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
    print("Llama Causal LM Example (Pydantic-native)")
    print("=" * 60)

    # Use the tiny preset for fast demo
    print("\n1. Creating LlamaConfig (Pydantic)...")
    config = LlamaConfig.tiny()

    # Show config introspection
    print("\n   LlamaConfig attributes:")
    print(f"     vocab_size: {config.vocab_size}")
    print(f"     hidden_size: {config.hidden_size}")
    print(f"     intermediate_size: {config.intermediate_size}")
    print(f"     num_hidden_layers: {config.num_hidden_layers}")
    print(f"     num_attention_heads: {config.num_attention_heads}")
    print(f"     num_key_value_heads: {config.num_key_value_heads}")
    print(f"     max_position_embeddings: {config.max_position_embeddings}")
    print(f"     rope_theta: {config.rope_theta}")
    print(f"     rms_norm_eps: {config.rms_norm_eps}")

    # Config is frozen (immutable)
    print(f"\n   Config is frozen: {config.model_config.get('frozen', True)}")

    # Create model
    print("\n2. Creating LlamaForCausalLM...")
    model = LlamaForCausalLM(config)

    total_params = count_params(model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Create dummy input
    print("\n3. Creating input...")
    batch_size = 2
    seq_len = 32
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"   Input shape: {input_ids.shape}")

    # Forward pass without labels
    print("\n4. Running forward pass (inference mode)...")
    output = model(input_ids)
    mx.eval(output.logits)

    print(f"   Output logits shape: {output.logits.shape}")
    print(f"   Expected: ({batch_size}, {seq_len}, {config.vocab_size})")

    # Forward pass with labels (for training)
    print("\n5. Running forward pass with labels (training mode)...")
    labels = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    output_with_loss = model(input_ids, labels=labels)
    mx.eval(output_with_loss.loss)

    print(f"   Loss: {float(output_with_loss.loss):.4f}")
    print(f"   (Cross-entropy on random labels, expected ~log({config.vocab_size}) = {mx.log(mx.array(config.vocab_size)).item():.2f})")

    # Generate next token prediction
    print("\n6. Next token prediction...")
    last_logits = output.logits[:, -1, :]  # (batch, vocab)
    probs = mx.softmax(last_logits, axis=-1)
    predicted_tokens = mx.argmax(probs, axis=-1)
    print(f"   Predicted next tokens: {predicted_tokens.tolist()}")
    print(f"   Prediction confidence: {[round(float(p), 4) for p in mx.max(probs, axis=-1)]}")

    # Show model structure
    print("\n7. Model structure:")
    print(f"   model.backbone: {type(model.backbone).__name__}")
    print(f"   model.lm_head: {type(model.lm_head).__name__}")
    print(f"   Number of transformer blocks: {len(model.backbone.layers)}")

    print("\n" + "=" * 60)
    print("SUCCESS! LlamaForCausalLM works correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
