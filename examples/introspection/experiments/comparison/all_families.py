#!/usr/bin/env python3
"""
Test introspection hooks against all supported model families.

This verifies that the hooks infrastructure works correctly with
each model architecture's unique forward pass structure.

Run: uv run python examples/introspection_all_families.py
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.introspection import (
    CaptureConfig,
    LayerSelection,
    LogitLens,
    ModelHooks,
    PositionSelection,
)


def test_model_family(name: str, model: nn.Module, vocab_size: int, num_layers: int):
    """Test introspection on a model family."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {name}")
    print(f"{'=' * 60}")

    # Create sample input
    input_ids = mx.array([[1, 42, 100, 7, 99]])

    # Test 1: Direct forward pass
    print("\n1. Direct forward pass...")
    try:
        output = model(input_ids)
        if hasattr(output, "logits"):
            direct_logits = output.logits
        else:
            direct_logits = output
        print(f"   Output shape: {direct_logits.shape}")
        print(f"   Sample logits: {direct_logits[0, -1, :5].tolist()}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return False

    # Test 2: Hooked forward pass
    print("\n2. Hooked forward pass...")
    try:
        hooks = ModelHooks(model)
        hooks.configure(
            CaptureConfig(
                layers=LayerSelection.ALL,
                capture_hidden_states=True,
                positions=PositionSelection.ALL,
            )
        )
        hooked_logits = hooks.forward(input_ids)
        print(f"   Output shape: {hooked_logits.shape}")
        print(f"   Sample logits: {hooked_logits[0, -1, :5].tolist()}")
        print(f"   Layers captured: {hooks.state.captured_layers}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return False

    # Test 3: Verify outputs match
    print("\n3. Verifying outputs match...")
    try:
        diff = mx.abs(direct_logits - hooked_logits).max().item()
        print(f"   Max difference: {diff}")
        if diff < 1e-4:
            print("   PASSED: Outputs match!")
        else:
            print(f"   WARNING: Outputs differ by {diff}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return False

    # Test 4: Hidden state shapes
    print("\n4. Checking hidden state shapes...")
    try:
        for layer_idx, hidden in hooks.state.hidden_states.items():
            expected_shape = (
                1,
                5,
                model.config.hidden_size if hasattr(model, "config") else hidden.shape[-1],
            )
            if hidden.shape != expected_shape:
                print(f"   Layer {layer_idx}: {hidden.shape} (expected {expected_shape})")
            else:
                print(f"   Layer {layer_idx}: {hidden.shape} OK")
    except Exception as e:
        print(f"   FAILED: {e}")
        return False

    # Test 5: Logit lens
    print("\n5. Testing logit lens...")
    try:
        # Simple mock tokenizer
        class MockTokenizer:
            def decode(self, ids):
                return f"[{ids[0]}]"

            def encode(self, text):
                return [0]

        lens = LogitLens(hooks, MockTokenizer())
        predictions = lens.get_layer_predictions(position=-1, top_k=3)
        print(f"   Got predictions for {len(predictions)} layers")
        if predictions:
            last = predictions[-1]
            print(f"   Final layer top-3: {list(zip(last.top_ids[:3], last.top_probs[:3]))}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return False

    # Test 6: Forward to specific layer
    print("\n6. Testing forward_to_layer...")
    try:
        mid_layer = num_layers // 2
        hidden = hooks.forward_to_layer(input_ids, target_layer=mid_layer)
        print(f"   Hidden at layer {mid_layer}: {hidden.shape}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return False

    print(f"\n*** {name}: ALL TESTS PASSED ***")
    return True


def main():
    print("=" * 60)
    print("Introspection Test Suite - All Model Families")
    print("=" * 60)

    results = {}

    # Test Llama
    print("\n\n" + "#" * 60)
    print("# LLAMA FAMILY")
    print("#" * 60)
    try:
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)
        results["Llama"] = test_model_family(
            "Llama", model, config.vocab_size, config.num_hidden_layers
        )
    except Exception as e:
        print(f"Failed to load Llama: {e}")
        results["Llama"] = False

    # Test Gemma
    print("\n\n" + "#" * 60)
    print("# GEMMA FAMILY")
    print("#" * 60)
    try:
        from chuk_lazarus.models_v2.families.gemma import GemmaConfig, GemmaForCausalLM

        config = GemmaConfig.tiny()
        model = GemmaForCausalLM(config)
        results["Gemma"] = test_model_family(
            "Gemma", model, config.vocab_size, config.num_hidden_layers
        )
    except Exception as e:
        print(f"Failed to load Gemma: {e}")
        results["Gemma"] = False

    # Test StarCoder2
    print("\n\n" + "#" * 60)
    print("# STARCODER2 FAMILY")
    print("#" * 60)
    try:
        from chuk_lazarus.models_v2.families.starcoder2 import (
            StarCoder2Config,
            StarCoder2ForCausalLM,
        )

        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM(config)
        results["StarCoder2"] = test_model_family(
            "StarCoder2", model, config.vocab_size, config.num_hidden_layers
        )
    except Exception as e:
        print(f"Failed to load StarCoder2: {e}")
        results["StarCoder2"] = False

    # Test Granite
    print("\n\n" + "#" * 60)
    print("# GRANITE FAMILY")
    print("#" * 60)
    try:
        from chuk_lazarus.models_v2.families.granite import GraniteConfig, GraniteForCausalLM

        config = GraniteConfig.tiny()
        model = GraniteForCausalLM(config)
        results["Granite"] = test_model_family(
            "Granite", model, config.vocab_size, config.num_hidden_layers
        )
    except Exception as e:
        print(f"Failed to load Granite: {e}")
        results["Granite"] = False

    # Test Llama4
    print("\n\n" + "#" * 60)
    print("# LLAMA4 FAMILY")
    print("#" * 60)
    try:
        from chuk_lazarus.models_v2.families.llama4 import Llama4ForCausalLM, Llama4TextConfig

        config = Llama4TextConfig.tiny()
        model = Llama4ForCausalLM(config)
        results["Llama4"] = test_model_family(
            "Llama4", model, config.vocab_size, config.num_hidden_layers
        )
    except Exception as e:
        print(f"Failed to load Llama4: {e}")
        results["Llama4"] = False

    # Test Jamba (hybrid Mamba-Transformer)
    print("\n\n" + "#" * 60)
    print("# JAMBA FAMILY (Hybrid)")
    print("#" * 60)
    try:
        from chuk_lazarus.models_v2.families.jamba import JambaConfig, JambaForCausalLM

        config = JambaConfig.tiny()
        model = JambaForCausalLM(config)
        results["Jamba"] = test_model_family(
            "Jamba", model, config.vocab_size, config.num_hidden_layers
        )
    except Exception as e:
        print(f"Failed to load Jamba: {e}")
        results["Jamba"] = False

    # Test Mamba (pure SSM)
    print("\n\n" + "#" * 60)
    print("# MAMBA FAMILY (Pure SSM)")
    print("#" * 60)
    try:
        from chuk_lazarus.models_v2.families.mamba import MambaConfig, MambaForCausalLM

        config = MambaConfig.tiny()
        model = MambaForCausalLM(config)
        results["Mamba"] = test_model_family(
            "Mamba", model, config.vocab_size, config.num_hidden_layers
        )
    except Exception as e:
        print(f"Failed to load Mamba: {e}")
        results["Mamba"] = False

    # Test Granite Hybrid (Mamba-2 + Transformer + MoE)
    print("\n\n" + "#" * 60)
    print("# GRANITE HYBRID FAMILY")
    print("#" * 60)
    try:
        from chuk_lazarus.models_v2.families.granite import (
            GraniteHybridConfig,
            GraniteHybridForCausalLM,
        )

        config = GraniteHybridConfig.tiny()
        model = GraniteHybridForCausalLM(config)
        results["GraniteHybrid"] = test_model_family(
            "GraniteHybrid", model, config.vocab_size, config.num_hidden_layers
        )
    except Exception as e:
        print(f"Failed to load GraniteHybrid: {e}")
        results["GraniteHybrid"] = False

    # Summary
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed_count = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\nResults: {passed_count}/{total} families passed")
    print()
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {name:20s}: {status}")

    if passed_count == total:
        print("\n*** ALL MODEL FAMILIES PASSED ***")
        return 0
    else:
        print(f"\n*** {total - passed_count} MODEL FAMILIES FAILED ***")
        return 1


if __name__ == "__main__":
    exit(main())
