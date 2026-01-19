#!/usr/bin/env python3
"""
Gemma Attention Head Ablation Study.

Since MLP neuron ablation showed no effect (20% neurons ablated = 0% accuracy drop),
this script tests whether ATTENTION HEADS are more localized and causally important.

Hypotheses:
1. Attention heads may be more specialized than MLP neurons
2. Specific heads may handle "lookup" of multiplication facts
3. Ablating key attention heads may break multiplication

Usage:
    uv run python examples/introspection/experiments/model_specific/gemma_attention_ablation.py
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


class AttentionAblationStudy:
    """Ablate attention heads and measure impact on multiplication."""

    def __init__(self, model_id: str = "mlx-community/gemma-3-4b-it-bf16"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.config = None

    def load_model(self):
        """Load the model."""
        print(f"Loading model: {self.model_id}")

        result = HFLoader.download(self.model_id)
        model_path = result.model_path

        with open(model_path / "config.json") as f:
            config_data = json.load(f)

        family_type = detect_model_family(config_data)
        family_info = get_family_info(family_type)
        self.config = family_info.config_class.from_hf_config(config_data)
        self.model = family_info.model_class(self.config)

        HFLoader.apply_weights_to_model(self.model, model_path, self.config, dtype=DType.BFLOAT16)
        self.tokenizer = HFLoader.load_tokenizer(model_path)

        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.head_dim

        print(f"  Layers: {self.num_layers}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Attention heads: {self.num_heads}")
        print(f"  Head dim: {self.head_dim}")

    def _get_components(self):
        """Get model components."""
        if hasattr(self.model, "model"):
            backbone = self.model.model
        else:
            backbone = self.model

        layers = list(backbone.layers)
        embed = backbone.embed_tokens
        norm = getattr(backbone, "norm", None)

        if hasattr(self.model, "lm_head"):
            head = self.model.lm_head
        else:
            head = None

        embed_scale = float(self.hidden_size**0.5)

        return layers, embed, norm, head, embed_scale

    def generate(self, prompt: str, max_tokens: int = 5) -> str:
        """Generate tokens from a prompt."""
        layers, embed, norm, head, embed_scale = self._get_components()

        input_ids = self.tokenizer.encode(prompt)
        input_ids = mx.array(input_ids)[None, :]

        for _ in range(max_tokens):
            seq_len = input_ids.shape[1]
            h = embed(input_ids) * embed_scale
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(h.dtype)

            for layer in layers:
                try:
                    out = layer(h, mask=mask)
                except TypeError:
                    out = layer(h)

                if hasattr(out, "hidden_states"):
                    h = out.hidden_states
                elif isinstance(out, tuple):
                    h = out[0]
                else:
                    h = out

            if norm is not None:
                h = norm(h)

            if head is not None:
                logits = head(h)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            else:
                logits = h @ embed.weight.T

            next_token = mx.argmax(logits[0, -1, :])
            input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

            if int(next_token) in [self.tokenizer.eos_token_id, 13, 10]:
                break

        output_ids = input_ids[0, len(self.tokenizer.encode(prompt)) :].tolist()
        return self.tokenizer.decode(output_ids).strip()

    def generate_with_head_ablation(
        self,
        prompt: str,
        layer_idx: int,
        heads_to_ablate: list[int],
        max_tokens: int = 5,
    ) -> str:
        """
        Generate with specific attention heads ablated.

        Ablation: Zero out the output of specific heads before the output projection.
        """
        layers, embed, norm, head, embed_scale = self._get_components()

        input_ids = self.tokenizer.encode(prompt)
        input_ids = mx.array(input_ids)[None, :]

        for _ in range(max_tokens):
            seq_len = input_ids.shape[1]
            h = embed(input_ids) * embed_scale
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(h.dtype)

            for i, layer in enumerate(layers):
                if i == layer_idx:
                    # Custom forward with head ablation
                    h = self._forward_with_head_ablation(layer, h, mask, heads_to_ablate)
                else:
                    try:
                        out = layer(h, mask=mask)
                    except TypeError:
                        out = layer(h)

                    if hasattr(out, "hidden_states"):
                        h = out.hidden_states
                    elif isinstance(out, tuple):
                        h = out[0]
                    else:
                        h = out

            if norm is not None:
                h = norm(h)

            if head is not None:
                logits = head(h)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            else:
                logits = h @ embed.weight.T

            next_token = mx.argmax(logits[0, -1, :])
            input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

            if int(next_token) in [self.tokenizer.eos_token_id, 13, 10]:
                break

        output_ids = input_ids[0, len(self.tokenizer.encode(prompt)) :].tolist()
        return self.tokenizer.decode(output_ids).strip()

    def _forward_with_head_ablation(
        self,
        layer: nn.Module,
        h: mx.array,
        mask: mx.array,
        heads_to_ablate: list[int],
    ) -> mx.array:
        """Forward through a layer with specific attention heads zeroed."""
        batch_size, seq_len, _ = h.shape

        # Input layernorm
        if hasattr(layer, "input_layernorm"):
            h_normed = layer.input_layernorm(h)
        else:
            h_normed = h

        attn = layer.self_attn

        # Manual attention with head ablation
        # Q, K, V projections
        queries = attn.q_proj(h_normed)
        keys = attn.k_proj(h_normed)
        values = attn.v_proj(h_normed)

        # Reshape to (batch, heads, seq, head_dim)
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        head_dim = attn.head_dim
        n_rep = num_heads // num_kv_heads

        queries = queries.reshape(batch_size, seq_len, num_heads, head_dim)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_size, seq_len, num_kv_heads, head_dim)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_len, num_kv_heads, head_dim)
        values = values.transpose(0, 2, 1, 3)

        # Query/key normalization (Gemma-specific)
        queries = attn.q_norm(queries)
        keys = attn.k_norm(keys)

        # RoPE
        queries = attn.rope(queries)
        keys = attn.rope(keys)

        # Repeat KV heads for GQA
        if n_rep > 1:
            keys = mx.repeat(keys, n_rep, axis=1)
            values = mx.repeat(values, n_rep, axis=1)

        # Scaled dot-product attention
        # Shape: (batch, heads, seq, seq)
        scale = attn.scale
        attn_weights = (queries @ keys.transpose(0, 1, 3, 2)) * scale

        # Apply mask
        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = mx.softmax(attn_weights, axis=-1)

        # Attention output: (batch, heads, seq, head_dim)
        attn_output = attn_weights @ values

        # ABLATE: Zero out specific heads
        heads_set = set(heads_to_ablate)
        ablation_mask = mx.array([0.0 if i in heads_set else 1.0 for i in range(num_heads)])
        ablation_mask = ablation_mask.reshape(1, num_heads, 1, 1)
        ablation_mask = ablation_mask.astype(attn_output.dtype)

        attn_output = attn_output * ablation_mask

        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # Output projection
        attn_output = attn.o_proj(attn_output)

        # Post-attention norm (Gemma-specific)
        if hasattr(layer, "post_attention_layernorm"):
            attn_output = layer.post_attention_layernorm(attn_output)

        # Residual
        h = h + attn_output

        # MLP (unchanged)
        if hasattr(layer, "pre_feedforward_layernorm"):
            mlp_input = layer.pre_feedforward_layernorm(h)
        else:
            mlp_input = h

        mlp_out = layer.mlp(mlp_input)

        if hasattr(layer, "post_feedforward_layernorm"):
            mlp_out = layer.post_feedforward_layernorm(mlp_out)

        h = h + mlp_out

        return h

    def test_multiplication_accuracy(
        self,
        ablate_heads: list[int] | None = None,
        ablate_layer: int | None = None,
    ) -> tuple[float, list[dict]]:
        """Test multiplication accuracy with optional head ablation."""
        test_cases = [
            (2, 3, 6),
            (3, 4, 12),
            (5, 6, 30),
            (7, 8, 56),
            (9, 9, 81),
            (4, 7, 28),
            (6, 8, 48),
            (3, 9, 27),
            (5, 5, 25),
            (8, 9, 72),
        ]

        results = []
        correct = 0

        for a, b, expected in test_cases:
            prompt = f"{a} * {b} = "

            if ablate_heads and ablate_layer is not None:
                output = self.generate_with_head_ablation(
                    prompt, ablate_layer, ablate_heads, max_tokens=5
                )
            else:
                output = self.generate(prompt, max_tokens=5)

            is_correct = str(expected) in output

            if is_correct:
                correct += 1

            results.append(
                {
                    "prompt": prompt,
                    "expected": str(expected),
                    "output": output,
                    "correct": is_correct,
                }
            )

        accuracy = correct / len(test_cases)
        return accuracy, results

    def run_ablation_study(self):
        """Run the attention head ablation study."""
        self.load_model()

        print("\n" + "=" * 70)
        print("ATTENTION HEAD ABLATION STUDY")
        print("=" * 70)

        # Baseline
        print("\n1. Computing baseline accuracy...")
        baseline_acc, baseline_results = self.test_multiplication_accuracy()
        print(f"   Baseline accuracy: {baseline_acc:.1%}")

        for r in baseline_results[:3]:
            status = "✓" if r["correct"] else "✗"
            print(f"   {status} {r['prompt']} -> {r['output']}")

        # Single head ablation at different layers
        print("\n2. Single head ablation tests...")
        print(f"\n{'Layer':<8} {'Head':<8} {'Accuracy':<12} {'Drop'}")
        print("-" * 40)

        single_results = []
        key_layers = [16, 20, 24, 28]

        for layer in key_layers:
            # Test first few heads
            for head in range(min(4, self.num_heads)):
                acc, _ = self.test_multiplication_accuracy(
                    ablate_heads=[head],
                    ablate_layer=layer,
                )
                drop = baseline_acc - acc
                if drop > 0:
                    print(f"L{layer:<7} H{head:<7} {acc:>10.1%} {drop:>+9.1%}")

                single_results.append(
                    {
                        "layer": layer,
                        "head": head,
                        "accuracy": acc,
                        "drop": drop,
                    }
                )

        # Multi-head ablation
        print("\n3. Multi-head ablation tests...")

        for layer in [20, 24]:
            for num_heads in [1, 2, 4, 8]:
                if num_heads > self.num_heads:
                    break

                heads = list(range(num_heads))
                acc, _ = self.test_multiplication_accuracy(
                    ablate_heads=heads,
                    ablate_layer=layer,
                )
                drop = baseline_acc - acc
                pct = num_heads / self.num_heads * 100
                print(
                    f"   L{layer} heads 0-{num_heads - 1} ({pct:.0f}%): {acc:.1%} (drop: {drop:+.1%})"
                )

        # Ablate ALL heads at a layer
        print("\n4. Complete layer attention ablation...")

        for layer in [16, 20, 24, 28]:
            all_heads = list(range(self.num_heads))
            acc, results = self.test_multiplication_accuracy(
                ablate_heads=all_heads,
                ablate_layer=layer,
            )
            drop = baseline_acc - acc
            print(f"   L{layer} ALL {self.num_heads} heads: {acc:.1%} (drop: {drop:+.1%})")

            if drop > 0.1:
                print("      Sample outputs:")
                for r in results[:3]:
                    print(f"      {r['prompt']} -> {r['output']}")

        # Detailed comparison
        print("\n5. Detailed comparison: 7 * 8 = 56")
        prompt = "7 * 8 = "
        baseline_out = self.generate(prompt, max_tokens=5)
        print(f"   Baseline: {baseline_out}")

        for layer in [20, 24]:
            for num_heads in [4, 8, self.num_heads]:
                heads = list(range(num_heads))
                out = self.generate_with_head_ablation(prompt, layer, heads, max_tokens=5)
                print(f"   L{layer} heads 0-{num_heads - 1}: {out}")

        # Summary
        print("\n" + "=" * 70)
        print("ATTENTION HEAD ABLATION SUMMARY")
        print("=" * 70)

        # Find most impactful
        impactful = [r for r in single_results if r["drop"] > 0]
        if impactful:
            print("\nMost impactful single heads:")
            for r in sorted(impactful, key=lambda x: -x["drop"])[:5]:
                print(f"  L{r['layer']} H{r['head']}: {r['drop']:+.1%} drop")
        else:
            print("\nNo single head showed impact (like MLP neurons)")

        print("\nComparison with MLP ablation:")
        print("  - MLP: 2000 neurons (20%) ablated = 0% drop")
        print("  - Attention: Results shown above")

        # Save
        results = {
            "model": self.model_id,
            "baseline_accuracy": baseline_acc,
            "num_heads": self.num_heads,
            "single_head_results": single_results,
        }

        output_path = Path("gemma_discovery_cache/attention_ablation.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return results


def main():
    study = AttentionAblationStudy()
    study.run_ablation_study()


if __name__ == "__main__":
    main()
