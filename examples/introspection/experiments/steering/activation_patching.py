#!/usr/bin/env python3
"""
Activation Patching: The Definitive Test for Causal Circuits.

If layer 20 truly computes the answer, then patching its activations
from "347 * 892 = " to "100 * 100 = " should make the model output
309524 instead of 10000.

This is the strongest causal evidence for computation localization.

Usage:
    uv run python examples/introspection/activation_patching.py

    # Custom prompts
    uv run python examples/introspection/activation_patching.py \
        --source "347 * 892 = " \
        --target "100 * 100 = " \
        --layer 20
"""

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any
import json

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


@dataclass
class PatchResult:
    """Result of patching activations from source to target."""
    source_prompt: str
    target_prompt: str
    patch_layer: int

    # What each prompt normally predicts
    source_prediction: str
    source_probability: float
    target_prediction: str
    target_probability: float

    # What target predicts after patching
    patched_prediction: str
    patched_probability: float

    # Did the answer transfer?
    answer_transferred: bool
    transfer_probability: float  # Probability of source's answer in patched output


class ActivationPatcher:
    """
    Patches activations between prompts at specific layers.

    The idea: if layer L computes the answer, then replacing
    layer L's activations from prompt A into prompt B should
    make B output A's answer.
    """

    def __init__(self, model: nn.Module, tokenizer: Any, config: Any = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @classmethod
    async def from_pretrained(cls, model_id: str) -> "ActivationPatcher":
        """Load model from HuggingFace."""
        # Download model
        result = HFLoader.download(model_id)
        model_path = result.model_path

        # Load config
        config_path = model_path / "config.json"
        with open(config_path) as f:
            config_data = json.load(f)

        # Detect model family
        family_type = detect_model_family(config_data)
        if family_type is None:
            raise ValueError(f"Unsupported model: {model_id}")

        family_info = get_family_info(family_type)
        config_class = family_info.config_class
        model_class = family_info.model_class

        # Create model
        config = config_class.from_hf_config(config_data)
        model = model_class(config)

        # Load weights
        HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)

        # Load tokenizer
        tokenizer = HFLoader.load_tokenizer(model_path)

        return cls(model, tokenizer, config)

    def _get_layers(self) -> list[nn.Module]:
        """Get transformer layers."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        if hasattr(self.model, "layers"):
            return list(self.model.layers)
        raise ValueError("Cannot find layers")

    def _get_embed_tokens(self) -> nn.Module:
        """Get embedding layer."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens
        if hasattr(self.model, "embed_tokens"):
            return self.model.embed_tokens
        raise ValueError("Cannot find embedding layer")

    def _get_final_norm(self) -> nn.Module:
        """Get final layer norm."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            return self.model.model.norm
        if hasattr(self.model, "norm"):
            return self.model.norm
        return None

    def _get_lm_head(self):
        """Get LM head."""
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        # Tied embeddings
        embed = self._get_embed_tokens()
        if hasattr(embed, "as_linear"):
            return embed.as_linear
        return None

    def _get_embedding_scale(self) -> float | None:
        """Get embedding scale."""
        if self.config and hasattr(self.config, "embedding_scale"):
            return self.config.embedding_scale
        return None

    def _forward_with_patch(
        self,
        input_ids: mx.array,
        patch_layer: int,
        patch_activations: mx.array,
    ) -> mx.array:
        """
        Run forward pass, patching in activations at a specific layer.

        Args:
            input_ids: Input token IDs
            patch_layer: Which layer to patch
            patch_activations: Activations to inject at patch_layer

        Returns:
            Logits after patched forward pass
        """
        layers = self._get_layers()
        embed = self._get_embed_tokens()
        final_norm = self._get_final_norm()
        lm_head = self._get_lm_head()

        # Embeddings
        h = embed(input_ids)
        embed_scale = self._get_embedding_scale()
        if embed_scale:
            h = h * embed_scale

        # Create causal mask
        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        # Process layers
        for layer_idx, layer in enumerate(layers):
            # PATCH: Replace activations at the specified layer
            if layer_idx == patch_layer:
                # Ensure shapes match
                if patch_activations.shape == h.shape:
                    h = patch_activations
                else:
                    # If shapes don't match, only patch last position
                    h = h.at[:, -1:, :].set(patch_activations[:, -1:, :])

            try:
                layer_out = layer(h, mask=mask)
            except TypeError:
                layer_out = layer(h)

            # Extract hidden state
            if hasattr(layer_out, "hidden_states"):
                h = layer_out.hidden_states
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
            else:
                h = layer_out

        # Final norm and LM head
        if final_norm:
            h = final_norm(h)

        if lm_head:
            head_out = lm_head(h)
            if hasattr(head_out, "logits"):
                return head_out.logits
            return head_out

        return h

    def _get_activations_at_layer(
        self,
        input_ids: mx.array,
        layer_idx: int,
    ) -> tuple[mx.array, mx.array]:
        """
        Run forward pass up to a layer and return hidden state + final logits.

        Returns:
            (hidden_at_layer, final_logits)
        """
        layers = self._get_layers()
        embed = self._get_embed_tokens()
        final_norm = self._get_final_norm()
        lm_head = self._get_lm_head()

        # Embeddings
        h = embed(input_ids)
        embed_scale = self._get_embedding_scale()
        if embed_scale:
            h = h * embed_scale

        # Create causal mask
        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        captured_h = None

        # Process layers
        for idx, layer in enumerate(layers):
            try:
                layer_out = layer(h, mask=mask)
            except TypeError:
                layer_out = layer(h)

            # Extract hidden state
            if hasattr(layer_out, "hidden_states"):
                h = layer_out.hidden_states
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
            else:
                h = layer_out

            # Capture at target layer
            if idx == layer_idx:
                captured_h = mx.array(h)  # Copy

        # Final norm and LM head
        if final_norm:
            h = final_norm(h)

        if lm_head:
            head_out = lm_head(h)
            if hasattr(head_out, "logits"):
                logits = head_out.logits
            else:
                logits = head_out
        else:
            logits = h

        return captured_h, logits

    def patch(
        self,
        source_prompt: str,
        target_prompt: str,
        patch_layer: int,
    ) -> PatchResult:
        """
        Patch activations from source into target at specified layer.

        Args:
            source_prompt: Prompt whose answer we want to inject
            target_prompt: Prompt to inject into
            patch_layer: Layer at which to perform the patch

        Returns:
            PatchResult with predictions before and after patching
        """
        # Tokenize both prompts
        source_ids = mx.array(self.tokenizer.encode(source_prompt))[None, :]
        target_ids = mx.array(self.tokenizer.encode(target_prompt))[None, :]

        # Get source activations at patch layer
        source_h, source_logits = self._get_activations_at_layer(source_ids, patch_layer)

        # Get target normal predictions
        _, target_logits = self._get_activations_at_layer(target_ids, patch_layer)

        # Get predictions
        source_probs = mx.softmax(source_logits[0, -1, :])
        source_top_idx = int(mx.argmax(source_probs))
        source_pred = self.tokenizer.decode([source_top_idx])
        source_prob = float(source_probs[source_top_idx])

        target_probs = mx.softmax(target_logits[0, -1, :])
        target_top_idx = int(mx.argmax(target_probs))
        target_pred = self.tokenizer.decode([target_top_idx])
        target_prob = float(target_probs[target_top_idx])

        # Now patch: run target with source's activations at patch_layer
        patched_logits = self._forward_with_patch(target_ids, patch_layer, source_h)

        patched_probs = mx.softmax(patched_logits[0, -1, :])
        patched_top_idx = int(mx.argmax(patched_probs))
        patched_pred = self.tokenizer.decode([patched_top_idx])
        patched_prob = float(patched_probs[patched_top_idx])

        # Check if source's answer transferred
        transfer_prob = float(patched_probs[source_top_idx])
        answer_transferred = patched_pred == source_pred or transfer_prob > 0.1

        return PatchResult(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            patch_layer=patch_layer,
            source_prediction=source_pred,
            source_probability=source_prob,
            target_prediction=target_pred,
            target_probability=target_prob,
            patched_prediction=patched_pred,
            patched_probability=patched_prob,
            answer_transferred=answer_transferred,
            transfer_probability=transfer_prob,
        )


async def run_patch_experiment(
    model_id: str,
    source_prompt: str,
    target_prompt: str,
    layers: list[int] | None = None,
):
    """Run activation patching experiment."""

    print(f"\n{'='*70}")
    print("ACTIVATION PATCHING EXPERIMENT")
    print(f"{'='*70}")
    print(f"Model: {model_id}")
    print(f"Source: {repr(source_prompt)}")
    print(f"Target: {repr(target_prompt)}")
    print(f"{'='*70}")

    print("\nLoading model...")
    patcher = await ActivationPatcher.from_pretrained(model_id)

    num_layers = len(patcher._get_layers())
    print(f"Model has {num_layers} layers")

    # Default: sweep layers around 20
    if layers is None:
        layers = list(range(0, num_layers, 4)) + [16, 18, 19, 20, 21, 22, 24]
        layers = sorted(set(l for l in layers if l < num_layers))

    print(f"\nPatching at layers: {layers}")
    print()

    # Header
    print(f"{'Layer':<8} {'Src Pred':<12} {'Tgt Pred':<12} {'Patched':<12} {'Transfer?':<12} {'Xfer Prob'}")
    print("-" * 70)

    results = []
    for layer in layers:
        result = patcher.patch(source_prompt, target_prompt, layer)
        results.append(result)

        xfer_str = "YES ✓" if result.answer_transferred else "no"
        print(
            f"{layer:<8} "
            f"{repr(result.source_prediction):<12} "
            f"{repr(result.target_prediction):<12} "
            f"{repr(result.patched_prediction):<12} "
            f"{xfer_str:<12} "
            f"{result.transfer_probability:.4f}"
        )

    # Find causal layers
    print()
    print("="*70)
    print("ANALYSIS")
    print("="*70)

    causal_layers = [r for r in results if r.answer_transferred]
    if causal_layers:
        layers_str = ", ".join(str(r.patch_layer) for r in causal_layers)
        print(f"\n✓ Answer transfers at layers: {layers_str}")

        # Find the earliest causal layer
        earliest = min(causal_layers, key=lambda r: r.patch_layer)
        print(f"✓ Earliest causal layer: {earliest.patch_layer}")
        print(f"  → This is where the arithmetic computation happens!")
    else:
        print("\n✗ Answer did not transfer at any layer")
        print("  → Computation may be more distributed")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Activation patching to find causal computation layers",
    )
    parser.add_argument(
        "--model", "-m",
        default="mlx-community/gemma-3-4b-it-bf16",
        help="Model to analyze",
    )
    parser.add_argument(
        "--source", "-s",
        default="347 * 892 = ",
        help="Source prompt (answer to transfer)",
    )
    parser.add_argument(
        "--target", "-t",
        default="100 * 100 = ",
        help="Target prompt (to inject into)",
    )
    parser.add_argument(
        "--layer", "-l",
        type=int,
        default=None,
        help="Specific layer to patch (otherwise sweeps)",
    )

    args = parser.parse_args()

    layers = [args.layer] if args.layer is not None else None

    asyncio.run(run_patch_experiment(
        args.model,
        args.source,
        args.target,
        layers,
    ))


if __name__ == "__main__":
    main()
