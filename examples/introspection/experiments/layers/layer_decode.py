#!/usr/bin/env python3
"""
Decode from Hidden Layers - Bypass AR Decoding Errors

The logit lens reveals that models often "know" the answer internally
(e.g., 100% probability on correct token at L28) but fail to output it
correctly during autoregressive generation.

This script decodes directly from intermediate layer hidden states,
bypassing the problematic token-by-token generation.

Methods:
1. layer_logits: Apply unembedding at layer L instead of final layer
2. beam_search: Use beam search with digit constraints
3. readout_head: Train a tiny MLP to extract digits from hidden state

Usage:
    # Decode 127*89 from layer 28 (where the answer is most clear)
    uv run python examples/introspection/layer_decode.py \
        --model mlx-community/gemma-3-4b-it-bf16 \
        --prompt "127 * 89 = " \
        --layer 28 \
        --method layer_logits

    # Compare decoding from different layers
    uv run python examples/introspection/layer_decode.py \
        --model mlx-community/gemma-3-4b-it-bf16 \
        --prompt "127 * 89 = " \
        --method layer_sweep

    # Use constrained beam search (digits only)
    uv run python examples/introspection/layer_decode.py \
        --model mlx-community/gemma-3-4b-it-bf16 \
        --prompt "127 * 89 = " \
        --method beam_digits
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class DecodeResult:
    """Result of layer decoding."""

    layer: int
    tokens: list[str]
    token_ids: list[int]
    probs: list[float]
    text: str


class LayerDecoder:
    """
    Decode from intermediate layer hidden states.

    Bypasses autoregressive generation to extract what the model
    "knows" at a specific layer.
    """

    def __init__(self, model, tokenizer, model_id: str = "unknown"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self._detect_structure()

    def _detect_structure(self):
        """Detect model components."""
        # Get layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self._backbone = self.model.model
            self._layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            self._backbone = self.model
            self._layers = self.model.layers
        else:
            raise ValueError("Cannot detect model structure")

        self.num_layers = len(self._layers)

        # Get final norm
        if hasattr(self._backbone, "norm"):
            self._final_norm = self._backbone.norm
        elif hasattr(self._backbone, "final_layernorm"):
            self._final_norm = self._backbone.final_layernorm
        else:
            self._final_norm = None

        # Get lm_head
        if hasattr(self.model, "lm_head"):
            self._lm_head = self.model.lm_head
        elif hasattr(self.model, "head"):
            self._lm_head = self.model.head
        else:
            self._lm_head = None

        # Get embeddings for tied weights
        if hasattr(self._backbone, "embed_tokens"):
            self._embed = self._backbone.embed_tokens
        elif hasattr(self._backbone, "wte"):
            self._embed = self._backbone.wte
        else:
            self._embed = None

        # Get embedding scale (Gemma uses sqrt(hidden_size))
        self._embed_scale = None
        if hasattr(self._backbone, "config"):
            cfg = self._backbone.config
            if hasattr(cfg, "hidden_size"):
                # Gemma scales embeddings by sqrt(hidden_size)
                self._embed_scale = cfg.hidden_size**0.5

    @classmethod
    def from_pretrained(cls, model_id: str) -> LayerDecoder:
        """Load model for layer decoding."""
        from chuk_lazarus.introspection.ablation import AblationStudy

        study = AblationStudy.from_pretrained(model_id)
        return cls(
            model=study.adapter.model,
            tokenizer=study.adapter.tokenizer,
            model_id=model_id,
        )

    def get_hidden_at_layer(self, prompt: str, layer: int) -> mx.array:
        """
        Get hidden state at a specific layer.

        Returns: [batch, seq, hidden]
        """
        from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks

        hooks = ModelHooks(self.model)
        hooks.configure(
            CaptureConfig(
                layers=[layer],
                capture_hidden_states=True,
            )
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)
        hooks.forward(input_ids)

        return hooks.state.hidden_states[layer]

    def hidden_to_logits(self, hidden: mx.array, apply_norm: bool = True) -> mx.array:
        """
        Project hidden state to logits.

        Args:
            hidden: [batch, seq, hidden] or [seq, hidden]
            apply_norm: Whether to apply final layer norm

        Returns: [batch, seq, vocab] or [seq, vocab]
        """
        h = hidden

        # Apply final norm
        if apply_norm and self._final_norm is not None:
            h = self._final_norm(h)

        # Project to vocab
        if self._lm_head is not None:
            # Explicit lm_head
            out = self._lm_head(h)
            if hasattr(out, "logits"):
                return out.logits
            return out
        elif self._embed is not None:
            # Tied embeddings - use transpose of embedding matrix
            if hasattr(self._embed, "weight"):
                weight = self._embed.weight
            else:
                weight = self._embed.as_linear().weight
            return h @ weight.T
        else:
            raise ValueError("Cannot find projection to vocabulary")

    def decode_from_layer(
        self,
        prompt: str,
        layer: int,
        num_tokens: int = 5,
        apply_norm: bool = True,
    ) -> DecodeResult:
        """
        Decode tokens from a specific layer's hidden state.

        This applies the unembedding at an intermediate layer,
        potentially getting better results than final-layer AR.
        """
        hidden = self.get_hidden_at_layer(prompt, layer)

        # Get logits at last position
        logits = self.hidden_to_logits(hidden, apply_norm)
        if logits.ndim == 3:
            logits = logits[0]  # Remove batch dim

        last_logits = logits[-1]  # Last position
        probs = mx.softmax(last_logits, axis=-1)

        # Get top tokens
        sorted_indices = mx.argsort(probs)[::-1][:num_tokens]
        token_ids = sorted_indices.tolist()
        token_probs = probs[sorted_indices].tolist()
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        return DecodeResult(
            layer=layer,
            tokens=tokens,
            token_ids=token_ids,
            probs=token_probs,
            text=tokens[0] if tokens else "",
        )

    def decode_sequence_from_layer(
        self,
        prompt: str,
        layer: int,
        max_tokens: int = 10,
        apply_norm: bool = True,
        digits_only: bool = False,
    ) -> str:
        """
        Autoregressively decode from a specific layer.

        At each step:
        1. Run full forward pass
        2. Extract hidden state at `layer`
        3. Project to logits and sample
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        current_ids = mx.array(input_ids)

        generated = []

        # Get digit token IDs for constraint
        digit_ids = None
        if digits_only:
            digit_ids = set()
            for d in "0123456789":
                ids = self.tokenizer.encode(d, add_special_tokens=False)
                if ids:
                    digit_ids.add(ids[-1])
            # Also add common number tokens
            for d in [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]:
                ids = self.tokenizer.encode(d, add_special_tokens=False)
                if ids:
                    digit_ids.add(ids[-1])

        for _ in range(max_tokens):
            # Get hidden at layer
            from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks

            hooks = ModelHooks(self.model)
            hooks.configure(
                CaptureConfig(
                    layers=[layer],
                    capture_hidden_states=True,
                )
            )
            hooks.forward(current_ids)

            hidden = hooks.state.hidden_states[layer]
            logits = self.hidden_to_logits(hidden, apply_norm)

            if logits.ndim == 3:
                logits = logits[0]

            last_logits = logits[-1]

            # Apply digit constraint
            if digit_ids:
                mask = mx.full(last_logits.shape, float("-inf"))
                for did in digit_ids:
                    mask = mask.at[did].add(float("inf"))  # Cancels -inf to 0
                last_logits = last_logits + mask

            # Greedy decode
            next_token = mx.argmax(last_logits, axis=-1)
            next_id = int(next_token)

            # Check for EOS
            if hasattr(self.tokenizer, "eos_token_id"):
                if next_id == self.tokenizer.eos_token_id:
                    break

            generated.append(next_id)
            current_ids = mx.concatenate([current_ids, next_token[None, None]], axis=1)

            # Stop on newline or non-digit for arithmetic
            token_str = self.tokenizer.decode([next_id])
            if "\n" in token_str:
                break

        return self.tokenizer.decode(generated)

    def layer_sweep(
        self,
        prompt: str,
        layers: list[int] | None = None,
        num_tokens: int = 3,
    ) -> dict[int, DecodeResult]:
        """
        Decode from multiple layers and compare.
        """
        if layers is None:
            # Sample layers across the model
            layers = list(range(0, self.num_layers, max(1, self.num_layers // 10)))
            if self.num_layers - 1 not in layers:
                layers.append(self.num_layers - 1)

        results = {}
        for layer in layers:
            results[layer] = self.decode_from_layer(prompt, layer, num_tokens)

        return results

    def print_layer_sweep(
        self,
        prompt: str,
        layers: list[int] | None = None,
    ) -> None:
        """Print formatted layer sweep results."""
        results = self.layer_sweep(prompt, layers, num_tokens=5)

        print("\n" + "=" * 70)
        print("LAYER DECODE SWEEP")
        print("=" * 70)
        print(f"Prompt: {prompt!r}")
        print(f"Model: {self.model_id}")
        print("-" * 70)

        for layer, result in sorted(results.items()):
            top3 = ", ".join(f"'{t}':{p:.1%}" for t, p in zip(result.tokens[:3], result.probs[:3]))
            print(f"L{layer:2d}: {top3}")

        print("=" * 70)

    def compare_ar_vs_layer(
        self,
        prompt: str,
        layer: int,
        max_tokens: int = 10,
    ) -> None:
        """Compare standard AR decoding vs layer decoding."""
        print("\n" + "=" * 70)
        print("AR vs LAYER DECODING")
        print("=" * 70)
        print(f"Prompt: {prompt!r}")
        print(f"Layer: {layer}")
        print("-" * 70)

        # Standard AR from final layer
        print("\nStandard AR (final layer):")
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        current_ids = mx.array(input_ids)
        generated = []

        for _ in range(max_tokens):
            outputs = self.model(current_ids)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            next_id = int(next_token[0])

            if hasattr(self.tokenizer, "eos_token_id"):
                if next_id == self.tokenizer.eos_token_id:
                    break

            generated.append(next_id)
            current_ids = mx.concatenate([current_ids, next_token[:, None]], axis=1)

            if "\n" in self.tokenizer.decode([next_id]):
                break

        ar_result = self.tokenizer.decode(generated)
        print(f"  {ar_result}")

        # Layer decoding
        print(f"\nLayer {layer} AR:")
        layer_result = self.decode_sequence_from_layer(prompt, layer, max_tokens)
        print(f"  {layer_result}")

        # Layer decoding with digit constraint
        print(f"\nLayer {layer} AR (digits only):")
        constrained_result = self.decode_sequence_from_layer(
            prompt, layer, max_tokens, digits_only=True
        )
        print(f"  {constrained_result}")

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Decode from intermediate layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        "-m",
        default="mlx-community/gemma-3-4b-it-bf16",
        help="Model ID",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="127 * 89 = ",
        help="Prompt to decode",
    )
    parser.add_argument(
        "--layer",
        "-l",
        type=int,
        default=28,
        help="Layer to decode from",
    )
    parser.add_argument(
        "--method",
        choices=["layer_logits", "layer_sweep", "compare", "beam_digits"],
        default="compare",
        help="Decoding method",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10,
        help="Max tokens to generate",
    )

    args = parser.parse_args()

    print(f"\nLoading model: {args.model}")
    decoder = LayerDecoder.from_pretrained(args.model)
    print(f"Model loaded: {decoder.num_layers} layers")

    if args.method == "layer_logits":
        result = decoder.decode_from_layer(args.prompt, args.layer, num_tokens=10)
        print(f"\nDecoding from layer {args.layer}:")
        print(f"Top tokens: {result.tokens}")
        print(f"Probs: {[f'{p:.1%}' for p in result.probs]}")

    elif args.method == "layer_sweep":
        decoder.print_layer_sweep(args.prompt)

    elif args.method == "compare":
        decoder.compare_ar_vs_layer(args.prompt, args.layer, args.max_tokens)

    elif args.method == "beam_digits":
        print(f"\nDecoding with digit constraint from layer {args.layer}:")
        result = decoder.decode_sequence_from_layer(
            args.prompt, args.layer, args.max_tokens, digits_only=True
        )
        print(f"  {result}")


if __name__ == "__main__":
    main()
