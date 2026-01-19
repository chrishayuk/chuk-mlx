#!/usr/bin/env python3
"""
Generic Activation Steering for Any Model

This script provides model-agnostic activation steering using contrastive prompts.
No pre-extracted directions needed - directions are computed on-the-fly.

The key insight: To steer a model, you need a DIRECTION in activation space.
This direction can come from:
1. Pre-extracted directions (circuit analysis)
2. Contrastive prompts (difference between two prompts)
3. Mean difference (difference between prompt categories)

Usage:
    # Contrastive steering: "make it more like prompt B than prompt A"
    uv run python examples/introspection/generic_steering.py \
        --model mlx-community/gemma-3-4b-it-bf16 \
        --prompt-a "I don't know the answer" \
        --prompt-b "The answer is 42" \
        --test-prompt "What is 6 * 7?" \
        --layer 24 \
        --coefficients "-2.0,0.0,2.0"

    # Steering toward confidence
    uv run python examples/introspection/generic_steering.py \
        --model mlx-community/Llama-3.2-3B-Instruct-4bit \
        --prompt-a "I'm not sure, but maybe" \
        --prompt-b "I am certain that" \
        --test-prompt "The capital of France is" \
        --layer 16

    # Steering toward helpfulness vs refusal
    uv run python examples/introspection/generic_steering.py \
        --model mlx-community/gemma-3-4b-it-bf16 \
        --prompt-a "I cannot help with that" \
        --prompt-b "I'd be happy to help" \
        --test-prompt "How do I" \
        --layer 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class ContrastiveDirection:
    """A steering direction derived from contrastive prompts."""

    direction: mx.array  # [hidden_size]
    layer: int
    prompt_a: str  # Negative pole
    prompt_b: str  # Positive pole

    @property
    def hidden_size(self) -> int:
        return self.direction.shape[0]


class GenericSteering:
    """
    Generic activation steering that works with any MLX model.

    Computes steering directions on-the-fly from contrastive prompts.
    """

    def __init__(self, model, tokenizer, model_id: str = "unknown"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id

        # Detect model structure
        self._detect_structure()

        # Cache for computed directions
        self._directions: dict[int, ContrastiveDirection] = {}
        self._original_layers: dict[int, object] = {}

    def _detect_structure(self):
        """Detect model layer structure."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self._layers = self.model.model.layers
            self._backbone = self.model.model
        elif hasattr(self.model, "layers"):
            self._layers = self.model.layers
            self._backbone = self.model
        elif hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "h"):
                self._layers = self.model.transformer.h
            else:
                self._layers = self.model.transformer.layers
            self._backbone = self.model.transformer
        else:
            raise ValueError("Cannot detect model layer structure")

        self.num_layers = len(self._layers)

        # Detect hidden size from first layer
        layer = self._layers[0]
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
            self.hidden_size = layer.mlp.down_proj.weight.shape[0]
        elif hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            self.hidden_size = layer.self_attn.o_proj.weight.shape[0]
        else:
            # Fallback: run a forward pass
            self.hidden_size = None

    @classmethod
    def from_pretrained(cls, model_id: str) -> GenericSteering:
        """Load a model for steering."""
        from chuk_lazarus.introspection.ablation import AblationStudy

        study = AblationStudy.from_pretrained(model_id)
        return cls(
            model=study.adapter.model,
            tokenizer=study.adapter.tokenizer,
            model_id=model_id,
        )

    def get_hidden_state(self, prompt: str, layer: int) -> mx.array:
        """Get hidden state at a specific layer for a prompt."""
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

        # Get last position hidden state
        h = hooks.state.hidden_states[layer]  # [batch, seq, hidden] or [seq, hidden]
        if h.ndim == 3:
            h = h[0, -1, :]  # Last position
        else:
            h = h[-1, :]

        return h

    def compute_direction(
        self,
        prompt_a: str,
        prompt_b: str,
        layer: int,
    ) -> ContrastiveDirection:
        """
        Compute steering direction from contrastive prompts.

        Direction points from A → B, so:
        - Positive coefficient = more like B
        - Negative coefficient = more like A
        """
        h_a = self.get_hidden_state(prompt_a, layer)
        h_b = self.get_hidden_state(prompt_b, layer)

        # Direction is B - A (pointing toward B)
        direction = h_b - h_a

        # Normalize
        norm = mx.sqrt(mx.sum(direction * direction))
        direction = direction / (norm + 1e-8)

        return ContrastiveDirection(
            direction=direction,
            layer=layer,
            prompt_a=prompt_a,
            prompt_b=prompt_b,
        )

    def add_direction(self, direction: ContrastiveDirection) -> None:
        """Add a steering direction."""
        self._directions[direction.layer] = direction

    def _wrap_layer(self, layer_idx: int, coefficient: float) -> None:
        """Wrap a layer to apply steering."""
        if layer_idx not in self._directions:
            return

        direction = self._directions[layer_idx]

        # Store original
        if layer_idx not in self._original_layers:
            self._original_layers[layer_idx] = self._layers[layer_idx]

        original_layer = self._original_layers[layer_idx]

        class SteeredWrapper:
            def __init__(self, layer, direction, coef):
                self._wrapped = layer
                self._direction = direction
                self._coef = coef

            def __call__(self, h, **kwargs):
                out = self._wrapped(h, **kwargs)

                # Get hidden states
                if hasattr(out, "hidden_states"):
                    hidden = out.hidden_states
                elif isinstance(out, tuple):
                    hidden = out[0]
                else:
                    hidden = out

                # Scale by activation norm for meaningful steering
                h_norm = mx.sqrt(mx.mean(hidden * hidden))
                steering = self._direction * self._coef * h_norm

                # Apply steering
                steered = hidden + steering

                # Return in same format
                if hasattr(out, "hidden_states"):
                    out.hidden_states = steered
                    return out
                elif isinstance(out, tuple):
                    return (steered,) + out[1:]
                else:
                    return steered

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

        self._layers[layer_idx] = SteeredWrapper(original_layer, direction.direction, coefficient)

    def _unwrap_layers(self) -> None:
        """Restore original layers."""
        for layer_idx, original in self._original_layers.items():
            self._layers[layer_idx] = original
        self._original_layers.clear()

    def generate(
        self,
        prompt: str,
        layer: int,
        coefficient: float = 1.0,
        max_new_tokens: int = 50,
    ) -> str:
        """Generate with steering applied."""
        try:
            self._wrap_layer(layer, coefficient)

            input_ids = self.tokenizer.encode(prompt, return_tensors="np")
            input_ids = mx.array(input_ids)

            generated = []
            current_ids = input_ids

            for _ in range(max_new_tokens):
                outputs = self.model(current_ids)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs

                next_token = mx.argmax(logits[:, -1, :], axis=-1)
                generated.append(int(next_token[0]))

                if hasattr(self.tokenizer, "eos_token_id"):
                    if generated[-1] == self.tokenizer.eos_token_id:
                        break

                current_ids = mx.concatenate([current_ids, next_token[:, None]], axis=1)

            return self.tokenizer.decode(generated)

        finally:
            self._unwrap_layers()

    def compare(
        self,
        prompt: str,
        layer: int,
        coefficients: list[float] = [-1.0, 0.0, 1.0],
        max_new_tokens: int = 50,
    ) -> dict[float, str]:
        """Compare outputs with different steering coefficients."""
        results = {}
        for coef in coefficients:
            results[coef] = self.generate(prompt, layer, coef, max_new_tokens)
        return results

    def print_comparison(
        self,
        prompt: str,
        layer: int,
        coefficients: list[float] = [-1.0, 0.0, 1.0],
        max_new_tokens: int = 50,
    ) -> None:
        """Print formatted comparison."""
        direction = self._directions.get(layer)

        print("\n" + "=" * 70)
        print("CONTRASTIVE STEERING")
        print("=" * 70)
        print(f"Model: {self.model_id}")
        print(f"Layer: {layer}")
        if direction:
            print(f"Direction: '{direction.prompt_a}' → '{direction.prompt_b}'")
        print(f"Test prompt: {prompt!r}")
        print("-" * 70)

        results = self.compare(prompt, layer, coefficients, max_new_tokens)

        for coef, output in sorted(results.items()):
            if direction:
                if coef < 0:
                    label = f"← '{direction.prompt_a[:20]}...'"
                elif coef > 0:
                    label = f"→ '{direction.prompt_b[:20]}...'"
                else:
                    label = "no steering"
            else:
                label = ""

            print(f"\nCoef {coef:+.1f} {label}:")
            print(f"  {output}")

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Generic activation steering using contrastive prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model ID (HuggingFace or local path)",
    )
    parser.add_argument(
        "--prompt-a",
        required=True,
        help="Negative pole prompt (steer away from this)",
    )
    parser.add_argument(
        "--prompt-b",
        required=True,
        help="Positive pole prompt (steer toward this)",
    )
    parser.add_argument(
        "--test-prompt",
        required=True,
        help="Prompt to test steering on",
    )
    parser.add_argument(
        "--layer",
        "-l",
        type=int,
        required=True,
        help="Layer to apply steering at",
    )
    parser.add_argument(
        "--coefficients",
        type=str,
        default="-1.0,0.0,1.0",
        help="Steering coefficients (comma-separated)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate",
    )

    args = parser.parse_args()
    coefficients = [float(x) for x in args.coefficients.split(",")]

    print(f"\nLoading model: {args.model}")
    steerer = GenericSteering.from_pretrained(args.model)
    print(f"Model loaded: {steerer.num_layers} layers")

    print(f"\nComputing direction at layer {args.layer}...")
    print(f"  A (negative): {args.prompt_a!r}")
    print(f"  B (positive): {args.prompt_b!r}")

    direction = steerer.compute_direction(
        args.prompt_a,
        args.prompt_b,
        args.layer,
    )
    steerer.add_direction(direction)

    print(f"Direction computed (norm=1.0, hidden_size={direction.hidden_size})")

    steerer.print_comparison(
        args.test_prompt,
        args.layer,
        coefficients,
        args.max_tokens,
    )


if __name__ == "__main__":
    main()
