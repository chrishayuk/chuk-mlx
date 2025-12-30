#!/usr/bin/env python3
"""
Activation Steering: Route to external tools based on internal uncertainty.

The idea:
1. Monitor activation patterns during inference
2. When the model shows uncertainty (digit doesn't emerge, gate stays closed),
   route to an external calculator
3. This creates a "virtual circuit" that leverages model introspection

Uncertainty signals:
- Space token never reaches high probability (no answer formatting)
- First digit probability stays low (computation failed)
- High entropy at computation layers (model confused)

Usage:
    uv run python examples/introspection/activation_steering.py \
        --prompt "347 * 892 = " \
        --model "mlx-community/gemma-3-4b-it-bf16"

    # Test routing decision
    uv run python examples/introspection/activation_steering.py \
        --prompt "What is 12345678 * 87654321?" \
        --threshold 0.3
"""

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


class RouteDecision(Enum):
    """Routing decision based on activation analysis."""
    MODEL = "model"           # Model is confident, use its output
    CALCULATOR = "calculator" # Route to external calculator
    UNCERTAIN = "uncertain"   # Low confidence, might need help


@dataclass
class ActivationProbe:
    """Results from probing activations."""
    prompt: str

    # Gate signal (space token)
    space_peak_prob: float
    space_peak_layer: int
    space_peak_rank: int | None

    # Computation signal (first digit)
    digit_peak_prob: float
    digit_peak_layer: int
    digit_peak_rank: int | None

    # Entropy at key layers
    entropy_at_computation: float  # Layer ~20
    entropy_at_output: float       # Final layer

    # Derived metrics
    gate_open: bool
    digit_computed: bool
    confidence: float

    # Decision
    route: RouteDecision
    model_output: str

    def explain(self) -> str:
        """Explain the routing decision."""
        lines = [
            f"Prompt: {repr(self.prompt)}",
            f"",
            f"Gate Signal (space token):",
            f"  Peak: {self.space_peak_prob:.1%} at L{self.space_peak_layer} "
            f"(rank {self.space_peak_rank or '>100'})",
            f"  Gate {'OPEN' if self.gate_open else 'CLOSED'}",
            f"",
            f"Computation Signal (first digit):",
            f"  Peak: {self.digit_peak_prob:.1%} at L{self.digit_peak_layer} "
            f"(rank {self.digit_peak_rank or '>100'})",
            f"  Digit {'COMPUTED' if self.digit_computed else 'NOT COMPUTED'}",
            f"",
            f"Entropy:",
            f"  At computation layers: {self.entropy_at_computation:.2f}",
            f"  At output: {self.entropy_at_output:.2f}",
            f"",
            f"Confidence: {self.confidence:.1%}",
            f"Decision: {self.route.value.upper()}",
            f"Model output: {repr(self.model_output)}",
        ]
        return "\n".join(lines)


class ActivationSteering:
    """Monitor activations and make routing decisions."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Any,
        model_id: str,
        gate_threshold: float = 0.1,
        digit_threshold: float = 0.01,
        confidence_threshold: float = 0.3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_id = model_id
        self.gate_threshold = gate_threshold
        self.digit_threshold = digit_threshold
        self.confidence_threshold = confidence_threshold

    @classmethod
    async def from_pretrained(
        cls,
        model_id: str,
        gate_threshold: float = 0.1,
        digit_threshold: float = 0.01,
        confidence_threshold: float = 0.3,
    ) -> "ActivationSteering":
        """Load model."""
        print(f"Loading model: {model_id}")

        result = HFLoader.download(model_id)
        model_path = result.model_path

        config_path = model_path / "config.json"
        with open(config_path) as f:
            config_data = json.load(f)

        family_type = detect_model_family(config_data)
        if family_type is None:
            raise ValueError(f"Unsupported model: {model_id}")

        family_info = get_family_info(family_type)
        config = family_info.config_class.from_hf_config(config_data)
        model = family_info.model_class(config)

        HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
        tokenizer = HFLoader.load_tokenizer(model_path)

        return cls(
            model, tokenizer, config, model_id,
            gate_threshold, digit_threshold, confidence_threshold
        )

    def _get_layers(self):
        if hasattr(self.model, "model"):
            return list(self.model.model.layers)
        return list(self.model.layers)

    def _get_embed(self):
        if hasattr(self.model, "model"):
            return self.model.model.embed_tokens
        return self.model.embed_tokens

    def _get_norm(self):
        if hasattr(self.model, "model"):
            return getattr(self.model.model, "norm", None)
        return getattr(self.model, "norm", None)

    def _get_head(self):
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        return self._get_embed().as_linear

    def _get_scale(self):
        return getattr(self.config, "embedding_scale", None)

    def _detect_arithmetic(self, prompt: str) -> tuple[bool, list[str]]:
        """Detect if prompt contains arithmetic and extract expected first digits."""
        # Pattern: number op number =
        patterns = [
            r"(\d+)\s*\+\s*(\d+)",
            r"(\d+)\s*-\s*(\d+)",
            r"(\d+)\s*\*\s*(\d+)",
            r"(\d+)\s*/\s*(\d+)",
            r"(\d+)\s*×\s*(\d+)",
            r"(\d+)\s*÷\s*(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt)
            if match:
                a, b = int(match.group(1)), int(match.group(2))

                # Compute expected result
                if "+" in prompt:
                    result = a + b
                elif "-" in prompt:
                    result = a - b
                elif "*" in prompt or "×" in prompt:
                    result = a * b
                elif "/" in prompt or "÷" in prompt:
                    result = a // b if b != 0 else 0
                else:
                    continue

                # Return possible first digits
                result_str = str(abs(result))
                return True, [result_str[0]]

        return False, []

    def probe(self, prompt: str, possible_digits: list[str] | None = None) -> ActivationProbe:
        """
        Probe activations for routing decision.

        Args:
            prompt: The input prompt
            possible_digits: Expected first digits (if None, auto-detect for arithmetic)
        """
        # Auto-detect arithmetic
        is_arithmetic, detected_digits = self._detect_arithmetic(prompt)
        if possible_digits is None:
            possible_digits = detected_digits if is_arithmetic else ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        # Get token IDs
        space_ids = self.tokenizer.encode(" ", add_special_tokens=False)
        space_id = space_ids[0] if space_ids else None

        digit_ids = {}
        for d in possible_digits:
            ids = self.tokenizer.encode(d, add_special_tokens=False)
            if ids:
                digit_ids[d] = ids[0]

        layers = self._get_layers()
        embed = self._get_embed()
        norm = self._get_norm()
        head = self._get_head()
        scale = self._get_scale()
        num_layers = len(layers)

        h = embed(input_ids)
        if scale:
            h = h * scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

        # Track metrics
        space_peak_prob = 0.0
        space_peak_layer = 0
        space_peak_rank = None

        digit_peak_prob = 0.0
        digit_peak_layer = 0
        digit_peak_rank = None
        best_digit = None

        entropy_at_computation = 0.0
        computation_layer = int(num_layers * 0.6)  # ~60% through

        final_logits = None

        for layer_idx, layer in enumerate(layers):
            try:
                out = layer(h, mask=mask)
            except TypeError:
                out = layer(h)

            h = out.hidden_states if hasattr(out, "hidden_states") else (out[0] if isinstance(out, tuple) else out)

            # Project to logits
            h_n = norm(h) if norm else h
            logits = head(h_n)
            if hasattr(logits, "logits"):
                logits = logits.logits

            probs = mx.softmax(logits[0, -1, :])
            sorted_idx = mx.argsort(probs)[::-1][:100].tolist()

            # Track space token
            if space_id:
                prob = float(probs[space_id])
                rank = sorted_idx.index(space_id) + 1 if space_id in sorted_idx else None
                if prob > space_peak_prob:
                    space_peak_prob = prob
                    space_peak_layer = layer_idx
                    space_peak_rank = rank

            # Track digit tokens
            for d, d_id in digit_ids.items():
                prob = float(probs[d_id])
                rank = sorted_idx.index(d_id) + 1 if d_id in sorted_idx else None
                if prob > digit_peak_prob:
                    digit_peak_prob = prob
                    digit_peak_layer = layer_idx
                    digit_peak_rank = rank
                    best_digit = d

            # Track entropy at computation layer
            if layer_idx == computation_layer:
                probs_np = probs.tolist()
                entropy = -sum(p * (float('inf') if p == 0 else mx.log(mx.array(p)).item())
                              for p in probs_np if p > 0)
                entropy_at_computation = entropy

            final_logits = logits

        # Final output
        final_probs = mx.softmax(final_logits[0, -1, :])
        final_top_idx = int(mx.argmax(final_probs))
        model_output = self.tokenizer.decode([final_top_idx])

        # Final entropy
        final_probs_list = final_probs.tolist()
        entropy_at_output = -sum(p * (float('inf') if p == 0 else mx.log(mx.array(p)).item())
                                 for p in final_probs_list if p > 0)

        # Derived metrics
        gate_open = space_peak_rank is not None and space_peak_rank <= 5 and space_peak_prob > self.gate_threshold
        digit_computed = digit_peak_rank is not None and digit_peak_rank <= 10 and digit_peak_prob > self.digit_threshold

        # Confidence combines gate and digit signals
        confidence = min(space_peak_prob, digit_peak_prob) if (gate_open and digit_computed) else 0.0

        # Routing decision
        if is_arithmetic:
            if gate_open and digit_computed and digit_peak_prob > self.confidence_threshold:
                route = RouteDecision.MODEL
            elif not gate_open and not digit_computed:
                route = RouteDecision.CALCULATOR  # Model can't handle this
            else:
                route = RouteDecision.UNCERTAIN
        else:
            route = RouteDecision.MODEL  # Non-arithmetic, let model handle

        return ActivationProbe(
            prompt=prompt,
            space_peak_prob=space_peak_prob,
            space_peak_layer=space_peak_layer,
            space_peak_rank=space_peak_rank,
            digit_peak_prob=digit_peak_prob,
            digit_peak_layer=digit_peak_layer,
            digit_peak_rank=digit_peak_rank,
            entropy_at_computation=entropy_at_computation,
            entropy_at_output=entropy_at_output,
            gate_open=gate_open,
            digit_computed=digit_computed,
            confidence=confidence,
            route=route,
            model_output=model_output,
        )

    def route_and_execute(self, prompt: str) -> tuple[str, RouteDecision, ActivationProbe]:
        """
        Route prompt to model or calculator based on activations.

        Returns:
            tuple of (result, route_decision, probe_data)
        """
        probe = self.probe(prompt)

        if probe.route == RouteDecision.MODEL:
            # Use model's output
            return probe.model_output, probe.route, probe

        elif probe.route == RouteDecision.CALCULATOR:
            # Extract numbers and compute
            is_arith, _ = self._detect_arithmetic(prompt)
            if is_arith:
                result = self._compute_arithmetic(prompt)
                if result is not None:
                    return str(result), probe.route, probe
            return "?", probe.route, probe

        else:  # UNCERTAIN
            # Could combine model and calculator, for now use calculator for arithmetic
            is_arith, _ = self._detect_arithmetic(prompt)
            if is_arith:
                result = self._compute_arithmetic(prompt)
                if result is not None:
                    return str(result), probe.route, probe
            return probe.model_output, probe.route, probe

    def _compute_arithmetic(self, prompt: str) -> int | None:
        """Compute arithmetic from prompt."""
        patterns = [
            (r"(\d+)\s*\+\s*(\d+)", lambda a, b: a + b),
            (r"(\d+)\s*-\s*(\d+)", lambda a, b: a - b),
            (r"(\d+)\s*\*\s*(\d+)", lambda a, b: a * b),
            (r"(\d+)\s*×\s*(\d+)", lambda a, b: a * b),
            (r"(\d+)\s*/\s*(\d+)", lambda a, b: a // b if b != 0 else None),
            (r"(\d+)\s*÷\s*(\d+)", lambda a, b: a // b if b != 0 else None),
        ]

        for pattern, op in patterns:
            match = re.search(pattern, prompt)
            if match:
                a, b = int(match.group(1)), int(match.group(2))
                return op(a, b)
        return None


def demo_routing(steering: ActivationSteering):
    """Demo the routing system."""
    print("\n" + "=" * 70)
    print("ACTIVATION STEERING DEMO")
    print("=" * 70)

    test_cases = [
        # Should use model (well-formatted)
        ("156 + 287 = ", "Model should handle"),
        ("347 * 892 = ", "Model should handle"),
        ("25 * 4 = ", "Model should handle"),

        # Should route to calculator (format issues)
        ("156 + 287 =", "Might need calculator"),
        ("347 * 892 =", "Might need calculator"),

        # Hard cases (should definitely route)
        ("12345678 * 87654321 = ", "Large numbers"),
        ("What is 999999 * 888888?", "Natural language"),
    ]

    print("\nRouting decisions:")
    print("-" * 70)

    for prompt, description in test_cases:
        result, route, probe = steering.route_and_execute(prompt)

        # Compute correct answer for verification
        correct = steering._compute_arithmetic(prompt)

        status = ""
        if correct is not None:
            if str(correct).startswith(result.strip()):
                status = "✅"
            else:
                status = "❌"

        print(f"{status} [{route.value:10}] {prompt:<30} → {result:<10} "
              f"(gate:{probe.space_peak_prob:.0%}, digit:{probe.digit_peak_prob:.0%})")

    print("-" * 70)


async def main(
    model_id: str,
    prompt: str | None = None,
    threshold: float = 0.3,
    demo: bool = False,
):
    """Run activation steering."""
    steering = await ActivationSteering.from_pretrained(
        model_id,
        confidence_threshold=threshold,
    )

    if demo or prompt is None:
        demo_routing(steering)
        return

    # Single prompt analysis
    probe = steering.probe(prompt)
    print(probe.explain())

    print("\n" + "=" * 70)
    result, route, _ = steering.route_and_execute(prompt)
    print(f"FINAL RESULT: {result}")
    print(f"Routed via: {route.value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16")
    parser.add_argument("--prompt", "-p", default=None)
    parser.add_argument("--threshold", "-t", type=float, default=0.3)
    parser.add_argument("--demo", "-d", action="store_true")
    args = parser.parse_args()

    asyncio.run(main(args.model, args.prompt, args.threshold, args.demo))
