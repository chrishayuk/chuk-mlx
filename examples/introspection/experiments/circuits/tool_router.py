#!/usr/bin/env python3
"""
Tool Router: Route to external tools when model is uncertain.

This demonstrates using the uncertainty detector to decide when to use
external tools (like a calculator) vs trusting the model's output.

The key insight: We can detect uncertainty BEFORE generation, avoiding
wasted compute on prompts the model will refuse.

Usage:
    uv run python examples/introspection/tool_router.py \
        --prompts "100 - 37 = " "100 - 37 =" "What is 999 * 888?"
"""

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    prompt: str
    route: str  # "MODEL" or "CALCULATOR"
    uncertainty_score: float
    result: str
    source: str  # "model" or "calculator"


def calculate(expr: str) -> str | None:
    """Simple calculator for arithmetic expressions."""
    # Extract arithmetic expression
    patterns = [
        r"(\d+)\s*\+\s*(\d+)",
        r"(\d+)\s*-\s*(\d+)",
        r"(\d+)\s*\*\s*(\d+)",
        r"(\d+)\s*[x×]\s*(\d+)",
        r"(\d+)\s*/\s*(\d+)",
        r"(\d+)\s*÷\s*(\d+)",
    ]

    ops = [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a * b,
        lambda a, b: a // b if b != 0 else None,
        lambda a, b: a // b if b != 0 else None,
    ]

    for pattern, op in zip(patterns, ops):
        match = re.search(pattern, expr)
        if match:
            a, b = int(match.group(1)), int(match.group(2))
            result = op(a, b)
            if result is not None:
                return str(result)

    return None


class ToolRouter:
    """Route between model and external tools based on uncertainty."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Any,
        model_id: str,
        detection_layer: int = 22,
        threshold: float = 0,  # Score below this → use tool
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_id = model_id
        self.detection_layer = detection_layer
        self.threshold = threshold

        # Calibration
        self.compute_center: np.ndarray | None = None
        self.refusal_center: np.ndarray | None = None
        self.is_calibrated = False

    @classmethod
    async def from_pretrained(
        cls,
        model_id: str,
        detection_layer: int = 22,
        threshold: float = 0,
    ) -> "ToolRouter":
        """Load model and create router."""
        print(f"Loading model: {model_id}")

        result = HFLoader.download(model_id)
        model_path = result.model_path

        with open(model_path / "config.json") as f:
            config_data = json.load(f)

        family_type = detect_model_family(config_data)
        if family_type is None:
            raise ValueError(f"Unsupported model: {model_id}")

        family_info = get_family_info(family_type)
        config = family_info.config_class.from_hf_config(config_data)
        model = family_info.model_class(config)

        HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
        tokenizer = HFLoader.load_tokenizer(model_path)

        return cls(model, tokenizer, config, model_id, detection_layer, threshold)

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

    def get_hidden_state(self, prompt: str) -> np.ndarray:
        """Get hidden state at detection layer."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        layers = self._get_layers()
        embed = self._get_embed()
        scale = self._get_scale()

        h = embed(input_ids)
        if scale:
            h = h * scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

        for idx, lyr in enumerate(layers):
            try:
                out = lyr(h, mask=mask)
            except TypeError:
                out = lyr(h)
            h = out.hidden_states if hasattr(out, "hidden_states") else (out[0] if isinstance(out, tuple) else out)

            if idx == self.detection_layer:
                return np.array(h[0, -1, :].tolist())

        return np.array(h[0, -1, :].tolist())

    def generate(self, prompt: str, max_tokens: int = 10) -> str:
        """Generate text from model."""
        input_ids = self.tokenizer.encode(prompt)

        layers = self._get_layers()
        embed = self._get_embed()
        norm = self._get_norm()
        head = self._get_head()
        scale = self._get_scale()

        generated = []
        for _ in range(max_tokens):
            x = mx.array(input_ids + generated)[None, :]
            h = embed(x)
            if scale:
                h = h * scale

            seq_len = x.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

            for layer in layers:
                try:
                    out = layer(h, mask=mask)
                except TypeError:
                    out = layer(h)
                h = out.hidden_states if hasattr(out, "hidden_states") else (out[0] if isinstance(out, tuple) else out)

            h_n = norm(h) if norm else h
            logits = head(h_n)
            if hasattr(logits, "logits"):
                logits = logits.logits

            next_token = int(mx.argmax(logits[0, -1, :]))
            generated.append(next_token)

            # Stop on newline or EOS
            decoded = self.tokenizer.decode([next_token])
            if decoded in ["\n", "</s>", "<eos>"]:
                break

        return self.tokenizer.decode(generated).strip()

    def calibrate_default(self):
        """Calibrate with default arithmetic examples."""
        working = ["100 - 37 = ", "50 + 25 = ", "10 * 10 = "]
        broken = ["100 - 37 =", "50 + 25 =", "10 * 10 ="]

        working_hiddens = [self.get_hidden_state(p) for p in working]
        broken_hiddens = [self.get_hidden_state(p) for p in broken]

        self.compute_center = np.mean(working_hiddens, axis=0)
        self.refusal_center = np.mean(broken_hiddens, axis=0)
        self.is_calibrated = True

        print("Calibrated on arithmetic examples")

    def get_uncertainty_score(self, prompt: str) -> float:
        """Get uncertainty score for prompt."""
        if not self.is_calibrated:
            raise ValueError("Router not calibrated")

        h = self.get_hidden_state(prompt)
        dist_compute = float(np.linalg.norm(h - self.compute_center))
        dist_refusal = float(np.linalg.norm(h - self.refusal_center))

        return dist_refusal - dist_compute

    def route(self, prompt: str) -> RoutingDecision:
        """Route prompt to model or calculator."""
        score = self.get_uncertainty_score(prompt)

        if score > self.threshold:
            # Model is confident, use it
            result = self.generate(prompt)
            return RoutingDecision(
                prompt=prompt,
                route="MODEL",
                uncertainty_score=score,
                result=result,
                source="model",
            )
        else:
            # Model is uncertain, try calculator
            calc_result = calculate(prompt)
            if calc_result is not None:
                return RoutingDecision(
                    prompt=prompt,
                    route="CALCULATOR",
                    uncertainty_score=score,
                    result=calc_result,
                    source="calculator",
                )
            else:
                # Can't calculate, fall back to model
                result = self.generate(prompt)
                return RoutingDecision(
                    prompt=prompt,
                    route="MODEL (fallback)",
                    uncertainty_score=score,
                    result=result,
                    source="model",
                )


async def main(model_id: str, prompts: list[str], threshold: float = 0):
    """Demo tool routing."""
    router = await ToolRouter.from_pretrained(model_id, threshold=threshold)
    router.calibrate_default()

    print("\n" + "=" * 80)
    print("TOOL ROUTING DEMO")
    print("=" * 80)
    print(f"Threshold: {threshold} (score below this → use calculator)")
    print()

    print(f"{'Prompt':<30} {'Score':>8} {'Route':<15} {'Result':<15} {'Source'}")
    print("-" * 85)

    for prompt in prompts:
        decision = router.route(prompt)
        print(
            f"{prompt:<30} {decision.uncertainty_score:>8.0f} "
            f"{decision.route:<15} {decision.result:<15} {decision.source}"
        )

    print()
    print("Key insight: We detect uncertainty BEFORE generation,")
    print("avoiding wasted compute on prompts the model would refuse.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Route between model and tools")
    parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16")
    parser.add_argument(
        "--prompts", "-p", nargs="+",
        default=[
            "100 - 37 = ",
            "100 - 37 =",
            "50 + 25 = ",
            "50 + 25 =",
            "What is 999 * 888?",
            "Calculate 144 / 12",
        ]
    )
    parser.add_argument("--threshold", "-t", type=float, default=0)

    args = parser.parse_args()
    asyncio.run(main(args.model, args.prompts, args.threshold))
