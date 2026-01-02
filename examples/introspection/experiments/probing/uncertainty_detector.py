#!/usr/bin/env python3
"""
Uncertainty Detector: Predict model confidence before generation.

Uses hidden state geometry to detect when a model is in "compute" vs "refusal"
subspace. This enables:
1. Routing to external tools when model is uncertain
2. Detecting format issues in prompts
3. Measuring confidence without running generation

Key Insight: At layer ~22, working prompts cluster near a "compute center"
while broken prompts cluster near a "refusal center". The distance ratio
perfectly predicts output behavior.

Usage:
    # Basic usage - check single prompt
    uv run python examples/introspection/uncertainty_detector.py \
        --prompt "100 - 37 = "

    # Check multiple prompts
    uv run python examples/introspection/uncertainty_detector.py \
        --prompts "100 - 37 = " "100 - 37 =" "50 + 25 = " "50 + 25 ="

    # Calibrate on custom examples
    uv run python examples/introspection/uncertainty_detector.py \
        --calibrate \
        --working "100 - 37 = ,50 + 25 = ,10 * 10 = " \
        --broken "100 - 37 =,50 + 25 =,10 * 10 ="

    # Interactive mode
    uv run python examples/introspection/uncertainty_detector.py --interactive
"""

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


@dataclass
class UncertaintyResult:
    """Result of uncertainty detection."""
    prompt: str
    score: float  # Positive = confident, Negative = uncertain
    prediction: str  # "CONFIDENT" or "UNCERTAIN"
    dist_to_compute: float
    dist_to_refusal: float
    actual_output: str | None = None
    correct_prediction: bool | None = None


class UncertaintyDetector:
    """Detect model uncertainty using hidden state geometry."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Any,
        model_id: str,
        detection_layer: int = 22,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_id = model_id
        self.detection_layer = detection_layer

        # Calibration centers (computed from examples)
        self.compute_center: np.ndarray | None = None
        self.refusal_center: np.ndarray | None = None
        self.direction: np.ndarray | None = None
        self.is_calibrated = False

    @classmethod
    async def from_pretrained(
        cls,
        model_id: str,
        detection_layer: int = 22,
    ) -> "UncertaintyDetector":
        """Load model and create detector."""
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

        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Detection layer: {detection_layer}")

        return cls(model, tokenizer, config, model_id, detection_layer)

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

    def get_output(self, prompt: str) -> str:
        """Get model output for prompt."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        layers = self._get_layers()
        embed = self._get_embed()
        norm = self._get_norm()
        head = self._get_head()
        scale = self._get_scale()

        h = embed(input_ids)
        if scale:
            h = h * scale

        seq_len = input_ids.shape[1]
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

        top_idx = int(mx.argmax(logits[0, -1, :]))
        return self.tokenizer.decode([top_idx])

    def calibrate(
        self,
        working_prompts: list[str],
        broken_prompts: list[str],
    ):
        """Calibrate detector using example prompts."""
        print(f"\nCalibrating on {len(working_prompts)} working + {len(broken_prompts)} broken examples...")

        # Collect hidden states
        working_hiddens = [self.get_hidden_state(p) for p in working_prompts]
        broken_hiddens = [self.get_hidden_state(p) for p in broken_prompts]

        # Compute centers
        self.compute_center = np.mean(working_hiddens, axis=0)
        self.refusal_center = np.mean(broken_hiddens, axis=0)

        # Direction from refusal to compute
        self.direction = self.compute_center - self.refusal_center
        self.direction = self.direction / np.linalg.norm(self.direction)

        self.is_calibrated = True

        # Compute separation
        separation = np.linalg.norm(self.compute_center - self.refusal_center)
        print(f"  Compute-Refusal separation: {separation:.0f}")
        print("  Calibration complete!")

    def calibrate_default(self):
        """Calibrate with default arithmetic examples."""
        working = [
            "100 - 37 = ",
            "50 + 25 = ",
            "10 * 10 = ",
            "200 - 50 = ",
            "25 * 4 = ",
        ]
        broken = [
            "100 - 37 =",
            "50 + 25 =",
            "10 * 10 =",
            "200 - 50 =",
            "25 * 4 =",
        ]
        self.calibrate(working, broken)

    def detect(self, prompt: str, verify: bool = False) -> UncertaintyResult:
        """Detect uncertainty for a prompt."""
        if not self.is_calibrated:
            raise ValueError("Detector not calibrated. Call calibrate() first.")

        h = self.get_hidden_state(prompt)

        # Compute distances
        dist_compute = float(np.linalg.norm(h - self.compute_center))
        dist_refusal = float(np.linalg.norm(h - self.refusal_center))

        # Score: positive = closer to compute (confident)
        score = dist_refusal - dist_compute
        prediction = "CONFIDENT" if score > 0 else "UNCERTAIN"

        result = UncertaintyResult(
            prompt=prompt,
            score=score,
            prediction=prediction,
            dist_to_compute=dist_compute,
            dist_to_refusal=dist_refusal,
        )

        # Optionally verify by running generation
        if verify:
            output = self.get_output(prompt)
            result.actual_output = output

            # Check if prediction was correct
            # "Confident" means we expect a digit/answer, "Uncertain" means space/?
            is_refusal = output in [" ", " ?", "?", "\n", "\n\n"]
            actual_confident = not is_refusal
            result.correct_prediction = (prediction == "CONFIDENT") == actual_confident

        return result

    def detect_batch(
        self,
        prompts: list[str],
        verify: bool = False,
    ) -> list[UncertaintyResult]:
        """Detect uncertainty for multiple prompts."""
        return [self.detect(p, verify=verify) for p in prompts]


def print_results(results: list[UncertaintyResult], show_distances: bool = False):
    """Print detection results in a table."""
    print("\n" + "=" * 80)
    print("UNCERTAINTY DETECTION RESULTS")
    print("=" * 80)

    # Header
    if show_distances:
        header = f"{'Prompt':<30} {'Score':>8} {'Prediction':<12}"
        if results[0].actual_output is not None:
            header += f" {'Output':<8} {'Correct'}"
        header += f" {'→Compute':>10} {'→Refusal':>10}"
    else:
        header = f"{'Prompt':<30} {'Score':>8} {'Prediction':<12}"
        if results[0].actual_output is not None:
            header += f" {'Output':<8} {'Correct'}"

    print(header)
    print("-" * len(header))

    for r in results:
        line = f"{r.prompt:<30} {r.score:>8.0f} {r.prediction:<12}"

        if r.actual_output is not None:
            correct_str = "✓" if r.correct_prediction else "✗"
            line += f" {repr(r.actual_output):<8} {correct_str}"

        if show_distances:
            line += f" {r.dist_to_compute:>10.0f} {r.dist_to_refusal:>10.0f}"

        print(line)

    # Summary
    if results[0].actual_output is not None:
        correct = sum(1 for r in results if r.correct_prediction)
        total = len(results)
        print("-" * len(header))
        print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")


async def interactive_mode(detector: UncertaintyDetector):
    """Interactive prompt checking."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Enter prompts to check uncertainty. Type 'quit' to exit.")
    print("Tip: Compare 'X = ' vs 'X =' to see format sensitivity.\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()
            if prompt.lower() in ["quit", "exit", "q"]:
                break
            if not prompt:
                continue

            result = detector.detect(prompt, verify=True)

            print(f"  Score: {result.score:>8.0f}")
            print(f"  Prediction: {result.prediction}")
            print(f"  Output: {repr(result.actual_output)}")
            correct = "✓" if result.correct_prediction else "✗"
            print(f"  Correct: {correct}")
            print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"  Error: {e}")


async def main(
    model_id: str,
    prompts: list[str] | None = None,
    working: list[str] | None = None,
    broken: list[str] | None = None,
    interactive: bool = False,
    verify: bool = True,
    layer: int = 22,
):
    """Run uncertainty detection."""
    detector = await UncertaintyDetector.from_pretrained(model_id, detection_layer=layer)

    # Calibrate
    if working and broken:
        detector.calibrate(working, broken)
    else:
        detector.calibrate_default()

    # Interactive mode
    if interactive:
        await interactive_mode(detector)
        return

    # Single or batch detection
    if prompts:
        results = detector.detect_batch(prompts, verify=verify)
        print_results(results, show_distances=True)
    else:
        # Demo mode
        demo_prompts = [
            "100 - 37 = ",
            "100 - 37 =",
            "50 + 25 = ",
            "50 + 25 =",
            "999 - 1 = ",
            "999 - 1 =",
            "12 * 12 = ",
            "12 * 12 =",
        ]
        print("\nRunning demo with arithmetic prompts...")
        results = detector.detect_batch(demo_prompts, verify=verify)
        print_results(results, show_distances=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect model uncertainty before generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check single prompt
  %(prog)s --prompt "100 - 37 = "

  # Compare working vs broken format
  %(prog)s --prompts "100 - 37 = " "100 - 37 ="

  # Interactive mode
  %(prog)s --interactive

  # Custom calibration
  %(prog)s --working "a = ,b = " --broken "a =,b ="
""",
    )
    parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16")
    parser.add_argument("--prompt", "-p", help="Single prompt to check")
    parser.add_argument("--prompts", "-P", nargs="+", help="Multiple prompts to check")
    parser.add_argument("--working", "-w", help="Comma-separated working examples for calibration")
    parser.add_argument("--broken", "-b", help="Comma-separated broken examples for calibration")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification (faster)")
    parser.add_argument("--layer", "-l", type=int, default=22, help="Detection layer")

    args = parser.parse_args()

    # Parse prompts
    prompts = None
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts:
        prompts = args.prompts

    # Parse calibration examples
    working = [x.strip() for x in args.working.split(",")] if args.working else None
    broken = [x.strip() for x in args.broken.split(",")] if args.broken else None

    asyncio.run(main(
        args.model,
        prompts=prompts,
        working=working,
        broken=broken,
        interactive=args.interactive,
        verify=not args.no_verify,
        layer=args.layer,
    ))
