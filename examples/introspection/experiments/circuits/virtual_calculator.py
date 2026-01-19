#!/usr/bin/env python3
"""
Virtual Calculator Circuit - Arithmetic Delegation Demo

Demonstrates the key insight from Gemma alignment circuit research:

1. Model computes answer internally but suppresses it (L24 destruction)
2. For hard problems, the model doesn't actually know the answer
3. Solution: Detect when to delegate and use external compute

This is the "virtual circuit" approach:
- Introspection detects confidence at key layers
- Low confidence → delegate to calculator
- High confidence → let model answer directly

Usage:
    # Demo: model alone vs model + delegation
    uv run python examples/introspection/virtual_calculator.py \
        --prompt "127 * 89 = "

    # Test multiple problems
    uv run python examples/introspection/virtual_calculator.py \
        --demo

    # Show introspection details
    uv run python examples/introspection/virtual_calculator.py \
        --prompt "127 * 89 = " \
        --verbose
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from enum import Enum

import mlx.core as mx


class OperationType(Enum):
    """Types of arithmetic operations."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    UNKNOWN = "?"


@dataclass
class ArithmeticProblem:
    """A parsed arithmetic problem."""

    a: int
    b: int
    op: OperationType
    prompt: str

    @property
    def expected(self) -> int:
        """Compute the correct answer."""
        if self.op == OperationType.ADD:
            return self.a + self.b
        elif self.op == OperationType.SUB:
            return self.a - self.b
        elif self.op == OperationType.MUL:
            return self.a * self.b
        elif self.op == OperationType.DIV:
            return self.a // self.b
        else:
            raise ValueError(f"Unknown operation: {self.op}")

    @property
    def difficulty(self) -> str:
        """Estimate problem difficulty."""
        if self.op == OperationType.MUL:
            if self.a >= 100 or self.b >= 100:
                return "hard"
            elif self.a >= 10 and self.b >= 10:
                return "medium"
            else:
                return "easy"
        elif self.op == OperationType.ADD or self.op == OperationType.SUB:
            if self.a >= 1000 or self.b >= 1000:
                return "medium"
            else:
                return "easy"
        return "unknown"


@dataclass
class IntrospectionResult:
    """Result of introspecting the model's computation."""

    layer_probs: dict[int, float]  # Layer -> probability of correct answer
    peak_layer: int
    peak_prob: float
    final_prob: float
    destruction_layer: int | None
    confidence: str  # "high", "medium", "low"


@dataclass
class DelegationDecision:
    """Decision about whether to delegate."""

    should_delegate: bool
    reason: str
    confidence_score: float


class VirtualCalculator:
    """
    Virtual calculator circuit that combines:
    1. LLM generation (for easy problems)
    2. Introspection (to assess confidence)
    3. External compute (for hard problems)
    """

    # Layers to check for confidence
    KEY_LAYERS = [20, 22, 24, 28, 30, 32]

    # Thresholds
    HIGH_CONFIDENCE = 0.7
    DELEGATION_THRESHOLD = 0.3

    def __init__(self, model, tokenizer, model_id: str = "unknown"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self._detect_structure()

    def _detect_structure(self):
        """Detect model structure."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self._backbone = self.model.model
            self._layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            self._backbone = self.model
            self._layers = self.model.layers
        else:
            raise ValueError("Cannot detect model structure")

        self.num_layers = len(self._layers)

        # Get norm and lm_head for logit lens
        self._norm = getattr(self._backbone, "norm", None)
        self._lm_head = getattr(self.model, "lm_head", None)

    @classmethod
    def from_pretrained(cls, model_id: str) -> VirtualCalculator:
        """Load model."""
        from chuk_lazarus.introspection.ablation import AblationStudy

        study = AblationStudy.from_pretrained(model_id)
        return cls(
            model=study.adapter.model,
            tokenizer=study.adapter.tokenizer,
            model_id=model_id,
        )

    # =========================================================================
    # Parsing
    # =========================================================================

    def parse_problem(self, prompt: str) -> ArithmeticProblem | None:
        """Parse an arithmetic problem from a prompt."""
        # Pattern: "a op b = " or "a op b"
        pattern = r"(\d+)\s*([+\-*/×÷])\s*(\d+)\s*=?\s*$"
        match = re.search(pattern, prompt.strip())

        if not match:
            return None

        a = int(match.group(1))
        op_str = match.group(2)
        b = int(match.group(3))

        op_map = {
            "+": OperationType.ADD,
            "-": OperationType.SUB,
            "*": OperationType.MUL,
            "×": OperationType.MUL,
            "/": OperationType.DIV,
            "÷": OperationType.DIV,
        }

        op = op_map.get(op_str, OperationType.UNKNOWN)

        return ArithmeticProblem(a=a, b=b, op=op, prompt=prompt)

    # =========================================================================
    # Introspection
    # =========================================================================

    def introspect(self, prompt: str, expected: int) -> IntrospectionResult:
        """
        Introspect the model's computation to assess confidence.

        Returns layer-by-layer probability of the correct answer.
        """
        from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks

        hooks = ModelHooks(self.model)
        hooks.configure(
            CaptureConfig(
                layers=self.KEY_LAYERS,
                capture_hidden_states=True,
            )
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)
        hooks.forward(input_ids)

        # Get first digit of expected answer
        first_digit = str(expected)[0]
        first_digit_ids = self.tokenizer.encode(first_digit, add_special_tokens=False)
        if not first_digit_ids:
            first_digit_id = None
        else:
            first_digit_id = first_digit_ids[-1]

        layer_probs = {}
        for layer in self.KEY_LAYERS:
            if layer not in hooks.state.hidden_states:
                continue

            h = hooks.state.hidden_states[layer]
            if h.ndim == 3:
                h = h[0, -1, :]
            else:
                h = h[-1, :]

            # Project to logits
            if self._norm is not None:
                h = self._norm(h)
            if self._lm_head is not None:
                logits = self._lm_head(h)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            else:
                continue

            probs = mx.softmax(logits, axis=-1)

            if first_digit_id is not None:
                layer_probs[layer] = float(probs[first_digit_id])
            else:
                layer_probs[layer] = 0.0

        # Analyze
        if layer_probs:
            peak_layer = max(layer_probs, key=layer_probs.get)
            peak_prob = layer_probs[peak_layer]
            final_prob = layer_probs.get(max(layer_probs.keys()), 0.0)
        else:
            peak_layer = 0
            peak_prob = 0.0
            final_prob = 0.0

        # Detect destruction
        destruction_layer = None
        prev_prob = 0.0
        for layer in sorted(layer_probs.keys()):
            prob = layer_probs[layer]
            if prev_prob > 0.3 and prob < 0.1:
                destruction_layer = layer
                break
            prev_prob = prob

        # Assess confidence
        if final_prob >= self.HIGH_CONFIDENCE:
            confidence = "high"
        elif final_prob >= self.DELEGATION_THRESHOLD:
            confidence = "medium"
        else:
            confidence = "low"

        return IntrospectionResult(
            layer_probs=layer_probs,
            peak_layer=peak_layer,
            peak_prob=peak_prob,
            final_prob=final_prob,
            destruction_layer=destruction_layer,
            confidence=confidence,
        )

    # =========================================================================
    # Delegation Decision
    # =========================================================================

    def should_delegate(
        self,
        problem: ArithmeticProblem,
        introspection: IntrospectionResult,
    ) -> DelegationDecision:
        """
        Decide whether to delegate to external calculator.

        Uses combination of:
        - Problem difficulty heuristic
        - Introspection confidence
        """
        reasons = []

        # Check difficulty
        if problem.difficulty == "hard":
            reasons.append(f"hard {problem.op.value} problem")

        # Check introspection
        if introspection.confidence == "low":
            reasons.append(f"low confidence ({introspection.final_prob:.1%})")

        if introspection.destruction_layer is not None:
            reasons.append(f"destruction at L{introspection.destruction_layer}")

        # Decision
        should = len(reasons) > 0

        return DelegationDecision(
            should_delegate=should,
            reason="; ".join(reasons) if reasons else "high confidence",
            confidence_score=introspection.final_prob,
        )

    # =========================================================================
    # Generation
    # =========================================================================

    def generate_direct(self, prompt: str, max_tokens: int = 10) -> str:
        """Generate answer directly from model."""
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

            # Stop on non-digit
            token_str = self.tokenizer.decode([next_id])
            if "\n" in token_str:
                break
            if generated and not any(c.isdigit() for c in token_str):
                break

        return self.tokenizer.decode(generated).strip()

    def compute_external(self, problem: ArithmeticProblem) -> int:
        """
        External calculator - the "virtual circuit".

        In a full implementation, this could be:
        - WASM module
        - Tool call to chuk-tool-processor
        - GPU kernel
        """
        return problem.expected

    # =========================================================================
    # Main Interface
    # =========================================================================

    def solve(
        self,
        prompt: str,
        verbose: bool = False,
    ) -> tuple[str, bool, dict]:
        """
        Solve an arithmetic problem with optional delegation.

        Returns:
            (answer, delegated, metadata)
        """
        # Parse
        problem = self.parse_problem(prompt)
        if problem is None:
            # Not arithmetic - just generate
            answer = self.generate_direct(prompt)
            return answer, False, {"reason": "not arithmetic"}

        # Introspect
        introspection = self.introspect(prompt, problem.expected)

        # Decide
        decision = self.should_delegate(problem, introspection)

        metadata = {
            "problem": f"{problem.a} {problem.op.value} {problem.b}",
            "expected": problem.expected,
            "difficulty": problem.difficulty,
            "introspection": {
                "peak_layer": introspection.peak_layer,
                "peak_prob": introspection.peak_prob,
                "final_prob": introspection.final_prob,
                "destruction_layer": introspection.destruction_layer,
                "confidence": introspection.confidence,
            },
            "decision": {
                "delegate": decision.should_delegate,
                "reason": decision.reason,
            },
        }

        if verbose:
            print("\n--- Introspection ---")
            for layer, prob in sorted(introspection.layer_probs.items()):
                bar = "█" * int(prob * 20)
                marker = " ← destruction" if layer == introspection.destruction_layer else ""
                print(f"  L{layer:2d}: {prob:5.1%} {bar}{marker}")
            print(f"\n  Confidence: {introspection.confidence}")
            print(f"  Decision: {'DELEGATE' if decision.should_delegate else 'DIRECT'}")
            print(f"  Reason: {decision.reason}")

        # Execute
        if decision.should_delegate:
            answer = str(self.compute_external(problem))
            delegated = True
        else:
            answer = self.generate_direct(prompt)
            delegated = False

        return answer, delegated, metadata

    def compare(self, prompt: str) -> None:
        """
        Compare model-only vs delegation for a problem.

        This is the key demo for the video.
        """
        problem = self.parse_problem(prompt)
        if problem is None:
            print("Could not parse arithmetic problem")
            return

        print("\n" + "=" * 70)
        print("VIRTUAL CALCULATOR DEMO")
        print("=" * 70)
        print(f"Prompt: {prompt}")
        print(f"Expected: {problem.expected}")
        print("-" * 70)

        # Model alone
        print("\n1. MODEL ALONE:")
        direct_answer = self.generate_direct(prompt)
        try:
            direct_num = int(re.search(r"\d+", direct_answer).group())
        except:
            direct_num = None

        if direct_num == problem.expected:
            print(f"   Output: {direct_answer} ✓")
        else:
            print(f"   Output: {direct_answer} ✗ (expected {problem.expected})")

        # Introspection
        print("\n2. INTROSPECTION (first digit confidence):")
        introspection = self.introspect(prompt, problem.expected)
        first_digit = str(problem.expected)[0]
        print(f"   Tracking: '{first_digit}' (first digit of {problem.expected})")
        print()
        for layer, prob in sorted(introspection.layer_probs.items()):
            bar = "█" * int(prob * 20)
            marker = ""
            if layer == introspection.destruction_layer:
                marker = " ← destruction"
            elif layer == introspection.peak_layer:
                marker = " ← peak"
            print(f"   L{layer:2d}: {prob:5.1%} {bar}{marker}")

        print(
            f"\n   The model is {introspection.final_prob:.0%} confident it starts with '{first_digit}'."
        )
        print("   It's right. But it has no idea what comes next.")

        # Decision
        decision = self.should_delegate(problem, introspection)
        print("\n3. DELEGATION DECISION:")
        print(f"   {'→ DELEGATE to calculator' if decision.should_delegate else '→ Trust model'}")
        print(f"   Reason: {decision.reason}")

        # With delegation
        print("\n4. MODEL + DELEGATION:")
        answer, delegated, _ = self.solve(prompt)
        try:
            answer_num = int(re.search(r"\d+", answer).group())
        except:
            answer_num = None

        source = "calculator" if delegated else "model"
        if answer_num == problem.expected:
            print(f"   Output: {answer} ✓ (from {source})")
        else:
            print(f"   Output: {answer} ✗ (from {source})")

        print("=" * 70)


def demo_problems():
    """Demo with a range of problems."""
    problems = [
        # Easy - model can handle
        ("6 * 7 = ", "trivial multiplication"),
        ("156 + 287 = ", "3-digit addition"),
        # Hard - model fails
        ("127 * 89 = ", "3-digit × 2-digit multiplication"),
        ("456 * 78 = ", "3-digit × 2-digit multiplication"),
        ("23 * 17 = ", "2-digit × 2-digit multiplication"),
    ]

    print("\n" + "=" * 70)
    print("ARITHMETIC DELEGATION DEMO")
    print("=" * 70)
    print(
        "\n{:<20} {:>10} {:>10} {:>10} {:>10}".format(
            "Problem", "Expected", "Model", "Delegated", "Result"
        )
    )
    print("-" * 70)

    calc = None

    for prompt, description in problems:
        if calc is None:
            print("\nLoading model...")
            calc = VirtualCalculator.from_pretrained("mlx-community/gemma-3-4b-it-bf16")

        problem = calc.parse_problem(prompt)

        # Model alone
        direct = calc.generate_direct(prompt)
        try:
            direct_num = int(re.search(r"\d+", direct).group())
        except:
            direct_num = None

        # With delegation
        answer, delegated, _ = calc.solve(prompt)
        try:
            answer_num = int(re.search(r"\d+", answer).group())
        except:
            answer_num = None

        model_status = "✓" if direct_num == problem.expected else "✗"
        final_status = "✓" if answer_num == problem.expected else "✗"
        delegate_str = "Yes" if delegated else "No"

        print(
            f"{prompt:<20} {problem.expected:>10} {direct_num:>10} {model_status} {delegate_str:>10} {answer_num:>10} {final_status}"
        )

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Virtual Calculator Circuit Demo",
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
        default=None,
        help="Arithmetic prompt to solve",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with multiple problems",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show introspection details",
    )

    args = parser.parse_args()

    if args.demo:
        demo_problems()
    elif args.prompt:
        print(f"\nLoading model: {args.model}")
        calc = VirtualCalculator.from_pretrained(args.model)
        print(f"Model loaded: {calc.num_layers} layers")

        calc.compare(args.prompt)
    else:
        # Default demo
        print(f"\nLoading model: {args.model}")
        calc = VirtualCalculator.from_pretrained(args.model)
        print(f"Model loaded: {calc.num_layers} layers")

        calc.compare("127 * 89 = ")


if __name__ == "__main__":
    main()
