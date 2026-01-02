#!/usr/bin/env python3
"""
Arithmetic Readout Head - Extract Answers from Hidden States

The model internally computes correct answers but fails to output them
due to autoregressive decoding errors. This script:

1. Trains a tiny MLP to read digits directly from hidden states
2. Uses beam search with calculator verification
3. Demonstrates that "the answer is in the latent"

Usage:
    # Beam search with verification (no training needed)
    uv run python examples/introspection/arithmetic_readout.py \
        --model mlx-community/gemma-3-4b-it-bf16 \
        --prompt "127 * 89 = " \
        --method beam_verify

    # Train a readout head on arithmetic examples
    uv run python examples/introspection/arithmetic_readout.py \
        --model mlx-community/gemma-3-4b-it-bf16 \
        --method train_head \
        --layer 28

    # Use trained head to extract answer
    uv run python examples/introspection/arithmetic_readout.py \
        --model mlx-community/gemma-3-4b-it-bf16 \
        --prompt "127 * 89 = " \
        --method use_head \
        --layer 28
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


@dataclass
class ArithmeticExample:
    """An arithmetic problem with solution."""
    prompt: str
    answer: int
    operation: str  # '+', '-', '*', '/'


def generate_arithmetic_dataset(
    n_examples: int = 500,
    max_value: int = 999,
    operations: list[str] = ["+", "-", "*"],
) -> list[ArithmeticExample]:
    """Generate arithmetic training examples."""
    examples = []

    for _ in range(n_examples):
        op = random.choice(operations)

        if op == "+":
            a = random.randint(1, max_value)
            b = random.randint(1, max_value)
            answer = a + b
        elif op == "-":
            a = random.randint(1, max_value)
            b = random.randint(1, a)  # Ensure positive result
            answer = a - b
        elif op == "*":
            a = random.randint(1, min(99, max_value))
            b = random.randint(1, min(99, max_value))
            answer = a * b
        else:  # division
            b = random.randint(1, min(99, max_value))
            answer = random.randint(1, min(99, max_value))
            a = b * answer  # Ensure clean division

        prompt = f"{a} {op} {b} = "
        examples.append(ArithmeticExample(
            prompt=prompt,
            answer=answer,
            operation=op,
        ))

    return examples


class DigitReadoutHead(nn.Module):
    """
    MLP that reads digits from a hidden state.

    Input: hidden state [hidden_size]
    Output: digit logits [max_digits, 10] + length logit [max_digits]
    """

    def __init__(
        self,
        hidden_size: int,
        max_digits: int = 8,
        intermediate_size: int = 512,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_digits = max_digits

        # Project hidden state
        self.proj = nn.Linear(hidden_size, intermediate_size)

        # Digit predictors (one per position)
        self.digit_heads = [
            nn.Linear(intermediate_size, 10)
            for _ in range(max_digits)
        ]

        # Length predictor (how many digits)
        self.length_head = nn.Linear(intermediate_size, max_digits + 1)

    def __call__(self, h: mx.array) -> tuple[mx.array, mx.array]:
        """
        Predict digits from hidden state.

        Args:
            h: [batch, hidden_size] or [hidden_size]

        Returns:
            digit_logits: [batch, max_digits, 10]
            length_logits: [batch, max_digits + 1]
        """
        if h.ndim == 1:
            h = h[None, :]  # Add batch dim

        # Project
        x = nn.gelu(self.proj(h))  # [batch, intermediate]

        # Predict each digit
        digit_logits = []
        for head in self.digit_heads:
            digit_logits.append(head(x))  # [batch, 10]

        digit_logits = mx.stack(digit_logits, axis=1)  # [batch, max_digits, 10]

        # Predict length
        length_logits = self.length_head(x)  # [batch, max_digits + 1]

        return digit_logits, length_logits

    def decode(self, h: mx.array) -> int:
        """Decode hidden state to integer."""
        digit_logits, length_logits = self(h)

        # Get predicted length
        length = int(mx.argmax(length_logits[0]))
        if length == 0:
            return 0

        # Get predicted digits
        digits = []
        for i in range(min(length, self.max_digits)):
            digit = int(mx.argmax(digit_logits[0, i]))
            digits.append(str(digit))

        if not digits:
            return 0

        return int("".join(digits))


class ArithmeticReadout:
    """
    Extract arithmetic answers from model hidden states.
    """

    def __init__(self, model, tokenizer, model_id: str = "unknown"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self._detect_structure()

        self.readout_head: DigitReadoutHead | None = None

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

        # Get hidden size
        layer = self._layers[0]
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
            self.hidden_size = layer.mlp.down_proj.weight.shape[0]
        elif hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            self.hidden_size = layer.self_attn.o_proj.weight.shape[0]
        else:
            self.hidden_size = 2560  # Fallback for Gemma-3-4b

    @classmethod
    def from_pretrained(cls, model_id: str) -> ArithmeticReadout:
        """Load model."""
        from chuk_lazarus.introspection.ablation import AblationStudy

        study = AblationStudy.from_pretrained(model_id)
        return cls(
            model=study.adapter.model,
            tokenizer=study.adapter.tokenizer,
            model_id=model_id,
        )

    def get_hidden_state(self, prompt: str, layer: int) -> mx.array:
        """Get hidden state at layer for last position."""
        from chuk_lazarus.introspection.hooks import ModelHooks, CaptureConfig

        hooks = ModelHooks(self.model)
        hooks.configure(CaptureConfig(
            layers=[layer],
            capture_hidden_states=True,
        ))

        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)
        hooks.forward(input_ids)

        h = hooks.state.hidden_states[layer]
        if h.ndim == 3:
            h = h[0, -1, :]  # Last position
        else:
            h = h[-1, :]

        return h

    # =========================================================================
    # Method 1: Beam Search with Verification
    # =========================================================================

    def beam_search_verified(
        self,
        prompt: str,
        beam_width: int = 10,
        max_digits: int = 8,
    ) -> tuple[int | None, list[tuple[str, float]]]:
        """
        Generate candidates with beam search, verify with calculator.

        Returns:
            (verified_answer, all_candidates)
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        current_ids = mx.array(input_ids)

        # Get digit token IDs
        digit_tokens = {}
        for d in "0123456789":
            ids = self.tokenizer.encode(d, add_special_tokens=False)
            if ids:
                digit_tokens[ids[-1]] = d

        # Initial beam: [(token_ids, log_prob)]
        beams = [([], 0.0)]

        for pos in range(max_digits):
            new_beams = []

            for tokens, log_prob in beams:
                # Build current sequence
                if tokens:
                    seq = mx.concatenate([
                        current_ids,
                        mx.array([[self.tokenizer.encode(t, add_special_tokens=False)[-1] for t in tokens]])
                    ], axis=1)
                else:
                    seq = current_ids

                # Get next token probs
                outputs = self.model(seq)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs

                last_logits = logits[0, -1, :]
                probs = mx.softmax(last_logits, axis=-1)

                # Get top digit probabilities
                for tid, digit in digit_tokens.items():
                    prob = float(probs[tid])
                    if prob > 0.001:  # Threshold
                        new_beams.append((
                            tokens + [digit],
                            log_prob + mx.log(probs[tid])
                        ))

                # Also consider stopping (newline/space)
                if tokens:  # At least one digit
                    new_beams.append((tokens, log_prob - 1.0))  # Penalty for stopping

            # Keep top beams
            new_beams.sort(key=lambda x: -x[1])
            beams = new_beams[:beam_width]

            if not beams:
                break

        # Convert to candidates
        candidates = []
        for tokens, log_prob in beams:
            if tokens:
                num_str = "".join(tokens)
                try:
                    num = int(num_str)
                    candidates.append((num_str, float(mx.exp(mx.array(log_prob)))))
                except ValueError:
                    pass

        # Parse the prompt to get expected answer
        expected = self._compute_expected(prompt)

        # Check if any candidate matches
        verified = None
        for num_str, prob in candidates:
            try:
                if int(num_str) == expected:
                    verified = int(num_str)
                    break
            except ValueError:
                pass

        return verified, candidates

    def _compute_expected(self, prompt: str) -> int | None:
        """Parse prompt and compute expected answer."""
        import re

        # Try to parse "a op b = " format
        match = re.match(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=", prompt.strip())
        if match:
            a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
            if op == "+":
                return a + b
            elif op == "-":
                return a - b
            elif op == "*":
                return a * b
            elif op == "/":
                return a // b

        return None

    # =========================================================================
    # Method 2: Trained Readout Head
    # =========================================================================

    def train_readout_head(
        self,
        layer: int,
        n_examples: int = 500,
        epochs: int = 50,
        lr: float = 1e-3,
        save_path: str | None = None,
    ) -> DigitReadoutHead:
        """
        Train a readout head to extract digits from hidden states.
        """
        print(f"\nGenerating {n_examples} training examples...")
        examples = generate_arithmetic_dataset(n_examples)

        print(f"Collecting hidden states at layer {layer}...")
        hidden_states = []
        targets = []  # (digits, length)

        for i, ex in enumerate(examples):
            if i % 50 == 0:
                print(f"  {i}/{len(examples)}")

            h = self.get_hidden_state(ex.prompt, layer)
            hidden_states.append(h)

            # Convert answer to digits
            answer_str = str(ex.answer)
            digits = [int(d) for d in answer_str]
            length = len(digits)

            # Pad to max_digits
            while len(digits) < 8:
                digits.append(0)

            targets.append((digits[:8], length))

        # Convert to arrays
        H = mx.stack(hidden_states)  # [n, hidden_size]
        digit_targets = mx.array([t[0] for t in targets])  # [n, 8]
        length_targets = mx.array([t[1] for t in targets])  # [n]

        print(f"\nTraining readout head...")
        print(f"  Hidden states: {H.shape}")
        print(f"  Digit targets: {digit_targets.shape}")

        # Create model
        head = DigitReadoutHead(
            hidden_size=self.hidden_size,
            max_digits=8,
            intermediate_size=512,
        )

        # Training
        optimizer = optim.Adam(learning_rate=lr)

        def loss_fn(head, H, digit_targets, length_targets):
            digit_logits, length_logits = head(H)

            # Digit loss (cross entropy per position)
            digit_loss = mx.array(0.0)
            for i in range(8):
                logits_i = digit_logits[:, i, :]  # [batch, 10]
                targets_i = digit_targets[:, i]  # [batch]
                # Only count loss for positions within actual length
                mask = (i < length_targets).astype(mx.float32)
                ce = nn.losses.cross_entropy(logits_i, targets_i, reduction="none")
                digit_loss = digit_loss + mx.sum(ce * mask) / mx.maximum(mx.sum(mask), mx.array(1.0))

            # Length loss
            length_loss = mx.mean(nn.losses.cross_entropy(length_logits, length_targets, reduction="none"))

            return digit_loss + length_loss

        loss_and_grad = nn.value_and_grad(head, loss_fn)

        for epoch in range(epochs):
            loss, grads = loss_and_grad(head, H, digit_targets, length_targets)
            optimizer.update(head, grads)
            mx.eval(head.parameters())

            if epoch % 10 == 0:
                # Test accuracy
                digit_logits, length_logits = head(H)
                pred_lengths = mx.argmax(length_logits, axis=-1)
                length_acc = float(mx.mean(pred_lengths == length_targets))

                correct = 0
                for i in range(len(examples)):
                    pred = head.decode(H[i])
                    if pred == examples[i].answer:
                        correct += 1

                print(f"  Epoch {epoch}: loss={float(loss):.4f}, length_acc={length_acc:.1%}, answer_acc={correct/len(examples):.1%}")

        self.readout_head = head

        # Save if requested
        if save_path:
            self._save_head(head, save_path, layer)

        return head

    def _save_head(self, head: DigitReadoutHead, path: str, layer: int):
        """Save readout head."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        # Save weights
        weights = {}
        weights["proj.weight"] = head.proj.weight
        weights["proj.bias"] = head.proj.bias
        weights["length_head.weight"] = head.length_head.weight
        weights["length_head.bias"] = head.length_head.bias
        for i, dh in enumerate(head.digit_heads):
            weights[f"digit_heads.{i}.weight"] = dh.weight
            weights[f"digit_heads.{i}.bias"] = dh.bias

        mx.savez(str(p / "weights.npz"), **weights)

        # Save config
        config = {
            "hidden_size": head.hidden_size,
            "max_digits": head.max_digits,
            "layer": layer,
            "model_id": self.model_id,
        }
        with open(p / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved readout head to {path}")

    def load_head(self, path: str) -> tuple[DigitReadoutHead, int]:
        """Load readout head."""
        p = Path(path)

        with open(p / "config.json") as f:
            config = json.load(f)

        head = DigitReadoutHead(
            hidden_size=config["hidden_size"],
            max_digits=config["max_digits"],
        )

        weights = dict(mx.load(str(p / "weights.npz")))
        head.proj.weight = weights["proj.weight"]
        head.proj.bias = weights["proj.bias"]
        head.length_head.weight = weights["length_head.weight"]
        head.length_head.bias = weights["length_head.bias"]
        for i in range(head.max_digits):
            head.digit_heads[i].weight = weights[f"digit_heads.{i}.weight"]
            head.digit_heads[i].bias = weights[f"digit_heads.{i}.bias"]

        self.readout_head = head
        return head, config["layer"]

    def extract_with_head(self, prompt: str, layer: int) -> int:
        """Extract answer using trained readout head."""
        if self.readout_head is None:
            raise ValueError("No readout head loaded. Train or load one first.")

        h = self.get_hidden_state(prompt, layer)
        return self.readout_head.decode(h)

    # =========================================================================
    # Comparison / Demo
    # =========================================================================

    def compare_methods(
        self,
        prompt: str,
        layer: int = 28,
        beam_width: int = 10,
    ):
        """Compare all extraction methods."""
        expected = self._compute_expected(prompt)

        print("\n" + "=" * 70)
        print("ARITHMETIC EXTRACTION COMPARISON")
        print("=" * 70)
        print(f"Prompt: {prompt!r}")
        print(f"Expected: {expected}")
        print("-" * 70)

        # Standard AR
        print("\n1. Standard Autoregressive:")
        ar_result = self._generate_ar(prompt, max_tokens=10)
        ar_num = self._extract_number(ar_result)
        status = "✓" if ar_num == expected else "✗"
        print(f"   Output: {ar_result!r}")
        print(f"   Extracted: {ar_num} {status}")

        # Beam search with verification
        print("\n2. Beam Search + Verification:")
        verified, candidates = self.beam_search_verified(prompt, beam_width)
        if verified is not None:
            print(f"   Verified: {verified} ✓")
        else:
            print(f"   No verified answer found")
        print(f"   Top candidates: {candidates[:5]}")

        # Readout head (if available)
        if self.readout_head is not None:
            print(f"\n3. Readout Head (layer {layer}):")
            head_result = self.extract_with_head(prompt, layer)
            status = "✓" if head_result == expected else "✗"
            print(f"   Extracted: {head_result} {status}")

        print("=" * 70)

    def _generate_ar(self, prompt: str, max_tokens: int = 10) -> str:
        """Standard autoregressive generation."""
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

            token_str = self.tokenizer.decode([next_id])
            if "\n" in token_str or not any(c.isdigit() for c in token_str):
                break

        return self.tokenizer.decode(generated)

    def _extract_number(self, text: str) -> int | None:
        """Extract first number from text."""
        import re
        match = re.search(r"\d+", text)
        if match:
            return int(match.group())
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract arithmetic answers from hidden states",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m",
        default="mlx-community/gemma-3-4b-it-bf16",
        help="Model ID",
    )
    parser.add_argument(
        "--prompt", "-p",
        default="127 * 89 = ",
        help="Arithmetic prompt",
    )
    parser.add_argument(
        "--layer", "-l",
        type=int,
        default=28,
        help="Layer for readout head",
    )
    parser.add_argument(
        "--method",
        choices=["beam_verify", "train_head", "use_head", "compare"],
        default="compare",
        help="Extraction method",
    )
    parser.add_argument(
        "--head-path",
        default="arithmetic_head",
        help="Path to save/load readout head",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=500,
        help="Number of training examples",
    )

    args = parser.parse_args()

    print(f"\nLoading model: {args.model}")
    extractor = ArithmeticReadout.from_pretrained(args.model)
    print(f"Model loaded: {extractor.num_layers} layers, hidden_size={extractor.hidden_size}")

    if args.method == "beam_verify":
        print(f"\nBeam search with verification on: {args.prompt}")
        verified, candidates = extractor.beam_search_verified(args.prompt)
        expected = extractor._compute_expected(args.prompt)
        print(f"Expected: {expected}")
        if verified is not None:
            print(f"Verified: {verified} ✓")
        else:
            print("No verified match found")
        print(f"Candidates: {candidates[:10]}")

    elif args.method == "train_head":
        extractor.train_readout_head(
            layer=args.layer,
            n_examples=args.n_examples,
            save_path=args.head_path,
        )

    elif args.method == "use_head":
        print(f"\nLoading readout head from {args.head_path}")
        head, layer = extractor.load_head(args.head_path)
        print(f"Loaded head for layer {layer}")

        result = extractor.extract_with_head(args.prompt, layer)
        expected = extractor._compute_expected(args.prompt)
        status = "✓" if result == expected else "✗"
        print(f"\nPrompt: {args.prompt}")
        print(f"Expected: {expected}")
        print(f"Extracted: {result} {status}")

    elif args.method == "compare":
        # Try to load head if it exists
        head_path = Path(args.head_path)
        if (head_path / "config.json").exists():
            print(f"Loading existing readout head from {args.head_path}")
            extractor.load_head(args.head_path)

        extractor.compare_methods(args.prompt, args.layer)


if __name__ == "__main__":
    main()
