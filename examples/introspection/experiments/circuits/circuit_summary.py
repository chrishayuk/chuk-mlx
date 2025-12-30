#!/usr/bin/env python3
"""
Circuit Summary: One-command analysis of how models compute answers.

This brings together all the insights from our circuit investigation:
1. Layer-by-layer token probability tracking
2. Computation window identification
3. Serialization detection
4. Format sensitivity analysis

Usage:
    uv run python examples/introspection/circuit_summary.py \
        --prompt "347 * 892 = " \
        --answer " 309524"

    # Quick summary
    uv run python examples/introspection/circuit_summary.py \
        --prompt "The capital of France is" \
        --answer " Paris"
"""

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def auto_detect_answer(prompt: str) -> str | None:
    """Auto-detect expected answer for arithmetic prompts."""
    patterns = [
        (r"(\d+)\s*\+\s*(\d+)", lambda a, b: a + b),
        (r"(\d+)\s*-\s*(\d+)", lambda a, b: a - b),
        (r"(\d+)\s*\*\s*(\d+)", lambda a, b: a * b),
        (r"(\d+)\s*×\s*(\d+)", lambda a, b: a * b),
        (r"(\d+)\s*/\s*(\d+)", lambda a, b: a // b if b != 0 else 0),
        (r"(\d+)\s*÷\s*(\d+)", lambda a, b: a // b if b != 0 else 0),
    ]

    for pattern, op in patterns:
        match = re.search(pattern, prompt)
        if match:
            a, b = int(match.group(1)), int(match.group(2))
            result = op(a, b)
            return f" {result}"

    return None

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


@dataclass
class CircuitSummary:
    """Summary of computation circuit."""
    prompt: str
    answer: str
    model_id: str
    num_layers: int

    # Key findings
    computation_layer: int | None  # Where answer peaks
    emergence_layer: int | None     # Where answer becomes #1
    serialization_layer: int | None # Where first-token takes over

    # Probabilities
    peak_probability: float
    final_probability: float
    final_prediction: str
    correct: bool

    # Computation type
    computation_type: str

    # Layer trajectory (layer, top_token, answer_prob, answer_rank)
    trajectory: list[tuple[int, str, float, int | None]]


def classify_computation(
    num_layers: int,
    emergence: int | None,
    peak: int | None,
    serialization: int | None,
    peak_prob: float,
) -> str:
    """Classify the computation pattern."""

    if emergence is None and peak_prob < 0.01:
        return "NOT_COMPUTED"  # Model never found the answer

    if serialization and peak and serialization > peak:
        return "HOLISTIC_THEN_SERIAL"  # Ghost answer pattern

    if emergence and emergence < num_layers * 0.3:
        return "EARLY_LOOKUP"  # Quick retrieval

    if emergence and emergence > num_layers * 0.7:
        return "LATE_COMPUTATION"  # Deep processing

    if peak_prob > 0.5 and emergence:
        return "CONFIDENT_COMPUTATION"  # Strong answer

    return "DISTRIBUTED"  # Spread across layers


class CircuitAnalyzer:
    """Analyze computation circuits."""

    def __init__(self, model: nn.Module, tokenizer: Any, config: Any, model_id: str):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_id = model_id

    @classmethod
    async def from_pretrained(cls, model_id: str) -> "CircuitAnalyzer":
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

        return cls(model, tokenizer, config, model_id)

    def _get_layers(self): return list(self.model.model.layers) if hasattr(self.model, "model") else list(self.model.layers)
    def _get_embed(self): return self.model.model.embed_tokens if hasattr(self.model, "model") else self.model.embed_tokens
    def _get_norm(self): return getattr(self.model.model if hasattr(self.model, "model") else self.model, "norm", None)
    def _get_head(self):
        if hasattr(self.model, "lm_head"): return self.model.lm_head
        return self._get_embed().as_linear if hasattr(self._get_embed(), "as_linear") else None
    def _get_scale(self): return getattr(self.config, "embedding_scale", None)

    def analyze(self, prompt: str, answer: str) -> CircuitSummary:
        """Run full circuit analysis."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        # Get answer token ID
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        answer_id = answer_ids[-1] if answer_ids else None

        # Also track first character for serialization detection
        first_char = answer.strip()[0] if answer.strip() else ""
        first_char_ids = self.tokenizer.encode(first_char, add_special_tokens=False)
        first_char_id = first_char_ids[-1] if first_char_ids else None

        layers = self._get_layers()
        embed = self._get_embed()
        norm = self._get_norm()
        head = self._get_head()
        scale = self._get_scale()
        num_layers = len(layers)

        h = embed(input_ids)
        if scale: h = h * scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

        trajectory = []
        peak_layer = None
        peak_prob = 0.0
        emergence_layer = None
        serialization_layer = None

        for layer_idx, layer in enumerate(layers):
            try:
                out = layer(h, mask=mask)
            except TypeError:
                out = layer(h)

            h = out.hidden_states if hasattr(out, "hidden_states") else (out[0] if isinstance(out, tuple) else out)

            # Project to logits
            h_n = norm(h) if norm else h
            logits = head(h_n)
            if hasattr(logits, "logits"): logits = logits.logits

            probs = mx.softmax(logits[0, -1, :])
            top_idx = int(mx.argmax(probs))
            top_token = self.tokenizer.decode([top_idx])

            answer_prob = float(probs[answer_id]) if answer_id else 0.0
            sorted_idx = mx.argsort(probs)[::-1][:100].tolist()
            answer_rank = sorted_idx.index(answer_id) + 1 if answer_id and answer_id in sorted_idx else None

            trajectory.append((layer_idx, top_token, answer_prob, answer_rank))

            # Track peak
            if answer_prob > peak_prob:
                peak_prob = answer_prob
                peak_layer = layer_idx

            # Track emergence
            if answer_rank == 1 and emergence_layer is None:
                emergence_layer = layer_idx

            # Track serialization (when first-char takes over from full answer)
            if first_char_id and emergence_layer and serialization_layer is None:
                if top_idx == first_char_id and answer_rank != 1:
                    serialization_layer = layer_idx

        # Final prediction
        final = trajectory[-1]
        final_pred = final[1]
        final_prob = final[2]
        correct = answer.strip() in final_pred or (answer_rank == 1 if answer_rank else False)

        # Classify
        comp_type = classify_computation(num_layers, emergence_layer, peak_layer, serialization_layer, peak_prob)

        return CircuitSummary(
            prompt=prompt,
            answer=answer,
            model_id=self.model_id,
            num_layers=num_layers,
            computation_layer=peak_layer,
            emergence_layer=emergence_layer,
            serialization_layer=serialization_layer,
            peak_probability=peak_prob,
            final_probability=final_prob,
            final_prediction=final_pred,
            correct=correct,
            computation_type=comp_type,
            trajectory=trajectory,
        )


def print_summary(s: CircuitSummary):
    """Print circuit summary."""
    print(f"\n{'═'*70}")
    print("⚡ CIRCUIT ANALYSIS SUMMARY")
    print(f"{'═'*70}")
    print(f"Prompt: {repr(s.prompt)}")
    print(f"Expected: {repr(s.answer)}")
    print(f"Model: {s.model_id} ({s.num_layers} layers)")
    print(f"{'─'*70}")

    # Key layers
    print("\n📍 KEY LAYERS:")
    if s.computation_layer is not None:
        print(f"   Computation peak:  Layer {s.computation_layer} ({s.peak_probability:.1%} probability)")
    else:
        print(f"   Computation peak:  Not found (answer probability < 1%)")

    if s.emergence_layer is not None:
        print(f"   Answer emerges:    Layer {s.emergence_layer} (becomes #1 prediction)")
    else:
        print(f"   Answer emerges:    Never (doesn't become #1)")

    if s.serialization_layer is not None:
        print(f"   Serialization:     Layer {s.serialization_layer} (switches to first token)")

    # Result
    print(f"\n📊 RESULT:")
    print(f"   Final prediction: {repr(s.final_prediction)} ({'✅ CORRECT' if s.correct else '❌ WRONG'})")
    print(f"   Computation type: {s.computation_type}")

    # Interpretation
    print(f"\n💡 INTERPRETATION:")
    if s.computation_type == "HOLISTIC_THEN_SERIAL":
        print("   The model computes the FULL answer holistically, then")
        print("   serializes it to output one token at a time.")
        print(f"   → Answer exists as '{s.answer}' at layer {s.computation_layer}")
        print(f"   → Switches to first token at layer {s.serialization_layer}")
    elif s.computation_type == "NOT_COMPUTED":
        print("   ⚠️  The model never computed this answer!")
        print("   This may be due to:")
        print("   - Tokenization issues (try adding trailing space)")
        print("   - Task not in training distribution")
    elif s.computation_type == "EARLY_LOOKUP":
        print("   Answer retrieved early (likely memorized pattern)")
    elif s.computation_type == "LATE_COMPUTATION":
        print("   Answer computed in deep layers (complex reasoning)")
    elif s.computation_type == "CONFIDENT_COMPUTATION":
        print("   Strong, confident computation")
    else:
        print("   Computation distributed across layers")

    # Mini visualization
    print(f"\n📈 TRAJECTORY (every 4 layers):")
    print(f"   {'Layer':<6} {'Top Token':<12} {'Answer Prob':<12} {'Rank'}")
    print(f"   {'─'*45}")
    for layer, top, prob, rank in s.trajectory[::4]:
        rank_str = f"#{rank}" if rank else ">100"
        bar = "█" * int(prob * 20)
        print(f"   L{layer:<4} {repr(top):<12} {prob:.3f} {bar:<8} {rank_str}")
    # Always show final
    layer, top, prob, rank = s.trajectory[-1]
    rank_str = f"#{rank}" if rank else ">100"
    print(f"   L{layer:<4} {repr(top):<12} {prob:.3f} {'█'*int(prob*20):<8} {rank_str}")


async def main(model_id: str, prompt: str, answer: str):
    """Run circuit analysis."""
    print(f"Loading {model_id}...")
    analyzer = await CircuitAnalyzer.from_pretrained(model_id)

    summary = analyzer.analyze(prompt, answer)
    print_summary(summary)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16")
    parser.add_argument("--prompt", "-p", default="347 * 892 = ")
    parser.add_argument("--answer", "-a", default=None,
                        help="Expected answer token. Auto-detected for arithmetic if not specified.")
    args = parser.parse_args()

    # Auto-detect answer for arithmetic
    answer = args.answer if args.answer else auto_detect_answer(args.prompt)
    if answer is None:
        print("Error: Could not auto-detect answer. Please specify --answer")
        exit(1)

    print(f"Tracking answer: {repr(answer)}")
    asyncio.run(main(args.model, args.prompt, answer))
