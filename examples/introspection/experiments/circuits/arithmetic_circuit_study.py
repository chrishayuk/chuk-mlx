#!/usr/bin/env python3
"""
Arithmetic Circuit Study: Systematic investigation of how models compute.

This script runs a battery of experiments to understand:
1. Does computation layer vary by difficulty?
2. Does operation type (add/mul) affect circuit location?
3. Is there a "number magnitude" effect?
4. How do attention patterns change at computation layers?
5. What tokens are most attended during computation?

Usage:
    uv run python examples/introspection/arithmetic_circuit_study.py

    # Quick mode (fewer tests)
    uv run python examples/introspection/arithmetic_circuit_study.py --quick

    # Different model
    uv run python examples/introspection/arithmetic_circuit_study.py --model "mlx-community/Llama-3.2-1B-Instruct-4bit"
"""

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


@dataclass
class ArithmeticTest:
    """A single arithmetic test case."""
    prompt: str
    expected_first_digit: str
    operation: str  # "add", "mul", "sub", "div"
    difficulty: str  # "easy", "medium", "hard"
    operand_magnitude: int  # max digits in operands


@dataclass
class TestResult:
    """Result of running a test."""
    test: ArithmeticTest
    emergence_layer: int | None
    peak_layer: int | None
    peak_probability: float
    final_prediction: str
    correct: bool
    computation_type: str
    layer_trajectory: list[tuple[int, str, float]]  # (layer, top_token, prob)


@dataclass
class StudyResult:
    """Complete study results."""
    model_id: str
    num_layers: int
    results: list[TestResult] = field(default_factory=list)

    def summary(self) -> dict:
        """Generate summary statistics."""
        by_operation = {}
        by_difficulty = {}
        by_magnitude = {}

        for r in self.results:
            # By operation
            op = r.test.operation
            if op not in by_operation:
                by_operation[op] = {"correct": 0, "total": 0, "avg_emergence": []}
            by_operation[op]["total"] += 1
            if r.correct:
                by_operation[op]["correct"] += 1
            if r.emergence_layer is not None:
                by_operation[op]["avg_emergence"].append(r.emergence_layer)

            # By difficulty
            diff = r.test.difficulty
            if diff not in by_difficulty:
                by_difficulty[diff] = {"correct": 0, "total": 0, "avg_emergence": []}
            by_difficulty[diff]["total"] += 1
            if r.correct:
                by_difficulty[diff]["correct"] += 1
            if r.emergence_layer is not None:
                by_difficulty[diff]["avg_emergence"].append(r.emergence_layer)

            # By magnitude
            mag = r.test.operand_magnitude
            if mag not in by_magnitude:
                by_magnitude[mag] = {"correct": 0, "total": 0, "avg_emergence": []}
            by_magnitude[mag]["total"] += 1
            if r.correct:
                by_magnitude[mag]["correct"] += 1
            if r.emergence_layer is not None:
                by_magnitude[mag]["avg_emergence"].append(r.emergence_layer)

        # Compute averages
        for d in [by_operation, by_difficulty, by_magnitude]:
            for k, v in d.items():
                if v["avg_emergence"]:
                    v["avg_emergence_layer"] = sum(v["avg_emergence"]) / len(v["avg_emergence"])
                else:
                    v["avg_emergence_layer"] = None
                del v["avg_emergence"]
                v["accuracy"] = v["correct"] / v["total"] if v["total"] > 0 else 0

        return {
            "by_operation": by_operation,
            "by_difficulty": by_difficulty,
            "by_magnitude": by_magnitude,
        }


class ArithmeticStudy:
    """Run systematic arithmetic circuit study."""

    def __init__(self, model: nn.Module, tokenizer: Any, config: Any, model_id: str):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_id = model_id

    @classmethod
    async def from_pretrained(cls, model_id: str) -> "ArithmeticStudy":
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

        print(f"  Family: {family_type.value}")

        family_info = get_family_info(family_type)
        config = family_info.config_class.from_hf_config(config_data)
        model = family_info.model_class(config)

        HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
        tokenizer = HFLoader.load_tokenizer(model_path)

        print(f"  Layers: {config.num_hidden_layers}")

        return cls(model, tokenizer, config, model_id)

    def _get_layers(self) -> list[nn.Module]:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        if hasattr(self.model, "layers"):
            return list(self.model.layers)
        raise ValueError("Cannot find layers")

    def _get_num_layers(self) -> int:
        return self.config.num_hidden_layers

    def _get_embed_tokens(self) -> nn.Module:
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens
        if hasattr(self.model, "embed_tokens"):
            return self.model.embed_tokens
        raise ValueError("Cannot find embed_tokens")

    def _get_final_norm(self) -> nn.Module | None:
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            return self.model.model.norm
        if hasattr(self.model, "norm"):
            return self.model.norm
        return None

    def _get_lm_head(self):
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        embed = self._get_embed_tokens()
        if hasattr(embed, "as_linear"):
            return embed.as_linear
        return None

    def _get_embedding_scale(self) -> float | None:
        if hasattr(self.config, "embedding_scale"):
            return self.config.embedding_scale
        return None

    def _probe_all_layers(self, prompt: str, target_token: str) -> list[tuple[int, str, float, int | None]]:
        """
        Probe all layers and return trajectory.

        Returns: list of (layer_idx, top_token, target_prob, target_rank)
        """
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        # Get target token ID
        target_ids = self.tokenizer.encode(target_token, add_special_tokens=False)
        target_id = target_ids[0] if target_ids else None

        layers = self._get_layers()
        embed = self._get_embed_tokens()
        final_norm = self._get_final_norm()
        lm_head = self._get_lm_head()

        # Embeddings
        h = embed(input_ids)
        embed_scale = self._get_embedding_scale()
        if embed_scale:
            h = h * embed_scale

        # Mask
        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        results = []

        for layer_idx, layer in enumerate(layers):
            try:
                layer_out = layer(h, mask=mask)
            except TypeError:
                layer_out = layer(h)

            if hasattr(layer_out, "hidden_states"):
                h = layer_out.hidden_states
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
            else:
                h = layer_out

            # Project to logits
            h_normed = final_norm(h) if final_norm else h
            if lm_head:
                head_out = lm_head(h_normed)
                logits = head_out.logits if hasattr(head_out, "logits") else head_out
            else:
                logits = h_normed

            # Get probs
            probs = mx.softmax(logits[0, -1, :])
            top_idx = int(mx.argmax(probs))
            top_token = self.tokenizer.decode([top_idx])

            target_prob = float(probs[target_id]) if target_id else 0.0

            # Find rank
            sorted_idx = mx.argsort(probs)[::-1][:100].tolist()
            target_rank = None
            if target_id and target_id in sorted_idx:
                target_rank = sorted_idx.index(target_id) + 1

            results.append((layer_idx, top_token, target_prob, target_rank))

        return results

    def run_test(self, test: ArithmeticTest) -> TestResult:
        """Run a single test case."""
        trajectory = self._probe_all_layers(test.prompt, test.expected_first_digit)

        # Find emergence layer
        emergence_layer = None
        for layer_idx, top_token, prob, rank in trajectory:
            if rank == 1:
                emergence_layer = layer_idx
                break

        # Find peak layer
        peak_layer = None
        peak_prob = 0.0
        for layer_idx, top_token, prob, rank in trajectory:
            if prob > peak_prob:
                peak_prob = prob
                peak_layer = layer_idx

        # Final prediction
        final = trajectory[-1] if trajectory else (0, "?", 0.0, None)
        final_prediction = final[1]
        correct = test.expected_first_digit in final_prediction

        # Determine computation type
        if emergence_layer is None:
            comp_type = "never_emerges"
        elif emergence_layer < self._get_num_layers() * 0.3:
            comp_type = "early"
        elif emergence_layer < self._get_num_layers() * 0.7:
            comp_type = "middle"
        else:
            comp_type = "late"

        return TestResult(
            test=test,
            emergence_layer=emergence_layer,
            peak_layer=peak_layer,
            peak_probability=peak_prob,
            final_prediction=final_prediction,
            correct=correct,
            computation_type=comp_type,
            layer_trajectory=[(l, t, p) for l, t, p, r in trajectory],
        )

    def run_study(self, tests: list[ArithmeticTest]) -> StudyResult:
        """Run full study."""
        results = []

        for i, test in enumerate(tests):
            print(f"\n[{i+1}/{len(tests)}] {test.prompt}", end=" -> ")
            result = self.run_test(test)
            print(f"{result.final_prediction} ({'✓' if result.correct else '✗'})", end="")
            if result.emergence_layer is not None:
                print(f" (emerges @ L{result.emergence_layer})")
            else:
                print(" (never emerges)")
            results.append(result)

        return StudyResult(
            model_id=self.model_id,
            num_layers=self._get_num_layers(),
            results=results,
        )


def generate_test_suite(quick: bool = False) -> list[ArithmeticTest]:
    """Generate test suite."""
    tests = []

    # Easy addition (1 digit)
    easy_add = [
        ("1 + 1 = ", "2", 1),
        ("2 + 3 = ", "5", 1),
        ("4 + 5 = ", "9", 1),
        ("7 + 2 = ", "9", 1),
    ]

    # Medium addition (2 digits)
    med_add = [
        ("12 + 34 = ", "4", 2),  # 46, first digit is 4
        ("25 + 17 = ", "4", 2),  # 42
        ("99 + 11 = ", "1", 2),  # 110
    ]

    # Hard addition (3 digits)
    hard_add = [
        ("156 + 287 = ", "4", 3),  # 443
        ("999 + 111 = ", "1", 3),  # 1110
    ]

    # Easy multiplication
    easy_mul = [
        ("2 * 3 = ", "6", 1),
        ("4 * 5 = ", "2", 1),  # 20
        ("7 * 8 = ", "5", 1),  # 56
    ]

    # Medium multiplication
    med_mul = [
        ("12 * 12 = ", "1", 2),  # 144
        ("25 * 4 = ", "1", 2),   # 100
    ]

    # Hard multiplication
    hard_mul = [
        ("123 * 456 = ", "5", 3),  # 56088
        ("347 * 892 = ", "3", 3),  # 309524
    ]

    # Subtraction
    sub_tests = [
        ("10 - 3 = ", "7", 1),
        ("100 - 37 = ", "6", 2),  # 63
    ]

    # Division
    div_tests = [
        ("10 / 2 = ", "5", 1),
        ("100 / 4 = ", "2", 2),  # 25
    ]

    for prompt, first_digit, mag in easy_add:
        tests.append(ArithmeticTest(prompt, first_digit, "add", "easy", mag))
    for prompt, first_digit, mag in med_add:
        tests.append(ArithmeticTest(prompt, first_digit, "add", "medium", mag))
    for prompt, first_digit, mag in hard_add:
        tests.append(ArithmeticTest(prompt, first_digit, "add", "hard", mag))

    for prompt, first_digit, mag in easy_mul:
        tests.append(ArithmeticTest(prompt, first_digit, "mul", "easy", mag))
    for prompt, first_digit, mag in med_mul:
        tests.append(ArithmeticTest(prompt, first_digit, "mul", "medium", mag))
    for prompt, first_digit, mag in hard_mul:
        tests.append(ArithmeticTest(prompt, first_digit, "mul", "hard", mag))

    for prompt, first_digit, mag in sub_tests:
        tests.append(ArithmeticTest(prompt, first_digit, "sub", "easy" if mag == 1 else "medium", mag))
    for prompt, first_digit, mag in div_tests:
        tests.append(ArithmeticTest(prompt, first_digit, "div", "easy" if mag == 1 else "medium", mag))

    if quick:
        # Just take a sample
        tests = tests[::3]

    return tests


def print_study_summary(study: StudyResult):
    """Print study summary."""
    summary = study.summary()

    print(f"\n{'='*70}")
    print("STUDY SUMMARY")
    print(f"{'='*70}")
    print(f"Model: {study.model_id} ({study.num_layers} layers)")
    print(f"Total tests: {len(study.results)}")

    # By operation
    print(f"\n{'--- By Operation ---'}")
    print(f"{'Operation':<12} {'Accuracy':<12} {'Avg Emergence Layer'}")
    print("-" * 45)
    for op, stats in summary["by_operation"].items():
        acc = f"{stats['accuracy']*100:.1f}%"
        emerg = f"L{stats['avg_emergence_layer']:.1f}" if stats['avg_emergence_layer'] else "N/A"
        print(f"{op:<12} {acc:<12} {emerg}")

    # By difficulty
    print(f"\n{'--- By Difficulty ---'}")
    print(f"{'Difficulty':<12} {'Accuracy':<12} {'Avg Emergence Layer'}")
    print("-" * 45)
    for diff, stats in summary["by_difficulty"].items():
        acc = f"{stats['accuracy']*100:.1f}%"
        emerg = f"L{stats['avg_emergence_layer']:.1f}" if stats['avg_emergence_layer'] else "N/A"
        print(f"{diff:<12} {acc:<12} {emerg}")

    # By magnitude
    print(f"\n{'--- By Operand Magnitude ---'}")
    print(f"{'Digits':<12} {'Accuracy':<12} {'Avg Emergence Layer'}")
    print("-" * 45)
    for mag, stats in sorted(summary["by_magnitude"].items()):
        acc = f"{stats['accuracy']*100:.1f}%"
        emerg = f"L{stats['avg_emergence_layer']:.1f}" if stats['avg_emergence_layer'] else "N/A"
        print(f"{mag}-digit     {acc:<12} {emerg}")


async def main(model_id: str, quick: bool = False):
    """Run study."""
    study = await ArithmeticStudy.from_pretrained(model_id)

    tests = generate_test_suite(quick)
    print(f"\nRunning {len(tests)} arithmetic tests...")

    result = study.run_study(tests)
    print_study_summary(result)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16")
    parser.add_argument("--quick", "-q", action="store_true")
    args = parser.parse_args()

    asyncio.run(main(args.model, args.quick))
