#!/usr/bin/env python3
"""
Ghost Hunter: Find cases where the model computes correctly but outputs wrong.

KEY INSIGHT FOR GEMMA: Numbers are tokenized digit-by-digit.
" 443" becomes [' ', '4', '4', '3'] - there's no single "443" token!

So the "ghost" manifests as:
1. Space token " " reaches high probability at some layer (formatting ready)
2. First digit reaches high probability at some layer (computation happened)
3. But the model outputs something wrong

The ghost is proven when:
- Space token " " peaks with good probability (model started to format answer)
- First digit peaks with good probability (model computed the answer)
- Final output is wrong (serialization path failed)

This distinguishes:
- TRUE GHOST: Model computed answer but couldn't output it
- NEVER COMPUTED: Model never had the answer (no ghost)

Usage:
    uv run python examples/introspection/ghost_hunter.py \
        --model "mlx-community/gemma-3-4b-it-bf16"
"""

import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


@dataclass
class GhostCase:
    """A potential ghost case."""

    prompt: str
    expected_answer: str  # Human-readable answer (e.g., "443")
    first_digit: str  # First digit to track
    model_output: str
    model_correct: bool

    # Space token tracking (formatting signal)
    space_prob_at_peak: float
    space_rank_at_peak: int | None
    space_peak_layer: int

    # First digit tracking (computation signal)
    digit_prob_at_peak: float
    digit_rank_at_peak: int | None
    digit_peak_layer: int

    # Final layer stats
    digit_prob_final: float
    digit_rank_final: int | None

    # Ghost detection
    is_ghost: bool  # Computed but wrong output
    ghost_evidence: str  # Explanation of why it's a ghost


class GhostHunter:
    """Hunt for ghost answers in hidden states."""

    def __init__(self, model: nn.Module, tokenizer: Any, config: Any, model_id: str):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_id = model_id

    @classmethod
    async def from_pretrained(cls, model_id: str) -> "GhostHunter":
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

        return cls(model, tokenizer, config, model_id)

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

    def analyze_case(self, prompt: str, expected_answer: str, first_digit: str) -> GhostCase:
        """
        Analyze a single case for ghost answers.

        For digit-by-digit tokenizers (like Gemma), we track:
        1. Space token " " - indicates model is ready to output formatted answer
        2. First digit (e.g., "4") - indicates model computed the answer

        A GHOST is when both space and digit appear with high probability
        at some layer, but the final output is wrong.
        """
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        # Get token IDs for space and first digit
        space_ids = self.tokenizer.encode(" ", add_special_tokens=False)
        space_id = space_ids[0] if space_ids else None

        digit_ids = self.tokenizer.encode(first_digit, add_special_tokens=False)
        digit_id = digit_ids[0] if digit_ids else None

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

        # Track space token through layers
        space_peak_prob = 0.0
        space_peak_rank = None
        space_peak_layer = 0

        # Track first digit through layers
        digit_peak_prob = 0.0
        digit_peak_rank = None
        digit_peak_layer = 0

        final_logits = None

        for layer_idx, layer in enumerate(layers):
            try:
                out = layer(h, mask=mask)
            except TypeError:
                out = layer(h)

            h = (
                out.hidden_states
                if hasattr(out, "hidden_states")
                else (out[0] if isinstance(out, tuple) else out)
            )

            # Project to logits at each layer
            h_n = norm(h) if norm else h
            logits = head(h_n)
            if hasattr(logits, "logits"):
                logits = logits.logits

            probs = mx.softmax(logits[0, -1, :])
            sorted_idx = mx.argsort(probs)[::-1][:100].tolist()

            # Track space token probability
            if space_id:
                prob = float(probs[space_id])
                rank = sorted_idx.index(space_id) + 1 if space_id in sorted_idx else None

                if prob > space_peak_prob:
                    space_peak_prob = prob
                    space_peak_rank = rank
                    space_peak_layer = layer_idx

            # Track first digit probability
            if digit_id:
                prob = float(probs[digit_id])
                rank = sorted_idx.index(digit_id) + 1 if digit_id in sorted_idx else None

                if prob > digit_peak_prob:
                    digit_peak_prob = prob
                    digit_peak_rank = rank
                    digit_peak_layer = layer_idx

            final_logits = logits

        # Get final output and first digit stats
        final_probs = mx.softmax(final_logits[0, -1, :])
        final_top_idx = int(mx.argmax(final_probs))
        model_output = self.tokenizer.decode([final_top_idx])
        model_correct = first_digit in model_output or model_output.strip() == first_digit

        # First digit at final layer
        digit_prob_final = float(final_probs[digit_id]) if digit_id else 0.0
        sorted_final = mx.argsort(final_probs)[::-1][:100].tolist()
        digit_rank_final = (
            sorted_final.index(digit_id) + 1 if digit_id and digit_id in sorted_final else None
        )

        # GHOST DETECTION:
        # Ghost exists when:
        # 1. Space token appeared with reasonable probability (formatting was ready)
        # 2. First digit appeared with reasonable probability (computation happened)
        # 3. But the output is wrong
        #
        # This proves the model "knew" the answer but couldn't serialize it

        space_was_ready = (
            space_peak_rank is not None and space_peak_rank <= 5 and space_peak_prob > 0.05
        )
        digit_was_computed = (
            digit_peak_rank is not None and digit_peak_rank <= 10 and digit_peak_prob > 0.01
        )

        is_ghost = space_was_ready and digit_was_computed and not model_correct

        # Build evidence string
        if is_ghost:
            ghost_evidence = (
                f"Space ' ' peaked at L{space_peak_layer} ({space_peak_prob:.1%}, rank {space_peak_rank}), "
                f"Digit '{first_digit}' peaked at L{digit_peak_layer} ({digit_peak_prob:.1%}, rank {digit_peak_rank})"
            )
        elif not model_correct and digit_was_computed:
            ghost_evidence = "Digit computed but no space formatting"
        elif not model_correct and space_was_ready:
            ghost_evidence = "Space ready but digit never computed"
        elif not model_correct:
            ghost_evidence = "Never computed (no ghost)"
        else:
            ghost_evidence = "Correct output"

        return GhostCase(
            prompt=prompt,
            expected_answer=expected_answer,
            first_digit=first_digit,
            model_output=model_output,
            model_correct=model_correct,
            space_prob_at_peak=space_peak_prob,
            space_rank_at_peak=space_peak_rank,
            space_peak_layer=space_peak_layer,
            digit_prob_at_peak=digit_peak_prob,
            digit_rank_at_peak=digit_peak_rank,
            digit_peak_layer=digit_peak_layer,
            digit_prob_final=digit_prob_final,
            digit_rank_final=digit_rank_final,
            is_ghost=is_ghost,
            ghost_evidence=ghost_evidence,
        )

    def hunt(self) -> list[GhostCase]:
        """Hunt for ghosts across various test cases."""
        cases = []

        # Format: (prompt, expected_answer, first_digit)
        # Category 1: Format variations (likely ghosts)
        format_tests = [
            # Missing trailing space - GHOSTS EXPECTED HERE
            ("156 + 287 =", "443", "4"),
            ("347 * 892 =", "309524", "3"),
            ("100 - 37 =", "63", "6"),
            ("25 * 4 =", "100", "1"),
            # With trailing space (control - should work)
            ("156 + 287 = ", "443", "4"),
            ("347 * 892 = ", "309524", "3"),
            ("100 - 37 = ", "63", "6"),
            ("25 * 4 = ", "100", "1"),
            # Different equals formatting
            ("156+287=", "443", "4"),
            ("156 +287= ", "443", "4"),
            ("156+ 287 =", "443", "4"),
        ]

        # Category 2: Large numbers (stress test)
        large_tests = [
            ("999 * 999 = ", "998001", "9"),
            ("12345 + 67890 = ", "80235", "8"),
            ("99999 - 11111 = ", "88888", "8"),
            ("1000 * 1000 = ", "1000000", "1"),
        ]

        # Category 3: Edge cases
        edge_tests = [
            ("0 + 0 = ", "0", "0"),
            ("1 * 0 = ", "0", "0"),
            ("100 - 100 = ", "0", "0"),
            ("999 + 1 = ", "1000", "1"),
        ]

        all_tests = format_tests + large_tests + edge_tests

        print(f"\nHunting ghosts across {len(all_tests)} test cases...")
        print("Tracking SPACE token and FIRST DIGIT separately")
        print("=" * 80)

        for prompt, expected_answer, first_digit in all_tests:
            case = self.analyze_case(prompt, expected_answer, first_digit)
            cases.append(case)

            status = "👻 GHOST!" if case.is_ghost else ("✅" if case.model_correct else "❌")

            # Show both space and digit tracking
            space_str = f"sp@L{case.space_peak_layer}({case.space_prob_at_peak:.0%})"
            digit_str = f"d@L{case.digit_peak_layer}({case.digit_prob_at_peak:.0%})"

            print(
                f"{status} {prompt:<20} expect={first_digit} got={case.model_output:<4} "
                f"{space_str} {digit_str}"
            )

        return cases


def print_ghost_summary(cases: list[GhostCase]):
    """Print summary of ghost hunting."""
    ghosts = [c for c in cases if c.is_ghost]
    correct = [c for c in cases if c.model_correct]
    wrong_no_ghost = [c for c in cases if not c.model_correct and not c.is_ghost]

    print(f"\n{'=' * 80}")
    print("👻 GHOST HUNTING SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total cases:      {len(cases)}")
    print(f"Model correct:    {len(correct)} ({100 * len(correct) / len(cases):.1f}%)")
    print(
        f"Wrong (no ghost): {len(wrong_no_ghost)} ({100 * len(wrong_no_ghost) / len(cases):.1f}%)"
    )
    print(f"GHOSTS FOUND:     {len(ghosts)} ({100 * len(ghosts) / len(cases):.1f}%)")

    if ghosts:
        print(f"\n{'=' * 80}")
        print("🔥 GHOST CASES (model computed answer but couldn't output it)")
        print(f"{'=' * 80}")
        for g in ghosts:
            print(f"\n  Prompt: {repr(g.prompt)}")
            print(f"  Expected: {g.expected_answer} (first digit: '{g.first_digit}')")
            print(f"  Model output: {repr(g.model_output)} ❌")
            print(f"  Evidence: {g.ghost_evidence}")
            print(f"  → The model KNEW '{g.first_digit}' but couldn't serialize it!")

    # Show wrong cases without ghosts
    if wrong_no_ghost:
        print(f"\n{'=' * 80}")
        print("❌ WRONG CASES (no ghost - model never computed answer)")
        print(f"{'=' * 80}")
        for c in wrong_no_ghost[:5]:  # Show first 5
            print(f"  {repr(c.prompt)}: {c.ghost_evidence}")
        if len(wrong_no_ghost) > 5:
            print(f"  ... and {len(wrong_no_ghost) - 5} more")

    # Analyze format sensitivity
    format_cases = [c for c in cases if "=" in c.prompt]
    with_space = [c for c in format_cases if c.prompt.endswith(" ")]
    without_space = [c for c in format_cases if not c.prompt.endswith(" ")]

    if with_space and without_space:
        ws_correct = sum(1 for c in with_space if c.model_correct)
        wos_correct = sum(1 for c in without_space if c.model_correct)
        wos_ghosts = sum(1 for c in without_space if c.is_ghost)

        print(f"\n{'=' * 80}")
        print("FORMAT SENSITIVITY ANALYSIS")
        print(f"{'=' * 80}")
        print(
            f"With trailing space:    {ws_correct}/{len(with_space)} correct ({100 * ws_correct / len(with_space):.0f}%)"
        )
        print(
            f"Without trailing space: {wos_correct}/{len(without_space)} correct ({100 * wos_correct / len(without_space):.0f}%)"
        )
        print(f"Ghosts in no-space:     {wos_ghosts}/{len(without_space)}")

        if wos_ghosts > 0:
            print(
                f"\n⚠️  {wos_ghosts} cases where model COMPUTED the answer but FORMAT blocked output!"
            )
            print("   This is the 'ghost in Layer 20' phenomenon.")
        elif wos_correct < len(without_space):
            print(
                f"\n⚠️  {len(without_space) - wos_correct} failures WITHOUT ghosts = model never computed answer"
            )
            print("   The format issue blocks computation entirely, not just serialization.")


async def main(model_id: str):
    """Run ghost hunting."""
    hunter = await GhostHunter.from_pretrained(model_id)
    cases = hunter.hunt()
    print_ghost_summary(cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16")
    args = parser.parse_args()

    asyncio.run(main(args.model))
