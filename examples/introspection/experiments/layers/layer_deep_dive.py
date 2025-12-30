#!/usr/bin/env python3
"""
Layer 20 Deep Dive: Investigating where arithmetic computation happens.

This script provides tools to test hypotheses about layer 20's role in arithmetic:

1. Attention Pattern Analysis - Where is position "=" looking?
2. MLP Knockout - Zero out layer 20's MLP and see if answer disappears
3. Activation Patching - Swap layer 20 activations between prompts
4. Linear Probe - Can we decode the full answer from layer 20's hidden state?

The key insight: Layer 20 shows the FULL answer " 309524" but by Layer 24
the model switches to outputting "3" (first digit for autoregressive generation).

Usage:
    uv run python examples/introspection/layer_deep_dive.py --prompt "347 * 892 = " --answer " 309524"

    # Test multiple prompts
    uv run python examples/introspection/layer_deep_dive.py --prompt "156 + 287 = " --answer " 443"

    # Run all experiments
    uv run python examples/introspection/layer_deep_dive.py --all-experiments
"""

import argparse
import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.introspection import (
    AnalysisConfig,
    LayerStrategy,
    ModelAnalyzer,
    PositionSelection,
)
from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks


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


@dataclass
class AttentionFocusResult:
    """Where a position is attending in a layer."""
    layer_idx: int
    query_position: int
    query_token: str
    top_attended: list[tuple[int, str, float]]  # (position, token, attention_weight)


@dataclass
class MLPKnockoutResult:
    """Result of knocking out an MLP at a layer."""
    layer_idx: int
    original_top_token: str
    original_probability: float
    ablated_top_token: str
    ablated_probability: float
    target_token: str
    target_original_prob: float
    target_ablated_prob: float
    answer_disappeared: bool


@dataclass
class ActivationPatchResult:
    """Result of patching activations from one prompt to another."""
    source_prompt: str
    target_prompt: str
    layer_idx: int
    original_prediction: str
    patched_prediction: str
    source_answer: str
    answer_transferred: bool


@dataclass
class LinearProbeResult:
    """Result of probing layer hidden states for answer decoding."""
    layer_idx: int
    hidden_state_norm: float
    top_5_tokens: list[tuple[str, float]]  # (token, probability)
    target_token: str
    target_probability: float
    target_rank: int | None
    can_decode_answer: bool


@dataclass
class DeepDiveResult:
    """Complete deep dive analysis."""
    prompt: str
    target_answer: str
    model_id: str

    # Layer predictions (logit lens style)
    layer_predictions: dict[int, tuple[str, float]] = field(default_factory=dict)

    # Attention analysis
    attention_focus: list[AttentionFocusResult] = field(default_factory=list)

    # MLP knockout results
    mlp_knockouts: list[MLPKnockoutResult] = field(default_factory=list)

    # Activation patching
    activation_patches: list[ActivationPatchResult] = field(default_factory=list)

    # Linear probe results
    linear_probes: list[LinearProbeResult] = field(default_factory=list)


class LayerDeepDive:
    """
    Deep investigation of layer behavior for arithmetic tasks.

    Tests the hypothesis that layer ~20 computes the answer holistically,
    and later layers serialize it for autoregressive output.
    """

    def __init__(self, model: nn.Module, tokenizer: Any, model_id: str = "unknown"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self._config = None

    @classmethod
    async def from_pretrained(cls, model_id: str) -> "LayerDeepDive":
        """Load model from HuggingFace."""
        from chuk_lazarus.inference.loader import DType, HFLoader
        from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info
        import json

        # Download model
        result = HFLoader.download(model_id)
        model_path = result.model_path

        # Load config
        config_path = model_path / "config.json"
        with open(config_path) as f:
            config_data = json.load(f)

        # Detect model family
        family_type = detect_model_family(config_data)
        if family_type is None:
            raise ValueError(f"Unsupported model: {model_id}")

        family_info = get_family_info(family_type)
        config_class = family_info.config_class
        model_class = family_info.model_class

        # Create model
        config = config_class.from_hf_config(config_data)
        model = model_class(config)

        # Load weights
        HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)

        # Load tokenizer
        tokenizer = HFLoader.load_tokenizer(model_path)

        instance = cls(model, tokenizer, model_id)
        instance._config = config
        return instance

    def _get_layers(self) -> list[nn.Module]:
        """Get transformer layers."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        if hasattr(self.model, "layers"):
            return list(self.model.layers)
        raise ValueError("Cannot find layers")

    def _get_num_layers(self) -> int:
        """Get number of layers."""
        return len(self._get_layers())

    def _is_moe_model(self) -> bool:
        """Check if model uses MoE (Mixture of Experts)."""
        # Check config for MoE indicators
        if self._config is not None:
            if hasattr(self._config, 'is_moe') and self._config.is_moe:
                return True
            if hasattr(self._config, 'num_local_experts') and self._config.num_local_experts > 1:
                return True
        # Check first layer for MoE structure
        layers = self._get_layers()
        if layers:
            layer = layers[0]
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                # GPT-OSS MoE has router and experts
                if hasattr(mlp, 'router') and hasattr(mlp, 'experts'):
                    return True
                # Other MoE patterns
                if hasattr(mlp, 'gate') and hasattr(mlp, 'experts'):
                    return True
        return False

    def _get_embedding_scale(self) -> float | None:
        """Get embedding scale for models like Gemma."""
        if self._config and hasattr(self._config, "embedding_scale"):
            return self._config.embedding_scale
        return None

    def analyze_attention_patterns(
        self,
        prompt: str,
        layers: list[int],
        query_position: int = -1,
        top_k: int = 5,
    ) -> list[AttentionFocusResult]:
        """
        Analyze where a position attends to in specific layers.

        For arithmetic, we expect the "=" position to gather from operand positions.
        """
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

        # Setup hooks to capture attention
        hooks = ModelHooks(self.model, model_config=self._config)
        hooks.configure(CaptureConfig(
            layers=layers,
            capture_hidden_states=True,
            capture_attention_weights=True,
            positions=PositionSelection.ALL,  # Need all positions for attention
        ))

        # Forward pass
        hooks.forward(input_ids)

        # Handle negative position
        seq_len = len(tokens)
        if query_position < 0:
            query_position = seq_len + query_position

        results = []
        for layer_idx in layers:
            attn_weights = hooks.state.attention_weights.get(layer_idx)
            if attn_weights is None:
                continue

            # attn_weights shape: [batch, heads, seq, seq]
            # Average across heads
            avg_attn = mx.mean(attn_weights[0], axis=0)  # [seq, seq]

            # Get attention for query position
            query_attn = avg_attn[query_position].tolist()

            # Get top-k attended positions
            indexed = list(enumerate(query_attn))
            indexed.sort(key=lambda x: x[1], reverse=True)

            top_attended = [
                (pos, tokens[pos], weight)
                for pos, weight in indexed[:top_k]
            ]

            results.append(AttentionFocusResult(
                layer_idx=layer_idx,
                query_position=query_position,
                query_token=tokens[query_position],
                top_attended=top_attended,
            ))

        return results

    def knockout_mlp(
        self,
        prompt: str,
        target_token: str,
        layers: list[int],
    ) -> list[MLPKnockoutResult]:
        """
        Zero out MLP at each layer and see effect on target token probability.

        If layer 20's MLP is causal for arithmetic, ablating it should destroy
        the answer.

        Note: MoE models are not supported for MLP knockout as the experts
        use quantized weights that cannot be easily zeroed.
        """
        # Check for MoE - not supported for ablation
        if self._is_moe_model():
            print("  (MLP knockout not supported for MoE models)")
            return []

        from chuk_lazarus.introspection.ablation import AblationStudy, ComponentType, AblationConfig

        # Create adapter manually
        from chuk_lazarus.introspection.ablation import ModelAdapter
        adapter = ModelAdapter(self.model, self.tokenizer, self._config)
        study = AblationStudy(adapter)

        # Get target token ID
        target_ids = self.tokenizer.encode(target_token)
        if len(target_ids) == 0:
            target_ids = self.tokenizer.encode(target_token.strip())
        target_id = target_ids[-1] if target_ids else None

        # Get original predictions
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        hooks = ModelHooks(self.model, model_config=self._config)
        hooks.configure(CaptureConfig(layers=[], capture_hidden_states=False))
        logits = hooks.forward(input_ids)

        probs = mx.softmax(logits[0, -1, :])
        top_idx = int(mx.argmax(probs))
        original_top_token = self.tokenizer.decode([top_idx])
        original_top_prob = float(probs[top_idx])
        original_target_prob = float(probs[target_id]) if target_id else 0.0

        results = []
        for layer_idx in layers:
            # Ablate MLP at this layer
            ablated_output = study.ablate_and_generate(
                prompt,
                layers=[layer_idx],
                component=ComponentType.MLP,
                config=AblationConfig(max_new_tokens=1, temperature=0.0),
            )

            # Get ablated predictions by running forward again with ablation
            # For now, use the generated output to determine effect
            ablated_top_token = ablated_output.strip()[:10] if ablated_output else "?"

            # For proper comparison, we need to get logits with ablation
            # This is a simplified version - full version would re-run hooks
            results.append(MLPKnockoutResult(
                layer_idx=layer_idx,
                original_top_token=original_top_token,
                original_probability=original_top_prob,
                ablated_top_token=ablated_top_token,
                ablated_probability=0.0,  # Would need full re-run
                target_token=target_token,
                target_original_prob=original_target_prob,
                target_ablated_prob=0.0,  # Would need full re-run
                answer_disappeared=ablated_top_token != original_top_token,
            ))

        return results

    def probe_layers(
        self,
        prompt: str,
        target_token: str,
        layers: list[int] | None = None,
    ) -> list[LinearProbeResult]:
        """
        Probe each layer to see if the target token can be decoded.

        This is the key test: if layer 20 "knows" the answer, projecting
        its hidden state through the LM head should give high probability
        to the answer token.
        """
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        num_layers = self._get_num_layers()
        if layers is None:
            layers = list(range(0, num_layers, 4)) + [num_layers - 1]
            layers = sorted(set(layers))

        # Get target token ID
        target_ids = self.tokenizer.encode(target_token)
        if len(target_ids) == 0:
            target_ids = self.tokenizer.encode(target_token.strip())
        target_id = target_ids[-1] if target_ids else None

        # Setup hooks
        hooks = ModelHooks(self.model, model_config=self._config)
        hooks.configure(CaptureConfig(
            layers=layers,
            capture_hidden_states=True,
            positions=PositionSelection.LAST,
        ))

        # Forward pass
        hooks.forward(input_ids)

        results = []
        for layer_idx in layers:
            # Get logits for this layer using logit lens
            layer_logits = hooks.get_layer_logits(layer_idx, normalize=True)
            if layer_logits is None:
                continue

            # Get last position logits
            if layer_logits.ndim == 3:
                pos_logits = layer_logits[0, -1, :]
            else:
                pos_logits = layer_logits[-1, :]

            probs = mx.softmax(pos_logits)

            # Get top 5 predictions
            sorted_idx = mx.argsort(probs)[::-1]
            top_5_idx = sorted_idx[:5].tolist()
            top_5 = [
                (self.tokenizer.decode([idx]), float(probs[idx]))
                for idx in top_5_idx
            ]

            # Get target token info
            target_prob = float(probs[target_id]) if target_id else 0.0

            # Find rank of target
            target_rank = None
            if target_id is not None:
                sorted_list = sorted_idx[:100].tolist()
                if target_id in sorted_list:
                    target_rank = sorted_list.index(target_id) + 1

            # Get hidden state norm
            hidden = hooks.state.hidden_states.get(layer_idx)
            if hidden is not None:
                if hidden.ndim == 3:
                    h = hidden[0, -1, :]
                else:
                    h = hidden[-1, :]
                hidden_norm = float(mx.sqrt(mx.sum(h * h)))
            else:
                hidden_norm = 0.0

            results.append(LinearProbeResult(
                layer_idx=layer_idx,
                hidden_state_norm=hidden_norm,
                top_5_tokens=top_5,
                target_token=target_token,
                target_probability=target_prob,
                target_rank=target_rank,
                can_decode_answer=target_rank == 1,
            ))

        return results


async def run_deep_dive(
    model_id: str,
    prompt: str,
    target_answer: str,
    focus_layers: list[int] | None = None,
) -> DeepDiveResult:
    """Run full deep dive analysis on a prompt."""

    print(f"\n{'='*60}")
    print("Layer Deep Dive Analysis")
    print(f"{'='*60}")
    print(f"Model: {model_id}")
    print(f"Prompt: {repr(prompt)}")
    print(f"Target Answer: {repr(target_answer)}")
    print(f"{'='*60}")

    # Load model
    print("\nLoading model...")
    dive = await LayerDeepDive.from_pretrained(model_id)

    num_layers = dive._get_num_layers()
    print(f"Model has {num_layers} layers")

    # Default focus layers around layer 20
    if focus_layers is None:
        focus_layers = [0, 8, 12, 16, 18, 19, 20, 21, 22, 24, 28, 32, num_layers - 1]
        focus_layers = [l for l in focus_layers if l < num_layers]

    result = DeepDiveResult(
        prompt=prompt,
        target_answer=target_answer,
        model_id=model_id,
    )

    # 1. Linear Probe Analysis
    print("\n" + "="*60)
    print("1. LINEAR PROBE ANALYSIS")
    print("   Testing: Can we decode the answer from each layer?")
    print("="*60)

    probes = dive.probe_layers(prompt, target_answer, focus_layers)
    result.linear_probes = probes

    print(f"\n{'Layer':<8} {'Top Token':<15} {'Target Prob':<12} {'Target Rank':<12} {'Decode?'}")
    print("-" * 60)

    emergence_layer = None
    for probe in probes:
        top_token = probe.top_5_tokens[0][0] if probe.top_5_tokens else "?"
        rank_str = str(probe.target_rank) if probe.target_rank else ">100"
        decode_str = "YES ✓" if probe.can_decode_answer else "no"

        if probe.can_decode_answer and emergence_layer is None:
            emergence_layer = probe.layer_idx

        print(f"{probe.layer_idx:<8} {repr(top_token):<15} {probe.target_probability:.4f}       {rank_str:<12} {decode_str}")

    if emergence_layer is not None:
        print(f"\n→ Answer becomes top-1 at layer {emergence_layer}")

    # 2. Attention Pattern Analysis
    print("\n" + "="*60)
    print("2. ATTENTION PATTERN ANALYSIS")
    print("   Testing: Where does the last position look?")
    print("="*60)

    attn_layers = [l for l in [16, 18, 20, 22, 24] if l < num_layers]
    attention_results = dive.analyze_attention_patterns(
        prompt,
        layers=attn_layers,
        query_position=-1,
        top_k=5,
    )
    result.attention_focus = attention_results

    for attn in attention_results:
        print(f"\nLayer {attn.layer_idx} - Position {attn.query_position} ({repr(attn.query_token)}) attends to:")
        for pos, token, weight in attn.top_attended:
            bar = "#" * int(weight * 50)
            print(f"  {pos:3d} {repr(token):10} {weight:.3f} {bar}")

    # 3. MLP Knockout (optional, slower)
    print("\n" + "="*60)
    print("3. MLP KNOCKOUT ANALYSIS")
    print("   Testing: Which layer's MLP is causal for the answer?")
    print("="*60)

    knockout_layers = [18, 19, 20, 21, 22]
    knockout_layers = [l for l in knockout_layers if l < num_layers]

    try:
        knockouts = dive.knockout_mlp(prompt, target_answer, knockout_layers)
        result.mlp_knockouts = knockouts

        print(f"\n{'Layer':<8} {'Original':<15} {'Ablated':<15} {'Changed?'}")
        print("-" * 50)

        for ko in knockouts:
            changed = "YES ✓" if ko.answer_disappeared else "no"
            print(f"{ko.layer_idx:<8} {repr(ko.original_top_token):<15} {repr(ko.ablated_top_token):<15} {changed}")
    except Exception as e:
        print(f"  (Knockout analysis failed: {e})")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Find the "computation layer"
    computation_layer = None
    for probe in probes:
        if probe.target_probability > 0.1 and probe.target_rank and probe.target_rank <= 3:
            computation_layer = probe.layer_idx
            break

    if computation_layer is not None:
        print(f"\n✓ Answer first appears strongly at layer {computation_layer}")

    if emergence_layer is not None:
        print(f"✓ Answer becomes top-1 at layer {emergence_layer}")

    # Check if later layers "serialize"
    later_probes = [p for p in probes if p.layer_idx >= 24]
    if later_probes and not later_probes[-1].can_decode_answer:
        first_digit = target_answer.strip()[0] if target_answer.strip() else "?"
        final_top = later_probes[-1].top_5_tokens[0][0] if later_probes[-1].top_5_tokens else "?"
        if first_digit in final_top:
            print(f"✓ Final layer outputs first digit '{first_digit}' instead of full answer")
            print("  → Confirms serialization hypothesis!")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Deep dive into layer behavior for arithmetic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        default="mlx-community/gemma-3-4b-it-bf16",
        help="Model to analyze",
    )
    parser.add_argument(
        "--prompt", "-p",
        default="347 * 892 = ",
        help="Arithmetic prompt to analyze",
    )
    parser.add_argument(
        "--answer", "-a",
        default=None,
        help="Expected answer token (auto-detected for arithmetic if not specified)",
    )
    parser.add_argument(
        "--all-experiments",
        action="store_true",
        help="Run on multiple arithmetic prompts",
    )

    args = parser.parse_args()

    if args.all_experiments:
        prompts = [
            ("347 * 892 = ", " 309524"),
            ("156 + 287 = ", " 443"),
            ("100 * 100 = ", " 10000"),
            ("25 * 4 = ", " 100"),
        ]
        for prompt, answer in prompts:
            asyncio.run(run_deep_dive(args.model, prompt, answer))
    else:
        # Auto-detect answer for arithmetic
        answer = args.answer if args.answer else auto_detect_answer(args.prompt)
        if answer is None:
            print("Error: Could not auto-detect answer. Please specify --answer")
            exit(1)

        print(f"Tracking answer: {repr(answer)}")
        asyncio.run(run_deep_dive(args.model, args.prompt, answer))


if __name__ == "__main__":
    main()
