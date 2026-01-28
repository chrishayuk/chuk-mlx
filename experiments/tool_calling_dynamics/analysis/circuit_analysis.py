#!/usr/bin/env python3
"""
Cross-Layer Circuit Analysis for Tool Calling

Identifies stable expert combinations across layers that form
functional units for tool-related computation.

Research Question: Do specific expert circuits form for different tool types?
"""

import gc
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


class CircuitAnalyzer:
    """
    Analyzes cross-layer expert circuits for tool calling.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model"]["primary"]
        self.correlation_threshold = config["circuits"]["correlation_threshold"]
        self.min_circuit_length = config["circuits"]["min_circuit_length"]
        self.consistency_threshold = config["circuits"]["consistency_threshold"]
        self.num_prompts = config["circuits"]["num_prompts"]

        self.model = None
        self.tokenizer = None
        self.top_k = config["moe"]["top_k"]

    def load_model(self):
        """Load the model and tokenizer."""
        from mlx_lm import load

        logger.info(f"Loading model: {self.model_name}")
        self.model, self.tokenizer = load(self.model_name)

        if hasattr(self.model, 'args'):
            self.top_k = getattr(self.model.args, 'num_experts_per_tok', self.top_k)
            self.hidden_size = getattr(self.model.args, 'hidden_size', 4096)

        logger.info("Model loaded successfully")

    def get_model_components(self):
        """Get model layers and embedding."""
        if hasattr(self.model, 'model'):
            return self.model.model.embed_tokens, self.model.model.layers
        return self.model.embed_tokens, self.model.layers

    def trace_expert_activations(
        self,
        prompt: str
    ) -> dict[int, set[int]]:
        """
        Trace which experts activate at each layer for a prompt.
        """
        embed_tokens, layers = self.get_model_components()

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        h = embed_tokens(input_ids)
        batch_size, seq_len, hidden_size = h.shape
        mask = mx.triu(mx.full((seq_len, seq_len), float('-inf'), dtype=h.dtype), k=1)

        activations = {}

        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, 'input_layernorm'):
                normed = layer.input_layernorm(h)
            else:
                normed = h

            mlp = layer.mlp
            if hasattr(mlp, 'router'):
                x_flat = normed.reshape(-1, hidden_size)
                router_out = mlp.router(x_flat)

                if isinstance(router_out, tuple):
                    _, top_k_indices = router_out
                else:
                    top_k_indices = mx.argsort(router_out, axis=-1)[:, -self.top_k:]

                mx.eval(top_k_indices)

                # Collect unique experts across all positions
                experts = set()
                for tok_idx in range(seq_len):
                    for k in range(min(self.top_k, top_k_indices.shape[-1])):
                        experts.add(int(top_k_indices[tok_idx, k]))

                activations[layer_idx] = experts

            h = layer(h, mask=mask)

        mx.eval(h)
        return activations

    def find_co_occurrence_patterns(
        self,
        prompts: list[str]
    ) -> dict[str, int]:
        """
        Find expert co-occurrence patterns across prompts.
        """
        circuit_counts = defaultdict(int)

        for prompt in prompts:
            activations = self.trace_expert_activations(prompt)

            # Build circuit signature from consecutive layer pairs
            sorted_layers = sorted(activations.keys())
            for i in range(len(sorted_layers) - 1):
                layer_a = sorted_layers[i]
                layer_b = sorted_layers[i + 1]

                for expert_a in activations[layer_a]:
                    for expert_b in activations[layer_b]:
                        circuit_sig = f"L{layer_a}-E{expert_a}->L{layer_b}-E{expert_b}"
                        circuit_counts[circuit_sig] += 1

        return dict(circuit_counts)

    def identify_stable_circuits(
        self,
        co_occurrence: dict[str, int],
        min_count: int = 3
    ) -> list[dict[str, Any]]:
        """
        Identify stable circuits that appear consistently.
        """
        stable = {k: v for k, v in co_occurrence.items() if v >= min_count}
        sorted_circuits = sorted(stable.items(), key=lambda x: -x[1])

        circuits = []
        for sig, count in sorted_circuits[:30]:
            parts = sig.split("->")
            if len(parts) != 2:
                continue

            try:
                layer_a, expert_a = parts[0].split("-E")
                layer_b, expert_b = parts[1].split("-E")

                circuits.append({
                    "signature": sig,
                    "start_layer": int(layer_a[1:]),
                    "start_expert": int(expert_a),
                    "end_layer": int(layer_b[1:]),
                    "end_expert": int(expert_b),
                    "count": count,
                    "frequency": count / max(self.num_prompts, 1),
                })
            except (ValueError, IndexError):
                continue

        return circuits

    def analyze_tool_specific_circuits(
        self,
        prompts_by_tool: dict[str, list[str]]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Find circuits that are specific to each tool type.
        """
        tool_circuits = {}

        for tool_name, prompts in prompts_by_tool.items():
            logger.info(f"Analyzing circuits for {tool_name}...")
            co_occurrence = self.find_co_occurrence_patterns(prompts)
            circuits = self.identify_stable_circuits(
                co_occurrence,
                min_count=max(2, len(prompts) // 5)
            )
            tool_circuits[tool_name] = circuits
            gc.collect()

        return tool_circuits

    def compute_circuit_consistency(
        self,
        tool_circuits: dict[str, list[dict[str, Any]]]
    ) -> dict[str, float]:
        """
        Compute how consistent circuits are across prompts of same tool.
        """
        consistency = {}

        for tool_name, circuits in tool_circuits.items():
            if not circuits:
                consistency[tool_name] = 0.0
                continue

            top_freqs = [c["frequency"] for c in circuits[:10]]
            consistency[tool_name] = float(np.mean(top_freqs)) if top_freqs else 0.0

        return consistency

    def find_shared_vs_unique_circuits(
        self,
        tool_circuits: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """
        Identify circuits shared across tools vs unique to specific tools.
        """
        tool_signatures = {
            tool: {c["signature"] for c in circuits}
            for tool, circuits in tool_circuits.items()
        }

        all_tools = list(tool_signatures.keys())
        if len(all_tools) < 2:
            return {"shared": [], "unique": tool_signatures, "num_shared": 0}

        shared = set.intersection(*tool_signatures.values()) if tool_signatures else set()

        unique = {}
        for tool, sigs in tool_signatures.items():
            other_sigs = set.union(*[
                s for t, s in tool_signatures.items() if t != tool
            ]) if len(all_tools) > 1 else set()
            unique[tool] = list(sigs - other_sigs)

        return {
            "shared": list(shared),
            "unique": unique,
            "num_shared": len(shared),
            "num_unique_per_tool": {t: len(u) for t, u in unique.items()}
        }

    def run(self) -> dict[str, Any]:
        """
        Run the full circuit analysis.
        """
        logger.info("Starting circuit analysis")

        # Load model
        self.load_model()

        # Organize prompts by tool
        prompts_by_tool = self.config["prompts"]["tool_required"]

        # Analyze tool-specific circuits
        tool_circuits = self.analyze_tool_specific_circuits(prompts_by_tool)

        # Compute consistency
        consistency = self.compute_circuit_consistency(tool_circuits)

        # Find shared vs unique
        shared_unique = self.find_shared_vs_unique_circuits(tool_circuits)

        # Overall statistics
        all_circuits = []
        for circuits in tool_circuits.values():
            all_circuits.extend(circuits)

        results = {
            "tool_circuits": tool_circuits,
            "consistency": consistency,
            "shared_unique_analysis": shared_unique,
            "num_circuits": len(set(c["signature"] for c in all_circuits)),
            "avg_consistency": float(np.mean(list(consistency.values()))) if consistency else 0.0,
            "tools_analyzed": list(prompts_by_tool.keys()),
        }

        logger.info(f"Found {results['num_circuits']} unique circuits")
        logger.info(f"Average consistency: {results['avg_consistency']:.1%}")

        return results


def main():
    """Run as standalone script."""
    import yaml

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    analyzer = CircuitAnalyzer(config)
    results = analyzer.run()

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
