#!/usr/bin/env python3
"""
Expert Pattern Analysis for Tool Calling

Analyzes which MoE experts specialize in tool-calling related tokens.

Research Question: Do specific experts handle tool-related computations?
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


class ExpertPatternAnalyzer:
    """
    Analyzes expert activation patterns for tool-related tokens.
    """

    # Token categories for analysis
    TOKEN_CATEGORIES = {
        "function_syntax": ["(", ")", ",", ":", "=>", "->"],
        "json_syntax": ["{", "}", "[", "]", '"', "true", "false", "null"],
        "tool_markers": ["tool", "function", "call", "invoke", "execute", "run"],
        "parameter_tokens": ["name", "arguments", "value", "params", "args", "input"],
        "math_operators": ["+", "-", "*", "/", "=", "calculate", "compute"],
        "search_tokens": ["search", "find", "lookup", "query", "get"],
    }

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model"]["primary"]
        self.concentration_threshold = config["expert_patterns"]["concentration_threshold"]
        self.min_activations = config["expert_patterns"]["min_activations"]

        self.model = None
        self.tokenizer = None
        self.num_experts = config["moe"]["num_experts"]
        self.num_layers = config["moe"]["num_layers"]
        self.top_k = config["moe"]["top_k"]

    def load_model(self):
        """Load the model and tokenizer."""
        from mlx_lm import load

        logger.info(f"Loading model: {self.model_name}")
        self.model, self.tokenizer = load(self.model_name)

        # Get actual model params
        if hasattr(self.model, 'args'):
            self.num_experts = getattr(self.model.args, 'num_local_experts', self.num_experts)
            self.top_k = getattr(self.model.args, 'num_experts_per_tok', self.top_k)
            self.hidden_size = getattr(self.model.args, 'hidden_size', 4096)

        logger.info("Model loaded successfully")

    def get_model_components(self):
        """Get model layers and embedding."""
        if hasattr(self.model, 'model'):
            return self.model.model.embed_tokens, self.model.model.layers
        return self.model.embed_tokens, self.model.layers

    def analyze_prompt_expert_activations(
        self,
        prompt: str,
        category: str
    ) -> dict[tuple[int, int], int]:
        """
        Analyze which experts activate for a prompt.

        Returns:
            Dictionary mapping (layer_idx, expert_idx) to activation count
        """
        embed_tokens, layers = self.get_model_components()

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        h = embed_tokens(input_ids)
        batch_size, seq_len, hidden_size = h.shape
        mask = mx.triu(mx.full((seq_len, seq_len), float('-inf'), dtype=h.dtype), k=1)

        activations = defaultdict(int)

        for layer_idx, layer in enumerate(layers):
            # Get pre-MLP hidden state
            if hasattr(layer, 'input_layernorm'):
                normed = layer.input_layernorm(h)
            else:
                normed = h

            mlp = layer.mlp
            if hasattr(mlp, 'router'):
                x_flat = normed.reshape(-1, hidden_size)
                # Router returns logits or (weights, indices)
                router_out = mlp.router(x_flat)

                if isinstance(router_out, tuple):
                    _, top_k_indices = router_out
                else:
                    # It's logits, get top-k
                    top_k_indices = mx.argsort(router_out, axis=-1)[:, -self.top_k:]

                mx.eval(top_k_indices)

                # Record activations
                for tok_idx in range(seq_len):
                    for k in range(self.top_k):
                        if k < top_k_indices.shape[-1]:
                            expert_idx = int(top_k_indices[tok_idx, k])
                            activations[(layer_idx, expert_idx)] += 1

            h = layer(h, mask=mask)

        mx.eval(h)
        return dict(activations)

    def analyze_category_activations(
        self,
        prompts: list[str],
        category: str
    ) -> dict[tuple[int, int], int]:
        """Analyze activations for all prompts in a category."""
        category_activations = defaultdict(int)

        for prompt in prompts:
            prompt_activations = self.analyze_prompt_expert_activations(prompt, category)
            for key, count in prompt_activations.items():
                category_activations[key] += count

        return dict(category_activations)

    def compute_specialization_scores(
        self,
        category_activations: dict[str, dict[tuple[int, int], int]]
    ) -> dict[str, Any]:
        """
        Compute specialization scores for each category.
        """
        specialization = {}

        for category, activations in category_activations.items():
            if not activations:
                continue

            # Group by layer
            layer_counts = defaultdict(lambda: defaultdict(int))
            for (layer_idx, expert_idx), count in activations.items():
                layer_counts[layer_idx][expert_idx] = count

            specialization[category] = {}
            for layer_idx, counts in layer_counts.items():
                total = sum(counts.values())
                if total < self.min_activations:
                    continue

                # Compute concentration (Herfindahl index)
                probs = np.array(list(counts.values())) / total
                concentration = float(np.sum(probs ** 2))

                # Find top experts
                sorted_experts = sorted(counts.items(), key=lambda x: -x[1])
                top_experts = [
                    {"expert": e, "count": c, "fraction": c / total}
                    for e, c in sorted_experts[:5]
                ]

                specialization[category][layer_idx] = {
                    "concentration": concentration,
                    "total_activations": total,
                    "top_experts": top_experts
                }

        return specialization

    def identify_tool_experts(
        self,
        specialization: dict[str, Any]
    ) -> dict[int, list[int]]:
        """
        Identify experts that specialize in tool-related tokens.
        """
        tool_categories = ["function_syntax", "json_syntax", "tool_markers", "math_operators"]
        tool_experts = defaultdict(set)

        for category in tool_categories:
            if category not in specialization:
                continue

            for layer_idx, stats in specialization[category].items():
                if stats["concentration"] >= self.concentration_threshold:
                    for expert_info in stats["top_experts"]:
                        if expert_info["fraction"] >= 0.1:
                            tool_experts[layer_idx].add(expert_info["expert"])

        return {k: list(v) for k, v in tool_experts.items()}

    def run(self) -> dict[str, Any]:
        """
        Run the full expert pattern analysis.
        """
        logger.info("Starting expert pattern analysis")

        # Load model
        self.load_model()

        # Collect prompts by category
        prompts = []
        for tool_prompts in self.config["prompts"]["tool_required"].values():
            prompts.extend(tool_prompts)
        prompts.extend(self.config["prompts"]["direct_answer"])

        logger.info(f"Analyzing {len(prompts)} prompts")

        # Analyze activations per tool category
        category_activations = {}
        for tool_name, tool_prompts in self.config["prompts"]["tool_required"].items():
            logger.info(f"Analyzing {tool_name}...")
            category_activations[tool_name] = self.analyze_category_activations(
                tool_prompts, tool_name
            )
            gc.collect()

        # Compute specialization
        specialization = self.compute_specialization_scores(category_activations)

        # Identify tool experts
        tool_experts = self.identify_tool_experts(specialization)

        # Summary statistics
        num_syntax_experts = sum(len(v) for v in tool_experts.values())

        all_concentrations = []
        for cat_stats in specialization.values():
            for stats in cat_stats.values():
                all_concentrations.append(stats["concentration"])

        avg_specialization = float(np.mean(all_concentrations)) if all_concentrations else 0.0

        results = {
            "specialization": specialization,
            "tool_experts": tool_experts,
            "num_syntax_experts": num_syntax_experts,
            "avg_specialization": avg_specialization,
            "num_prompts": len(prompts),
            "categories_analyzed": list(category_activations.keys()),
        }

        logger.info(f"Found {num_syntax_experts} tool-specialized experts")
        logger.info(f"Average specialization: {avg_specialization:.3f}")

        return results


def main():
    """Run as standalone script."""
    import yaml

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    analyzer = ExpertPatternAnalyzer(config)
    results = analyzer.run()

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
