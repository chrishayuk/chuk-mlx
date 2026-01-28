#!/usr/bin/env python3
"""
Generation Dynamics Analysis for Tool Calling

Analyzes how expert routing changes during tool call generation.

Research Question: How does expert routing evolve during tool call generation?
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


class GenerationDynamicsAnalyzer:
    """
    Analyzes expert routing dynamics during tool call generation.
    """

    DEFAULT_PHASES = {
        "intent": (0, 5),
        "selection": (5, 15),
        "format": (15, 30),
        "arguments": (30, 100),
    }

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model"]["primary"]
        self.max_tokens = config["generation_dynamics"]["max_tokens"]
        self.track_layers = config["generation_dynamics"]["track_layers"]
        self.phases = dict(self.DEFAULT_PHASES)

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
        """Get model components."""
        if hasattr(self.model, 'model'):
            embed_tokens = self.model.model.embed_tokens
            layers = self.model.model.layers
            norm = getattr(self.model.model, 'norm', None)
        else:
            embed_tokens = self.model.embed_tokens
            layers = self.model.layers
            norm = getattr(self.model, 'norm', None)

        lm_head = getattr(self.model, 'lm_head', None)
        return embed_tokens, layers, norm, lm_head

    def capture_routing_for_step(
        self,
        h: mx.array,
        layers: list,
        mask: mx.array
    ) -> dict[int, dict[str, Any]]:
        """
        Capture routing decisions at tracked layers for current hidden state.
        """
        routing = {}
        hidden_size = h.shape[-1]

        h_current = h
        for layer_idx, layer in enumerate(layers):
            if layer_idx in self.track_layers:
                if hasattr(layer, 'input_layernorm'):
                    normed = layer.input_layernorm(h_current)
                else:
                    normed = h_current

                mlp = layer.mlp
                if hasattr(mlp, 'router'):
                    x_flat = normed[:, -1, :].reshape(-1, hidden_size)  # Last position only
                    router_out = mlp.router(x_flat)

                    if isinstance(router_out, tuple):
                        weights, indices = router_out
                    else:
                        weights = mx.softmax(router_out, axis=-1)
                        indices = mx.argsort(router_out, axis=-1)[:, -self.top_k:]
                        weights = mx.take_along_axis(weights, indices, axis=-1)

                    mx.eval(indices)
                    mx.eval(weights)

                    routing[layer_idx] = {
                        "experts": indices[0].tolist() if indices.ndim > 1 else indices.tolist(),
                        "weights": [float(w) for w in weights[0].tolist()] if weights.ndim > 1 else [float(w) for w in weights.tolist()],
                    }

            h_current = layer(h_current, mask=mask)

        return routing

    def generate_with_routing_trace(
        self,
        prompt: str,
        max_tokens: int = None
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Generate text while capturing expert routing at each step.
        """
        max_tokens = max_tokens or self.max_tokens
        embed_tokens, layers, norm, lm_head = self.get_model_components()

        tokens = self.tokenizer.encode(prompt)
        routing_trace = []
        generated_tokens = []

        for step in range(max_tokens):
            input_ids = mx.array([tokens + generated_tokens])
            h = embed_tokens(input_ids)
            seq_len = h.shape[1]
            mask = mx.triu(mx.full((seq_len, seq_len), float('-inf'), dtype=h.dtype), k=1)

            # Capture routing
            step_routing = self.capture_routing_for_step(h, layers, mask)

            # Get next token prediction
            for layer in layers:
                h = layer(h, mask=mask)

            if norm is not None:
                h = norm(h)

            if lm_head is not None:
                logits = lm_head(h)
            else:
                logits = h @ embed_tokens.weight.T

            mx.eval(logits)

            # Greedy decode
            next_token = int(mx.argmax(logits[0, -1, :]))

            # Check for EOS
            eos_id = getattr(self.tokenizer, 'eos_token_id', None)
            if eos_id is not None and next_token == eos_id:
                break

            generated_tokens.append(next_token)

            try:
                token_text = self.tokenizer.decode([next_token])
            except Exception:
                token_text = f"<{next_token}>"

            step_routing["token_id"] = next_token
            step_routing["token_text"] = token_text
            step_routing["step"] = step
            step_routing["layers"] = step_routing.copy()
            routing_trace.append(step_routing)

        generated_text = self.tokenizer.decode(generated_tokens)
        return generated_text, routing_trace

    def classify_token_phase(self, step: int) -> str:
        """Classify which phase a token belongs to."""
        for phase_name, (start, end) in self.phases.items():
            if start <= step < end:
                return phase_name
        return "arguments"

    def analyze_phase_transitions(
        self,
        routing_trace: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Analyze how expert routing changes across phases.
        """
        phase_experts = defaultdict(lambda: defaultdict(list))

        for step_info in routing_trace:
            phase = self.classify_token_phase(step_info.get("step", 0))
            for layer, routing in step_info.get("layers", {}).items():
                if isinstance(routing, dict) and "experts" in routing:
                    experts = routing["experts"]
                    if isinstance(experts, list):
                        phase_experts[phase][layer].extend(experts)

        phase_stats = {}
        for phase, layer_experts in phase_experts.items():
            phase_stats[phase] = {}
            for layer, experts in layer_experts.items():
                if not experts:
                    continue

                unique, counts = np.unique(experts, return_counts=True)
                sorted_idx = np.argsort(-counts)
                top_experts = [
                    {"expert": int(unique[i]), "count": int(counts[i])}
                    for i in sorted_idx[:5]
                ]

                probs = counts / counts.sum()
                concentration = float(np.sum(probs ** 2))

                phase_stats[phase][layer] = {
                    "top_experts": top_experts,
                    "concentration": concentration,
                    "num_tokens": len(experts) // max(self.top_k, 1),
                }

        return phase_stats

    def compute_expert_consistency(
        self,
        routing_trace: list[dict[str, Any]]
    ) -> dict[int, float]:
        """
        Compute how consistent expert selection is across generation.
        """
        consistency = {}

        for layer in self.track_layers:
            experts_per_step = []
            for step_info in routing_trace:
                layers_info = step_info.get("layers", {})
                if layer in layers_info:
                    routing = layers_info[layer]
                    if isinstance(routing, dict) and "experts" in routing:
                        experts_per_step.append(set(routing["experts"]))

            if len(experts_per_step) < 2:
                continue

            similarities = []
            for i in range(len(experts_per_step) - 1):
                a, b = experts_per_step[i], experts_per_step[i + 1]
                if a or b:
                    jaccard = len(a & b) / len(a | b)
                    similarities.append(jaccard)

            consistency[layer] = float(np.mean(similarities)) if similarities else 0.0

        return consistency

    def run(self) -> dict[str, Any]:
        """
        Run the full generation dynamics analysis.
        """
        logger.info("Starting generation dynamics analysis")

        # Load model
        self.load_model()

        results = {
            "prompts": [],
            "overall_consistency": {},
            "phases_detected": 0,
        }

        all_consistency = defaultdict(list)

        # Analyze a few prompts from each tool
        for tool_name, prompts in self.config["prompts"]["tool_required"].items():
            for prompt in prompts[:2]:
                logger.info(f"Analyzing: {prompt[:50]}...")

                try:
                    generated, trace = self.generate_with_routing_trace(prompt, max_tokens=30)
                    phase_stats = self.analyze_phase_transitions(trace)
                    consistency = self.compute_expert_consistency(trace)

                    prompt_result = {
                        "prompt": prompt,
                        "tool": tool_name,
                        "generated_text": generated[:100],
                        "num_tokens": len(trace),
                        "phase_statistics": phase_stats,
                        "expert_consistency": consistency,
                    }
                    results["prompts"].append(prompt_result)

                    for layer, score in consistency.items():
                        all_consistency[layer].append(score)

                except Exception as e:
                    logger.warning(f"Failed to analyze prompt: {e}")

                gc.collect()

        # Compute overall statistics
        results["overall_consistency"] = {
            layer: float(np.mean(scores))
            for layer, scores in all_consistency.items()
        }
        results["expert_consistency"] = float(
            np.mean(list(results["overall_consistency"].values()))
        ) if results["overall_consistency"] else 0.0

        logger.info(f"Analyzed {len(results['prompts'])} prompts")
        logger.info(f"Average consistency: {results['expert_consistency']:.1%}")

        return results


def main():
    """Run as standalone script."""
    import yaml

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    analyzer = GenerationDynamicsAnalyzer(config)
    results = analyzer.run()

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
