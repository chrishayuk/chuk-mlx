#!/usr/bin/env python3
"""
Vocab Alignment Analysis for Tool Names

Tests whether tool names are represented in vocabulary space at intermediate layers.

Research Question: Are tool names vocab-aligned at intermediate layers?
"""

import gc
import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


class VocabAlignmentAnalyzer:
    """
    Analyzes vocabulary alignment for tool names.
    """

    TOOL_TOKENS = {
        "calculator": ["calculator", "calculate", "compute", "math", "arithmetic"],
        "search": ["search", "find", "lookup", "query", "retrieve"],
        "code_exec": ["execute", "run", "code", "python", "script"],
        "get_weather": ["weather", "temperature", "forecast", "climate"],
    }

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model"]["primary"]
        self.test_layers = config["vocab_alignment"]["test_layers"]
        self.top_k_tokens = config["vocab_alignment"]["top_k_tokens"]
        self.apply_layer_norm = config["vocab_alignment"]["apply_layer_norm"]

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and tokenizer."""
        from mlx_lm import load

        logger.info(f"Loading model: {self.model_name}")
        self.model, self.tokenizer = load(self.model_name)
        logger.info("Model loaded successfully")

    def get_model_components(self):
        """Get model layers, embedding and lm_head."""
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

    def get_hidden_state(
        self,
        prompt: str,
        layer: int
    ) -> mx.array:
        """
        Get hidden state at a specific layer for a prompt.
        """
        embed_tokens, layers, norm, _ = self.get_model_components()

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        h = embed_tokens(input_ids)
        batch_size, seq_len, hidden_dim = h.shape
        mask = mx.triu(mx.full((seq_len, seq_len), float('-inf'), dtype=h.dtype), k=1)

        for i, layer_module in enumerate(layers):
            h = layer_module(h, mask=mask)
            if i == layer:
                break

        mx.eval(h)
        return h[0, -1, :]  # Last token

    def project_to_vocab(
        self,
        hidden_state: mx.array
    ) -> tuple[mx.array, mx.array]:
        """
        Project hidden state to vocabulary space.
        """
        _, _, norm, lm_head = self.get_model_components()

        # Apply layer normalization if configured
        if self.apply_layer_norm and norm is not None:
            hidden_state = norm(hidden_state.reshape(1, -1))
            hidden_state = hidden_state.reshape(-1)

        # Project to vocabulary
        if lm_head is not None:
            logits = lm_head(hidden_state.reshape(1, -1))
            logits = logits.reshape(-1)
        else:
            # Fallback: use embed_tokens weight transposed
            embed_tokens, _, _, _ = self.get_model_components()
            logits = hidden_state @ embed_tokens.weight.T

        probs = mx.softmax(logits)
        mx.eval(probs)
        return logits, probs

    def find_token_ranks(
        self,
        probs: mx.array,
        target_tokens: list[str]
    ) -> dict[str, dict[str, Any]]:
        """
        Find the rank and probability of target tokens.
        """
        results = {}

        probs_np = np.array(probs.astype(mx.float32))
        sorted_indices = np.argsort(-probs_np)

        for token in target_tokens:
            variants = [token, f" {token}", f"{token}"]
            best_rank = float("inf")
            best_prob = 0.0
            best_variant = token

            for variant in variants:
                try:
                    token_ids = self.tokenizer.encode(variant)
                    for tid in token_ids:
                        rank = int(np.where(sorted_indices == tid)[0][0]) if tid in sorted_indices else -1
                        if rank >= 0:
                            prob = float(probs_np[tid])
                            if rank < best_rank:
                                best_rank = rank
                                best_prob = prob
                                best_variant = self.tokenizer.decode([tid])
                except Exception:
                    continue

            results[token] = {
                "rank": int(best_rank) if best_rank < float("inf") else -1,
                "probability": best_prob,
                "token_text": best_variant,
                "in_top_k": best_rank < self.top_k_tokens,
            }

        return results

    def get_top_tokens(
        self,
        probs: mx.array,
        k: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get the top-k tokens by probability.
        """
        probs_np = np.array(probs.astype(mx.float32))
        top_k_indices = np.argsort(-probs_np)[:k]

        results = []
        for rank, idx in enumerate(top_k_indices):
            prob = float(probs_np[idx])
            try:
                token_text = self.tokenizer.decode([int(idx)])
            except Exception:
                token_text = f"<token_{idx}>"
            results.append({
                "rank": rank + 1,
                "token_id": int(idx),
                "token_text": token_text,
                "probability": prob,
            })

        return results

    def analyze_prompt(
        self,
        prompt: str,
        expected_tool: str,
        layer: int
    ) -> dict[str, Any]:
        """
        Analyze vocab alignment for a single prompt.
        """
        hidden_state = self.get_hidden_state(prompt, layer)
        _, probs = self.project_to_vocab(hidden_state)

        # Check alignment for expected tool tokens
        tool_tokens = self.TOOL_TOKENS.get(expected_tool, [])
        token_ranks = self.find_token_ranks(probs, tool_tokens)

        # Get top tokens
        top_tokens = self.get_top_tokens(probs, k=10)

        # Check if any tool token is in top-k
        aligned = any(info["in_top_k"] for info in token_ranks.values())
        best_rank = min(
            (info["rank"] for info in token_ranks.values() if info["rank"] >= 0),
            default=-1
        )

        return {
            "prompt": prompt,
            "expected_tool": expected_tool,
            "layer": layer,
            "aligned": aligned,
            "best_tool_token_rank": best_rank,
            "token_ranks": token_ranks,
            "top_10_tokens": top_tokens,
        }

    def run(self) -> dict[str, Any]:
        """
        Run the full vocab alignment analysis.
        """
        logger.info("Starting vocab alignment analysis")

        # Load model
        self.load_model()

        results = {
            "layers": {},
            "per_tool_summary": {},
            "test_layers": self.test_layers,
        }

        # Analyze each layer
        for layer in self.test_layers:
            logger.info(f"Analyzing layer {layer}...")
            layer_results = []

            for tool_name, prompts in self.config["prompts"]["tool_required"].items():
                for prompt in prompts[:3]:
                    try:
                        analysis = self.analyze_prompt(prompt, tool_name, layer)
                        layer_results.append(analysis)
                    except Exception as e:
                        logger.warning(f"Failed to analyze prompt: {e}")

            results["layers"][layer] = layer_results

            aligned_count = sum(1 for r in layer_results if r["aligned"])
            logger.info(f"Layer {layer}: {aligned_count}/{len(layer_results)} aligned")

            gc.collect()

        # Per-tool summary
        tool_names = list(self.config["prompts"]["tool_required"].keys())
        for tool_name in tool_names:
            tool_results = []
            for layer, layer_results in results["layers"].items():
                for r in layer_results:
                    if r["expected_tool"] == tool_name:
                        tool_results.append(r)

            if tool_results:
                aligned_count = sum(1 for r in tool_results if r["aligned"])
                best_ranks = [
                    r["best_tool_token_rank"]
                    for r in tool_results
                    if r["best_tool_token_rank"] >= 0
                ]
                results["per_tool_summary"][tool_name] = {
                    "aligned_fraction": aligned_count / len(tool_results),
                    "avg_best_rank": float(np.mean(best_ranks)) if best_ranks else -1,
                    "num_samples": len(tool_results),
                }

        # Overall summary
        all_aligned = sum(
            1 for layer_results in results["layers"].values()
            for r in layer_results if r["aligned"]
        )
        total_samples = sum(len(lr) for lr in results["layers"].values())

        results["num_aligned"] = all_aligned
        results["num_tools"] = len(tool_names)
        results["total_samples"] = total_samples

        if results["per_tool_summary"]:
            best_tool = max(
                results["per_tool_summary"].items(),
                key=lambda x: x[1]["aligned_fraction"]
            )
            results["best_tool"] = best_tool[0]
            results["best_rank"] = int(best_tool[1]["avg_best_rank"])

        logger.info(f"Overall: {all_aligned}/{total_samples} aligned")

        return results


def main():
    """Run as standalone script."""
    import yaml

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    analyzer = VocabAlignmentAnalyzer(config)
    results = analyzer.run()

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
