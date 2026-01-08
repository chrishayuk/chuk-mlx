"""Memory analysis service for CLI commands.

This module provides the MemoryAnalysisService class that handles
all business logic for memory analysis, keeping CLI commands thin.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ...datasets import FactType


class MemoryAnalysisConfig(BaseModel):
    """Configuration for memory analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    facts: list[dict[str, Any]] = Field(..., description="Facts to analyze")
    fact_type: FactType = Field(..., description="Type of facts")
    layer: int | None = Field(default=None, description="Target layer")
    layer_depth_ratio: float | None = Field(default=None, description="Layer depth ratio")
    top_k: int = Field(default=10, description="Top-k predictions")
    classify: bool = Field(default=False, description="Classify memorization levels")

    # Memorization thresholds
    memorized_prob_threshold: float = Field(default=0.1)
    partial_prob_threshold: float = Field(default=0.01)
    weak_prob_threshold: float = Field(default=0.001)
    memorized_rank: int = Field(default=1)
    partial_rank: int = Field(default=5)
    weak_rank: int = Field(default=15)


class MemoryAnalysisResult(BaseModel):
    """Result of memory analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(..., description="Model ID")
    fact_type: str = Field(..., description="Fact type analyzed")
    layer: int = Field(..., description="Layer analyzed")
    num_facts: int = Field(..., description="Number of facts")

    # Accuracy metrics
    top1_accuracy: int = Field(default=0)
    top5_accuracy: int = Field(default=0)
    not_found: int = Field(default=0)

    # Attractor analysis
    attractors: list[dict[str, Any]] = Field(default_factory=list)

    # Category accuracy
    category_accuracy: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Classification results
    memorized: list[dict[str, Any]] = Field(default_factory=list)
    partial: list[dict[str, Any]] = Field(default_factory=list)
    weak: list[dict[str, Any]] = Field(default_factory=list)
    not_memorized: list[dict[str, Any]] = Field(default_factory=list)

    # Raw results
    results: list[dict[str, Any]] = Field(default_factory=list)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            f"MEMORY STRUCTURE ANALYSIS: {self.fact_type}",
            f"{'=' * 70}",
            f"\n1. RETRIEVAL ACCURACY",
            f"   Top-1: {self.top1_accuracy}/{self.num_facts} ({100 * self.top1_accuracy / self.num_facts:.1f}%)",
            f"   Top-5: {self.top5_accuracy}/{self.num_facts} ({100 * self.top5_accuracy / self.num_facts:.1f}%)",
            f"   Not found: {self.not_found}/{self.num_facts} ({100 * self.not_found / self.num_facts:.1f}%)",
        ]

        if self.category_accuracy:
            lines.append("\n2. ACCURACY BY CATEGORY")
            for cat, metrics in sorted(self.category_accuracy.items()):
                lines.append(
                    f"   {cat}: {metrics['top1']}/{metrics['total']} top-1, "
                    f"avg_prob={metrics['avg_prob']:.3f}"
                )

        if self.attractors:
            lines.append("\n3. TOP ATTRACTOR NODES")
            for attr in self.attractors[:10]:
                lines.append(
                    f"   '{attr['answer']}': appears {attr['count']} times, "
                    f"avg_prob={attr['avg_prob']:.4f}"
                )

        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    def save_plot(self, path: str | Path) -> None:
        """Save analysis plot."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Plot 1: Accuracy summary
            ax = axes[0, 0]
            labels = ["Top-1", "Top-5", "Not Found"]
            values = [self.top1_accuracy, self.top5_accuracy, self.not_found]
            ax.bar(labels, values)
            ax.set_ylabel("Count")
            ax.set_title("Retrieval Accuracy")

            # Plot 2: Category accuracy
            if self.category_accuracy:
                ax = axes[0, 1]
                cats = sorted(self.category_accuracy.keys())
                accs = [self.category_accuracy[c]["top1"] / self.category_accuracy[c]["total"] * 100
                        for c in cats]
                ax.bar(cats, accs)
                ax.set_ylabel("Top-1 Accuracy (%)")
                ax.set_title("Accuracy by Category")
                ax.tick_params(axis="x", rotation=45)

            # Plot 3: Top attractors
            if self.attractors:
                ax = axes[1, 0]
                answers = [a["answer"] for a in self.attractors[:10]]
                counts = [a["count"] for a in self.attractors[:10]]
                ax.barh(answers, counts)
                ax.set_xlabel("Co-activation Count")
                ax.set_title("Top Attractor Nodes")

            plt.suptitle(f"Memory Analysis: {self.fact_type} @ Layer {self.layer}")
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()

        except ImportError:
            pass  # matplotlib not available


class MemoryAnalysisService:
    """Service class for memory analysis operations."""

    @classmethod
    async def analyze(cls, config: MemoryAnalysisConfig) -> MemoryAnalysisResult:
        """Analyze model's memory of facts.

        Args:
            config: Analysis configuration.

        Returns:
            MemoryAnalysisResult with analysis metrics.
        """
        from ...models_v2 import load_model

        # Load model
        load_result = load_model(config.model)
        model = load_result.model
        tokenizer = load_result.tokenizer
        model_config = load_result.config

        # Determine target layer
        num_layers = getattr(model_config, "num_hidden_layers", 32)
        if config.layer is not None:
            target_layer = config.layer
        elif config.layer_depth_ratio is not None:
            target_layer = int(num_layers * config.layer_depth_ratio)
        else:
            target_layer = int(num_layers * 0.8)

        # Get model components
        def get_layers():
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                return list(model.model.layers)
            return list(model.layers)

        def get_embed():
            if hasattr(model, "model"):
                return model.model.embed_tokens
            return model.embed_tokens

        def get_norm():
            if hasattr(model, "model") and hasattr(model.model, "norm"):
                return model.model.norm
            if hasattr(model, "norm"):
                return model.norm
            return None

        def get_lm_head():
            if hasattr(model, "lm_head"):
                return model.lm_head
            return None

        def get_predictions_at_layer(prompt: str, layer: int, k: int) -> list:
            """Get top-k predictions at specific layer."""
            input_ids = mx.array(tokenizer.encode(prompt))[None, :]
            layers = get_layers()
            embed = get_embed()
            norm = get_norm()
            lm_head = get_lm_head()
            scale = getattr(model_config, "embedding_scale", None)

            h = embed(input_ids)
            if scale:
                h = h * scale

            seq_len = input_ids.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

            for idx, lyr in enumerate(layers):
                try:
                    out = lyr(h, mask=mask)
                except TypeError:
                    out = lyr(h)
                h = (
                    out.hidden_states
                    if hasattr(out, "hidden_states")
                    else (out[0] if isinstance(out, tuple) else out)
                )
                if idx == layer:
                    break

            if norm is not None:
                h = norm(h)
            if lm_head is not None:
                outputs = lm_head(h)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
            else:
                logits = h @ embed.weight.T

            probs = mx.softmax(logits[0, -1, :], axis=-1)
            top_indices = mx.argsort(probs)[-k:][::-1]
            top_probs = probs[top_indices]

            predictions = []
            for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
                token = tokenizer.decode([idx])
                predictions.append({
                    "token": token,
                    "token_id": idx,
                    "prob": prob,
                })
            return predictions

        # Build answer vocabulary
        answer_vocab = {fact["answer"]: fact for fact in config.facts}

        # Analyze each fact
        results = []
        for fact in config.facts:
            query = fact["query"]
            correct_answer = fact["answer"]

            predictions = get_predictions_at_layer(query, target_layer, config.top_k)

            # Find correct answer rank
            correct_rank = None
            correct_prob = None
            for j, pred in enumerate(predictions):
                if pred["token"].strip() == correct_answer or correct_answer in pred["token"]:
                    correct_rank = j + 1
                    correct_prob = pred["prob"]
                    break

            # Categorize predictions
            neighborhood = {
                "correct_rank": correct_rank,
                "correct_prob": correct_prob,
                "same_category": [],
                "other_answers": [],
            }

            for pred in predictions:
                token = pred["token"].strip()
                if token == correct_answer:
                    continue
                if token in answer_vocab:
                    other_fact = answer_vocab[token]
                    if "category" in fact and fact.get("category") == other_fact.get("category"):
                        neighborhood["same_category"].append({
                            "answer": token,
                            "prob": pred["prob"],
                        })
                    else:
                        neighborhood["other_answers"].append({
                            "answer": token,
                            "prob": pred["prob"],
                        })

            results.append({
                **fact,
                "predictions": predictions[:10],
                "neighborhood": neighborhood,
            })

        # Compute metrics
        top1 = sum(1 for r in results if r["neighborhood"]["correct_rank"] == 1)
        top5 = sum(
            1 for r in results
            if r["neighborhood"]["correct_rank"] and r["neighborhood"]["correct_rank"] <= 5
        )
        not_found = sum(1 for r in results if r["neighborhood"]["correct_rank"] is None)

        # Category accuracy
        category_accuracy = {}
        if "category" in config.facts[0]:
            categories = list({f["category"] for f in config.facts})
            for cat in categories:
                cat_facts = [r for r in results if r.get("category") == cat]
                cat_top1 = sum(1 for r in cat_facts if r["neighborhood"]["correct_rank"] == 1)
                cat_avg_prob = np.mean([r["neighborhood"]["correct_prob"] or 0 for r in cat_facts])
                category_accuracy[cat] = {
                    "top1": cat_top1,
                    "total": len(cat_facts),
                    "avg_prob": float(cat_avg_prob),
                }

        # Attractor analysis
        answer_counts: dict[str, int] = defaultdict(int)
        answer_probs: dict[str, list[float]] = defaultdict(list)
        for r in results:
            for cat in ["same_category", "other_answers"]:
                for item in r["neighborhood"].get(cat, []):
                    answer_counts[item["answer"]] += 1
                    answer_probs[item["answer"]].append(item["prob"])

        attractors = [
            {
                "answer": answer,
                "count": count,
                "avg_prob": float(np.mean(answer_probs[answer])),
            }
            for answer, count in sorted(answer_counts.items(), key=lambda x: -x[1])[:20]
        ]

        return MemoryAnalysisResult(
            model_id=config.model,
            fact_type=config.fact_type.value,
            layer=target_layer,
            num_facts=len(config.facts),
            top1_accuracy=top1,
            top5_accuracy=top5,
            not_found=not_found,
            attractors=attractors,
            category_accuracy=category_accuracy,
            results=results,
        )


__all__ = [
    "MemoryAnalysisConfig",
    "MemoryAnalysisResult",
    "MemoryAnalysisService",
]
