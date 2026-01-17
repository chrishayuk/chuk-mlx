"""Attention-based routing prediction service.

Predicts MoE expert routing from attention patterns alone,
without running the actual router. This enables:
- Understanding what attention patterns drive routing decisions
- Predicting routing for prefetch/speculation
- Analyzing the relationship between attention and expert selection
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .expert_router import ExpertRouter

logger = logging.getLogger(__name__)


class AttentionFeatures(BaseModel):
    """Features extracted from attention patterns for prediction."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    position: int = Field(ge=0)
    self_attention: float = Field(description="Self-attention weight")
    entropy: float = Field(description="Attention entropy")
    top_k_concentration: float = Field(description="Concentration in top-k tokens")
    recent_window_mass: float = Field(description="Attention mass in recent window")
    early_token_mass: float = Field(description="Attention mass to early tokens")


class RoutingPrediction(BaseModel):
    """Predicted routing from attention patterns."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    position: int = Field(ge=0)
    predicted_experts: tuple[int, ...] = Field(description="Predicted expert indices")
    predicted_weights: tuple[float, ...] = Field(description="Predicted expert weights")
    confidence: float = Field(ge=0, le=1, description="Prediction confidence")
    attention_features: AttentionFeatures = Field(description="Features used")


class PredictionEvaluation(BaseModel):
    """Evaluation of routing predictions vs actuals."""

    model_config = ConfigDict(frozen=True)

    num_predictions: int = Field(ge=0)
    top1_accuracy: float = Field(ge=0, le=1, description="Top-1 expert match rate")
    topk_overlap: float = Field(ge=0, le=1, description="Avg overlap in top-k experts")
    weight_correlation: float = Field(ge=-1, le=1, description="Weight correlation")
    layer_accuracies: dict[int, float] = Field(description="Accuracy by layer")


class AttentionRoutingCorrelation(BaseModel):
    """Correlation between attention features and routing."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    self_attention_vs_expert: dict[int, float] = Field(
        description="Correlation of self-attention with each expert's activation"
    )
    entropy_vs_diversity: float = Field(
        description="Correlation between attention entropy and expert diversity"
    )
    dominant_attention_pattern: str = Field(
        description="Description of dominant attention pattern for this layer"
    )


class AttentionPredictionAnalysis(BaseModel):
    """Complete attention-based prediction analysis."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(description="Model identifier")
    num_layers_analyzed: int = Field(ge=0)
    num_prompts: int = Field(ge=0)
    evaluation: PredictionEvaluation = Field(description="Prediction evaluation")
    correlations: tuple[AttentionRoutingCorrelation, ...] = Field(
        description="Per-layer correlations"
    )
    predictability_score: float = Field(
        ge=0, le=1, description="Overall predictability from attention"
    )


@dataclass
class AttentionPredictionService:
    """Service for predicting routing from attention patterns."""

    router: ExpertRouter
    _learned_mappings: dict[int, np.ndarray] = field(default_factory=dict)

    async def extract_attention_features(
        self,
        prompt: str,
        layer_idx: int,
    ) -> list[AttentionFeatures]:
        """Extract attention features for all positions in a prompt.

        Args:
            prompt: Input prompt
            layer_idx: Layer to analyze

        Returns:
            List of AttentionFeatures per position
        """
        from .attention_routing_service import AttentionRoutingService

        # Capture attention weights
        result = AttentionRoutingService.capture_attention_weights(
            self.router, prompt, layer_idx
        )

        if result.attention_weights is None:
            return []

        features = []
        attn_weights = result.attention_weights  # (num_heads, seq_len, seq_len)
        seq_len = attn_weights.shape[-1]

        # Average across heads
        avg_attn = mx.mean(attn_weights, axis=0)  # (seq_len, seq_len)

        for pos in range(seq_len):
            pos_attn = avg_attn[pos, : pos + 1]  # Only valid positions (causal)

            # Self-attention
            self_attn = float(pos_attn[pos]) if pos < len(pos_attn) else 0.0

            # Entropy
            pos_attn_np = np.array(pos_attn.tolist())
            pos_attn_np = pos_attn_np + 1e-10  # Avoid log(0)
            entropy = float(-np.sum(pos_attn_np * np.log(pos_attn_np)))

            # Top-k concentration (top 3)
            sorted_attn = np.sort(pos_attn_np)[::-1]
            top_k = min(3, len(sorted_attn))
            top_k_concentration = float(np.sum(sorted_attn[:top_k]))

            # Recent window mass (last 5 tokens)
            recent_window = min(5, pos + 1)
            recent_mass = float(np.sum(pos_attn_np[-recent_window:]))

            # Early token mass (first 3 tokens)
            early_tokens = min(3, len(pos_attn_np))
            early_mass = float(np.sum(pos_attn_np[:early_tokens]))

            features.append(
                AttentionFeatures(
                    layer_idx=layer_idx,
                    position=pos,
                    self_attention=self_attn,
                    entropy=entropy,
                    top_k_concentration=top_k_concentration,
                    recent_window_mass=recent_mass,
                    early_token_mass=early_mass,
                )
            )

        return features

    async def predict_routing(
        self,
        prompt: str,
        layer_idx: int,
        use_learned: bool = True,
    ) -> list[RoutingPrediction]:
        """Predict routing from attention patterns.

        Args:
            prompt: Input prompt
            layer_idx: Layer to predict for
            use_learned: Whether to use learned mappings if available

        Returns:
            List of RoutingPrediction per position
        """
        features = await self.extract_attention_features(prompt, layer_idx)
        if not features:
            return []

        num_experts = self.router.info.num_experts
        k = self.router.info.num_experts_per_tok

        predictions = []
        for feat in features:
            # Simple heuristic-based prediction:
            # High self-attention -> specialist experts
            # High entropy -> generalist experts
            # High early token mass -> position-encoding experts

            # Score each expert based on attention features
            expert_scores = np.zeros(num_experts)

            if use_learned and layer_idx in self._learned_mappings:
                # Use learned mapping
                feat_vec = np.array([
                    feat.self_attention,
                    feat.entropy,
                    feat.top_k_concentration,
                    feat.recent_window_mass,
                    feat.early_token_mass,
                ])
                expert_scores = self._learned_mappings[layer_idx] @ feat_vec
            else:
                # Heuristic: distribute based on attention entropy
                # Lower entropy -> prefer lower-indexed experts (often specialists)
                # Higher entropy -> prefer higher-indexed experts (often generalists)
                normalized_entropy = min(feat.entropy / 3.0, 1.0)
                for i in range(num_experts):
                    base_score = 1.0 - abs(i / num_experts - normalized_entropy)
                    # Boost by self-attention for certain experts
                    if feat.self_attention > 0.3:
                        base_score *= (1 + feat.self_attention) if i < num_experts // 2 else 1.0
                    expert_scores[i] = base_score

            # Softmax to get weights
            exp_scores = np.exp(expert_scores - np.max(expert_scores))
            probs = exp_scores / np.sum(exp_scores)

            # Select top-k
            top_k_idx = np.argsort(probs)[-k:][::-1]
            top_k_weights = probs[top_k_idx]
            top_k_weights = top_k_weights / np.sum(top_k_weights)

            # Confidence based on how peaked the distribution is
            confidence = float(np.max(probs) - np.mean(probs))

            predictions.append(
                RoutingPrediction(
                    layer_idx=layer_idx,
                    position=feat.position,
                    predicted_experts=tuple(int(i) for i in top_k_idx),
                    predicted_weights=tuple(float(w) for w in top_k_weights),
                    confidence=confidence,
                    attention_features=feat,
                )
            )

        return predictions

    async def learn_mapping(
        self,
        prompts: list[str],
        layer_idx: int,
    ) -> None:
        """Learn a mapping from attention features to routing.

        Uses simple linear regression to learn the relationship.

        Args:
            prompts: Training prompts
            layer_idx: Layer to learn mapping for
        """
        all_features = []
        all_targets = []

        for prompt in prompts:
            # Get attention features
            features = await self.extract_attention_features(prompt, layer_idx)

            # Get actual routing
            weights_list = await self.router.capture_router_weights(prompt)
            layer_weights = [w for w in weights_list if w.layer_idx == layer_idx]

            if not layer_weights or not features:
                continue

            lw = layer_weights[0]
            for pos_idx, (feat, pos) in enumerate(zip(features, lw.positions)):
                if pos_idx >= len(lw.positions):
                    break

                feat_vec = np.array([
                    feat.self_attention,
                    feat.entropy,
                    feat.top_k_concentration,
                    feat.recent_window_mass,
                    feat.early_token_mass,
                ])

                # Target: one-hot for top expert
                target = np.zeros(self.router.info.num_experts)
                for exp_idx, weight in zip(pos.expert_indices, pos.weights):
                    if 0 <= exp_idx < len(target):
                        target[exp_idx] = weight

                all_features.append(feat_vec)
                all_targets.append(target)

        if len(all_features) < 10:
            logger.warning(f"Insufficient data for learning ({len(all_features)} samples)")
            return

        # Simple least squares
        X = np.stack(all_features)  # (N, 5)
        Y = np.stack(all_targets)  # (N, num_experts)

        # Solve: Y = X @ W^T => W = (X^T X)^-1 X^T Y
        XtX = X.T @ X + 0.01 * np.eye(X.shape[1])  # Regularization
        XtY = X.T @ Y
        W = np.linalg.solve(XtX, XtY).T  # (num_experts, 5)

        self._learned_mappings[layer_idx] = W
        logger.info(f"Learned attention->routing mapping for layer {layer_idx}")

    async def evaluate_predictions(
        self,
        prompts: list[str],
        layers: list[int] | None = None,
    ) -> PredictionEvaluation:
        """Evaluate prediction accuracy against actual routing.

        Args:
            prompts: Prompts to evaluate on
            layers: Layers to evaluate (None for all MoE layers)

        Returns:
            PredictionEvaluation with accuracy metrics
        """
        if layers is None:
            layers = list(self.router.info.moe_layers)

        total_predictions = 0
        top1_matches = 0
        topk_overlaps = []
        weight_correlations = []
        layer_matches: dict[int, list[bool]] = defaultdict(list)

        for prompt in prompts:
            # Get actual routing
            weights_list = await self.router.capture_router_weights(prompt)

            for layer_idx in layers:
                # Get predictions
                predictions = await self.predict_routing(prompt, layer_idx, use_learned=True)

                # Get actual
                actuals = [w for w in weights_list if w.layer_idx == layer_idx]
                if not actuals or not predictions:
                    continue

                actual_layer = actuals[0]

                for pred, actual_pos in zip(predictions, actual_layer.positions):
                    total_predictions += 1

                    # Top-1 accuracy
                    actual_top1 = actual_pos.expert_indices[0] if actual_pos.expert_indices else -1
                    pred_top1 = pred.predicted_experts[0] if pred.predicted_experts else -1
                    is_match = actual_top1 == pred_top1
                    top1_matches += int(is_match)
                    layer_matches[layer_idx].append(is_match)

                    # Top-k overlap
                    actual_set = set(actual_pos.expert_indices)
                    pred_set = set(pred.predicted_experts)
                    if actual_set and pred_set:
                        overlap = len(actual_set & pred_set) / len(actual_set | pred_set)
                        topk_overlaps.append(overlap)

                    # Weight correlation
                    if len(actual_pos.weights) > 1 and len(pred.predicted_weights) > 1:
                        # Align by expert index
                        common = actual_set & pred_set
                        if len(common) > 1:
                            actual_w = []
                            pred_w = []
                            for exp in common:
                                if exp in actual_pos.expert_indices:
                                    idx = list(actual_pos.expert_indices).index(exp)
                                    actual_w.append(actual_pos.weights[idx])
                                if exp in pred.predicted_experts:
                                    idx = list(pred.predicted_experts).index(exp)
                                    pred_w.append(pred.predicted_weights[idx])
                            if len(actual_w) > 1:
                                corr = np.corrcoef(actual_w, pred_w)[0, 1]
                                if not np.isnan(corr):
                                    weight_correlations.append(corr)

        # Compute layer accuracies
        layer_accuracies = {
            layer: np.mean(matches) if matches else 0.0
            for layer, matches in layer_matches.items()
        }

        return PredictionEvaluation(
            num_predictions=total_predictions,
            top1_accuracy=top1_matches / total_predictions if total_predictions else 0.0,
            topk_overlap=np.mean(topk_overlaps) if topk_overlaps else 0.0,
            weight_correlation=np.mean(weight_correlations) if weight_correlations else 0.0,
            layer_accuracies=layer_accuracies,
        )

    async def analyze_correlations(
        self,
        prompts: list[str],
        layer_idx: int,
    ) -> AttentionRoutingCorrelation:
        """Analyze correlation between attention features and expert activation.

        Args:
            prompts: Prompts to analyze
            layer_idx: Layer to analyze

        Returns:
            AttentionRoutingCorrelation with correlation stats
        """
        self_attns = []
        entropies = []
        expert_activations: dict[int, list[float]] = defaultdict(list)

        for prompt in prompts:
            features = await self.extract_attention_features(prompt, layer_idx)
            weights_list = await self.router.capture_router_weights(prompt)

            layer_weights = [w for w in weights_list if w.layer_idx == layer_idx]
            if not layer_weights or not features:
                continue

            for feat, pos in zip(features, layer_weights[0].positions):
                self_attns.append(feat.self_attention)
                entropies.append(feat.entropy)

                for exp_idx, weight in zip(pos.expert_indices, pos.weights):
                    expert_activations[exp_idx].append(weight)

        # Compute correlations
        self_attn_vs_expert = {}
        for exp_idx, activations in expert_activations.items():
            if len(activations) >= 10 and len(self_attns) >= 10:
                # Pad or truncate to match lengths
                min_len = min(len(self_attns), len(activations))
                corr = np.corrcoef(self_attns[:min_len], activations[:min_len])[0, 1]
                if not np.isnan(corr):
                    self_attn_vs_expert[exp_idx] = float(corr)

        # Entropy vs expert diversity (using entropy of expert distribution)
        entropy_vs_diversity = 0.0
        # TODO: compute proper expert diversity correlation

        # Determine dominant pattern
        if np.mean(self_attns) > 0.3:
            dominant_pattern = "High self-attention (local processing)"
        elif np.mean(entropies) > 2.0:
            dominant_pattern = "High entropy (distributed attention)"
        else:
            dominant_pattern = "Mixed attention patterns"

        return AttentionRoutingCorrelation(
            layer_idx=layer_idx,
            self_attention_vs_expert=self_attn_vs_expert,
            entropy_vs_diversity=entropy_vs_diversity,
            dominant_attention_pattern=dominant_pattern,
        )

    async def analyze(
        self,
        prompts: list[str],
        layers: list[int] | None = None,
        learn_mappings: bool = True,
        model_id: str = "unknown",
    ) -> AttentionPredictionAnalysis:
        """Complete attention-based prediction analysis.

        Args:
            prompts: Prompts to analyze
            layers: Layers to analyze (None for subset)
            learn_mappings: Whether to learn mappings before evaluation

        Returns:
            AttentionPredictionAnalysis with full results
        """
        if layers is None:
            moe_layers = list(self.router.info.moe_layers)
            # Sample 3 layers: early, middle, late
            if len(moe_layers) >= 3:
                layers = [moe_layers[0], moe_layers[len(moe_layers) // 2], moe_layers[-1]]
            else:
                layers = moe_layers

        # Optionally learn mappings
        if learn_mappings:
            print("Learning attention->routing mappings...")
            train_prompts = prompts[: len(prompts) // 2]  # Use half for training
            for layer_idx in layers:
                await self.learn_mapping(train_prompts, layer_idx)

        # Evaluate
        print("Evaluating prediction accuracy...")
        eval_prompts = prompts[len(prompts) // 2:] if learn_mappings else prompts
        evaluation = await self.evaluate_predictions(eval_prompts, layers)

        # Analyze correlations
        print("Analyzing attention-routing correlations...")
        correlations = []
        for layer_idx in layers:
            corr = await self.analyze_correlations(prompts, layer_idx)
            correlations.append(corr)

        # Overall predictability score
        predictability = (
            evaluation.top1_accuracy * 0.5
            + evaluation.topk_overlap * 0.3
            + max(0, evaluation.weight_correlation) * 0.2
        )

        return AttentionPredictionAnalysis(
            model_id=model_id,
            num_layers_analyzed=len(layers),
            num_prompts=len(prompts),
            evaluation=evaluation,
            correlations=tuple(correlations),
            predictability_score=predictability,
        )


def print_prediction_analysis(analysis: AttentionPredictionAnalysis) -> None:
    """Print attention prediction analysis results."""
    print("\n" + "=" * 70)
    print("ATTENTION-BASED ROUTING PREDICTION ANALYSIS")
    print("=" * 70)

    print(f"\nModel: {analysis.model_id}")
    print(f"Layers analyzed: {analysis.num_layers_analyzed}")
    print(f"Prompts: {analysis.num_prompts}")

    print("\n" + "-" * 70)
    print("PREDICTION ACCURACY")
    print("-" * 70)

    eval = analysis.evaluation
    print(f"Total predictions: {eval.num_predictions}")
    print(f"Top-1 accuracy: {eval.top1_accuracy:.1%}")
    print(f"Top-k overlap: {eval.topk_overlap:.1%}")
    print(f"Weight correlation: {eval.weight_correlation:.3f}")

    print("\n" + "-" * 70)
    print("ACCURACY BY LAYER")
    print("-" * 70)

    for layer_idx, acc in sorted(eval.layer_accuracies.items()):
        bar_len = int(acc * 20)
        bar = "*" * bar_len + "." * (20 - bar_len)
        print(f"Layer {layer_idx:2d}: {bar} {acc:.1%}")

    print("\n" + "-" * 70)
    print("ATTENTION-ROUTING CORRELATIONS")
    print("-" * 70)

    for corr in analysis.correlations:
        print(f"\nLayer {corr.layer_idx}: {corr.dominant_attention_pattern}")
        if corr.self_attention_vs_expert:
            top_corrs = sorted(
                corr.self_attention_vs_expert.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            for exp_idx, c in top_corrs:
                direction = "+" if c > 0 else ""
                print(f"  Expert {exp_idx}: self_attn corr = {direction}{c:.3f}")

    print("\n" + "-" * 70)
    print("OVERALL PREDICTABILITY")
    print("-" * 70)

    score = analysis.predictability_score
    bar_len = int(score * 40)
    bar = "*" * bar_len + "." * (40 - bar_len)
    print(f"Score: {bar} {score:.1%}")

    if score > 0.7:
        print("-> Routing is HIGHLY predictable from attention patterns")
    elif score > 0.4:
        print("-> Routing is MODERATELY predictable from attention patterns")
    else:
        print("-> Routing has LOW predictability from attention patterns")
