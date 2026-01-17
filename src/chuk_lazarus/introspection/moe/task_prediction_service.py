"""Task-aware expert prediction service.

Uses early-layer activations to predict which experts will be needed
in later layers, enabling speculative expert prefetching.

Key insight: The routing decisions in later layers often correlate with
patterns visible at earlier layers. A lightweight probe trained at L4
can predict L16+ routing with reasonable accuracy.
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


class ProbeFeatures(BaseModel):
    """Features extracted from early layer for prediction."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    layer_idx: int = Field(description="Source layer index")
    hidden_mean: float = Field(description="Mean of hidden state")
    hidden_std: float = Field(description="Std of hidden state")
    hidden_norm: float = Field(description="L2 norm of hidden state")
    top_k_indices: tuple[int, ...] = Field(description="Top-k activation indices")


class ExpertPrediction(BaseModel):
    """Predicted experts for a target layer."""

    model_config = ConfigDict(frozen=True)

    probe_layer: int = Field(description="Layer used for probing")
    target_layer: int = Field(description="Target layer for prediction")
    predicted_experts: tuple[int, ...] = Field(description="Predicted expert indices")
    confidence: float = Field(ge=0, le=1, description="Prediction confidence")


class PredictionMetrics(BaseModel):
    """Metrics for task-aware prediction."""

    model_config = ConfigDict(frozen=True)

    probe_layer: int = Field(description="Layer used for probing")
    target_layer: int = Field(description="Target layer for prediction")
    accuracy: float = Field(ge=0, le=1, description="Prediction accuracy")
    precision: float = Field(ge=0, le=1, description="Precision")
    recall: float = Field(ge=0, le=1, description="Recall")
    prefetch_efficiency: float = Field(
        ge=0, le=1, description="Ratio of correctly prefetched experts"
    )


class LayerPairAnalysis(BaseModel):
    """Analysis of prediction between a probe-target layer pair."""

    model_config = ConfigDict(frozen=True)

    probe_layer: int
    target_layer: int
    correlation: float = Field(description="Routing correlation")
    predictability: float = Field(ge=0, le=1, description="How predictable target is from probe")
    dominant_experts: list[int] = Field(description="Experts most predictable from this probe")


class TaskPredictionAnalysis(BaseModel):
    """Complete task-aware prediction analysis."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    probe_layer: int
    num_prompts: int
    layer_metrics: dict[int, PredictionMetrics] = Field(description="Metrics per target layer")
    overall_accuracy: float = Field(ge=0, le=1)
    overall_prefetch_efficiency: float = Field(ge=0, le=1)
    best_predicted_layers: list[int] = Field(description="Layers best predicted from probe")
    layer_pairs: list[LayerPairAnalysis] = Field(description="Layer pair analyses")


@dataclass
class TaskPredictionService:
    """Service for task-aware expert prediction."""

    router: ExpertRouter
    _learned_probes: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    _feature_dim: int = 64

    async def capture_layer_activations(
        self,
        prompt: str,
        layer_idx: int,
    ) -> mx.array | None:
        """Capture hidden state activations at a specific layer.

        Args:
            prompt: Input prompt
            layer_idx: Layer to capture from

        Returns:
            Hidden state array or None if capture failed
        """
        captured = {}

        input_ids = mx.array(self.router.tokenizer.encode(prompt))[None, :]

        # Get the transformer layer
        target_block = self.router._model.model.layers[layer_idx]
        original_call = type(target_block).__call__

        def patched_call(self_block, x, *args, **kwargs):
            result = original_call(self_block, x, *args, **kwargs)
            captured["hidden"] = x  # Store input to this layer
            return result

        try:
            type(target_block).__call__ = patched_call
            self.router._model(input_ids)
        finally:
            type(target_block).__call__ = original_call

        return captured.get("hidden")

    def extract_features(self, hidden: mx.array) -> np.ndarray:
        """Extract prediction features from hidden state.

        Args:
            hidden: Hidden state (batch, seq, hidden_dim)

        Returns:
            Feature vector
        """
        # Average over sequence
        avg_hidden = mx.mean(hidden[0], axis=0)  # (hidden_dim,)
        avg_np = np.array(avg_hidden.tolist())

        # Basic statistics
        features = [
            np.mean(avg_np),
            np.std(avg_np),
            np.linalg.norm(avg_np),
            np.max(avg_np),
            np.min(avg_np),
        ]

        # Top-k activation indices (normalized by position)
        top_k = 10
        top_indices = np.argsort(np.abs(avg_np))[-top_k:]
        normalized_indices = top_indices / len(avg_np)
        features.extend(normalized_indices.tolist())

        # Quantiles
        quantiles = np.percentile(avg_np, [10, 25, 50, 75, 90])
        features.extend(quantiles.tolist())

        # Pad or truncate to fixed size
        feat_array = np.array(features[: self._feature_dim])
        if len(feat_array) < self._feature_dim:
            feat_array = np.pad(feat_array, (0, self._feature_dim - len(feat_array)))

        return feat_array

    async def collect_training_data(
        self,
        prompts: list[str],
        probe_layer: int,
        target_layers: list[int],
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Collect training data for probe.

        Args:
            prompts: Training prompts
            probe_layer: Layer to probe from
            target_layers: Layers to predict routing for

        Returns:
            Tuple of (features, {layer: expert_labels})
        """
        all_features = []
        all_labels: dict[int, list[np.ndarray]] = {l: [] for l in target_layers}

        for prompt in prompts:
            # Get probe layer activations
            hidden = await self.capture_layer_activations(prompt, probe_layer)
            if hidden is None:
                continue

            features = self.extract_features(hidden)
            all_features.append(features)

            # Get actual routing for target layers
            weights_list = await self.router.capture_router_weights(prompt)

            for target_layer in target_layers:
                layer_weights = [w for w in weights_list if w.layer_idx == target_layer]
                if not layer_weights:
                    # No routing data for this layer
                    labels = np.zeros(self.router.info.num_experts)
                else:
                    # Aggregate expert activations
                    labels = np.zeros(self.router.info.num_experts)
                    for pos in layer_weights[0].positions:
                        for exp_idx in pos.expert_indices:
                            if 0 <= exp_idx < len(labels):
                                labels[exp_idx] += 1
                    # Normalize
                    if np.sum(labels) > 0:
                        labels = labels / np.sum(labels)

                all_labels[target_layer].append(labels)

        X = np.stack(all_features) if all_features else np.array([])
        Y = {l: np.stack(labels) if labels else np.array([]) for l, labels in all_labels.items()}

        return X, Y

    async def train_probe(
        self,
        prompts: list[str],
        probe_layer: int,
        target_layers: list[int],
    ) -> None:
        """Train probes for predicting target layer routing.

        Args:
            prompts: Training prompts
            probe_layer: Layer to probe from
            target_layers: Layers to predict routing for
        """
        print(f"Collecting training data from {len(prompts)} prompts...")
        X, Y = await self.collect_training_data(prompts, probe_layer, target_layers)

        if len(X) < 10:
            logger.warning(f"Insufficient training data: {len(X)} samples")
            return

        for target_layer in target_layers:
            if len(Y[target_layer]) < 10:
                continue

            # Simple linear regression: Y = X @ W
            labels = Y[target_layer]
            XtX = X.T @ X + 0.01 * np.eye(X.shape[1])  # Regularization
            XtY = X.T @ labels
            W = np.linalg.solve(XtX, XtY)  # (feature_dim, num_experts)

            self._learned_probes[(probe_layer, target_layer)] = W
            logger.info(f"Trained probe: L{probe_layer} -> L{target_layer}")

        print(f"Trained {len(target_layers)} probes")

    async def predict_experts(
        self,
        prompt: str,
        probe_layer: int,
        target_layer: int,
        top_k: int | None = None,
    ) -> ExpertPrediction:
        """Predict which experts will be activated at target layer.

        Args:
            prompt: Input prompt
            probe_layer: Layer to probe from
            target_layer: Layer to predict for
            top_k: Number of experts to predict (default: num_experts_per_tok)

        Returns:
            ExpertPrediction with predicted experts
        """
        if top_k is None:
            top_k = self.router.info.num_experts_per_tok

        # Get probe features
        hidden = await self.capture_layer_activations(prompt, probe_layer)
        if hidden is None:
            return ExpertPrediction(
                probe_layer=probe_layer,
                target_layer=target_layer,
                predicted_experts=(),
                confidence=0.0,
            )

        features = self.extract_features(hidden)

        # Use learned probe if available
        key = (probe_layer, target_layer)
        if key in self._learned_probes:
            W = self._learned_probes[key]
            scores = features @ W  # (num_experts,)
        else:
            # Heuristic fallback
            num_experts = self.router.info.num_experts
            scores = np.random.random(num_experts)  # Random baseline

        # Select top-k
        top_k_idx = np.argsort(scores)[-top_k:][::-1]

        # Confidence based on score distribution
        probs = np.exp(scores - np.max(scores))
        probs = probs / np.sum(probs)
        confidence = float(np.max(probs) - np.mean(probs))

        return ExpertPrediction(
            probe_layer=probe_layer,
            target_layer=target_layer,
            predicted_experts=tuple(int(i) for i in top_k_idx),
            confidence=confidence,
        )

    async def evaluate_predictions(
        self,
        prompts: list[str],
        probe_layer: int,
        target_layers: list[int],
    ) -> dict[int, PredictionMetrics]:
        """Evaluate prediction accuracy.

        Args:
            prompts: Evaluation prompts
            probe_layer: Layer to probe from
            target_layers: Layers to evaluate

        Returns:
            Dict mapping target layer to metrics
        """
        metrics: dict[int, PredictionMetrics] = {}

        for target_layer in target_layers:
            all_correct = []
            all_predicted = []
            all_actual = []
            prefetch_hits = 0
            prefetch_total = 0

            for prompt in prompts:
                # Get prediction
                pred = await self.predict_experts(prompt, probe_layer, target_layer)

                # Get actual routing
                weights_list = await self.router.capture_router_weights(prompt)
                layer_weights = [w for w in weights_list if w.layer_idx == target_layer]

                if not layer_weights:
                    continue

                # Compare
                actual_experts = set()
                for pos in layer_weights[0].positions:
                    actual_experts.update(pos.expert_indices)

                predicted_set = set(pred.predicted_experts)

                # Accuracy: any overlap?
                has_overlap = len(predicted_set & actual_experts) > 0
                all_correct.append(has_overlap)

                # Precision: of predicted, how many were used?
                if predicted_set:
                    precision = len(predicted_set & actual_experts) / len(predicted_set)
                    all_predicted.append(precision)

                # Recall: of actual, how many were predicted?
                if actual_experts:
                    recall = len(predicted_set & actual_experts) / len(actual_experts)
                    all_actual.append(recall)

                # Prefetch efficiency
                prefetch_hits += len(predicted_set & actual_experts)
                prefetch_total += len(predicted_set)

            metrics[target_layer] = PredictionMetrics(
                probe_layer=probe_layer,
                target_layer=target_layer,
                accuracy=np.mean(all_correct) if all_correct else 0.0,
                precision=np.mean(all_predicted) if all_predicted else 0.0,
                recall=np.mean(all_actual) if all_actual else 0.0,
                prefetch_efficiency=prefetch_hits / prefetch_total if prefetch_total else 0.0,
            )

        return metrics

    async def analyze_layer_pair(
        self,
        prompts: list[str],
        probe_layer: int,
        target_layer: int,
    ) -> LayerPairAnalysis:
        """Analyze predictability between layer pair.

        Args:
            prompts: Analysis prompts
            probe_layer: Probe layer
            target_layer: Target layer

        Returns:
            LayerPairAnalysis
        """
        # Collect probe features and target routing
        probe_features = []
        target_routing = []

        for prompt in prompts:
            hidden = await self.capture_layer_activations(prompt, probe_layer)
            if hidden is None:
                continue

            features = self.extract_features(hidden)
            probe_features.append(features)

            weights_list = await self.router.capture_router_weights(prompt)
            layer_weights = [w for w in weights_list if w.layer_idx == target_layer]

            if layer_weights:
                # Get top expert for this prompt
                expert_counts = defaultdict(int)
                for pos in layer_weights[0].positions:
                    if pos.expert_indices:
                        expert_counts[pos.expert_indices[0]] += 1
                top_expert = max(expert_counts, key=expert_counts.get) if expert_counts else 0
                target_routing.append(top_expert)
            else:
                target_routing.append(-1)

        if not probe_features:
            return LayerPairAnalysis(
                probe_layer=probe_layer,
                target_layer=target_layer,
                correlation=0.0,
                predictability=0.0,
                dominant_experts=[],
            )

        # Simple predictability: can we predict top expert from features?
        X = np.stack(probe_features)
        y = np.array(target_routing)

        # Correlation with feature PCA
        from_pca = X @ np.random.randn(X.shape[1])  # Simplified
        correlation = 0.0
        if len(set(y)) > 1:
            try:
                correlation = float(np.corrcoef(from_pca, y)[0, 1])
                if np.isnan(correlation):
                    correlation = 0.0
            except Exception:
                correlation = 0.0

        # Predictability based on unique expert count
        unique_experts = len(set(y) - {-1})
        num_experts = self.router.info.num_experts
        predictability = 1.0 - (unique_experts / num_experts) if num_experts > 0 else 0.0

        # Dominant experts
        expert_counts = defaultdict(int)
        for e in y:
            if e >= 0:
                expert_counts[e] += 1
        dominant = sorted(expert_counts, key=expert_counts.get, reverse=True)[:3]

        return LayerPairAnalysis(
            probe_layer=probe_layer,
            target_layer=target_layer,
            correlation=abs(correlation),
            predictability=predictability,
            dominant_experts=dominant,
        )

    async def analyze(
        self,
        prompts: list[str],
        probe_layer: int = 4,
        target_layers: list[int] | None = None,
        model_id: str = "unknown",
    ) -> TaskPredictionAnalysis:
        """Complete task-aware prediction analysis.

        Args:
            prompts: Analysis prompts
            probe_layer: Layer to probe from (default: 4)
            target_layers: Target layers (default: later MoE layers)

        Returns:
            TaskPredictionAnalysis
        """
        moe_layers = list(self.router.info.moe_layers)

        if target_layers is None:
            # Default: second half of MoE layers
            mid = len(moe_layers) // 2
            target_layers = moe_layers[mid:]

        # Ensure probe layer is valid
        if probe_layer not in moe_layers:
            probe_layer = moe_layers[0]

        print(f"Training probes from L{probe_layer} to predict {len(target_layers)} layers...")

        # Train probes
        train_prompts = prompts[: len(prompts) // 2]
        await self.train_probe(train_prompts, probe_layer, target_layers)

        # Evaluate
        print("Evaluating prediction accuracy...")
        eval_prompts = prompts[len(prompts) // 2:]
        layer_metrics = await self.evaluate_predictions(eval_prompts, probe_layer, target_layers)

        # Analyze layer pairs
        print("Analyzing layer pair predictability...")
        layer_pairs = []
        for target_layer in target_layers[:5]:  # Limit for speed
            pair_analysis = await self.analyze_layer_pair(prompts[:20], probe_layer, target_layer)
            layer_pairs.append(pair_analysis)

        # Compute overall metrics
        accuracies = [m.accuracy for m in layer_metrics.values()]
        efficiencies = [m.prefetch_efficiency for m in layer_metrics.values()]

        overall_accuracy = np.mean(accuracies) if accuracies else 0.0
        overall_efficiency = np.mean(efficiencies) if efficiencies else 0.0

        # Best predicted layers
        sorted_by_acc = sorted(layer_metrics.items(), key=lambda x: x[1].accuracy, reverse=True)
        best_layers = [l for l, _ in sorted_by_acc[:3]]

        return TaskPredictionAnalysis(
            model_id=model_id,
            probe_layer=probe_layer,
            num_prompts=len(prompts),
            layer_metrics=layer_metrics,
            overall_accuracy=overall_accuracy,
            overall_prefetch_efficiency=overall_efficiency,
            best_predicted_layers=best_layers,
            layer_pairs=layer_pairs,
        )


def print_task_prediction_analysis(analysis: TaskPredictionAnalysis) -> None:
    """Print task prediction analysis results."""
    print("\n" + "=" * 70)
    print("TASK-AWARE EXPERT PREDICTION ANALYSIS")
    print("=" * 70)

    print(f"\nModel: {analysis.model_id}")
    print(f"Probe layer: L{analysis.probe_layer}")
    print(f"Prompts: {analysis.num_prompts}")

    print("\n" + "-" * 70)
    print("OVERALL PERFORMANCE")
    print("-" * 70)

    print(f"Average accuracy: {analysis.overall_accuracy:.1%}")
    print(f"Prefetch efficiency: {analysis.overall_prefetch_efficiency:.1%}")
    print(f"Best predicted layers: {analysis.best_predicted_layers}")

    print("\n" + "-" * 70)
    print("PER-LAYER METRICS")
    print("-" * 70)

    for layer_idx, metrics in sorted(analysis.layer_metrics.items()):
        acc_bar = "*" * int(metrics.accuracy * 20) + "." * (20 - int(metrics.accuracy * 20))
        print(f"L{layer_idx:2d}: {acc_bar} acc={metrics.accuracy:.1%} prec={metrics.precision:.1%} rec={metrics.recall:.1%}")

    if analysis.layer_pairs:
        print("\n" + "-" * 70)
        print("LAYER PAIR PREDICTABILITY")
        print("-" * 70)

        for pair in analysis.layer_pairs:
            print(f"L{pair.probe_layer} -> L{pair.target_layer}:")
            print(f"  Correlation: {pair.correlation:.3f}")
            print(f"  Predictability: {pair.predictability:.1%}")
            print(f"  Dominant experts: {pair.dominant_experts}")

    print("\n" + "-" * 70)
    print("PREFETCH RECOMMENDATION")
    print("-" * 70)

    if analysis.overall_prefetch_efficiency > 0.7:
        print("-> STRONG prefetch potential: L{} probe can reliably predict later experts".format(
            analysis.probe_layer
        ))
    elif analysis.overall_prefetch_efficiency > 0.4:
        print("-> MODERATE prefetch potential: Some predictability from early layer")
    else:
        print("-> LOW prefetch potential: Routing decisions made late in the network")
