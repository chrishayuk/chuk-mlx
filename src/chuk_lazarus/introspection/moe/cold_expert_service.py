"""Cold Expert Analysis Service.

Provides comprehensive analysis of rarely-activated (cold) experts:
- Identify cold experts across layers
- Find tokens that trigger cold experts
- Measure ablation impact
- Suggest pruning candidates
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .expert_router import ExpertRouter

logger = logging.getLogger(__name__)


class ColdExpertInfo(BaseModel):
    """Information about a cold expert."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    expert_idx: int = Field(ge=0)
    activation_rate: float = Field(ge=0, le=1)
    activation_count: int = Field(ge=0)
    total_samples: int = Field(ge=0)
    trigger_tokens: tuple[str, ...] = Field(default_factory=tuple)
    trigger_contexts: tuple[str, ...] = Field(default_factory=tuple)


class AblationImpact(BaseModel):
    """Impact of ablating a cold expert."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    expert_idx: int = Field(ge=0)
    baseline_quality: float = Field(ge=0)
    ablated_quality: float = Field(ge=0)
    quality_delta: float
    is_safe_to_prune: bool
    sample_outputs: tuple[tuple[str, str], ...] = Field(default_factory=tuple)


class ColdExpertAnalysis(BaseModel):
    """Complete cold expert analysis results."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    total_experts: int = Field(ge=0)
    cold_threshold: float = Field(ge=0, le=1)
    num_prompts_analyzed: int = Field(ge=0)
    cold_experts: tuple[ColdExpertInfo, ...] = Field(default_factory=tuple)
    cold_expert_count: int = Field(ge=0)
    cold_expert_percentage: float = Field(ge=0, le=1)
    ablation_impacts: tuple[AblationImpact, ...] = Field(default_factory=tuple)
    pruning_recommendations: tuple[tuple[int, int], ...] = Field(default_factory=tuple)

    def get_cold_experts_for_layer(self, layer_idx: int) -> tuple[ColdExpertInfo, ...]:
        """Get cold experts for a specific layer."""
        return tuple(e for e in self.cold_experts if e.layer_idx == layer_idx)


class ColdExpertService:
    """Service for analyzing cold (rarely-activated) experts."""

    def __init__(self, router: ExpertRouter):
        """Initialize service.

        Args:
            router: ExpertRouter instance with loaded model
        """
        self._router = router
        self._model = router._model
        self._tokenizer = router._tokenizer
        self._info = router._info

    async def analyze(
        self,
        prompts: list[str],
        cold_threshold: float = 0.01,
        analyze_triggers: bool = True,
        analyze_ablation: bool = False,
    ) -> ColdExpertAnalysis:
        """Run comprehensive cold expert analysis.

        Args:
            prompts: Prompts to analyze for activation patterns
            cold_threshold: Activation rate below which expert is "cold"
            analyze_triggers: Whether to find trigger tokens for cold experts
            analyze_ablation: Whether to test ablation impact (slow)

        Returns:
            ColdExpertAnalysis with results
        """
        logger.info(f"Analyzing cold experts with threshold {cold_threshold:.2%}")

        # Step 1: Collect activation frequencies
        activation_counts, total_activations = await self._collect_activation_stats(prompts)

        # Step 2: Identify cold experts
        cold_experts: list[ColdExpertInfo] = []
        total_experts = 0

        for layer_idx in self._info.moe_layers:
            layer_counts = activation_counts.get(layer_idx, {})
            layer_total = total_activations.get(layer_idx, 0)

            for expert_idx in range(self._info.num_experts):
                total_experts += 1
                count = layer_counts.get(expert_idx, 0)
                rate = count / layer_total if layer_total > 0 else 0.0

                if rate < cold_threshold:
                    trigger_tokens: tuple[str, ...] = ()
                    trigger_contexts: tuple[str, ...] = ()

                    if analyze_triggers and count > 0:
                        triggers = await self._find_triggers(
                            layer_idx, expert_idx, prompts[:50]
                        )
                        trigger_tokens = tuple(triggers["tokens"][:10])
                        trigger_contexts = tuple(triggers["contexts"][:5])

                    cold_experts.append(
                        ColdExpertInfo(
                            layer_idx=layer_idx,
                            expert_idx=expert_idx,
                            activation_rate=rate,
                            activation_count=count,
                            total_samples=layer_total,
                            trigger_tokens=trigger_tokens,
                            trigger_contexts=trigger_contexts,
                        )
                    )

        # Step 3: Analyze ablation impact for top cold experts
        ablation_impacts: list[AblationImpact] = []
        if analyze_ablation:
            # Test ablation on a subset of cold experts
            test_experts = cold_experts[:10]  # Top 10 coldest
            for cold_info in test_experts:
                impact = await self._test_ablation_impact(
                    cold_info.layer_idx,
                    cold_info.expert_idx,
                    prompts[:10],
                )
                ablation_impacts.append(impact)

        # Step 4: Generate pruning recommendations
        pruning_recommendations: list[tuple[int, int]] = []
        for cold_info in cold_experts:
            # Recommend pruning if activation rate is very low
            if cold_info.activation_rate < cold_threshold / 10:
                pruning_recommendations.append(
                    (cold_info.layer_idx, cold_info.expert_idx)
                )

        cold_count = len(cold_experts)
        cold_percentage = cold_count / total_experts if total_experts > 0 else 0.0

        logger.info(
            f"Found {cold_count}/{total_experts} cold experts ({cold_percentage:.1%})"
        )

        return ColdExpertAnalysis(
            model_id=self._router.info.architecture.value,
            total_experts=total_experts,
            cold_threshold=cold_threshold,
            num_prompts_analyzed=len(prompts),
            cold_experts=tuple(cold_experts),
            cold_expert_count=cold_count,
            cold_expert_percentage=cold_percentage,
            ablation_impacts=tuple(ablation_impacts),
            pruning_recommendations=tuple(pruning_recommendations),
        )

    async def _collect_activation_stats(
        self, prompts: list[str]
    ) -> tuple[dict[int, dict[int, int]], dict[int, int]]:
        """Collect activation statistics across prompts."""
        activation_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        total_activations: dict[int, int] = defaultdict(int)

        for prompt in prompts:
            weights_list = await self._router.capture_router_weights(prompt)

            for layer_weights in weights_list:
                layer_idx = layer_weights.layer_idx

                for pos in layer_weights.positions:
                    total_activations[layer_idx] += 1
                    for exp_idx in pos.expert_indices:
                        activation_counts[layer_idx][exp_idx] += 1

        return dict(activation_counts), dict(total_activations)

    async def _find_triggers(
        self,
        layer_idx: int,
        expert_idx: int,
        prompts: list[str],
    ) -> dict[str, list[str]]:
        """Find tokens and contexts that trigger a specific expert."""
        trigger_tokens: list[str] = []
        trigger_contexts: list[str] = []

        for prompt in prompts:
            weights_list = await self._router.capture_router_weights(
                prompt, layers=[layer_idx]
            )

            if not weights_list:
                continue

            positions = weights_list[0].positions
            for i, pos in enumerate(positions):
                if expert_idx in pos.expert_indices:
                    trigger_tokens.append(pos.token)

                    # Get surrounding context
                    start = max(0, i - 2)
                    end = min(len(positions), i + 3)
                    context = "".join(p.token for p in positions[start:end])
                    trigger_contexts.append(context)

        return {
            "tokens": trigger_tokens,
            "contexts": trigger_contexts,
        }

    async def _test_ablation_impact(
        self,
        layer_idx: int,
        expert_idx: int,
        test_prompts: list[str],
    ) -> AblationImpact:
        """Test impact of ablating a specific expert."""
        baseline_outputs: list[str] = []
        ablated_outputs: list[str] = []
        sample_outputs: list[tuple[str, str]] = []

        for prompt in test_prompts:
            # Baseline generation
            baseline = self._router._generate_normal_sync(prompt, max_tokens=20)
            baseline_outputs.append(baseline)

            # Ablated generation
            ablated, _ = await self._router.generate_with_ablation(
                prompt,
                expert_indices=[expert_idx],
                layers=[layer_idx],
                max_tokens=20,
            )
            ablated_outputs.append(ablated)

            sample_outputs.append((baseline, ablated))

        # Compute quality metrics
        baseline_quality = self._compute_quality(baseline_outputs)
        ablated_quality = self._compute_quality(ablated_outputs)
        quality_delta = ablated_quality - baseline_quality

        # Safe to prune if quality delta is small
        is_safe = abs(quality_delta) < 0.05

        return AblationImpact(
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            baseline_quality=baseline_quality,
            ablated_quality=ablated_quality,
            quality_delta=quality_delta,
            is_safe_to_prune=is_safe,
            sample_outputs=tuple(sample_outputs[:3]),
        )

    def _compute_quality(self, outputs: list[str]) -> float:
        """Compute simple quality metric for outputs."""
        if not outputs:
            return 0.0

        # Simple heuristic: average length normalized
        avg_len = sum(len(o) for o in outputs) / len(outputs)
        return min(avg_len / 100, 1.0)

    async def find_inputs_for_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        candidate_tokens: list[str] | None = None,
        max_attempts: int = 100,
    ) -> list[str]:
        """Find inputs that activate a specific cold expert.

        Args:
            layer_idx: Target layer
            expert_idx: Target expert
            candidate_tokens: Tokens to try (default: diverse set)
            max_attempts: Maximum attempts to find activating inputs

        Returns:
            List of inputs that activate the expert
        """
        if candidate_tokens is None:
            # Diverse token set
            candidate_tokens = [
                "0", "1", "100", "3.14",  # Numbers
                "+", "-", "*", "/", "=",  # Operators
                "def", "class", "import", "return",  # Code
                "the", "a", "is", "of", "and",  # Function words
                ".", ",", "!", "?", ";",  # Punctuation
                "Hello", "World", "Python", "Math",  # Misc
                "\n", "\t", "    ",  # Whitespace
            ]

        activating_inputs: list[str] = []

        for token in candidate_tokens[:max_attempts]:
            weights_list = await self._router.capture_router_weights(
                token, layers=[layer_idx]
            )

            if weights_list:
                for pos in weights_list[0].positions:
                    if expert_idx in pos.expert_indices:
                        activating_inputs.append(token)
                        break

        return activating_inputs


def print_cold_expert_report(analysis: ColdExpertAnalysis) -> None:
    """Print formatted cold expert analysis report."""
    print("\n" + "=" * 70)
    print("COLD EXPERT ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nModel: {analysis.model_id}")
    print(f"Threshold: < {analysis.cold_threshold:.2%} activation rate")
    print(f"Prompts analyzed: {analysis.num_prompts_analyzed}")

    print(f"\nCold Experts: {analysis.cold_expert_count}/{analysis.total_experts}")
    print(f"             ({analysis.cold_expert_percentage:.1%} of all experts)")

    # Group by layer
    by_layer: dict[int, list[ColdExpertInfo]] = defaultdict(list)
    for cold in analysis.cold_experts:
        by_layer[cold.layer_idx].append(cold)

    print("\n" + "-" * 70)
    print("COLD EXPERTS BY LAYER")
    print("-" * 70)

    for layer_idx in sorted(by_layer.keys()):
        experts = by_layer[layer_idx]
        expert_list = ", ".join(f"E{e.expert_idx}" for e in experts[:10])
        if len(experts) > 10:
            expert_list += f" (+{len(experts) - 10} more)"
        print(f"Layer {layer_idx:2d}: {len(experts):3d} cold | {expert_list}")

    # Show top coldest experts with triggers
    print("\n" + "-" * 70)
    print("TOP 10 COLDEST EXPERTS (with triggers)")
    print("-" * 70)

    sorted_cold = sorted(analysis.cold_experts, key=lambda x: x.activation_rate)
    for i, cold in enumerate(sorted_cold[:10]):
        rate_str = f"{cold.activation_rate:.4%}" if cold.activation_rate > 0 else "0%"
        print(f"\n{i+1}. Layer {cold.layer_idx}, Expert {cold.expert_idx}")
        print(f"   Activation rate: {rate_str} ({cold.activation_count} activations)")
        if cold.trigger_tokens:
            tokens = ", ".join(repr(t) for t in cold.trigger_tokens[:5])
            print(f"   Triggers: {tokens}")

    # Pruning recommendations
    if analysis.pruning_recommendations:
        print("\n" + "-" * 70)
        print("PRUNING RECOMMENDATIONS")
        print("-" * 70)
        print(f"Safe to prune: {len(analysis.pruning_recommendations)} experts")
        for layer_idx, expert_idx in analysis.pruning_recommendations[:10]:
            print(f"  - Layer {layer_idx}, Expert {expert_idx}")

    # Ablation results
    if analysis.ablation_impacts:
        print("\n" + "-" * 70)
        print("ABLATION IMPACT ANALYSIS")
        print("-" * 70)
        for impact in analysis.ablation_impacts:
            status = "✓ Safe" if impact.is_safe_to_prune else "✗ Risky"
            print(
                f"Layer {impact.layer_idx}, Expert {impact.expert_idx}: "
                f"Δ={impact.quality_delta:+.3f} [{status}]"
            )

    print("\n" + "=" * 70)
