"""Routing manipulation service.

Tools for analyzing and crafting inputs that produce specific routing patterns:
- Find inputs that activate specific experts
- Analyze what input modifications change routing
- Discover "routing triggers" for each expert
- Study routing stability under perturbations
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


class ExpertTrigger(BaseModel):
    """A trigger pattern that activates a specific expert."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    expert_idx: int = Field(ge=0)
    trigger_tokens: tuple[str, ...] = Field(description="Tokens that trigger this expert")
    activation_rate: float = Field(ge=0, le=1, description="How often trigger activates expert")
    specificity: float = Field(ge=0, le=1, description="How specific trigger is to this expert")
    example_prompts: tuple[str, ...] = Field(description="Example prompts using trigger")


class RoutingPerturbation(BaseModel):
    """Result of perturbing input and observing routing change."""

    model_config = ConfigDict(frozen=True)

    original_prompt: str
    perturbed_prompt: str
    perturbation_type: str = Field(description="Type of perturbation applied")
    original_experts: dict[int, tuple[int, ...]] = Field(description="Layer -> experts")
    perturbed_experts: dict[int, tuple[int, ...]] = Field(description="Layer -> experts")
    routing_changed: bool
    changed_layers: list[int] = Field(description="Layers where routing changed")


class RoutingControlResult(BaseModel):
    """Result of attempting to control routing."""

    model_config = ConfigDict(frozen=True)

    target_layer: int
    target_expert: int
    success: bool
    crafted_input: str
    achieved_weight: float = Field(ge=0, le=1, description="Weight achieved for target expert")
    attempts: int


class ExpertInfluenceMap(BaseModel):
    """Map of what influences each expert's activation."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int
    expert_idx: int
    token_influences: dict[str, float] = Field(
        description="Token -> influence score on activation"
    )
    position_preference: str = Field(
        description="Position preference (early, middle, late, any)"
    )
    context_sensitivity: float = Field(
        ge=0, le=1, description="How much context affects activation"
    )


class RoutingManipulationAnalysis(BaseModel):
    """Complete routing manipulation analysis."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    num_experts: int
    num_layers: int
    triggers: tuple[ExpertTrigger, ...] = Field(description="Discovered expert triggers")
    perturbation_results: tuple[RoutingPerturbation, ...] = Field(
        description="Perturbation experiment results"
    )
    controllability_score: float = Field(
        ge=0, le=1, description="Overall routing controllability"
    )
    stable_experts: list[int] = Field(description="Experts with stable routing")
    volatile_experts: list[int] = Field(description="Experts with volatile routing")


# Token pools for trigger discovery
TOKEN_POOLS = {
    "math": ["1", "2", "3", "+", "-", "*", "/", "=", "sum", "calculate", "solve"],
    "code": ["def", "class", "import", "return", "if", "for", "while", "(", ")", "{"],
    "language": ["the", "a", "an", "is", "are", "was", "were", "will", "be", "have"],
    "punctuation": [".", ",", "!", "?", ":", ";", '"', "'", "-", "_"],
    "names": ["John", "Paris", "Google", "Python", "Alice", "Bob", "New", "York"],
}


@dataclass
class RoutingManipulationService:
    """Service for routing manipulation and analysis."""

    router: ExpertRouter
    _trigger_cache: dict[tuple[int, int], list[str]] = field(default_factory=dict)

    async def get_routing_for_prompt(
        self,
        prompt: str,
    ) -> dict[int, list[int]]:
        """Get expert routing for a prompt.

        Args:
            prompt: Input prompt

        Returns:
            Dict mapping layer index to list of expert indices
        """
        weights_list = await self.router.capture_router_weights(prompt)

        routing: dict[int, list[int]] = {}
        for lw in weights_list:
            experts = set()
            for pos in lw.positions:
                experts.update(pos.expert_indices)
            routing[lw.layer_idx] = sorted(experts)

        return routing

    async def find_trigger_tokens(
        self,
        layer_idx: int,
        expert_idx: int,
        token_pool: list[str] | None = None,
        num_candidates: int = 20,
    ) -> list[str]:
        """Find tokens that trigger a specific expert.

        Args:
            layer_idx: Layer index
            expert_idx: Expert to find triggers for
            token_pool: Pool of tokens to search (default: combined pools)
            num_candidates: Number of candidates to return

        Returns:
            List of trigger tokens sorted by effectiveness
        """
        if token_pool is None:
            token_pool = []
            for pool in TOKEN_POOLS.values():
                token_pool.extend(pool)

        scores: dict[str, float] = {}

        for token in token_pool:
            # Test token in different contexts
            prompts = [
                token,
                f"The {token}",
                f"{token} is",
                f"About {token}:",
            ]

            activation_count = 0
            total_tests = 0

            for prompt in prompts:
                try:
                    routing = await self.get_routing_for_prompt(prompt)
                    if layer_idx in routing and expert_idx in routing[layer_idx]:
                        activation_count += 1
                    total_tests += 1
                except Exception:
                    continue

            if total_tests > 0:
                scores[token] = activation_count / total_tests

        # Sort by score
        sorted_tokens = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        triggers = [t for t, s in sorted_tokens[:num_candidates] if s > 0]

        # Cache results
        self._trigger_cache[(layer_idx, expert_idx)] = triggers

        return triggers

    async def analyze_expert_trigger(
        self,
        layer_idx: int,
        expert_idx: int,
        test_prompts: list[str],
    ) -> ExpertTrigger:
        """Analyze what triggers a specific expert.

        Args:
            layer_idx: Layer index
            expert_idx: Expert to analyze
            test_prompts: Prompts to test

        Returns:
            ExpertTrigger with trigger information
        """
        # Find trigger tokens
        trigger_tokens = await self.find_trigger_tokens(layer_idx, expert_idx)

        # Test on prompts
        activation_count = 0
        example_prompts = []

        for prompt in test_prompts[:50]:
            routing = await self.get_routing_for_prompt(prompt)
            if layer_idx in routing and expert_idx in routing[layer_idx]:
                activation_count += 1
                if len(example_prompts) < 3:
                    example_prompts.append(prompt[:50])

        activation_rate = activation_count / len(test_prompts) if test_prompts else 0.0

        # Compute specificity (how unique are triggers to this expert)
        specificity = 0.5  # Default

        if trigger_tokens:
            # Test if triggers also activate other experts
            other_expert_activations = 0
            for token in trigger_tokens[:5]:
                routing = await self.get_routing_for_prompt(token)
                if layer_idx in routing:
                    other_experts = [e for e in routing[layer_idx] if e != expert_idx]
                    other_expert_activations += len(other_experts)

            # Lower is more specific
            specificity = 1.0 / (1 + other_expert_activations / 5)

        return ExpertTrigger(
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            trigger_tokens=tuple(trigger_tokens[:5]),
            activation_rate=activation_rate,
            specificity=specificity,
            example_prompts=tuple(example_prompts),
        )

    async def perturb_and_observe(
        self,
        prompt: str,
        perturbation_type: str = "suffix",
    ) -> RoutingPerturbation:
        """Perturb input and observe routing changes.

        Args:
            prompt: Original prompt
            perturbation_type: Type of perturbation

        Returns:
            RoutingPerturbation with results
        """
        # Get original routing
        original = await self.get_routing_for_prompt(prompt)

        # Apply perturbation
        if perturbation_type == "suffix":
            perturbed_prompt = prompt + " Please"
        elif perturbation_type == "prefix":
            perturbed_prompt = "Note: " + prompt
        elif perturbation_type == "case":
            perturbed_prompt = prompt.upper() if prompt.islower() else prompt.lower()
        elif perturbation_type == "spacing":
            perturbed_prompt = "  " + prompt + "  "
        elif perturbation_type == "punctuation":
            perturbed_prompt = prompt.rstrip(".!?") + "..."
        else:
            perturbed_prompt = prompt + "."

        # Get perturbed routing
        perturbed = await self.get_routing_for_prompt(perturbed_prompt)

        # Compare
        changed_layers = []
        for layer_idx in set(original.keys()) | set(perturbed.keys()):
            orig_experts = set(original.get(layer_idx, []))
            pert_experts = set(perturbed.get(layer_idx, []))
            if orig_experts != pert_experts:
                changed_layers.append(layer_idx)

        return RoutingPerturbation(
            original_prompt=prompt,
            perturbed_prompt=perturbed_prompt,
            perturbation_type=perturbation_type,
            original_experts={k: tuple(v) for k, v in original.items()},
            perturbed_experts={k: tuple(v) for k, v in perturbed.items()},
            routing_changed=len(changed_layers) > 0,
            changed_layers=changed_layers,
        )

    async def craft_input_for_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        max_attempts: int = 20,
    ) -> RoutingControlResult:
        """Craft an input that activates a specific expert.

        Args:
            layer_idx: Target layer
            expert_idx: Target expert
            max_attempts: Maximum attempts

        Returns:
            RoutingControlResult with crafted input
        """
        # Get trigger tokens
        triggers = await self.find_trigger_tokens(layer_idx, expert_idx)

        best_input = ""
        best_weight = 0.0
        attempts = 0

        # Try trigger combinations
        for i in range(min(max_attempts, len(triggers) + 10)):
            attempts += 1

            # Build candidate input
            if i < len(triggers):
                candidate = triggers[i]
            else:
                # Combine triggers
                idx = i - len(triggers)
                if idx < len(triggers) - 1:
                    candidate = f"{triggers[idx]} {triggers[idx + 1]}"
                else:
                    candidate = " ".join(triggers[: min(3, len(triggers))])

            # Test candidate
            weights_list = await self.router.capture_router_weights(candidate)

            for lw in weights_list:
                if lw.layer_idx == layer_idx:
                    for pos in lw.positions:
                        if expert_idx in pos.expert_indices:
                            idx = list(pos.expert_indices).index(expert_idx)
                            weight = pos.weights[idx] if idx < len(pos.weights) else 0.0
                            if weight > best_weight:
                                best_weight = weight
                                best_input = candidate

            # Early success if weight is high enough
            if best_weight > 0.8:
                break

        return RoutingControlResult(
            target_layer=layer_idx,
            target_expert=expert_idx,
            success=best_weight > 0.1,
            crafted_input=best_input,
            achieved_weight=best_weight,
            attempts=attempts,
        )

    async def analyze_routing_stability(
        self,
        prompts: list[str],
    ) -> tuple[list[int], list[int]]:
        """Analyze which experts have stable vs volatile routing.

        Args:
            prompts: Test prompts

        Returns:
            Tuple of (stable_experts, volatile_experts) as lists of indices
        """
        expert_consistency: dict[int, list[bool]] = defaultdict(list)

        perturbation_types = ["suffix", "prefix", "spacing", "punctuation"]

        for prompt in prompts[:20]:
            original = await self.get_routing_for_prompt(prompt)

            for pert_type in perturbation_types:
                result = await self.perturb_and_observe(prompt, pert_type)

                # Check each expert
                for layer_idx in original:
                    for expert_idx in original[layer_idx]:
                        # Was expert still active after perturbation?
                        still_active = (
                            layer_idx in result.perturbed_experts
                            and expert_idx in result.perturbed_experts[layer_idx]
                        )
                        expert_consistency[expert_idx].append(still_active)

        # Classify experts
        stable = []
        volatile = []

        for expert_idx, consistencies in expert_consistency.items():
            if len(consistencies) > 5:
                stability = np.mean(consistencies)
                if stability > 0.8:
                    stable.append(expert_idx)
                elif stability < 0.5:
                    volatile.append(expert_idx)

        return sorted(stable), sorted(volatile)

    async def analyze(
        self,
        prompts: list[str],
        layers_to_analyze: list[int] | None = None,
        experts_per_layer: int = 5,
        model_id: str = "unknown",
    ) -> RoutingManipulationAnalysis:
        """Complete routing manipulation analysis.

        Args:
            prompts: Test prompts
            layers_to_analyze: Specific layers (None for sample)
            experts_per_layer: Number of experts to analyze per layer

        Returns:
            RoutingManipulationAnalysis
        """
        moe_layers = list(self.router.info.moe_layers)

        if layers_to_analyze is None:
            # Sample 3 layers
            if len(moe_layers) >= 3:
                layers_to_analyze = [
                    moe_layers[0],
                    moe_layers[len(moe_layers) // 2],
                    moe_layers[-1],
                ]
            else:
                layers_to_analyze = moe_layers

        print(f"Analyzing routing manipulation for {len(layers_to_analyze)} layers...")

        # Discover triggers
        triggers = []
        for layer_idx in layers_to_analyze:
            print(f"  Discovering triggers for layer {layer_idx}...")
            for expert_idx in range(min(experts_per_layer, self.router.info.num_experts)):
                trigger = await self.analyze_expert_trigger(
                    layer_idx, expert_idx, prompts[:30]
                )
                triggers.append(trigger)

        # Perturbation experiments
        print("Running perturbation experiments...")
        perturbations = []
        for prompt in prompts[:10]:
            for pert_type in ["suffix", "prefix", "case", "punctuation"]:
                result = await self.perturb_and_observe(prompt, pert_type)
                perturbations.append(result)

        # Analyze stability
        print("Analyzing routing stability...")
        stable_experts, volatile_experts = await self.analyze_routing_stability(prompts)

        # Compute controllability score
        successful_controls = 0
        control_attempts = 0

        for layer_idx in layers_to_analyze[:2]:
            for expert_idx in range(min(3, self.router.info.num_experts)):
                result = await self.craft_input_for_expert(layer_idx, expert_idx, max_attempts=10)
                if result.success:
                    successful_controls += 1
                control_attempts += 1

        controllability = successful_controls / control_attempts if control_attempts else 0.0

        return RoutingManipulationAnalysis(
            model_id=model_id,
            num_experts=self.router.info.num_experts,
            num_layers=len(moe_layers),
            triggers=tuple(triggers),
            perturbation_results=tuple(perturbations),
            controllability_score=controllability,
            stable_experts=stable_experts,
            volatile_experts=volatile_experts,
        )


def print_manipulation_analysis(analysis: RoutingManipulationAnalysis) -> None:
    """Print routing manipulation analysis results."""
    print("\n" + "=" * 70)
    print("ROUTING MANIPULATION ANALYSIS")
    print("=" * 70)

    print(f"\nModel: {analysis.model_id}")
    print(f"Experts: {analysis.num_experts}")
    print(f"MoE Layers: {analysis.num_layers}")

    print("\n" + "-" * 70)
    print("CONTROLLABILITY")
    print("-" * 70)

    score = analysis.controllability_score
    bar_len = int(score * 30)
    bar = "*" * bar_len + "." * (30 - bar_len)
    print(f"Score: {bar} {score:.1%}")

    if score > 0.7:
        print("-> Routing is HIGHLY controllable via input crafting")
    elif score > 0.4:
        print("-> Routing is MODERATELY controllable")
    else:
        print("-> Routing is DIFFICULT to control")

    print("\n" + "-" * 70)
    print("EXPERT STABILITY")
    print("-" * 70)

    print(f"Stable experts (consistent routing): {analysis.stable_experts[:10]}")
    print(f"Volatile experts (routing varies): {analysis.volatile_experts[:10]}")

    print("\n" + "-" * 70)
    print("TOP EXPERT TRIGGERS")
    print("-" * 70)

    # Group triggers by effectiveness
    sorted_triggers = sorted(
        analysis.triggers,
        key=lambda t: t.activation_rate * t.specificity,
        reverse=True,
    )

    for trigger in sorted_triggers[:10]:
        tokens = ", ".join(trigger.trigger_tokens[:3]) if trigger.trigger_tokens else "none"
        print(
            f"L{trigger.layer_idx}:E{trigger.expert_idx}: "
            f"[{tokens}] "
            f"(act={trigger.activation_rate:.1%}, spec={trigger.specificity:.2f})"
        )

    print("\n" + "-" * 70)
    print("PERTURBATION SENSITIVITY")
    print("-" * 70)

    # Summarize perturbation results
    by_type: dict[str, list[bool]] = defaultdict(list)
    for p in analysis.perturbation_results:
        by_type[p.perturbation_type].append(p.routing_changed)

    for pert_type, results in sorted(by_type.items()):
        change_rate = np.mean(results) if results else 0.0
        bar_len = int(change_rate * 20)
        bar = "!" * bar_len + "." * (20 - bar_len)
        print(f"{pert_type:12s}: {bar} {change_rate:.1%} routing changed")
