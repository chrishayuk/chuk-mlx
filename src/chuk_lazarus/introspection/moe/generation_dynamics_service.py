"""Generation Dynamics Service.

Analyzes expert routing behavior during autoregressive generation:
- Token-by-token routing traces
- Expert handoff patterns
- Phase transitions
- Error correlation with routing anomalies
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .expert_router import ExpertRouter

logger = logging.getLogger(__name__)


class TokenRoutingSnapshot(BaseModel):
    """Routing snapshot at a single generation step."""

    model_config = ConfigDict(frozen=True)

    step: int = Field(ge=0, description="Generation step (0 = first generated token)")
    token: str = Field(description="Generated token")
    token_id: int = Field(ge=0)
    layer_experts: dict[int, tuple[int, ...]] = Field(
        default_factory=dict,
        description="Layer -> selected expert indices",
    )
    layer_weights: dict[int, tuple[float, ...]] = Field(
        default_factory=dict,
        description="Layer -> expert weights",
    )


class ExpertHandoff(BaseModel):
    """Handoff between experts across generation steps."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    step: int = Field(ge=0)
    from_expert: int = Field(ge=-1, description="-1 if no previous")
    to_expert: int = Field(ge=0)
    from_token: str = ""
    to_token: str = ""


class GenerationTrace(BaseModel):
    """Complete routing trace for one generation."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    generated_text: str
    snapshots: tuple[TokenRoutingSnapshot, ...] = Field(default_factory=tuple)
    handoffs: tuple[ExpertHandoff, ...] = Field(default_factory=tuple)
    consistency_score: float = Field(ge=0, le=1, description="Expert consistency across steps")
    phase_boundaries: tuple[int, ...] = Field(
        default_factory=tuple,
        description="Steps where routing pattern shifts significantly",
    )


class GenerationDynamicsAnalysis(BaseModel):
    """Analysis results across multiple generations."""

    model_config = ConfigDict(frozen=True)

    num_traces: int = Field(ge=0)
    avg_consistency: float = Field(ge=0, le=1)
    avg_handoffs_per_token: float = Field(ge=0)
    common_handoff_patterns: tuple[tuple[int, int, int], ...] = Field(
        default_factory=tuple,
        description="(layer, from_expert, to_expert) patterns",
    )
    layer_stability: dict[int, float] = Field(
        default_factory=dict,
        description="Layer -> stability score (how often same expert selected)",
    )
    phase_pattern_detected: bool = False
    traces: tuple[GenerationTrace, ...] = Field(default_factory=tuple)


class GenerationDynamicsService:
    """Service for analyzing expert routing during generation."""

    def __init__(self, router: ExpertRouter):
        """Initialize service.

        Args:
            router: ExpertRouter instance with loaded model
        """
        self._router = router
        self._model = router._model
        self._tokenizer = router._tokenizer
        self._info = router._info

    async def analyze_generation(
        self,
        prompt: str,
        max_tokens: int = 50,
        track_layers: list[int] | None = None,
    ) -> GenerationTrace:
        """Analyze routing during a single generation.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            track_layers: Layers to track (default: all MoE layers)

        Returns:
            GenerationTrace with routing snapshots and analysis
        """
        if track_layers is None:
            track_layers = list(self._info.moe_layers)

        # Generate with routing capture
        snapshots, generated_tokens = await self._generate_with_capture(
            prompt, max_tokens, track_layers
        )

        # Compute handoffs
        handoffs = self._compute_handoffs(snapshots, track_layers)

        # Compute consistency
        consistency = self._compute_consistency(snapshots, track_layers)

        # Detect phase boundaries
        phase_boundaries = self._detect_phase_boundaries(snapshots, track_layers)

        generated_text = self._tokenizer.decode(generated_tokens)

        return GenerationTrace(
            prompt=prompt,
            generated_text=generated_text,
            snapshots=tuple(snapshots),
            handoffs=tuple(handoffs),
            consistency_score=consistency,
            phase_boundaries=tuple(phase_boundaries),
        )

    async def analyze_batch(
        self,
        prompts: list[str],
        max_tokens: int = 50,
        track_layers: list[int] | None = None,
    ) -> GenerationDynamicsAnalysis:
        """Analyze routing across multiple generations.

        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens per generation
            track_layers: Layers to track

        Returns:
            GenerationDynamicsAnalysis with aggregate statistics
        """
        if track_layers is None:
            track_layers = list(self._info.moe_layers)

        traces: list[GenerationTrace] = []
        all_handoffs: list[ExpertHandoff] = []

        for prompt in prompts:
            trace = await self.analyze_generation(prompt, max_tokens, track_layers)
            traces.append(trace)
            all_handoffs.extend(trace.handoffs)

        # Compute aggregate statistics
        avg_consistency = (
            sum(t.consistency_score for t in traces) / len(traces) if traces else 0.0
        )

        total_handoffs = len(all_handoffs)
        total_tokens = sum(len(t.snapshots) for t in traces)
        avg_handoffs_per_token = total_handoffs / total_tokens if total_tokens > 0 else 0.0

        # Find common handoff patterns
        pattern_counts: dict[tuple[int, int, int], int] = defaultdict(int)
        for handoff in all_handoffs:
            pattern = (handoff.layer_idx, handoff.from_expert, handoff.to_expert)
            pattern_counts[pattern] += 1

        common_patterns = sorted(
            pattern_counts.items(), key=lambda x: x[1], reverse=True
        )[:20]

        # Compute layer stability
        layer_stability: dict[int, float] = {}
        for layer_idx in track_layers:
            stability = self._compute_layer_stability(traces, layer_idx)
            layer_stability[layer_idx] = stability

        # Check for phase patterns
        phase_detected = any(len(t.phase_boundaries) > 0 for t in traces)

        return GenerationDynamicsAnalysis(
            num_traces=len(traces),
            avg_consistency=avg_consistency,
            avg_handoffs_per_token=avg_handoffs_per_token,
            common_handoff_patterns=tuple(p[0] for p in common_patterns),
            layer_stability=layer_stability,
            phase_pattern_detected=phase_detected,
            traces=tuple(traces),
        )

    async def _generate_with_capture(
        self,
        prompt: str,
        max_tokens: int,
        track_layers: list[int],
    ) -> tuple[list[TokenRoutingSnapshot], list[int]]:
        """Generate tokens while capturing routing at each step."""
        input_ids = mx.array(self._tokenizer.encode(prompt))[None, :]
        generated: list[int] = []
        snapshots: list[TokenRoutingSnapshot] = []

        cache = None
        target_layers = set(track_layers)

        # Get the MLP class for patching
        mlp_class = type(self._model.model.layers[0].mlp)
        original_call = mlp_class.__call__

        for step in range(max_tokens):
            # Capture routing at this step
            step_routing: dict[int, list[int]] = {}
            step_weights: dict[int, list[float]] = {}

            def make_patched_call(capture_routing: dict, capture_weights: dict):
                def patched_call(mlp_self: Any, x: mx.array) -> mx.array:
                    # Find layer
                    layer_idx = -1
                    for i, layer in enumerate(self._model.model.layers):
                        if layer.mlp is mlp_self:
                            layer_idx = i
                            break

                    if layer_idx in target_layers:
                        # Capture router output
                        router = mlp_self.router
                        if x.ndim == 3:
                            x_flat = x.reshape(-1, x.shape[-1])
                        else:
                            x_flat = x

                        router_result = router(x_flat)
                        if isinstance(router_result, tuple):
                            weights, indices = router_result
                            # Take last position (for generation, that's the new token)
                            last_indices = indices[-1].tolist()
                            last_weights = [float(w) for w in weights[-1].tolist()]
                            capture_routing[layer_idx] = last_indices
                            capture_weights[layer_idx] = last_weights

                    return original_call(mlp_self, x)

                return patched_call

            try:
                mlp_class.__call__ = make_patched_call(step_routing, step_weights)

                output = self._model(input_ids, cache=cache)
                if hasattr(output, "logits"):
                    logits = output.logits
                    cache = getattr(output, "cache", None)
                elif isinstance(output, tuple):
                    logits, cache = output
                else:
                    logits = output
                    cache = None

            finally:
                mlp_class.__call__ = original_call

            # Sample next token
            next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            generated.append(next_token)

            # Create snapshot
            token_str = self._tokenizer.decode([next_token])
            snapshot = TokenRoutingSnapshot(
                step=step,
                token=token_str,
                token_id=next_token,
                layer_experts={k: tuple(v) for k, v in step_routing.items()},
                layer_weights={k: tuple(v) for k, v in step_weights.items()},
            )
            snapshots.append(snapshot)

            if next_token == self._tokenizer.eos_token_id:
                break

            input_ids = mx.array([[next_token]])

        return snapshots, generated

    def _compute_handoffs(
        self,
        snapshots: list[TokenRoutingSnapshot],
        track_layers: list[int],
    ) -> list[ExpertHandoff]:
        """Compute expert handoffs between consecutive steps."""
        handoffs: list[ExpertHandoff] = []

        for step in range(1, len(snapshots)):
            prev = snapshots[step - 1]
            curr = snapshots[step]

            for layer_idx in track_layers:
                prev_experts = prev.layer_experts.get(layer_idx, ())
                curr_experts = curr.layer_experts.get(layer_idx, ())

                if prev_experts and curr_experts:
                    # Check if primary expert changed
                    prev_primary = prev_experts[0]
                    curr_primary = curr_experts[0]

                    if prev_primary != curr_primary:
                        handoffs.append(
                            ExpertHandoff(
                                layer_idx=layer_idx,
                                step=step,
                                from_expert=prev_primary,
                                to_expert=curr_primary,
                                from_token=prev.token,
                                to_token=curr.token,
                            )
                        )

        return handoffs

    def _compute_consistency(
        self,
        snapshots: list[TokenRoutingSnapshot],
        track_layers: list[int],
    ) -> float:
        """Compute how consistent expert routing is across steps."""
        if len(snapshots) < 2:
            return 1.0

        matches = 0
        total = 0

        for step in range(1, len(snapshots)):
            prev = snapshots[step - 1]
            curr = snapshots[step]

            for layer_idx in track_layers:
                prev_experts = set(prev.layer_experts.get(layer_idx, ()))
                curr_experts = set(curr.layer_experts.get(layer_idx, ()))

                if prev_experts and curr_experts:
                    overlap = len(prev_experts & curr_experts)
                    max_size = max(len(prev_experts), len(curr_experts))
                    matches += overlap
                    total += max_size

        return matches / total if total > 0 else 1.0

    def _detect_phase_boundaries(
        self,
        snapshots: list[TokenRoutingSnapshot],
        track_layers: list[int],
        threshold: float = 0.5,
    ) -> list[int]:
        """Detect steps where routing pattern shifts significantly."""
        if len(snapshots) < 3:
            return []

        boundaries: list[int] = []

        for step in range(2, len(snapshots)):
            # Compare routing at step vs previous window
            prev_window = snapshots[max(0, step - 3) : step]
            curr = snapshots[step]

            divergence = 0.0
            count = 0

            for layer_idx in track_layers:
                curr_experts = set(curr.layer_experts.get(layer_idx, ()))
                prev_experts: set[int] = set()
                for s in prev_window:
                    prev_experts.update(s.layer_experts.get(layer_idx, ()))

                if prev_experts and curr_experts:
                    overlap = len(prev_experts & curr_experts)
                    union = len(prev_experts | curr_experts)
                    divergence += 1.0 - (overlap / union if union > 0 else 0)
                    count += 1

            avg_divergence = divergence / count if count > 0 else 0.0

            if avg_divergence > threshold:
                boundaries.append(step)

        return boundaries

    def _compute_layer_stability(
        self,
        traces: list[GenerationTrace],
        layer_idx: int,
    ) -> float:
        """Compute stability score for a layer across traces."""
        total_consistent = 0
        total_pairs = 0

        for trace in traces:
            for step in range(1, len(trace.snapshots)):
                prev = trace.snapshots[step - 1].layer_experts.get(layer_idx, ())
                curr = trace.snapshots[step].layer_experts.get(layer_idx, ())

                if prev and curr:
                    if prev[0] == curr[0]:
                        total_consistent += 1
                    total_pairs += 1

        return total_consistent / total_pairs if total_pairs > 0 else 1.0


def print_generation_trace(trace: GenerationTrace) -> None:
    """Print formatted generation trace."""
    print("\n" + "=" * 70)
    print("GENERATION TRACE")
    print("=" * 70)

    print(f"\nPrompt: {trace.prompt[:50]}...")
    print(f"Generated: {trace.generated_text[:100]}...")
    print(f"Tokens generated: {len(trace.snapshots)}")
    print(f"Consistency score: {trace.consistency_score:.2%}")
    print(f"Handoffs: {len(trace.handoffs)}")

    if trace.phase_boundaries:
        print(f"Phase boundaries at steps: {trace.phase_boundaries}")

    print("\n" + "-" * 70)
    print("ROUTING EVOLUTION (first 10 tokens)")
    print("-" * 70)

    for snapshot in trace.snapshots[:10]:
        experts_str = " | ".join(
            f"L{layer}:E{exps[0]}" if exps else f"L{layer}:?"
            for layer, exps in sorted(snapshot.layer_experts.items())
        )
        print(f"Step {snapshot.step:3d}: {repr(snapshot.token):10s} → {experts_str}")

    if trace.handoffs:
        print("\n" + "-" * 70)
        print("EXPERT HANDOFFS")
        print("-" * 70)
        for handoff in trace.handoffs[:10]:
            print(
                f"Step {handoff.step}: L{handoff.layer_idx} "
                f"E{handoff.from_expert}→E{handoff.to_expert} "
                f"({repr(handoff.from_token)} → {repr(handoff.to_token)})"
            )


def print_dynamics_analysis(analysis: GenerationDynamicsAnalysis) -> None:
    """Print formatted dynamics analysis."""
    print("\n" + "=" * 70)
    print("GENERATION DYNAMICS ANALYSIS")
    print("=" * 70)

    print(f"\nTraces analyzed: {analysis.num_traces}")
    print(f"Average consistency: {analysis.avg_consistency:.2%}")
    print(f"Average handoffs per token: {analysis.avg_handoffs_per_token:.2f}")
    print(f"Phase patterns detected: {'Yes' if analysis.phase_pattern_detected else 'No'}")

    print("\n" + "-" * 70)
    print("LAYER STABILITY")
    print("-" * 70)

    for layer_idx, stability in sorted(analysis.layer_stability.items()):
        bar_len = int(stability * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"Layer {layer_idx:2d}: {bar} {stability:.2%}")

    if analysis.common_handoff_patterns:
        print("\n" + "-" * 70)
        print("COMMON HANDOFF PATTERNS")
        print("-" * 70)
        for layer, from_exp, to_exp in analysis.common_handoff_patterns[:10]:
            print(f"  L{layer}: E{from_exp} → E{to_exp}")
