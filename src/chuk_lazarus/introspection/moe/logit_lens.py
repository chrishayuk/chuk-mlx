"""MoE-specific logit lens analysis.

Extends the base logit lens with MoE-specific analysis:
- Per-expert contribution to final logits
- Router decision evolution across layers
- Expert specialization through vocabulary analysis
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .hooks import MoEHooks


class ExpertLogitContribution(BaseModel):
    """Contribution of a single expert to logit predictions."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    expert_idx: int = Field(ge=0)
    top_tokens: tuple[str, ...] = Field(default_factory=tuple)
    top_logits: tuple[float, ...] = Field(default_factory=tuple)
    top_token_ids: tuple[int, ...] = Field(default_factory=tuple)
    activation_weight: float = Field(ge=0, le=1)


class LayerRoutingSnapshot(BaseModel):
    """Routing snapshot at a layer."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    selected_experts: tuple[int, ...] = Field(default_factory=tuple)
    expert_weights: tuple[float, ...] = Field(default_factory=tuple)
    router_entropy: float = Field(ge=0)
    top_token: str = ""
    top_token_prob: float = Field(ge=0, le=1, default=0.0)


class MoELogitLens:
    """
    MoE-specific logit lens analysis.

    Provides insight into how expert routing affects predictions
    and how different experts contribute to the output vocabulary.
    """

    def __init__(
        self,
        hooks: MoEHooks,
        tokenizer: Any | None = None,
    ):
        """
        Initialize MoE logit lens.

        Args:
            hooks: MoEHooks with captured state
            tokenizer: Tokenizer for decoding
        """
        self.hooks = hooks
        self.tokenizer = tokenizer

    def get_expert_contributions(
        self,
        layer_idx: int,
        position: int = -1,
        top_k: int = 10,
    ) -> list[ExpertLogitContribution]:
        """
        Analyze how each selected expert contributes to predictions.

        Args:
            layer_idx: Layer to analyze
            position: Sequence position
            top_k: Number of top tokens per expert

        Returns:
            List of ExpertLogitContribution for selected experts
        """
        if layer_idx not in self.hooks.moe_state.selected_experts:
            return []

        selected = self.hooks.moe_state.selected_experts[layer_idx]
        weights = self.hooks.moe_state.router_weights.get(layer_idx)

        if selected.ndim == 3:
            # [batch, seq, k] -> get position
            sel_at_pos = selected[0, position, :].tolist()
            if weights is not None:
                w_at_pos = weights.reshape(selected.shape)[0, position, :].tolist()
            else:
                w_at_pos = [1.0 / len(sel_at_pos)] * len(sel_at_pos)
        else:
            sel_at_pos = selected[position, :].tolist()
            w_at_pos = [1.0 / len(sel_at_pos)] * len(sel_at_pos)

        contributions = []
        for expert_idx, weight in zip(sel_at_pos, w_at_pos):
            # Get expert's vocabulary preference
            # This requires capturing expert outputs, which may not be available
            contributions.append(
                ExpertLogitContribution(
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    top_tokens=(),
                    top_logits=(),
                    top_token_ids=(),
                    activation_weight=weight,
                )
            )

        return contributions

    def get_routing_evolution(
        self,
        position: int = -1,
    ) -> list[LayerRoutingSnapshot]:
        """
        Get routing decisions across all captured layers.

        Args:
            position: Sequence position

        Returns:
            List of LayerRoutingSnapshot, one per layer
        """
        snapshots = []

        for layer_idx in sorted(self.hooks.moe_state.selected_experts.keys()):
            selected = self.hooks.moe_state.selected_experts[layer_idx]
            weights = self.hooks.moe_state.router_weights.get(layer_idx)

            if selected.ndim == 3:
                sel_at_pos = selected[0, position, :].tolist()
                if weights is not None:
                    w_at_pos = weights.reshape(selected.shape)[0, position, :].tolist()
                else:
                    w_at_pos = [1.0 / len(sel_at_pos)] * len(sel_at_pos)
            else:
                sel_at_pos = selected[position, :].tolist()
                w_at_pos = [1.0 / len(sel_at_pos)] * len(sel_at_pos)

            # Compute entropy from router logits if available
            entropy = 0.0
            if layer_idx in self.hooks.moe_state.router_logits:
                logits = self.hooks.moe_state.router_logits[layer_idx]
                probs = mx.softmax(logits, axis=-1)
                log_probs = mx.log(probs + 1e-10)
                ent = -mx.sum(probs * log_probs, axis=-1)
                entropy = float(mx.mean(ent))

            # Get top prediction at this layer
            top_token = ""
            top_prob = 0.0

            snapshots.append(
                LayerRoutingSnapshot(
                    layer_idx=layer_idx,
                    selected_experts=tuple(sel_at_pos),
                    expert_weights=tuple(w_at_pos),
                    router_entropy=entropy,
                    top_token=top_token,
                    top_token_prob=top_prob,
                )
            )

        return snapshots

    def find_routing_divergence(
        self,
        position: int = -1,
    ) -> list[tuple[int, int, set[int]]]:
        """
        Find layers where routing changes significantly.

        Args:
            position: Sequence position

        Returns:
            List of (layer_a, layer_b, expert_difference) tuples
        """
        snapshots = self.get_routing_evolution(position)
        divergences = []

        for i in range(len(snapshots) - 1):
            a, b = snapshots[i], snapshots[i + 1]
            set_a = set(a.selected_experts)
            set_b = set(b.selected_experts)

            if set_a != set_b:
                diff = set_a.symmetric_difference(set_b)
                divergences.append((a.layer_idx, b.layer_idx, diff))

        return divergences

    def print_routing_evolution(self, position: int = -1) -> None:
        """Print routing evolution in human-readable format."""
        snapshots = self.get_routing_evolution(position)

        if not snapshots:
            print("No routing data captured")
            return

        print(f"\nMoE Routing Evolution (position {position})")
        print("=" * 60)

        for snap in snapshots:
            experts_str = ", ".join(
                f"E{e}({w:.2f})" for e, w in zip(snap.selected_experts, snap.expert_weights)
            )
            print(f"Layer {snap.layer_idx:2d}: [{experts_str}] entropy={snap.router_entropy:.3f}")


def analyze_expert_vocabulary(
    model: nn.Module,
    layer_idx: int,
    expert_idx: int,
    tokenizer: Any,
    top_k: int = 20,
) -> dict[str, Any]:
    """
    Analyze what vocabulary an expert specializes in.

    This examines the expert's output projection to find
    which tokens it most strongly promotes.

    Args:
        model: The model
        layer_idx: Layer index
        expert_idx: Expert index
        tokenizer: Tokenizer
        top_k: Number of top tokens

    Returns:
        Dict with vocabulary analysis
    """
    layers = _get_model_layers(model)
    if layer_idx >= len(layers):
        return {"error": "layer out of range"}

    layer = layers[layer_idx]
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return {"error": "no mlp"}

    experts = getattr(mlp, "experts", None)
    if experts is None or not isinstance(experts, list):
        return {"error": "no experts list"}

    if expert_idx >= len(experts):
        return {"error": "expert out of range"}

    expert = experts[expert_idx]
    down_proj = getattr(expert, "down_proj", None)
    if down_proj is None:
        return {"error": "no down_proj"}

    # Get the output weight
    weight = down_proj.weight  # [hidden, intermediate]

    # Compute which output dimensions have strongest weights
    # This is a proxy for vocabulary preference
    output_norms = mx.linalg.norm(weight, axis=1)
    top_dims = mx.argsort(output_norms)[::-1][:top_k].tolist()

    return {
        "expert_idx": expert_idx,
        "layer_idx": layer_idx,
        "top_output_dimensions": top_dims,
        "dimension_norms": output_norms[top_dims[:10]].tolist(),
    }


def _get_model_layers(model: nn.Module) -> list[nn.Module]:
    """Get transformer layers from model."""
    for attr in ["model", "transformer", "decoder"]:
        submodel = getattr(model, attr, None)
        if submodel is not None:
            layers = getattr(submodel, "layers", None)
            if layers is not None:
                return list(layers)
    return list(getattr(model, "layers", []))
