"""
Experimental MoE Architectures.

Based on empirical findings from MoE dynamics analysis of GPT-OSS-20B:
- 12.9% cold experts → Tiered allocation
- 15 stable pipelines, 87.5% consistency → Circuit-based routing
- k=4 essential, k=1 breaks model → Expert teams (guaranteed cooperation)
- 0.906 attention-expert correlation → Compact non-linear router
- 94% early prediction accuracy → Adaptive-k routing

See experiments/moe_dynamics/ARCHITECTURE_PROPOSALS.md for details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from ...core.config import FFNConfig
from .glu import GLU


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ExperimentalMoEConfig:
    """Configuration for experimental MoE variants."""

    # Base settings
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_experts: int = 32
    num_experts_per_tok: int = 4
    bias: bool = False

    # Variant selection
    variant: Literal[
        "standard",
        "compact_router",
        "compact_router_16",  # Bottleneck=16 for actual savings
        "tiered",
        "circuit",
        "layer_pair_circuit",  # New: uses 87.5% layer-pair consistency
        "team",
        "lightweight_team",
        "tiered_lightweight",  # New: combines tiered + lightweight teams
        "adaptive_k",
        "attention_augmented",
        "tiered_circuit_teams",
    ] = "standard"

    # Compact router settings
    router_bottleneck: int = 64

    # Tiered settings (expert counts per layer phase)
    tiered_early_experts: int = 16  # L0-L7
    tiered_middle_experts: int = 32  # L8-L17
    tiered_late_experts: int = 24  # L18+

    # Circuit settings
    num_circuits: int = 15
    circuit_definitions: list[list[int]] = field(default_factory=list)

    # Team settings
    team_size: int = 4
    num_teams: int = 8
    use_lightweight_teams: bool = True  # Use LightweightTeam vs ExpertTeam

    # Adaptive-k settings
    min_k: int = 2
    max_k: int = 8

    # Attention-augmented router settings
    attention_feature_size: int = 4  # Number of attention features


# =============================================================================
# Proposal 1: Compact Non-Linear Router
# =============================================================================


class CompactNonlinearRouter(nn.Module):
    """
    Bottlenecked router that captures non-linear attention→routing mapping.

    Motivation: 0.906 correlation suggests structure exists that a smaller
    model could capture. Bottleneck forces compression of the non-linear mapping.

    Args:
        hidden_size: Input dimension
        num_experts: Number of experts to route to
        bottleneck: Bottleneck dimension (default 64)
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        bottleneck: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.bottleneck = bottleneck

        # Bottlenecked projection
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, num_experts, bias=False)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Route through bottleneck."""
        # Non-linear bottleneck
        h = nn.gelu(self.down(x))
        logits = self.up(h)

        # Top-k selection
        indices = mx.argsort(logits, axis=-1)[:, :, -self.num_experts_per_tok:]
        indices = indices[:, :, ::-1]

        weights = mx.softmax(logits, axis=-1)
        top_k_weights = mx.take_along_axis(weights, indices, axis=-1)
        top_k_weights = top_k_weights / (mx.sum(top_k_weights, axis=-1, keepdims=True) + 1e-6)

        return top_k_weights, indices

    @property
    def param_count(self) -> int:
        """Parameter count for this router."""
        return self.hidden_size * self.bottleneck + self.bottleneck * self.num_experts

    @property
    def param_savings_vs_standard(self) -> float:
        """Fraction of parameters saved vs standard router."""
        standard = self.hidden_size * self.num_experts
        return 1.0 - (self.param_count / standard)


# =============================================================================
# Proposal 2: Circuit-Aware MoE
# =============================================================================


class CircuitRouter(nn.Module):
    """
    Route to functional circuits, not individual experts.

    Motivation: 15 stable pipelines with 87.5% consistency span all 24 layers.
    Rather than routing independently at each layer, route to circuits once.

    A circuit defines a fixed expert path across all layers.
    """

    def __init__(
        self,
        hidden_size: int,
        num_circuits: int,
        num_circuits_per_tok: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_circuits = num_circuits
        self.num_circuits_per_tok = num_circuits_per_tok

        # Route to circuits, not experts
        self.gate = nn.Linear(hidden_size, num_circuits, bias=False)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Select circuits."""
        logits = self.gate(x)

        indices = mx.argsort(logits, axis=-1)[:, :, -self.num_circuits_per_tok:]
        indices = indices[:, :, ::-1]

        weights = mx.softmax(logits, axis=-1)
        top_k_weights = mx.take_along_axis(weights, indices, axis=-1)
        top_k_weights = top_k_weights / (mx.sum(top_k_weights, axis=-1, keepdims=True) + 1e-6)

        return top_k_weights, indices


class CircuitMoE(nn.Module):
    """
    MoE with circuit-based routing.

    Instead of routing to experts at each layer, route to circuits once
    at the first layer. Each circuit defines which expert to use at each layer.

    Benefits:
    - 24 routing decisions → 1 routing decision
    - Guaranteed coherent cross-layer paths
    - Matches discovered functional structure
    """

    def __init__(self, config: ExperimentalMoEConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_layers = 24  # Assumed, could be configurable

        # Circuit router (only used at layer 0)
        if layer_idx == 0:
            self.circuit_router = CircuitRouter(
                hidden_size=config.hidden_size,
                num_circuits=config.num_circuits,
                num_circuits_per_tok=config.num_experts_per_tok,
            )

        # Circuit definitions: circuit_idx -> [expert_idx_at_layer_0, ..., expert_idx_at_layer_23]
        # If not provided, generate default (each circuit uses same expert across layers)
        if config.circuit_definitions:
            self.circuit_definitions = config.circuit_definitions
        else:
            # Default: circuit i uses expert i at all layers
            self.circuit_definitions = [
                [i % config.num_experts] * self.num_layers
                for i in range(config.num_circuits)
            ]

        # Experts for this layer
        expert_config = FFNConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.bias,
        )
        self.experts = [GLU(expert_config) for _ in range(config.num_experts)]

        # Cache for circuit routing (set by layer 0, used by subsequent layers)
        self._cached_circuit_weights: mx.array | None = None
        self._cached_circuit_indices: mx.array | None = None

    def set_circuit_routing(self, weights: mx.array, indices: mx.array) -> None:
        """Set circuit routing from layer 0 for use by subsequent layers."""
        self._cached_circuit_weights = weights
        self._cached_circuit_indices = indices

    def get_circuit_routing(self) -> tuple[mx.array, mx.array] | None:
        """Get cached circuit routing."""
        if self._cached_circuit_weights is None:
            return None
        return self._cached_circuit_weights, self._cached_circuit_indices

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass using circuit-based routing."""
        batch_size, seq_len, hidden_size = x.shape

        # Get circuit routing (from cache or compute at layer 0)
        if self.layer_idx == 0:
            circuit_weights, circuit_indices = self.circuit_router(x)
            self._cached_circuit_weights = circuit_weights
            self._cached_circuit_indices = circuit_indices
        else:
            if self._cached_circuit_weights is None:
                raise RuntimeError("Circuit routing not set. Call layer 0 first.")
            circuit_weights = self._cached_circuit_weights
            circuit_indices = self._cached_circuit_indices

        # Map circuits to experts at this layer
        x_flat = x.reshape(-1, hidden_size)
        circuit_indices_flat = circuit_indices.reshape(-1, circuit_indices.shape[-1])
        circuit_weights_flat = circuit_weights.reshape(-1, circuit_weights.shape[-1])

        output = mx.zeros_like(x_flat)

        for circuit_idx in range(self.config.num_circuits):
            # Which expert does this circuit use at this layer?
            expert_idx = self.circuit_definitions[circuit_idx][self.layer_idx]

            # Find tokens routed to this circuit
            circuit_mask = circuit_indices_flat == circuit_idx
            circuit_w = mx.sum(circuit_weights_flat * circuit_mask.astype(circuit_weights_flat.dtype), axis=-1)

            if mx.any(circuit_w > 0):
                expert_out = self.experts[expert_idx](x_flat)
                output = output + expert_out * circuit_w[:, None]

        return output.reshape(batch_size, seq_len, hidden_size)


# =============================================================================
# Proposal 3: Adaptive-k Routing
# =============================================================================


class AdaptiveKRouter(nn.Module):
    """
    Predict required expert count per token.

    Motivation: k=1 breaks output, k=4 works. But k=4 for all tokens may
    be wasteful. Task-aware prediction (94% accuracy) suggests complexity
    is predictable.

    Simple tokens get k=2, complex tokens get k=6-8.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        min_k: int = 2,
        max_k: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.min_k = min_k
        self.max_k = max_k

        # Expert router
        self.expert_gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Complexity probe (predicts how many experts needed)
        self.complexity_probe = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """
        Route with adaptive k.

        Returns:
            weights: (batch, seq, max_k) - routing weights
            indices: (batch, seq, max_k) - expert indices
            k_values: (batch, seq) - predicted k per position
        """
        # Predict complexity (0-1)
        complexity = mx.sigmoid(self.complexity_probe(x)).squeeze(-1)  # (batch, seq)

        # Map to k value
        k_float = self.min_k + complexity * (self.max_k - self.min_k)
        k_values = mx.floor(k_float).astype(mx.int32)

        # Get expert logits
        logits = self.expert_gate(x)

        # Get top-max_k (we'll mask based on actual k)
        indices = mx.argsort(logits, axis=-1)[:, :, -self.max_k:]
        indices = indices[:, :, ::-1]

        weights = mx.softmax(logits, axis=-1)
        top_k_weights = mx.take_along_axis(weights, indices, axis=-1)

        # Create mask for actual k per position
        # Shape: (batch, seq, max_k)
        k_expanded = k_values[:, :, None]  # (batch, seq, 1)
        position_indices = mx.arange(self.max_k)[None, None, :]  # (1, 1, max_k)
        mask = position_indices < k_expanded  # (batch, seq, max_k)

        # Apply mask
        top_k_weights = top_k_weights * mask.astype(top_k_weights.dtype)

        # Renormalize
        top_k_weights = top_k_weights / (mx.sum(top_k_weights, axis=-1, keepdims=True) + 1e-6)

        return top_k_weights, indices, k_values


class AdaptiveKMoE(nn.Module):
    """MoE with adaptive expert count per token."""

    def __init__(self, config: ExperimentalMoEConfig):
        super().__init__()
        self.config = config

        self.router = AdaptiveKRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            min_k=config.min_k,
            max_k=config.max_k,
        )

        expert_config = FFNConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.bias,
        )
        self.experts = [GLU(expert_config) for _ in range(config.num_experts)]

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """
        Forward pass with adaptive k.

        Returns:
            output: (batch, seq, hidden)
            k_values: (batch, seq) - k used per position (for analysis)
        """
        batch_size, seq_len, hidden_size = x.shape

        weights, indices, k_values = self.router(x)

        x_flat = x.reshape(-1, hidden_size)
        indices_flat = indices.reshape(-1, self.config.max_k)
        weights_flat = weights.reshape(-1, self.config.max_k)

        output = mx.zeros_like(x_flat)

        for expert_idx, expert in enumerate(self.experts):
            expert_mask = indices_flat == expert_idx
            expert_weights = mx.sum(weights_flat * expert_mask.astype(weights_flat.dtype), axis=-1)

            if mx.any(expert_weights > 0):
                expert_out = expert(x_flat)
                output = output + expert_out * expert_weights[:, None]

        output = output.reshape(batch_size, seq_len, hidden_size)
        return output, k_values


# =============================================================================
# Proposal 4: Tiered MoE
# =============================================================================


class TieredMoE(nn.Module):
    """
    Non-uniform expert allocation by layer phase.

    Motivation: 12.9% cold experts, and middle layers (L8-L17) show maximum
    differentiation. Allocate more experts where they're actually used.

    Default allocation:
    - Early (L0-L7): 16 experts
    - Middle (L8-L17): 32 experts (full)
    - Late (L18+): 24 experts
    """

    def __init__(self, config: ExperimentalMoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Determine expert count for this layer
        if layer_idx < 8:
            num_experts = config.tiered_early_experts
        elif layer_idx < 18:
            num_experts = config.tiered_middle_experts
        else:
            num_experts = config.tiered_late_experts

        self.num_experts = num_experts

        # Router for this layer's expert count
        self.router = nn.Linear(config.hidden_size, num_experts, bias=False)
        self.num_experts_per_tok = min(config.num_experts_per_tok, num_experts)

        # Experts
        expert_config = FFNConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.bias,
        )
        self.experts = [GLU(expert_config) for _ in range(num_experts)]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with tiered expert count."""
        batch_size, seq_len, hidden_size = x.shape

        # Route
        logits = self.router(x)
        indices = mx.argsort(logits, axis=-1)[:, :, -self.num_experts_per_tok:]
        indices = indices[:, :, ::-1]

        weights = mx.softmax(logits, axis=-1)
        top_k_weights = mx.take_along_axis(weights, indices, axis=-1)
        top_k_weights = top_k_weights / (mx.sum(top_k_weights, axis=-1, keepdims=True) + 1e-6)

        # Process
        x_flat = x.reshape(-1, hidden_size)
        indices_flat = indices.reshape(-1, self.num_experts_per_tok)
        weights_flat = top_k_weights.reshape(-1, self.num_experts_per_tok)

        output = mx.zeros_like(x_flat)

        for expert_idx, expert in enumerate(self.experts):
            expert_mask = indices_flat == expert_idx
            expert_weights = mx.sum(weights_flat * expert_mask.astype(weights_flat.dtype), axis=-1)

            if mx.any(expert_weights > 0):
                expert_out = expert(x_flat)
                output = output + expert_out * expert_weights[:, None]

        return output.reshape(batch_size, seq_len, hidden_size)

    @property
    def expert_savings(self) -> float:
        """Fraction of experts saved vs full allocation."""
        full = self.config.num_experts
        return 1.0 - (self.num_experts / full)


# =============================================================================
# Proposal 5: Expert Teams
# =============================================================================


class ExpertTeam(nn.Module):
    """
    Team of experts that always activate together.

    Motivation: k=4 is essential, suggesting 4-expert teams are the
    functional unit. Rather than hoping routing finds good combinations,
    structure cooperation explicitly.

    Note: Full combiner version - see LightweightTeam for parameter-efficient variant.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        team_size: int = 4,
        bias: bool = False,
    ):
        super().__init__()
        self.team_size = team_size
        self.hidden_size = hidden_size

        # Team members (each is a smaller expert)
        member_intermediate = intermediate_size // team_size
        self.members = [
            nn.Sequential(
                nn.Linear(hidden_size, member_intermediate, bias=bias),
                nn.GELU(),
                nn.Linear(member_intermediate, hidden_size, bias=bias),
            )
            for _ in range(team_size)
        ]

        # Learned combination of team outputs
        self.combiner = nn.Linear(team_size * hidden_size, hidden_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        """All team members process input, then combine."""
        # Each member processes the input
        outputs = [member(x) for member in self.members]

        # Concatenate and combine
        combined = mx.concatenate(outputs, axis=-1)
        return self.combiner(combined)


class LightweightTeam(nn.Module):
    """
    Parameter-efficient team with learned mixing weights.

    Instead of a full combiner (team_size * hidden_size -> hidden_size),
    just learn how to weight the expert outputs.

    4 parameters instead of 67M (for team_size=4, hidden_dim=4096).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        team_size: int = 4,
        bias: bool = False,
    ):
        super().__init__()
        self.team_size = team_size
        self.hidden_size = hidden_size

        # Team members
        member_intermediate = intermediate_size // team_size
        self.members = [
            nn.Sequential(
                nn.Linear(hidden_size, member_intermediate, bias=bias),
                nn.GELU(),
                nn.Linear(member_intermediate, hidden_size, bias=bias),
            )
            for _ in range(team_size)
        ]

        # Just learn mixing weights - 4 params instead of 67M
        self._mix_weights = mx.ones((team_size,)) / team_size

    @property
    def mix_weights(self) -> mx.array:
        """Normalized mixing weights."""
        return mx.softmax(self._mix_weights, axis=0)

    def __call__(self, x: mx.array) -> mx.array:
        """Weighted sum of expert outputs."""
        weights = self.mix_weights
        outputs = [member(x) for member in self.members]
        return sum(w * out for w, out in zip(weights.tolist(), outputs))


class TeamMoE(nn.Module):
    """
    MoE with teams instead of individual experts.

    Route to teams of cooperating experts rather than individual experts.
    Each team internally handles the 4-way cooperation.

    Benefits:
    - Cooperation is guaranteed by design
    - Fewer routing decisions (8 teams vs 32 experts)
    - Team combination is learned, not assumed to be weighted sum
    """

    def __init__(self, config: ExperimentalMoEConfig):
        super().__init__()
        self.config = config

        # Teams
        self.teams = [
            ExpertTeam(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                team_size=config.team_size,
                bias=config.bias,
            )
            for _ in range(config.num_teams)
        ]

        # Team router (route to 1 or 2 teams)
        self.team_router = nn.Linear(config.hidden_size, config.num_teams, bias=False)
        self.num_teams_per_tok = max(1, config.num_experts_per_tok // config.team_size)

    def __call__(self, x: mx.array) -> mx.array:
        """Route to teams."""
        batch_size, seq_len, hidden_size = x.shape

        # Route to teams
        logits = self.team_router(x)
        indices = mx.argsort(logits, axis=-1)[:, :, -self.num_teams_per_tok:]
        indices = indices[:, :, ::-1]

        weights = mx.softmax(logits, axis=-1)
        top_k_weights = mx.take_along_axis(weights, indices, axis=-1)
        top_k_weights = top_k_weights / (mx.sum(top_k_weights, axis=-1, keepdims=True) + 1e-6)

        # Process
        x_flat = x.reshape(-1, hidden_size)
        indices_flat = indices.reshape(-1, self.num_teams_per_tok)
        weights_flat = top_k_weights.reshape(-1, self.num_teams_per_tok)

        output = mx.zeros_like(x_flat)

        for team_idx, team in enumerate(self.teams):
            team_mask = indices_flat == team_idx
            team_weights = mx.sum(weights_flat * team_mask.astype(weights_flat.dtype), axis=-1)

            if mx.any(team_weights > 0):
                team_out = team(x_flat)
                output = output + team_out * team_weights[:, None]

        return output.reshape(batch_size, seq_len, hidden_size)


# =============================================================================
# Proposal 6: Attention-Augmented Router
# =============================================================================


class AttentionAugmentedRouter(nn.Module):
    """
    Router that uses both hidden state AND attention pattern features.

    Motivation: 0.906 correlation but 4.3% prediction accuracy means
    attention contains routing information but the mapping is non-linear.
    Give the router explicit access to attention features.
    """

    def __init__(
        self,
        hidden_size: int,
        attention_feature_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        bottleneck: int = 128,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_feature_size = attention_feature_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Combined router using both hidden state and attention features
        combined_size = hidden_size + attention_feature_size
        self.router = nn.Sequential(
            nn.Linear(combined_size, bottleneck, bias=False),
            nn.GELU(),
            nn.Linear(bottleneck, num_experts, bias=False),
        )

    def __call__(
        self, hidden: mx.array, attention_features: mx.array
    ) -> tuple[mx.array, mx.array]:
        """
        Route using combined hidden state and attention features.

        Args:
            hidden: Hidden state (batch, seq, hidden_size)
            attention_features: Extracted attention features (batch, seq, attention_feature_size)

        Returns:
            weights, indices for top-k experts
        """
        # Combine inputs
        combined = mx.concatenate([hidden, attention_features], axis=-1)

        # Route
        logits = self.router(combined)

        indices = mx.argsort(logits, axis=-1)[:, :, -self.num_experts_per_tok:]
        indices = indices[:, :, ::-1]

        weights = mx.softmax(logits, axis=-1)
        top_k_weights = mx.take_along_axis(weights, indices, axis=-1)
        top_k_weights = top_k_weights / (mx.sum(top_k_weights, axis=-1, keepdims=True) + 1e-6)

        return top_k_weights, indices

    @staticmethod
    def extract_attention_features(
        attention_weights: mx.array,
        position: int = -1,
    ) -> mx.array:
        """
        Extract routing-relevant features from attention weights.

        Args:
            attention_weights: (batch, num_heads, seq, seq) attention weights
            position: Position to extract features for (-1 = last)

        Returns:
            Features (batch, feature_size) where feature_size = 4
        """
        # Average across heads
        avg_attn = mx.mean(attention_weights, axis=1)  # (batch, seq, seq)

        if position == -1:
            position = avg_attn.shape[1] - 1

        # Extract features for the target position
        pos_attn = avg_attn[:, position, :]  # (batch, seq)

        # Feature 1: Self-attention weight
        self_attn = pos_attn[:, position:position+1]  # (batch, 1)

        # Feature 2: Max attention to other positions
        max_attn = mx.max(pos_attn, axis=-1, keepdims=True)  # (batch, 1)

        # Feature 3: Entropy of attention distribution
        entropy = -mx.sum(pos_attn * mx.log(pos_attn + 1e-10), axis=-1, keepdims=True)

        # Feature 4: Attention spread (std)
        mean_attn = mx.mean(pos_attn, axis=-1, keepdims=True)
        std_attn = mx.sqrt(mx.mean((pos_attn - mean_attn) ** 2, axis=-1, keepdims=True))

        return mx.concatenate([self_attn, max_attn, entropy, std_attn], axis=-1)


# =============================================================================
# Proposal 7: Tiered Circuit Teams (Hybrid)
# =============================================================================


class TieredCircuitTeams(nn.Module):
    """
    Hybrid combining tiered allocation, circuits, and teams.

    - Tiered: Fewer teams in early/late layers (low differentiation)
    - Circuit: Each team defines a full-depth expert path
    - Teams: Cooperation is structural, not emergent

    Benefits:
    - 24 routing decisions → 1 (circuits)
    - Guaranteed coherent paths (circuits)
    - Parameter savings (tiered)
    - Cooperation by design (teams)
    """

    # Team counts per layer phase
    TEAM_COUNTS = {
        'early': 4,    # L0-L7: low differentiation anyway
        'middle': 15,  # L8-L17: full circuit coverage
        'late': 8,     # L18-L23: convergence phase
    }

    def __init__(
        self,
        config: ExperimentalMoEConfig,
        layer_idx: int,
        circuit_definitions: list[list[int]] | None = None,
        use_lightweight_teams: bool = True,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.use_lightweight_teams = use_lightweight_teams

        # Determine team count for this layer
        if layer_idx < 8:
            self.num_teams = self.TEAM_COUNTS['early']
            self.phase = 'early'
        elif layer_idx < 18:
            self.num_teams = self.TEAM_COUNTS['middle']
            self.phase = 'middle'
        else:
            self.num_teams = self.TEAM_COUNTS['late']
            self.phase = 'late'

        # Circuit definitions (team_idx -> expert assignments across layers)
        # Each circuit maps to a team of experts at each layer
        if circuit_definitions:
            self.circuit_definitions = circuit_definitions
        else:
            # Default: circuit i uses expert i at all layers
            self.circuit_definitions = [
                [i % config.num_experts] * 24
                for i in range(max(self.TEAM_COUNTS.values()))
            ]

        # Single routing decision at layer 0
        if layer_idx == 0:
            max_teams = max(self.TEAM_COUNTS.values())
            self.team_router = nn.Linear(config.hidden_size, max_teams, bias=False)

        # Teams for this layer (using lightweight version by default)
        team_class = LightweightTeam if use_lightweight_teams else ExpertTeam
        self.teams = [
            team_class(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                team_size=config.team_size,
                bias=config.bias,
            )
            for _ in range(self.num_teams)
        ]

        # Cache for routing
        self._cached_team_weights: mx.array | None = None
        self._cached_team_indices: mx.array | None = None

    def set_team_routing(self, weights: mx.array, indices: mx.array) -> None:
        """Set routing from layer 0."""
        self._cached_team_weights = weights
        self._cached_team_indices = indices

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        batch_size, seq_len, hidden_size = x.shape

        # Get team routing
        if self.layer_idx == 0:
            logits = self.team_router(x)
            indices = mx.argsort(logits, axis=-1)[:, :, -2:]  # Top-2 teams
            indices = indices[:, :, ::-1]

            weights = mx.softmax(logits, axis=-1)
            team_weights = mx.take_along_axis(weights, indices, axis=-1)
            team_weights = team_weights / (mx.sum(team_weights, axis=-1, keepdims=True) + 1e-6)

            self._cached_team_weights = team_weights
            self._cached_team_indices = indices
        else:
            if self._cached_team_weights is None:
                raise RuntimeError("Team routing not set. Call layer 0 first.")
            team_weights = self._cached_team_weights
            indices = self._cached_team_indices

        # Map team indices to this layer's available teams
        # (early/late layers have fewer teams)
        indices_clipped = mx.clip(indices, 0, self.num_teams - 1)

        # Process through teams
        x_flat = x.reshape(-1, hidden_size)
        indices_flat = indices_clipped.reshape(-1, 2)
        weights_flat = team_weights.reshape(-1, 2)

        output = mx.zeros_like(x_flat)

        for team_idx, team in enumerate(self.teams):
            team_mask = indices_flat == team_idx
            team_w = mx.sum(weights_flat * team_mask.astype(weights_flat.dtype), axis=-1)

            if mx.any(team_w > 0):
                team_out = team(x_flat)
                output = output + team_out * team_w[:, None]

        return output.reshape(batch_size, seq_len, hidden_size)


# =============================================================================
# Proposal 8: Tiered Lightweight MoE (Hybrid)
# =============================================================================


class TieredLightweightMoE(nn.Module):
    """
    Combines tiered allocation with lightweight teams.

    Based on validation results:
    - Tiered MoE: 50% param reduction, 37% faster
    - Lightweight Team: 83% param reduction, 5x faster

    Combined prediction: ~90% param reduction, ~6-7x faster.

    Allocation:
    - Early (L0-L7): 4 lightweight teams
    - Middle (L8-L17): 8 lightweight teams
    - Late (L18+): 6 lightweight teams
    """

    TEAM_COUNTS = {
        'early': 4,   # L0-L7: low differentiation
        'middle': 8,  # L8-L17: full allocation
        'late': 6,    # L18+: convergence phase
    }

    def __init__(self, config: ExperimentalMoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Determine team count for this layer
        if layer_idx < 8:
            self.num_teams = self.TEAM_COUNTS['early']
            self.phase = 'early'
        elif layer_idx < 18:
            self.num_teams = self.TEAM_COUNTS['middle']
            self.phase = 'middle'
        else:
            self.num_teams = self.TEAM_COUNTS['late']
            self.phase = 'late'

        # Team router
        self.team_router = nn.Linear(config.hidden_size, self.num_teams, bias=False)
        self.num_teams_per_tok = min(2, self.num_teams)  # Route to up to 2 teams

        # Lightweight teams
        self.teams = [
            LightweightTeam(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                team_size=config.team_size,
                bias=config.bias,
            )
            for _ in range(self.num_teams)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with tiered lightweight teams."""
        batch_size, seq_len, hidden_size = x.shape

        # Route to teams
        logits = self.team_router(x)
        indices = mx.argsort(logits, axis=-1)[:, :, -self.num_teams_per_tok:]
        indices = indices[:, :, ::-1]

        weights = mx.softmax(logits, axis=-1)
        top_k_weights = mx.take_along_axis(weights, indices, axis=-1)
        top_k_weights = top_k_weights / (mx.sum(top_k_weights, axis=-1, keepdims=True) + 1e-6)

        # Process
        x_flat = x.reshape(-1, hidden_size)
        indices_flat = indices.reshape(-1, self.num_teams_per_tok)
        weights_flat = top_k_weights.reshape(-1, self.num_teams_per_tok)

        output = mx.zeros_like(x_flat)

        for team_idx, team in enumerate(self.teams):
            team_mask = indices_flat == team_idx
            team_weights = mx.sum(weights_flat * team_mask.astype(weights_flat.dtype), axis=-1)

            if mx.any(team_weights > 0):
                team_out = team(x_flat)
                output = output + team_out * team_weights[:, None]

        return output.reshape(batch_size, seq_len, hidden_size)

    @property
    def param_summary(self) -> dict:
        """Summary of parameters by component."""
        router_params = self.config.hidden_size * self.num_teams
        # Each team: team_size * (hidden*intermediate + intermediate*hidden)
        member_intermediate = self.config.intermediate_size // self.config.team_size
        member_params = 2 * self.config.hidden_size * member_intermediate
        team_params = self.config.team_size * member_params + self.config.team_size  # +team_size for mix weights
        total_expert_params = self.num_teams * team_params

        return {
            'phase': self.phase,
            'num_teams': self.num_teams,
            'router_params': router_params,
            'expert_params': total_expert_params,
            'total_params': router_params + total_expert_params,
        }


# =============================================================================
# Proposal 9: Layer-Pair Circuit MoE
# =============================================================================


class LayerPairCircuitMoE(nn.Module):
    """
    Route based on previous layer's choice, not L0 commitment.

    Key insight from Experiment 9:
    - Full-path consistency: only 4% (0.875^23)
    - Layer-pair consistency: 87.5%

    Instead of committing to a full circuit at L0, bias routing toward
    consistency with the previous layer. This preserves local structure
    (87.5%) without forcing global commitment (4%).

    The transition matrix P(expert_j at L+1 | expert_i at L) captures
    the natural routing inertia discovered in the model.
    """

    def __init__(self, config: ExperimentalMoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Standard expert router
        self.expert_router = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # Transition bias: P(expert_j | expert_i was chosen at prev layer)
        # Initialized to identity = strong bias toward consistency
        self._transition_logits = mx.zeros((config.num_experts, config.num_experts))
        # Add identity bias (prefer same expert as previous layer)
        self._transition_bias_strength = 2.0  # Tunable

        self.num_experts_per_tok = config.num_experts_per_tok

        # Experts
        expert_config = FFNConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.bias,
        )
        self.experts = [GLU(expert_config) for _ in range(config.num_experts)]

        # Cache for previous layer's routing
        self._prev_expert_indices: mx.array | None = None

    def set_prev_routing(self, indices: mx.array) -> None:
        """Set previous layer's routing for layer-pair consistency."""
        self._prev_expert_indices = indices

    def get_routing(self) -> mx.array | None:
        """Get this layer's routing for use by next layer."""
        return self._prev_expert_indices

    @property
    def transition_matrix(self) -> mx.array:
        """Get the transition bias matrix."""
        # Identity + learned adjustments
        identity = mx.eye(self.config.num_experts) * self._transition_bias_strength
        return self._transition_logits + identity

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with layer-pair routing bias."""
        batch_size, seq_len, hidden_size = x.shape

        # Base routing logits
        base_logits = self.expert_router(x)

        # Apply transition bias if we have previous layer's routing
        if self._prev_expert_indices is not None and self.layer_idx > 0:
            # Get primary expert from previous layer
            prev_primary = self._prev_expert_indices[:, :, 0]  # (batch, seq)

            # Look up transition biases for each previous expert
            # transition_matrix[prev_expert, :] gives bias toward each next expert
            transition = self.transition_matrix

            # Gather transition biases for each position's previous expert
            # prev_primary: (batch, seq) -> indices into transition matrix
            bias = mx.zeros_like(base_logits)
            for i in range(self.config.num_experts):
                mask = (prev_primary == i).astype(base_logits.dtype)[:, :, None]
                bias = bias + mask * transition[i:i+1, :]

            logits = base_logits + bias
        else:
            logits = base_logits

        # Top-k selection
        indices = mx.argsort(logits, axis=-1)[:, :, -self.num_experts_per_tok:]
        indices = indices[:, :, ::-1]

        weights = mx.softmax(logits, axis=-1)
        top_k_weights = mx.take_along_axis(weights, indices, axis=-1)
        top_k_weights = top_k_weights / (mx.sum(top_k_weights, axis=-1, keepdims=True) + 1e-6)

        # Store routing for next layer
        self._prev_expert_indices = indices

        # Process through experts
        x_flat = x.reshape(-1, hidden_size)
        indices_flat = indices.reshape(-1, self.num_experts_per_tok)
        weights_flat = top_k_weights.reshape(-1, self.num_experts_per_tok)

        output = mx.zeros_like(x_flat)

        for expert_idx, expert in enumerate(self.experts):
            expert_mask = indices_flat == expert_idx
            expert_weights = mx.sum(weights_flat * expert_mask.astype(weights_flat.dtype), axis=-1)

            if mx.any(expert_weights > 0):
                expert_out = expert(x_flat)
                output = output + expert_out * expert_weights[:, None]

        return output.reshape(batch_size, seq_len, hidden_size)

    @property
    def consistency_strength(self) -> float:
        """Return the bias strength toward routing consistency."""
        return self._transition_bias_strength


# =============================================================================
# Circuit Discovery and Transfer Utilities
# =============================================================================


def discover_circuits(
    model,
    prompts: list[str],
    num_circuits: int = 15,
    consistency_threshold: float = 0.8,
) -> list[list[int]]:
    """
    Discover expert circuits from a trained model.

    Args:
        model: Trained MoE model with routing
        prompts: Prompts to analyze
        num_circuits: Number of circuits to discover
        consistency_threshold: Minimum consistency to include circuit

    Returns:
        List of circuit definitions [circuit_idx][layer_idx] = expert_idx
    """
    # This would use the expert_circuits analysis from the dynamics experiments
    # For now, return a placeholder implementation
    # Real implementation would:
    # 1. Run prompts through model
    # 2. Track expert co-occurrence across layers
    # 3. Cluster into consistent pipelines
    # 4. Return top-N most consistent circuits

    # Placeholder: identity mapping
    num_layers = 24
    num_experts = 32
    return [
        [i % num_experts] * num_layers
        for i in range(num_circuits)
    ]


def compute_circuit_overlap(
    circuits_a: list[list[int]],
    circuits_b: list[list[int]],
) -> float:
    """
    Compute overlap between two sets of discovered circuits.

    Used to test if circuits transfer across model sizes.

    Args:
        circuits_a: Circuits from model A
        circuits_b: Circuits from model B

    Returns:
        Overlap score (0-1), where 1 = identical circuits
    """
    if not circuits_a or not circuits_b:
        return 0.0

    # Convert to sets of tuples for comparison
    set_a = {tuple(c) for c in circuits_a}
    set_b = {tuple(c) for c in circuits_b}

    # Jaccard similarity
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0


def validate_cold_experts(
    model,
    cold_expert_list: list[tuple[int, int]],  # [(layer, expert), ...]
    rare_inputs: list[str],
) -> dict[tuple[int, int], float]:
    """
    Validate that cold experts are truly cold on rare/adversarial inputs.

    Args:
        model: MoE model
        cold_expert_list: List of (layer, expert) tuples identified as cold
        rare_inputs: Edge case inputs to test

    Returns:
        Dict mapping (layer, expert) to activation rate on rare inputs
    """
    # Placeholder - real implementation would:
    # 1. Run rare_inputs through model
    # 2. Track activation of cold experts
    # 3. Report which "cold" experts actually activate on rare cases
    return {exp: 0.0 for exp in cold_expert_list}


# =============================================================================
# Factory Function
# =============================================================================


def create_experimental_moe(
    config: ExperimentalMoEConfig,
    layer_idx: int = 0,
) -> nn.Module:
    """
    Create an experimental MoE variant.

    Args:
        config: Configuration specifying variant and parameters
        layer_idx: Layer index (needed for tiered and circuit variants)

    Returns:
        MoE module of specified variant

    Available variants:
        - standard: Standard MoE
        - compact_router: Standard MoE with bottlenecked router
        - tiered: Non-uniform expert allocation by layer phase
        - circuit: Route to circuits once at L0
        - team: Route to expert teams (guaranteed cooperation)
        - lightweight_team: Teams with learned mixing weights (4 params vs 67M)
        - adaptive_k: Variable k per token based on complexity
        - attention_augmented: Router uses attention features
        - tiered_circuit_teams: Hybrid combining all three
    """
    if config.variant == "compact_router":
        # Standard MoE with compact router (bottleneck=64, may be worse)
        from .moe import MoE

        ffn_config = FFNConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            bias=config.bias,
        )
        moe = MoE(ffn_config)

        # Replace router with compact version
        moe.router = CompactNonlinearRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            bottleneck=config.router_bottleneck,
        )
        return moe

    elif config.variant == "compact_router_16":
        # Compact router with bottleneck=16 for actual savings (50%)
        from .moe import MoE

        ffn_config = FFNConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            bias=config.bias,
        )
        moe = MoE(ffn_config)

        # Replace router with compact version (bottleneck=16)
        moe.router = CompactNonlinearRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            bottleneck=16,  # Fixed at 16 for 50% savings
        )
        return moe

    elif config.variant == "tiered":
        return TieredMoE(config, layer_idx)

    elif config.variant == "circuit":
        return CircuitMoE(config, layer_idx)

    elif config.variant == "layer_pair_circuit":
        return LayerPairCircuitMoE(config, layer_idx)

    elif config.variant == "team":
        return TeamMoE(config)

    elif config.variant == "lightweight_team":
        # TeamMoE but using LightweightTeam internally
        config_copy = ExperimentalMoEConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            bias=config.bias,
            variant="team",
            team_size=config.team_size,
            num_teams=config.num_teams,
            use_lightweight_teams=True,
        )
        # Create TeamMoE with LightweightTeam members
        team_moe = TeamMoE(config_copy)
        # Replace teams with lightweight versions
        team_moe.teams = [
            LightweightTeam(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                team_size=config.team_size,
                bias=config.bias,
            )
            for _ in range(config.num_teams)
        ]
        return team_moe

    elif config.variant == "adaptive_k":
        return AdaptiveKMoE(config)

    elif config.variant == "tiered_lightweight":
        return TieredLightweightMoE(config, layer_idx)

    elif config.variant == "tiered_circuit_teams":
        return TieredCircuitTeams(
            config,
            layer_idx,
            circuit_definitions=config.circuit_definitions if config.circuit_definitions else None,
            use_lightweight_teams=config.use_lightweight_teams,
        )

    else:  # standard
        from .moe import MoE

        ffn_config = FFNConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            bias=config.bias,
        )
        return MoE(ffn_config)


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Config
    "ExperimentalMoEConfig",
    # Routers
    "CompactNonlinearRouter",
    "AdaptiveKRouter",
    "AttentionAugmentedRouter",
    "CircuitRouter",
    # MoE Variants
    "TieredMoE",
    "CircuitMoE",
    "LayerPairCircuitMoE",  # New: uses 87.5% layer-pair consistency
    "TeamMoE",
    "AdaptiveKMoE",
    "TieredLightweightMoE",  # New: ~90% param reduction
    "TieredCircuitTeams",
    # Teams
    "ExpertTeam",
    "LightweightTeam",
    # Utilities
    "discover_circuits",
    "compute_circuit_overlap",
    "validate_cold_experts",
    # Factory
    "create_experimental_moe",
]
