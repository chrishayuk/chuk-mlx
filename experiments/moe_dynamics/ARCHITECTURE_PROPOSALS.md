# MoE Architecture Proposals Based on Empirical Findings

Based on the comprehensive MoE dynamics analysis of GPT-OSS-20B, this document proposes architectural improvements that make discovered structures explicit.

## Findings → Architecture Mapping

| Discovered Property | Finding | Architectural Implication |
|--------------------|---------|---------------------------|
| Non-linear attention→routing | 4.3% linear prediction, 0.906 correlation | Router does necessary non-linear work |
| Expert cooperation (k=4 essential) | k=1 breaks output | Cannot simplify to sparse attention |
| Stable circuits (15 pipelines) | 87.5% cross-layer consistency | Structure can be made explicit |
| Cold experts (12.9%) | 50 prunable | Non-uniform allocation viable |
| Middle-layer importance | L8-L17 most differentiated | Depth-varying expert counts |

---

## Proposal 1: Compact Non-Linear Router

**Motivation**: The router isn't redundant—it performs a necessary non-linear transformation. But the full `hidden_dim → num_experts` projection may be overparameterized.

**Evidence**: 0.906 correlation between attention and specific experts suggests structure exists that a smaller model could capture.

```python
class CompactNonlinearRouter(nn.Module):
    """Bottlenecked router that captures non-linear attention→routing mapping."""

    def __init__(self, hidden_dim: int, num_experts: int, bottleneck: int = 64):
        super().__init__()
        # Compress the non-linear mapping through bottleneck
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.up = nn.Linear(bottleneck, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Non-linear compression preserves routing structure
        return self.up(F.gelu(self.down(x)))
```

**Parameters saved**: For hidden_dim=4096, num_experts=32:
- Original: 4096 × 32 = 131K
- Bottleneck=64: 4096 × 64 + 64 × 32 = 264K (worse)
- Bottleneck=32: 4096 × 32 + 32 × 32 = 132K (neutral)
- Bottleneck=16: 4096 × 16 + 16 × 32 = 66K (50% savings)

**Experiment**: Train with bottleneck=16,32,64 and measure routing accuracy vs full router.

---

## Proposal 2: Circuit-Aware Architecture

**Motivation**: We found 15 stable pipelines with 87.5% consistency spanning all 24 layers. Rather than routing independently at each layer, route to functional circuits.

**Evidence**: E6, E15, E31 pipelines have 100% layer coverage with 1.00 consistency.

```python
class CircuitMoE(nn.Module):
    """Route to functional circuits, not individual experts."""

    def __init__(self, hidden_dim: int, num_circuits: int = 15):
        super().__init__()
        # Discovered circuits from analysis
        # Each circuit defines expert assignments across all layers
        self.circuit_definitions = discovered_circuits  # [(l0_e, l1_e, ..., l23_e), ...]

        # Single router decision at layer 0 determines entire path
        self.circuit_router = nn.Linear(hidden_dim, num_circuits)

        # Expert modules per layer
        self.layer_experts = nn.ModuleList([
            nn.ModuleList([Expert() for _ in range(32)])
            for _ in range(24)
        ])

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if layer_idx == 0:
            # Route to circuits once at layer 0
            circuit_weights = F.softmax(self.circuit_router(x), dim=-1)
            self._cached_circuits = torch.topk(circuit_weights, k=4)

        # Look up which expert this circuit uses at this layer
        outputs = []
        for circuit_idx, weight in zip(*self._cached_circuits):
            expert_idx = self.circuit_definitions[circuit_idx][layer_idx]
            outputs.append(weight * self.layer_experts[layer_idx][expert_idx](x))

        return sum(outputs)
```

**Benefits**:
- 24 routing decisions → 1 routing decision
- Guaranteed coherent cross-layer paths
- Matches discovered functional structure

**Experiment**: Extract circuit definitions from trained model, train Circuit-MoE, compare convergence and quality.

---

## Proposal 3: Adaptive-k Routing

**Motivation**: k=1 breaks output, k=4 works. But k=4 for all tokens may be wasteful. Some tokens may need k=2, others k=6.

**Evidence**: Task-aware prediction achieves 94% accuracy at L4, suggesting complexity is predictable early.

```python
class AdaptiveKRouter(nn.Module):
    """Predict required expert count per token."""

    def __init__(self, hidden_dim: int, num_experts: int, min_k: int = 2, max_k: int = 8):
        super().__init__()
        self.expert_router = nn.Linear(hidden_dim, num_experts)
        self.complexity_probe = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.min_k = min_k
        self.max_k = max_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Predict complexity (0-1)
        complexity = self.complexity_probe(x)  # (batch, seq, 1)

        # Map to k value
        k = self.min_k + (complexity * (self.max_k - self.min_k)).int()

        # Get expert logits
        logits = self.expert_router(x)

        # Variable top-k per position
        # (Implementation would need scatter/gather for variable k)
        return logits, k
```

**Expected behavior**:
- Simple tokens (articles, punctuation): k=2
- Standard tokens: k=4
- Complex tokens (math, code): k=6-8

**Experiment**: Train adaptive-k, measure average k used and quality vs fixed k=4.

---

## Proposal 4: Pruned + Tiered MoE

**Motivation**: 12.9% of experts are cold, and middle layers (L8-L17) show maximum differentiation. Non-uniform expert allocation matches actual usage.

**Evidence**:
- Early layers: 4-8% stability, low differentiation
- Middle layers: 10-18% stability, 10/16 unique experts
- Late layers: 6-13% stability, partial convergence

```python
class TieredMoE(nn.Module):
    """Non-uniform expert allocation by layer phase."""

    # Expert counts per layer (example for 24-layer model)
    EXPERT_COUNTS = {
        'early': 16,    # L0-L7: fewer experts (prune cold ones)
        'middle': 32,   # L8-L17: full expert set
        'late': 24,     # L18-L23: moderate experts
    }

    def __init__(self, hidden_dim: int, num_layers: int = 24):
        super().__init__()
        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            if layer_idx < 8:
                num_experts = self.EXPERT_COUNTS['early']
            elif layer_idx < 18:
                num_experts = self.EXPERT_COUNTS['middle']
            else:
                num_experts = self.EXPERT_COUNTS['late']

            self.layers.append(MoELayer(hidden_dim, num_experts))
```

**Parameter savings**:
- Original: 24 layers × 32 experts = 768 experts
- Tiered: 8×16 + 10×32 + 6×24 = 128 + 320 + 144 = 592 experts
- **23% expert reduction**

**Experiment**: Train tiered MoE, validate quality matches full model.

---

## Proposal 5: Cooperative Expert Teams

**Motivation**: Experts must cooperate (k=1 breaks model). Rather than hoping routing finds good combinations, structure cooperation explicitly.

**Evidence**: k=4 is essential, suggesting 4-expert teams are the functional unit.

```python
class ExpertTeam(nn.Module):
    """Team of experts that always activate together."""

    def __init__(self, hidden_dim: int, team_size: int = 4, expert_dim: int = None):
        super().__init__()
        expert_dim = expert_dim or hidden_dim

        # Team members
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, hidden_dim)
            )
            for _ in range(team_size)
        ])

        # Learned combination of team outputs
        self.combiner = nn.Linear(team_size * hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # All team members process input
        outputs = [expert(x) for expert in self.experts]

        # Learned combination
        combined = torch.cat(outputs, dim=-1)
        return self.combiner(combined)


class TeamMoE(nn.Module):
    """MoE with teams instead of individual experts."""

    def __init__(self, hidden_dim: int, num_teams: int = 8):
        super().__init__()
        self.teams = nn.ModuleList([
            ExpertTeam(hidden_dim, team_size=4)
            for _ in range(num_teams)
        ])
        self.team_router = nn.Linear(hidden_dim, num_teams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Route to top-1 team (team handles the 4-way cooperation internally)
        team_logits = self.team_router(x)
        team_weights = F.softmax(team_logits, dim=-1)

        # Weight teams (or just take argmax for hard routing)
        output = sum(
            weight * team(x)
            for weight, team in zip(team_weights.unbind(-1), self.teams)
        )
        return output
```

**Benefits**:
- Cooperation is guaranteed by design
- Fewer routing decisions (route to 8 teams vs 32 experts)
- Team combination is learned, not assumed to be weighted sum

**Experiment**: Compare Team-MoE (8 teams × 4 experts) vs standard MoE (32 experts, k=4).

---

## Experiment Priority

Based on empirical findings, recommended priority:

| Priority | Proposal | Why |
|----------|----------|-----|
| 1 | Tiered MoE | Immediate 23% savings, low risk |
| 2 | Circuit-Aware | Strongest empirical support (87.5% consistency) |
| 3 | Expert Teams | Matches cooperation requirement |
| 4 | Compact Router | Parameter savings, easy to test |
| 5 | Adaptive-k | Most complex, needs careful implementation |

---

## Validation Framework

To validate any proposal:

1. **Train standard MoE** on dataset X
2. **Extract discovered structures** (circuits, cold experts, k-requirements)
3. **Train proposed architecture** with structural priors
4. **Compare**:
   - Convergence speed (training loss curve)
   - Final quality (perplexity, downstream tasks)
   - Parameter efficiency (params per quality point)
   - Inference efficiency (FLOPs, latency)

**Success criteria**: Match or exceed standard MoE quality with fewer parameters or faster training.

---

## Key Insight

The MoE dynamics analysis reveals that **MoE training discovers structure that could be made explicit**:

| Discovered | Explicit Alternative |
|------------|---------------------|
| Non-linear routing | Bottlenecked router |
| Expert cooperation | Expert teams |
| Stable circuits | Circuit-based routing |
| Cold experts | Pruned allocation |
| Layer-phase importance | Tiered expert counts |

The cooperation finding (k=4 essential) is most architecturally significant—it rules out simple sparse attention alternatives and confirms multi-expert mixing does real work.
