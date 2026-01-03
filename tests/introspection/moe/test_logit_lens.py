"""Tests for MoE logit lens analysis."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.introspection.moe.config import MoECaptureConfig
from chuk_lazarus.introspection.moe.hooks import MoEHooks
from chuk_lazarus.introspection.moe.logit_lens import (
    ExpertLogitContribution,
    LayerRoutingSnapshot,
    MoELogitLens,
    analyze_expert_vocabulary,
)

# =============================================================================
# Mock Models
# =============================================================================


class MockRouter(nn.Module):
    """Mock router for testing."""

    def __init__(self, num_experts: int = 4, num_experts_per_tok: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.weight = mx.random.normal((num_experts, 32)) * 0.02
        self.bias = mx.zeros((num_experts,))


class MockExpert(nn.Module):
    """Mock expert for testing."""

    def __init__(self, hidden_size: int = 32, intermediate_size: int = 64):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(mx.maximum(self.up_proj(x), 0))


class MockMoE(nn.Module):
    """Mock MoE layer for testing."""

    def __init__(self, hidden_size: int = 32, num_experts: int = 4):
        super().__init__()
        self.router = MockRouter(num_experts)
        self.experts = [MockExpert(hidden_size) for _ in range(num_experts)]

    def __call__(self, x: mx.array) -> mx.array:
        return x


class MockLayer(nn.Module):
    """Mock transformer layer."""

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.mlp = MockMoE(hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp(x)


class MockMoEModel(nn.Module):
    """Mock MoE model for testing."""

    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 32,
        num_layers: int = 2,
        num_experts: int = 4,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = [MockLayer(hidden_size) for _ in range(num_layers)]
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3, 4, 5]

    def decode(self, ids) -> str:
        return "token"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def moe_model():
    """Create mock MoE model."""
    return MockMoEModel(vocab_size=100, hidden_size=32, num_layers=2, num_experts=4)


@pytest.fixture
def hooks_with_data(moe_model):
    """Create hooks with pre-populated test data."""
    hooks = MoEHooks(moe_model)
    hooks.configure(MoECaptureConfig())

    # Populate state
    hooks.moe_state.selected_experts[0] = mx.array(
        [
            [[0, 1], [0, 2], [1, 3], [0, 1], [0, 1]],
        ]
    )
    hooks.moe_state.selected_experts[1] = mx.array(
        [
            [[2, 3], [1, 2], [0, 3], [2, 3], [1, 2]],
        ]
    )
    hooks.moe_state.router_weights[0] = mx.array(
        [
            [0.6, 0.4],
            [0.5, 0.5],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.6, 0.4],
        ]
    )
    hooks.moe_state.router_logits[0] = mx.array(
        [
            [1.0, 2.0, 0.5, 0.3],
            [1.5, 1.5, 1.0, 0.5],
            [0.5, 2.0, 1.0, 1.5],
            [1.0, 2.0, 0.5, 0.3],
            [1.0, 2.0, 0.5, 0.3],
        ]
    )

    return hooks


# =============================================================================
# Tests
# =============================================================================


class TestExpertLogitContribution:
    """Tests for ExpertLogitContribution model."""

    def test_creation(self):
        """Test model creation."""
        contrib = ExpertLogitContribution(
            layer_idx=0,
            expert_idx=2,
            top_tokens=("def", "class"),
            top_logits=(1.5, 1.2),
            top_token_ids=(100, 101),
            activation_weight=0.6,
        )
        assert contrib.layer_idx == 0
        assert contrib.expert_idx == 2
        assert contrib.activation_weight == 0.6

    def test_defaults(self):
        """Test default values."""
        contrib = ExpertLogitContribution(
            layer_idx=0,
            expert_idx=0,
            activation_weight=0.5,
        )
        assert contrib.top_tokens == ()
        assert contrib.top_logits == ()


class TestLayerRoutingSnapshot:
    """Tests for LayerRoutingSnapshot model."""

    def test_creation(self):
        """Test model creation."""
        snapshot = LayerRoutingSnapshot(
            layer_idx=4,
            selected_experts=(0, 2),
            expert_weights=(0.6, 0.4),
            router_entropy=1.5,
            top_token="hello",
            top_token_prob=0.9,
        )
        assert snapshot.layer_idx == 4
        assert snapshot.selected_experts == (0, 2)
        assert snapshot.router_entropy == 1.5

    def test_defaults(self):
        """Test default values."""
        snapshot = LayerRoutingSnapshot(
            layer_idx=0,
            router_entropy=1.0,
        )
        assert snapshot.selected_experts == ()
        assert snapshot.top_token == ""
        assert snapshot.top_token_prob == 0.0


class TestMoELogitLens:
    """Tests for MoELogitLens class."""

    def test_initialization(self, hooks_with_data):
        """Test logit lens initialization."""
        lens = MoELogitLens(hooks_with_data, MockTokenizer())

        assert lens.hooks is hooks_with_data
        assert lens.tokenizer is not None

    def test_initialization_no_tokenizer(self, hooks_with_data):
        """Test initialization without tokenizer."""
        lens = MoELogitLens(hooks_with_data)

        assert lens.hooks is hooks_with_data
        assert lens.tokenizer is None

    def test_get_expert_contributions(self, hooks_with_data):
        """Test getting expert contributions."""
        lens = MoELogitLens(hooks_with_data)
        contributions = lens.get_expert_contributions(layer_idx=0, position=-1)

        assert isinstance(contributions, list)
        for contrib in contributions:
            assert isinstance(contrib, ExpertLogitContribution)

    def test_get_expert_contributions_missing_layer(self, hooks_with_data):
        """Test with missing layer."""
        lens = MoELogitLens(hooks_with_data)
        contributions = lens.get_expert_contributions(layer_idx=99)

        assert contributions == []

    def test_get_routing_evolution(self, hooks_with_data):
        """Test getting routing evolution."""
        lens = MoELogitLens(hooks_with_data)
        evolution = lens.get_routing_evolution(position=-1)

        assert isinstance(evolution, list)
        assert len(evolution) == 2  # Two layers with data
        for snapshot in evolution:
            assert isinstance(snapshot, LayerRoutingSnapshot)

    def test_get_routing_evolution_empty(self, moe_model):
        """Test routing evolution with no data."""
        hooks = MoEHooks(moe_model)
        lens = MoELogitLens(hooks)
        evolution = lens.get_routing_evolution()

        assert evolution == []

    def test_find_routing_divergence(self, hooks_with_data):
        """Test finding routing divergence."""
        lens = MoELogitLens(hooks_with_data)
        divergences = lens.find_routing_divergence(position=-1)

        assert isinstance(divergences, list)
        for div in divergences:
            assert isinstance(div, tuple)
            assert len(div) == 3  # (layer_a, layer_b, diff_set)

    def test_print_routing_evolution(self, hooks_with_data, capsys):
        """Test printing routing evolution."""
        lens = MoELogitLens(hooks_with_data)
        lens.print_routing_evolution(position=-1)

        captured = capsys.readouterr()
        assert "Routing Evolution" in captured.out or "Layer" in captured.out

    def test_print_routing_evolution_empty(self, moe_model, capsys):
        """Test printing with no data."""
        hooks = MoEHooks(moe_model)
        lens = MoELogitLens(hooks)
        lens.print_routing_evolution()

        captured = capsys.readouterr()
        assert "No routing data" in captured.out


class TestAnalyzeExpertVocabulary:
    """Tests for analyze_expert_vocabulary function."""

    def test_basic_analysis(self, moe_model):
        """Test basic vocabulary analysis."""
        result = analyze_expert_vocabulary(
            moe_model,
            layer_idx=0,
            expert_idx=0,
            tokenizer=MockTokenizer(),
            top_k=10,
        )

        assert isinstance(result, dict)
        assert "expert_idx" in result
        assert result["expert_idx"] == 0

    def test_layer_out_of_range(self, moe_model):
        """Test with out of range layer."""
        result = analyze_expert_vocabulary(
            moe_model,
            layer_idx=99,
            expert_idx=0,
            tokenizer=MockTokenizer(),
        )

        assert "error" in result

    def test_expert_out_of_range(self, moe_model):
        """Test with out of range expert."""
        result = analyze_expert_vocabulary(
            moe_model,
            layer_idx=0,
            expert_idx=99,
            tokenizer=MockTokenizer(),
        )

        assert "error" in result
