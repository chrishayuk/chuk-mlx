"""Tests for MoE router analysis."""

import pytest
import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.introspection.moe.config import MoECaptureConfig
from chuk_lazarus.introspection.moe.hooks import MoECapturedState, MoEHooks
from chuk_lazarus.introspection.moe.models import ExpertUtilization
from chuk_lazarus.introspection.moe.router import (
    analyze_coactivation,
    compare_routing,
    compute_routing_diversity,
    get_dominant_experts,
    get_rare_experts,
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

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        logits = x @ self.weight.T + self.bias
        k = self.num_experts_per_tok
        indices = mx.argsort(logits, axis=-1)[:, -k:]
        weights = mx.softmax(mx.take_along_axis(logits, indices, axis=-1), axis=-1)
        return weights, indices


class MockMoE(nn.Module):
    """Mock MoE layer for testing."""

    def __init__(self, hidden_size: int = 32, num_experts: int = 4):
        super().__init__()
        self.router = MockRouter(num_experts)
        self.experts = [nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)]

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

    # Manually populate state for testing
    hooks.moe_state.selected_experts[0] = mx.array([
        [[0, 1], [0, 2], [1, 3], [0, 1], [0, 1]],
    ])  # batch=1, seq=5, k=2

    hooks.moe_state.selected_experts[1] = mx.array([
        [[2, 3], [1, 2], [0, 3], [2, 3], [1, 2]],
    ])

    hooks.moe_state.router_logits[0] = mx.array([
        [1.0, 2.0, 0.5, 0.3],
        [1.5, 1.5, 1.0, 0.5],
        [0.5, 2.0, 1.0, 1.5],
        [1.0, 2.0, 0.5, 0.3],
        [1.0, 2.0, 0.5, 0.3],
    ])

    return hooks


# =============================================================================
# Tests
# =============================================================================


class TestAnalyzeCoactivation:
    """Tests for analyze_coactivation function."""

    def test_basic_analysis(self, hooks_with_data):
        """Test basic coactivation analysis."""
        analysis = analyze_coactivation(hooks_with_data, layer_idx=0)

        assert analysis is not None
        assert analysis.layer_idx == 0
        assert analysis.total_activations == 5

    def test_finds_top_pairs(self, hooks_with_data):
        """Test finds frequently coactivating pairs."""
        analysis = analyze_coactivation(hooks_with_data, layer_idx=0)

        assert analysis is not None
        assert len(analysis.top_pairs) > 0
        # (0, 1) appears 3 times, should be top pair
        top_pair = analysis.top_pairs[0]
        assert (top_pair.expert_a == 0 and top_pair.expert_b == 1) or \
               (top_pair.expert_a == 1 and top_pair.expert_b == 0)

    def test_missing_layer(self, hooks_with_data):
        """Test with layer not in state."""
        analysis = analyze_coactivation(hooks_with_data, layer_idx=99)
        assert analysis is None


class TestComputeRoutingDiversity:
    """Tests for compute_routing_diversity function."""

    def test_returns_float(self, hooks_with_data):
        """Test diversity returns a float."""
        diversity = compute_routing_diversity(hooks_with_data, layer_idx=0)
        assert isinstance(diversity, float)

    def test_diversity_bounds(self, hooks_with_data):
        """Test diversity is between 0 and 1."""
        diversity = compute_routing_diversity(hooks_with_data, layer_idx=0)
        assert 0.0 <= diversity <= 1.0

    def test_missing_data_returns_zero(self, moe_model):
        """Test returns 0 when no data."""
        hooks = MoEHooks(moe_model)
        diversity = compute_routing_diversity(hooks, layer_idx=0)
        assert diversity == 0.0


class TestGetDominantExperts:
    """Tests for get_dominant_experts function."""

    def test_returns_list(self, hooks_with_data):
        """Test returns list of tuples."""
        dominant = get_dominant_experts(hooks_with_data, layer_idx=0, top_k=2)

        assert isinstance(dominant, list)
        for item in dominant:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_top_k_limit(self, hooks_with_data):
        """Test respects top_k limit."""
        dominant = get_dominant_experts(hooks_with_data, layer_idx=0, top_k=2)
        assert len(dominant) <= 2

    def test_empty_when_no_data(self, moe_model):
        """Test returns empty list when no data."""
        hooks = MoEHooks(moe_model)
        dominant = get_dominant_experts(hooks, layer_idx=0)
        assert dominant == []


class TestGetRareExperts:
    """Tests for get_rare_experts function."""

    def test_returns_list(self, hooks_with_data):
        """Test returns list of expert indices."""
        rare = get_rare_experts(hooks_with_data, layer_idx=0, threshold=0.1)

        assert isinstance(rare, list)
        for item in rare:
            assert isinstance(item, int)

    def test_threshold_filtering(self, hooks_with_data):
        """Test threshold filtering works."""
        # With very high threshold, all experts should be rare
        rare = get_rare_experts(hooks_with_data, layer_idx=0, threshold=1.0)
        assert len(rare) == 4  # All 4 experts

    def test_empty_when_no_data(self, moe_model):
        """Test returns empty list when no data."""
        hooks = MoEHooks(moe_model)
        rare = get_rare_experts(hooks, layer_idx=0)
        assert rare == []


class TestCompareRouting:
    """Tests for compare_routing function."""

    def test_compare_with_self(self, hooks_with_data):
        """Test comparing hooks with itself."""
        result = compare_routing(hooks_with_data, hooks_with_data, layer_idx=0)

        assert isinstance(result, dict)
        assert "overlap_rate" in result
        assert result["overlap_rate"] == 1.0  # Identical data

    def test_compare_different_layers(self, hooks_with_data):
        """Test comparing different hooks."""
        # Create second hooks with different data
        hooks2 = MoEHooks(MockMoEModel())
        hooks2.configure(MoECaptureConfig())
        hooks2.moe_state.selected_experts[0] = mx.array([
            [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        ])

        result = compare_routing(hooks_with_data, hooks2, layer_idx=0)

        assert isinstance(result, dict)
        assert "overlap_rate" in result
        assert result["overlap_rate"] < 1.0  # Different data

    def test_missing_layer_returns_empty(self, hooks_with_data, moe_model):
        """Test with missing layer data."""
        hooks2 = MoEHooks(moe_model)

        result = compare_routing(hooks_with_data, hooks2, layer_idx=0)
        assert result == {}

    def test_shape_mismatch(self, hooks_with_data, moe_model):
        """Test with shape mismatch."""
        hooks2 = MoEHooks(moe_model)
        hooks2.configure(MoECaptureConfig())
        # Different shape
        hooks2.moe_state.selected_experts[0] = mx.array([
            [[0, 1], [0, 1]],  # Only 2 positions
        ])

        result = compare_routing(hooks_with_data, hooks2, layer_idx=0)
        assert result.get("shape_mismatch") == 1.0
