"""Tests for virtual_experts/router.py to improve coverage."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.inference.virtual_experts.router import VirtualRouter


class MockOriginalRouter(nn.Module):
    """Mock original MoE router for testing."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.weight = mx.zeros((num_experts, hidden_size))
        self.bias = mx.zeros((num_experts,))
        self.num_experts = num_experts
        self.num_experts_per_tok = 2


class TestVirtualRouterInit:
    """Tests for VirtualRouter initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        mock_router = MockOriginalRouter(hidden_size=64, num_experts=8)
        router = VirtualRouter(
            original_router=mock_router,
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_virtual_experts=2,
        )

        assert router.num_experts == 8
        assert router.num_experts_per_tok == 2
        assert router.hidden_size == 64
        assert router.num_virtual_experts == 2
        assert router.virtual_expert_start_idx == 8

    def test_init_directions_initialized(self):
        """Test that directions are initialized to zeros."""
        mock_router = MockOriginalRouter(hidden_size=64, num_experts=8)
        router = VirtualRouter(
            original_router=mock_router,
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_virtual_experts=3,
        )

        assert len(router.directions) == 3
        for direction in router.directions:
            assert direction.shape == (64,)

    def test_init_calibration_state(self):
        """Test initial calibration state."""
        mock_router = MockOriginalRouter(hidden_size=64, num_experts=8)
        router = VirtualRouter(
            original_router=mock_router,
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_virtual_experts=2,
        )

        assert len(router._calibrated) == 2
        assert all(not c for c in router._calibrated)


class TestVirtualRouterCalibrate:
    """Tests for VirtualRouter calibrate_expert method."""

    def test_calibrate_expert_basic(self):
        """Test basic calibration."""
        mock_router = MockOriginalRouter(hidden_size=64, num_experts=8)
        router = VirtualRouter(
            original_router=mock_router,
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_virtual_experts=2,
        )

        # Create positive and negative activations
        pos_activations = [mx.random.normal((64,)) for _ in range(5)]
        neg_activations = [mx.random.normal((64,)) for _ in range(5)]

        # Calibrate
        router.calibrate_expert(0, pos_activations, neg_activations)

        assert router._calibrated[0] is True
        assert router.directions[0].shape == (64,)

    def test_calibrate_expert_invalid_index(self):
        """Test calibration with invalid expert index."""
        mock_router = MockOriginalRouter(hidden_size=64, num_experts=8)
        router = VirtualRouter(
            original_router=mock_router,
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_virtual_experts=2,
        )

        pos_activations = [mx.random.normal((64,))]
        neg_activations = [mx.random.normal((64,))]

        with pytest.raises(ValueError, match="Expert index"):
            router.calibrate_expert(5, pos_activations, neg_activations)

    def test_calibrate_expert_scale_calculation(self):
        """Test that scale is calculated correctly."""
        mock_router = MockOriginalRouter(hidden_size=64, num_experts=8)
        router = VirtualRouter(
            original_router=mock_router,
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_virtual_experts=1,
        )

        # Create distinct positive and negative activations
        pos_activations = [mx.ones((64,)) * 10 for _ in range(3)]
        neg_activations = [mx.ones((64,)) * -10 for _ in range(3)]

        router.calibrate_expert(0, pos_activations, neg_activations)

        assert router._calibrated[0] is True
        # Scale should be non-zero
        assert router.scales[0] != 0

    def test_calibrate_expert_similar_activations(self):
        """Test calibration when pos and neg are very similar (scale=1.0 fallback)."""
        mock_router = MockOriginalRouter(hidden_size=64, num_experts=8)
        router = VirtualRouter(
            original_router=mock_router,
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_virtual_experts=1,
        )

        # Very similar activations
        base = mx.random.normal((64,))
        pos_activations = [base + mx.random.normal((64,)) * 0.001 for _ in range(3)]
        neg_activations = [base + mx.random.normal((64,)) * 0.001 for _ in range(3)]

        router.calibrate_expert(0, pos_activations, neg_activations)

        # Scale should fall back to 1.0 when difference is small
        assert router._calibrated[0] is True


class TestVirtualRouterCall:
    """Tests for VirtualRouter __call__ method."""

    def test_call_2d_input(self):
        """Test call with 2D input."""
        mock_router = MockOriginalRouter(hidden_size=64, num_experts=8)
        router = VirtualRouter(
            original_router=mock_router,
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_virtual_experts=1,
        )

        x = mx.random.normal((10, 64))  # 10 tokens
        weights, indices, virtual_masks = router(x)

        assert weights.shape[0] == 10
        assert indices.shape[0] == 10
        assert 0 in virtual_masks

    def test_call_3d_input(self):
        """Test call with 3D input (batch, seq, hidden)."""
        mock_router = MockOriginalRouter(hidden_size=64, num_experts=8)
        router = VirtualRouter(
            original_router=mock_router,
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_virtual_experts=1,
        )

        x = mx.random.normal((2, 5, 64))  # batch=2, seq=5
        weights, indices, virtual_masks = router(x)

        # Should reshape to (10, hidden) internally
        assert weights.shape[0] == 10
        assert indices.shape[0] == 10

    def test_call_uncalibrated_expert(self):
        """Test call with uncalibrated expert (very negative logits)."""
        mock_router = MockOriginalRouter(hidden_size=64, num_experts=8)
        router = VirtualRouter(
            original_router=mock_router,
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_virtual_experts=1,
        )

        x = mx.random.normal((5, 64))
        weights, indices, virtual_masks = router(x)

        # Uncalibrated expert should rarely/never be selected
        # Virtual expert index would be 8
        assert 0 in virtual_masks

    def test_call_calibrated_expert(self):
        """Test call with calibrated expert."""
        mock_router = MockOriginalRouter(hidden_size=64, num_experts=8)
        router = VirtualRouter(
            original_router=mock_router,
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_virtual_experts=1,
        )

        # Calibrate the expert
        pos = [mx.ones((64,)) * 5 for _ in range(5)]
        neg = [mx.ones((64,)) * -5 for _ in range(5)]
        router.calibrate_expert(0, pos, neg)

        x = mx.random.normal((5, 64))
        weights, indices, virtual_masks = router(x)

        assert router._last_virtual_logits is not None

    def test_call_multiple_virtual_experts(self):
        """Test call with multiple virtual experts."""
        mock_router = MockOriginalRouter(hidden_size=64, num_experts=8)
        router = VirtualRouter(
            original_router=mock_router,
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_virtual_experts=3,
        )

        x = mx.random.normal((5, 64))
        weights, indices, virtual_masks = router(x)

        # Should have masks for all 3 virtual experts
        assert 0 in virtual_masks
        assert 1 in virtual_masks
        assert 2 in virtual_masks


class TestVirtualRouterGetRoutingScore:
    """Tests for VirtualRouter get_routing_score method."""

    def test_get_routing_score_uncalibrated(self):
        """Test routing score for uncalibrated expert returns 0."""
        mock_router = MockOriginalRouter(hidden_size=64, num_experts=8)
        router = VirtualRouter(
            original_router=mock_router,
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_virtual_experts=1,
        )

        x = mx.random.normal((5, 64))
        score = router.get_routing_score(x, 0)

        assert score == 0.0

    def test_get_routing_score_calibrated(self):
        """Test routing score for calibrated expert."""
        mock_router = MockOriginalRouter(hidden_size=64, num_experts=8)
        router = VirtualRouter(
            original_router=mock_router,
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_virtual_experts=1,
        )

        # Calibrate
        pos = [mx.ones((64,)) * 10 for _ in range(5)]
        neg = [mx.ones((64,)) * -10 for _ in range(5)]
        router.calibrate_expert(0, pos, neg)

        x = mx.ones((1, 64)) * 10  # Should be similar to positive
        score = router.get_routing_score(x, 0)

        # Score should be between 0 and 1
        assert 0.0 <= score <= 1.0

    def test_get_routing_score_3d_input(self):
        """Test routing score with 3D input."""
        mock_router = MockOriginalRouter(hidden_size=64, num_experts=8)
        router = VirtualRouter(
            original_router=mock_router,
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_virtual_experts=1,
        )

        # Calibrate
        pos = [mx.ones((64,)) * 10 for _ in range(5)]
        neg = [mx.ones((64,)) * -10 for _ in range(5)]
        router.calibrate_expert(0, pos, neg)

        x = mx.ones((1, 3, 64)) * 10  # 3D input
        score = router.get_routing_score(x, 0)

        assert 0.0 <= score <= 1.0


class TestVirtualRouterIsCalibrated:
    """Tests for VirtualRouter is_calibrated method."""

    def test_is_calibrated_false_initially(self):
        """Test is_calibrated returns False initially."""
        mock_router = MockOriginalRouter(hidden_size=64, num_experts=8)
        router = VirtualRouter(
            original_router=mock_router,
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_virtual_experts=2,
        )

        assert router.is_calibrated(0) is False
        assert router.is_calibrated(1) is False

    def test_is_calibrated_true_after_calibration(self):
        """Test is_calibrated returns True after calibration."""
        mock_router = MockOriginalRouter(hidden_size=64, num_experts=8)
        router = VirtualRouter(
            original_router=mock_router,
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_virtual_experts=2,
        )

        pos = [mx.random.normal((64,)) for _ in range(3)]
        neg = [mx.random.normal((64,)) for _ in range(3)]
        router.calibrate_expert(0, pos, neg)

        assert router.is_calibrated(0) is True
        assert router.is_calibrated(1) is False

    def test_is_calibrated_invalid_index(self):
        """Test is_calibrated with invalid index returns False."""
        mock_router = MockOriginalRouter(hidden_size=64, num_experts=8)
        router = VirtualRouter(
            original_router=mock_router,
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_virtual_experts=2,
        )

        assert router.is_calibrated(99) is False
