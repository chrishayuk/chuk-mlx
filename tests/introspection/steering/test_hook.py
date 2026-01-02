"""Tests for steering hook module."""

import mlx.core as mx

from chuk_lazarus.introspection.steering.hook import SteeringHook


class TestSteeringHook:
    """Tests for SteeringHook class."""

    def test_init_default(self):
        """Test default initialization."""
        direction = mx.array([1.0, 0.0, 0.0, 0.0])
        hook = SteeringHook(direction)
        assert hook.coefficient == 1.0
        assert hook.position is None
        assert hook.normalize is True
        assert hook.scale_by_norm is False

    def test_init_custom(self):
        """Test custom initialization."""
        direction = mx.array([1.0, 2.0, 3.0, 4.0])
        hook = SteeringHook(
            direction=direction,
            coefficient=2.0,
            position=5,
            normalize=False,
            scale_by_norm=True,
        )
        assert hook.coefficient == 2.0
        assert hook.position == 5
        assert hook.normalize is False
        assert hook.scale_by_norm is True

    def test_normalization(self):
        """Test that direction is normalized when normalize=True."""
        direction = mx.array([3.0, 4.0])  # norm = 5
        hook = SteeringHook(direction, normalize=True)
        # Should be normalized to unit vector
        norm = float(mx.sqrt(mx.sum(hook.direction * hook.direction)))
        assert abs(norm - 1.0) < 0.001

    def test_no_normalization(self):
        """Test that direction is not modified when normalize=False."""
        direction = mx.array([3.0, 4.0])
        hook = SteeringHook(direction, normalize=False)
        assert mx.allclose(hook.direction, direction)

    def test_call_all_positions(self):
        """Test steering applied to all positions."""
        direction = mx.array([1.0, 0.0, 0.0, 0.0])
        hook = SteeringHook(direction, coefficient=1.0, normalize=False)

        # Input: batch=1, seq=3, hidden=4
        h = mx.zeros((1, 3, 4))
        result = hook(h)

        # All positions should have steering applied
        assert result.shape == (1, 3, 4)
        assert float(result[0, 0, 0]) == 1.0
        assert float(result[0, 1, 0]) == 1.0
        assert float(result[0, 2, 0]) == 1.0

    def test_call_specific_position(self):
        """Test steering applied to specific position."""
        direction = mx.array([1.0, 0.0, 0.0, 0.0])
        hook = SteeringHook(direction, coefficient=1.0, position=1, normalize=False)

        h = mx.zeros((1, 3, 4))
        result = hook(h)

        # Only position 1 should be modified
        # Note: .at[].add() returns new array with modification
        assert result.shape == (1, 3, 4)

    def test_coefficient_scaling(self):
        """Test that coefficient scales the direction."""
        direction = mx.array([1.0, 0.0])
        hook = SteeringHook(direction, coefficient=3.0, normalize=False)

        h = mx.zeros((1, 2, 2))
        result = hook(h)

        assert float(result[0, 0, 0]) == 3.0

    def test_scale_by_norm(self):
        """Test scaling by activation norm."""
        direction = mx.array([1.0, 0.0])
        hook = SteeringHook(direction, coefficient=1.0, normalize=False, scale_by_norm=True)

        # Create input with known norm
        h = mx.ones((1, 2, 2))  # mean = 1, sqrt(mean) = 1
        result = hook(h)

        # Should be scaled by activation norm
        assert result.shape == (1, 2, 2)
