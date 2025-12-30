"""Tests for analyzer utils module."""

import mlx.core as mx

from chuk_lazarus.introspection.analyzer.utils import (
    compute_entropy,
    compute_js_divergence,
    compute_kl_divergence,
    get_layers_to_capture,
)


class TestComputeEntropy:
    """Tests for compute_entropy function."""

    def test_uniform_distribution(self):
        """Test entropy of uniform distribution."""
        # Uniform over 4 elements: entropy = ln(4) ≈ 1.386
        probs = mx.array([0.25, 0.25, 0.25, 0.25])
        entropy = compute_entropy(probs)
        assert abs(entropy - 1.386) < 0.01

    def test_deterministic_distribution(self):
        """Test entropy of deterministic distribution."""
        # One element has probability 1
        probs = mx.array([1.0, 0.0, 0.0, 0.0])
        entropy = compute_entropy(probs)
        assert entropy < 0.001  # Should be close to 0

    def test_peaked_distribution(self):
        """Test entropy of peaked distribution."""
        probs = mx.array([0.9, 0.05, 0.03, 0.02])
        entropy = compute_entropy(probs)
        # Should be low but not zero
        assert 0 < entropy < 1.0


class TestComputeKLDivergence:
    """Tests for compute_kl_divergence function."""

    def test_same_distribution(self):
        """Test KL divergence of identical distributions."""
        p = mx.array([0.5, 0.3, 0.2])
        q = mx.array([0.5, 0.3, 0.2])
        kl = compute_kl_divergence(p, q)
        assert abs(kl) < 0.001

    def test_different_distributions(self):
        """Test KL divergence of different distributions."""
        p = mx.array([0.9, 0.05, 0.05])
        q = mx.array([0.33, 0.33, 0.34])
        kl = compute_kl_divergence(p, q)
        assert kl > 0  # KL should be positive

    def test_non_negative(self):
        """Test that KL is always non-negative."""
        p = mx.array([0.5, 0.5])
        q = mx.array([0.8, 0.2])
        kl = compute_kl_divergence(p, q)
        assert kl >= 0


class TestComputeJSDivergence:
    """Tests for compute_js_divergence function."""

    def test_same_distribution(self):
        """Test JS divergence of identical distributions."""
        p = mx.array([0.5, 0.3, 0.2])
        q = mx.array([0.5, 0.3, 0.2])
        js = compute_js_divergence(p, q)
        assert abs(js) < 0.001

    def test_different_distributions(self):
        """Test JS divergence of different distributions."""
        p = mx.array([0.9, 0.1])
        q = mx.array([0.1, 0.9])
        js = compute_js_divergence(p, q)
        assert js > 0

    def test_symmetry(self):
        """Test that JS divergence is symmetric."""
        p = mx.array([0.7, 0.3])
        q = mx.array([0.4, 0.6])
        js_pq = compute_js_divergence(p, q)
        js_qp = compute_js_divergence(q, p)
        assert abs(js_pq - js_qp) < 0.001


class TestGetLayersToCapture:
    """Tests for get_layers_to_capture function."""

    def test_all_strategy(self):
        """Test 'all' strategy."""
        layers = get_layers_to_capture(12, "all")
        assert layers == list(range(12))

    def test_first_last_strategy(self):
        """Test 'first_last' strategy."""
        layers = get_layers_to_capture(12, "first_last")
        assert layers == [0, 11]

    def test_evenly_spaced_default(self):
        """Test 'evenly_spaced' with default step."""
        layers = get_layers_to_capture(12, "evenly_spaced", layer_step=4)
        assert 0 in layers
        assert 11 in layers  # Last layer always included
        assert len(layers) == 4  # 0, 4, 8, 11

    def test_evenly_spaced_custom_step(self):
        """Test 'evenly_spaced' with custom step."""
        layers = get_layers_to_capture(10, "evenly_spaced", layer_step=2)
        assert layers == [0, 2, 4, 6, 8, 9]

    def test_custom_strategy_with_layers(self):
        """Test 'custom' strategy with specified layers."""
        layers = get_layers_to_capture(12, "custom", custom_layers=[0, 5, 10])
        assert layers == [0, 5, 10]

    def test_custom_strategy_no_layers(self):
        """Test 'custom' strategy without layers falls back to first_last."""
        layers = get_layers_to_capture(12, "custom")
        assert layers == [0, 11]

    def test_custom_strategy_deduplicates(self):
        """Test 'custom' removes duplicates and sorts."""
        layers = get_layers_to_capture(12, "custom", custom_layers=[5, 0, 5, 10, 0])
        assert layers == [0, 5, 10]
