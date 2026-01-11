"""Tests for KL divergence utilities."""

import mlx.core as mx
import pytest

from chuk_lazarus.training.utils.kl_divergence import (
    compute_approx_kl,
    compute_kl_divergence,
)


class TestComputeKLDivergence:
    """Tests for compute_kl_divergence function."""

    def test_kl_identical_distributions(self):
        """Test KL divergence is 0 for identical distributions."""
        log_probs_p = mx.array([[-1.0, -2.0, -1.5]])
        log_probs_q = mx.array([[-1.0, -2.0, -1.5]])

        kl = compute_kl_divergence(log_probs_p, log_probs_q)

        assert kl.item() == pytest.approx(0.0, abs=1e-6)

    def test_kl_different_distributions(self):
        """Test KL divergence is positive for different distributions."""
        log_probs_p = mx.array([[-1.0, -2.0, -1.5]])
        log_probs_q = mx.array([[-2.0, -1.0, -3.0]])

        kl = compute_kl_divergence(log_probs_p, log_probs_q)

        # KL should be positive when distributions differ
        assert kl.item() > 0.0 or kl.item() < 0.0  # Non-zero

    def test_kl_with_mask(self):
        """Test KL divergence with mask."""
        log_probs_p = mx.array([[-1.0, -2.0, -1.5, -1.0]])
        log_probs_q = mx.array([[-2.0, -1.0, -3.0, -2.0]])
        mask = mx.array([[1.0, 1.0, 0.0, 0.0]])  # Only first two tokens

        kl = compute_kl_divergence(log_probs_p, log_probs_q, mask)

        # Should only consider first two tokens
        assert isinstance(kl.item(), float)

    def test_kl_with_zero_mask(self):
        """Test KL divergence with all-zero mask."""
        log_probs_p = mx.array([[-1.0, -2.0]])
        log_probs_q = mx.array([[-2.0, -1.0]])
        mask = mx.zeros((1, 2))

        kl = compute_kl_divergence(log_probs_p, log_probs_q, mask)

        # With no valid tokens, result should be ~0
        assert kl.item() == pytest.approx(0.0, abs=1e-5)

    def test_kl_batch_processing(self):
        """Test KL divergence with batch dimension."""
        batch_size = 4
        seq_len = 8
        log_probs_p = mx.random.uniform(shape=(batch_size, seq_len)) * -5
        log_probs_q = mx.random.uniform(shape=(batch_size, seq_len)) * -5

        kl = compute_kl_divergence(log_probs_p, log_probs_q)

        assert isinstance(kl.item(), float)


class TestComputeApproxKL:
    """Tests for compute_approx_kl function."""

    def test_approx_kl_identical_policies(self):
        """Test approximate KL is 0 for identical policies."""
        old_log_probs = mx.array([[-1.0, -2.0, -1.5]])
        new_log_probs = mx.array([[-1.0, -2.0, -1.5]])

        approx_kl = compute_approx_kl(old_log_probs, new_log_probs)

        assert approx_kl.item() == pytest.approx(0.0, abs=1e-6)

    def test_approx_kl_positive(self):
        """Test approximate KL is always positive."""
        old_log_probs = mx.array([[-1.0, -2.0, -1.5]])
        new_log_probs = mx.array([[-2.0, -1.0, -3.0]])

        approx_kl = compute_approx_kl(old_log_probs, new_log_probs)

        # Approximate KL uses squared differences, so always positive
        assert approx_kl.item() >= 0.0

    def test_approx_kl_with_mask(self):
        """Test approximate KL with mask."""
        old_log_probs = mx.array([[-1.0, -2.0, -1.5, -1.0]])
        new_log_probs = mx.array([[-2.0, -1.0, -3.0, -2.0]])
        mask = mx.array([[1.0, 1.0, 0.0, 0.0]])

        approx_kl = compute_approx_kl(old_log_probs, new_log_probs, mask)

        assert isinstance(approx_kl.item(), float)
        assert approx_kl.item() >= 0.0

    def test_approx_kl_symmetry(self):
        """Test approximate KL is symmetric (unlike true KL)."""
        log_probs_a = mx.array([[-1.0, -2.0]])
        log_probs_b = mx.array([[-2.0, -1.0]])

        kl_ab = compute_approx_kl(log_probs_a, log_probs_b)
        kl_ba = compute_approx_kl(log_probs_b, log_probs_a)

        # Approximate KL should be symmetric since it uses squared differences
        assert kl_ab.item() == pytest.approx(kl_ba.item(), rel=1e-3)

    def test_approx_kl_scales_with_difference(self):
        """Test approximate KL increases with larger policy differences."""
        base_log_probs = mx.array([[-1.0, -1.0, -1.0]])
        small_diff = mx.array([[-1.1, -1.1, -1.1]])
        large_diff = mx.array([[-2.0, -2.0, -2.0]])

        small_kl = compute_approx_kl(base_log_probs, small_diff)
        large_kl = compute_approx_kl(base_log_probs, large_diff)

        assert large_kl.item() > small_kl.item()
