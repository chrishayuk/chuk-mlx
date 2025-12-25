"""
Tests for Rotary Position Embeddings (RoPE).
"""

import mlx.core as mx

from chuk_lazarus.models_v2.components.embeddings.rope import (
    RoPE,
    apply_rope_manual,
    compute_rope_frequencies,
)
from chuk_lazarus.models_v2.core.config import RoPEConfig


class TestRoPE:
    """Tests for Rotary Position Embeddings."""

    def test_basic_creation(self):
        """Test basic RoPE creation."""
        config = RoPEConfig(theta=10000.0, max_position_embeddings=4096)
        rope = RoPE(config, dims=128)
        assert rope.dims == 128
        assert rope.theta == 10000.0

    def test_forward(self):
        """Test forward pass."""
        config = RoPEConfig(theta=10000.0)
        rope = RoPE(config, dims=64)

        x = mx.random.normal((2, 8, 10, 64))  # (batch, heads, seq, head_dim)
        output = rope(x)
        assert output.shape == x.shape

    def test_forward_with_offset(self):
        """Test forward with offset for KV cache."""
        config = RoPEConfig(theta=10000.0)
        rope = RoPE(config, dims=64)

        x = mx.random.normal((2, 8, 1, 64))  # Single position
        output = rope(x, offset=50)
        assert output.shape == x.shape

    def test_rotate_half(self):
        """Test rotate_half method."""
        config = RoPEConfig(theta=10000.0)
        rope = RoPE(config, dims=64)

        x = mx.random.normal((2, 8, 10, 64))
        rotated = rope.rotate_half(x)
        assert rotated.shape == x.shape

    def test_from_config(self):
        """Test from_config class method."""
        config = RoPEConfig(theta=10000.0, max_position_embeddings=2048)
        rope = RoPE.from_config(config, head_dim=128)
        assert rope.dims == 128

    def test_scaling_factor(self):
        """Test RoPE with scaling factor."""
        config = RoPEConfig(theta=10000.0, scaling_factor=2.0)
        rope = RoPE(config, dims=64)
        assert rope.scaling_factor == 2.0

    def test_traditional_mode(self):
        """Test traditional vs non-traditional mode."""
        config_trad = RoPEConfig(theta=10000.0, traditional=True)
        config_new = RoPEConfig(theta=10000.0, traditional=False)

        rope_trad = RoPE(config_trad, dims=64)
        rope_new = RoPE(config_new, dims=64)

        assert rope_trad.traditional is True
        assert rope_new.traditional is False


class TestRoPEFunctions:
    """Tests for RoPE functional API."""

    def test_compute_rope_frequencies(self):
        """Test frequency computation."""
        cos, sin = compute_rope_frequencies(dim=64, max_seq_len=100)
        assert cos.shape == (100, 64)
        assert sin.shape == (100, 64)

    def test_compute_rope_frequencies_with_scaling(self):
        """Test frequency computation with scaling."""
        cos, sin = compute_rope_frequencies(
            dim=64,
            max_seq_len=100,
            scaling_factor=2.0,
        )
        assert cos.shape == (100, 64)
        assert sin.shape == (100, 64)

    def test_apply_rope_manual(self):
        """Test manual RoPE application."""
        dim = 64
        seq_len = 20
        cos, sin = compute_rope_frequencies(dim=dim, max_seq_len=100)

        q = mx.random.normal((2, 8, seq_len, dim))
        k = mx.random.normal((2, 8, seq_len, dim))

        q_rot, k_rot = apply_rope_manual(q, k, cos, sin, offset=0)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_apply_rope_manual_with_offset(self):
        """Test manual RoPE with offset."""
        dim = 64
        cos, sin = compute_rope_frequencies(dim=dim, max_seq_len=100)

        q = mx.random.normal((2, 8, 5, dim))
        k = mx.random.normal((2, 8, 5, dim))

        q_rot, k_rot = apply_rope_manual(q, k, cos, sin, offset=20)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestRoPEGradients:
    """Tests for gradient flow through RoPE."""

    def test_rope_gradients(self):
        """Test gradients flow through RoPE."""
        config = RoPEConfig(theta=10000.0)
        rope = RoPE(config, dims=32)

        x = mx.random.normal((1, 4, 5, 32))

        def loss_fn(x):
            out = rope(x)
            return mx.mean(out**2)

        loss = loss_fn(x)
        assert loss.item() > 0
