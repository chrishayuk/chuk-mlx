"""
Tests for sinusoidal position embeddings.
"""

import mlx.core as mx
import pytest

from chuk_lazarus.models_v2.components.embeddings.sinusoidal import (
    SinusoidalPositionEmbedding,
    create_sinusoidal_position_embedding,
)


class TestSinusoidalPositionEmbedding:
    """Tests for sinusoidal position embeddings."""

    def test_basic_creation(self):
        """Test basic creation."""
        pos_embed = SinusoidalPositionEmbedding(
            max_position_embeddings=512,
            hidden_size=256,
        )
        assert pos_embed.max_position_embeddings == 512
        assert pos_embed.hidden_size == 256

    def test_odd_hidden_size_raises(self):
        """Test that odd hidden_size raises error."""
        with pytest.raises(ValueError, match="must be even"):
            SinusoidalPositionEmbedding(
                max_position_embeddings=512,
                hidden_size=255,
            )

    def test_forward(self):
        """Test forward pass."""
        pos_embed = SinusoidalPositionEmbedding(
            max_position_embeddings=512,
            hidden_size=256,
        )
        output = pos_embed(seq_len=10)
        assert output.shape == (1, 10, 256)

    def test_forward_with_offset(self):
        """Test forward with offset."""
        pos_embed = SinusoidalPositionEmbedding(
            max_position_embeddings=512,
            hidden_size=256,
        )
        output = pos_embed(seq_len=5, offset=10)
        assert output.shape == (1, 5, 256)

    def test_forward_with_input(self):
        """Test forward matching input shape."""
        pos_embed = SinusoidalPositionEmbedding(
            max_position_embeddings=512,
            hidden_size=256,
        )
        input_ids = mx.zeros((4, 20), dtype=mx.int32)
        output = pos_embed.forward_with_input(input_ids)
        assert output.shape == (4, 20, 256)

    def test_forward_with_input_and_offset(self):
        """Test forward_with_input with offset."""
        pos_embed = SinusoidalPositionEmbedding(
            max_position_embeddings=512,
            hidden_size=256,
        )
        input_ids = mx.zeros((2, 10), dtype=mx.int32)
        output = pos_embed.forward_with_input(input_ids, offset=5)
        assert output.shape == (2, 10, 256)

    def test_factory_function(self):
        """Test factory function."""
        pos_embed = create_sinusoidal_position_embedding(
            max_position_embeddings=1024,
            hidden_size=512,
        )
        assert isinstance(pos_embed, SinusoidalPositionEmbedding)

    def test_embeddings_are_fixed(self):
        """Test that embeddings are fixed (not learnable)."""
        pos_embed = SinusoidalPositionEmbedding(
            max_position_embeddings=100,
            hidden_size=64,
        )
        # Access internal embeddings
        embed1 = pos_embed(seq_len=10)
        embed2 = pos_embed(seq_len=10)
        assert mx.allclose(embed1, embed2)
