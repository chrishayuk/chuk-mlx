"""
Tests for learned position embeddings.
"""

import mlx.core as mx

from chuk_lazarus.models_v2.components.embeddings.learned import (
    LearnedPositionEmbedding,
    create_learned_position_embedding,
)


class TestLearnedPositionEmbedding:
    """Tests for learned position embeddings."""

    def test_basic_creation(self):
        """Test basic creation."""
        pos_embed = LearnedPositionEmbedding(
            max_position_embeddings=512,
            hidden_size=768,
        )
        assert pos_embed.max_position_embeddings == 512
        assert pos_embed.hidden_size == 768

    def test_forward(self):
        """Test forward pass."""
        pos_embed = LearnedPositionEmbedding(
            max_position_embeddings=512,
            hidden_size=256,
        )
        output = pos_embed(seq_len=10)
        assert output.shape == (1, 10, 256)

    def test_forward_with_offset(self):
        """Test forward with offset for cached generation."""
        pos_embed = LearnedPositionEmbedding(
            max_position_embeddings=512,
            hidden_size=256,
        )
        output = pos_embed(seq_len=5, offset=10)
        assert output.shape == (1, 5, 256)

    def test_forward_with_input(self):
        """Test forward matching input shape."""
        pos_embed = LearnedPositionEmbedding(
            max_position_embeddings=512,
            hidden_size=256,
        )
        input_ids = mx.zeros((4, 20), dtype=mx.int32)
        output = pos_embed.forward_with_input(input_ids)
        assert output.shape == (4, 20, 256)

    def test_forward_with_input_and_offset(self):
        """Test forward_with_input with offset."""
        pos_embed = LearnedPositionEmbedding(
            max_position_embeddings=512,
            hidden_size=256,
        )
        input_ids = mx.zeros((2, 10), dtype=mx.int32)
        output = pos_embed.forward_with_input(input_ids, offset=5)
        assert output.shape == (2, 10, 256)

    def test_factory_function(self):
        """Test factory function."""
        pos_embed = create_learned_position_embedding(
            max_position_embeddings=1024,
            hidden_size=512,
        )
        assert isinstance(pos_embed, LearnedPositionEmbedding)
        assert pos_embed.max_position_embeddings == 1024
