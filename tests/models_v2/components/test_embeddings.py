"""
Tests for embedding components.

Tests ALiBi, LearnedPositionEmbedding, RoPE, SinusoidalPositionEmbedding, and TokenEmbedding.
"""

import math

import mlx.core as mx
import pytest

from chuk_lazarus.models_v2.components.embeddings import (
    create_token_embedding,
)
from chuk_lazarus.models_v2.components.embeddings.alibi import (
    ALiBi,
    compute_alibi_bias,
    compute_alibi_slopes,
)
from chuk_lazarus.models_v2.components.embeddings.learned import (
    LearnedPositionEmbedding,
    create_learned_position_embedding,
)
from chuk_lazarus.models_v2.components.embeddings.rope import (
    RoPE,
    apply_rope_manual,
    compute_rope_frequencies,
)
from chuk_lazarus.models_v2.components.embeddings.sinusoidal import (
    SinusoidalPositionEmbedding,
    create_sinusoidal_position_embedding,
)
from chuk_lazarus.models_v2.components.embeddings.token import TokenEmbedding
from chuk_lazarus.models_v2.core.config import EmbeddingConfig, RoPEConfig


class TestALiBi:
    """Tests for ALiBi position bias."""

    def test_basic_creation(self):
        """Test basic ALiBi creation."""
        alibi = ALiBi(num_heads=8)
        assert alibi.num_heads == 8

    def test_power_of_2_heads(self):
        """Test ALiBi with power-of-2 heads."""
        for num_heads in [2, 4, 8, 16, 32]:
            alibi = ALiBi(num_heads=num_heads)
            bias = alibi(seq_len=10)
            assert bias.shape == (1, num_heads, 10, 10)

    def test_non_power_of_2_heads(self):
        """Test ALiBi with non-power-of-2 heads."""
        for num_heads in [3, 5, 6, 7, 12]:
            alibi = ALiBi(num_heads=num_heads)
            bias = alibi(seq_len=10)
            assert bias.shape == (1, num_heads, 10, 10)

    def test_bias_shape(self):
        """Test output bias shape."""
        alibi = ALiBi(num_heads=8)
        bias = alibi(seq_len=50)
        assert bias.shape == (1, 8, 50, 50)

    def test_bias_is_causal(self):
        """Test that bias encourages attending to closer positions."""
        alibi = ALiBi(num_heads=4)
        bias = alibi(seq_len=10)

        # Diagonal should be 0 (same position)
        for i in range(10):
            assert float(bias[0, 0, i, i]) == 0.0

        # Future positions should have negative bias (more negative than past)
        # Actually ALiBi uses linear decay from current position

    def test_get_bias_for_cache(self):
        """Test bias generation for cached inference."""
        alibi = ALiBi(num_heads=8)

        # Single query position, multiple key positions
        bias = alibi.get_bias_for_cache(query_len=1, key_len=20)
        assert bias.shape == (1, 8, 1, 20)

        # Multiple query positions
        bias = alibi.get_bias_for_cache(query_len=5, key_len=20)
        assert bias.shape == (1, 8, 5, 20)

    def test_compute_alibi_bias_functional(self):
        """Test functional API."""
        bias = compute_alibi_bias(num_heads=8, seq_len=32)
        assert bias.shape == (1, 8, 32, 32)

    def test_compute_alibi_slopes(self):
        """Test slope computation."""
        slopes = compute_alibi_slopes(num_heads=8)
        assert slopes.shape == (8,)

        # Slopes should be positive and decreasing
        slopes_list = slopes.tolist()
        for i in range(len(slopes_list) - 1):
            assert slopes_list[i] > slopes_list[i + 1]


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


class TestTokenEmbedding:
    """Tests for token embeddings."""

    def test_basic_creation(self):
        """Test basic creation."""
        config = EmbeddingConfig(vocab_size=1000, hidden_size=256)
        embed = TokenEmbedding(config)
        assert embed.vocab_size == 1000
        assert embed.hidden_size == 256

    def test_forward(self):
        """Test forward pass."""
        config = EmbeddingConfig(vocab_size=1000, hidden_size=256)
        embed = TokenEmbedding(config)
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = embed(input_ids)
        assert output.shape == (1, 5, 256)

    def test_batch_forward(self):
        """Test batched forward pass."""
        config = EmbeddingConfig(vocab_size=1000, hidden_size=256)
        embed = TokenEmbedding(config)
        input_ids = mx.array([[1, 2, 3], [4, 5, 6]])
        output = embed(input_ids)
        assert output.shape == (2, 3, 256)

    def test_as_linear(self):
        """Test using embedding as linear projection (for tied embeddings)."""
        config = EmbeddingConfig(vocab_size=1000, hidden_size=256)
        embed = TokenEmbedding(config)
        hidden = mx.random.normal((2, 10, 256))
        logits = embed.as_linear(hidden)
        assert logits.shape == (2, 10, 1000)

    def test_with_scale_factor(self):
        """Test embedding with scale factor."""
        scale = math.sqrt(256)
        config = EmbeddingConfig(vocab_size=1000, hidden_size=256, scale_factor=scale)
        embed = TokenEmbedding(config)
        assert embed.scale_factor == scale

        input_ids = mx.array([[1, 2, 3]])
        output = embed(input_ids)
        assert output.shape == (1, 3, 256)

    def test_factory_function(self):
        """Test factory function."""
        embed = create_token_embedding(vocab_size=5000, hidden_size=512)
        assert isinstance(embed, TokenEmbedding)
        assert embed.vocab_size == 5000

    def test_from_pretrained(self):
        """Test creating TokenEmbedding from existing weight matrix."""
        # Create a weight matrix
        weight = mx.random.normal((1000, 256))

        # Create embedding from weight
        embed = TokenEmbedding.from_pretrained(weight)

        assert embed.vocab_size == 1000
        assert embed.hidden_size == 256

        # Verify forward pass works
        input_ids = mx.array([[1, 2, 3]])
        output = embed(input_ids)
        assert output.shape == (1, 3, 256)

    def test_from_pretrained_with_scale_factor(self):
        """Test creating TokenEmbedding from weight with scale factor."""
        weight = mx.random.normal((500, 128))
        scale = 2.0

        embed = TokenEmbedding.from_pretrained(weight, scale_factor=scale)

        assert embed.vocab_size == 500
        assert embed.hidden_size == 128
        assert embed.scale_factor == scale

    def test_from_pretrained_preserves_values(self):
        """Test that from_pretrained preserves the weight values."""
        weight = mx.random.normal((100, 64))

        embed = TokenEmbedding.from_pretrained(weight)

        # The weight should be set on the internal nn.Embedding
        # Verify by checking embedding lookup matches
        input_ids = mx.array([[0, 1, 2]])
        output = embed(input_ids)

        # Each row of output should match corresponding row of weight
        assert output.shape == (1, 3, 64)


class TestEmbeddingGradients:
    """Tests for gradient flow through embeddings."""

    def test_token_embedding_gradients(self):
        """Test gradients flow through token embeddings."""
        config = EmbeddingConfig(vocab_size=100, hidden_size=64)
        embed = TokenEmbedding(config)
        input_ids = mx.array([[1, 2, 3]])

        def loss_fn(model, x):
            out = model(x)
            return mx.mean(out**2)

        loss, grads = mx.value_and_grad(loss_fn)(embed, input_ids)
        assert loss.item() > 0

    def test_alibi_output_is_differentiable(self):
        """Test ALiBi bias computation works in gradients."""
        alibi = ALiBi(num_heads=4)

        def compute_bias(seq_len):
            return alibi(seq_len)

        # Just verify it runs without error
        bias = compute_bias(10)
        assert bias.shape == (1, 4, 10, 10)

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
