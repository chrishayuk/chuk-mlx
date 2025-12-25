"""
Tests for ALiBi position bias.
"""

from chuk_lazarus.models_v2.components.embeddings.alibi import (
    ALiBi,
    compute_alibi_bias,
    compute_alibi_slopes,
)


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
