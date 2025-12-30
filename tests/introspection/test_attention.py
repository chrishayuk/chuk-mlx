"""Tests for attention analysis module."""

import mlx.core as mx
import pytest

from chuk_lazarus.introspection import CapturedState
from chuk_lazarus.introspection.attention import (
    AggregationStrategy,
    AttentionAnalyzer,
    AttentionFocus,
    AttentionPattern,
)


class TestAttentionPattern:
    """Tests for AttentionPattern dataclass."""

    @pytest.fixture
    def sample_weights(self):
        """Create sample attention weights [batch, heads, seq, seq]."""
        # 1 batch, 4 heads, 5 tokens
        return mx.random.uniform(shape=(1, 4, 5, 5))

    @pytest.fixture
    def pattern(self, sample_weights):
        """Create a sample AttentionPattern."""
        return AttentionPattern(
            layer_idx=0,
            weights=sample_weights,
            tokens=["The", " cat", " sat", " on", " mat"],
        )

    def test_num_heads(self, pattern):
        assert pattern.num_heads == 4

    def test_seq_len(self, pattern):
        assert pattern.seq_len == 5

    def test_aggregate_mean(self, pattern):
        agg = pattern.aggregate(AggregationStrategy.MEAN)
        assert agg.shape == (1, 5, 5)  # [batch, seq, seq]

    def test_aggregate_max(self, pattern):
        agg = pattern.aggregate(AggregationStrategy.MAX)
        assert agg.shape == (1, 5, 5)

    def test_aggregate_min(self, pattern):
        agg = pattern.aggregate(AggregationStrategy.MIN)
        assert agg.shape == (1, 5, 5)

    def test_aggregate_head(self, pattern):
        agg = pattern.aggregate(AggregationStrategy.HEAD, head_idx=2)
        assert agg.shape == (1, 5, 5)

    def test_aggregate_head_requires_idx(self, pattern):
        with pytest.raises(ValueError, match="head_idx required"):
            pattern.aggregate(AggregationStrategy.HEAD)

    def test_get_head(self, pattern):
        head_weights = pattern.get_head(0)
        assert head_weights.shape == (1, 5, 5)


class TestAttentionFocus:
    """Tests for AttentionFocus dataclass."""

    @pytest.fixture
    def focus(self):
        """Create a sample AttentionFocus."""
        # Attention distribution where position 1 gets most attention
        weights = mx.array([0.1, 0.5, 0.2, 0.15, 0.05])
        return AttentionFocus(
            query_position=4,
            query_token=" mat",
            layer_idx=0,
            attention_weights=weights,
            tokens=["The", " cat", " sat", " on", " mat"],
        )

    def test_top_attended_positions(self, focus):
        top = focus.top_attended_positions

        # First should be position 1 with weight ~0.5
        assert top[0][0] == 1
        assert abs(top[0][1] - 0.5) < 1e-5
        # Second should be position 2 with weight ~0.2
        assert top[1][0] == 2
        assert abs(top[1][1] - 0.2) < 1e-5

    def test_top_attended_tokens(self, focus):
        top = focus.top_attended_tokens

        # First should be " cat" with weight ~0.5
        assert top[0][0] == " cat"
        assert abs(top[0][1] - 0.5) < 1e-5
        # Second should be " sat" with weight ~0.2
        assert top[1][0] == " sat"
        assert abs(top[1][1] - 0.2) < 1e-5

    def test_summary(self, focus):
        summary = focus.summary(top_k=3)

        assert "Layer 0" in summary
        assert "position 4" in summary
        assert "mat" in summary
        assert "cat" in summary


class TestAttentionAnalyzer:
    """Tests for AttentionAnalyzer class."""

    @pytest.fixture
    def state_with_attention(self):
        """Create a CapturedState with attention weights."""
        state = CapturedState()
        state.input_ids = mx.array([[1, 2, 3, 4, 5]])

        # Create attention weights for layer 0
        # Shape: [batch, heads, seq, seq]
        weights = mx.zeros((1, 4, 5, 5))
        # Make position 4 attend mostly to position 1
        weights = weights.at[:, :, 4, 1].add(0.5)
        weights = weights.at[:, :, 4, 2].add(0.3)
        weights = weights.at[:, :, 4, 3].add(0.2)

        state.attention_weights[0] = weights
        state.attention_weights[4] = weights  # Another layer

        return state

    @pytest.fixture
    def analyzer(self, state_with_attention):
        """Create an analyzer with mock tokenizer."""

        class MockTokenizer:
            def decode(self, ids):
                token_map = {1: "The", 2: " cat", 3: " sat", 4: " on", 5: " mat"}
                return "".join(token_map.get(i, f"[{i}]") for i in ids)

        return AttentionAnalyzer(state_with_attention, MockTokenizer())

    def test_tokens_property(self, analyzer):
        tokens = analyzer.tokens

        assert len(tokens) == 5
        assert tokens[0] == "The"
        assert tokens[1] == " cat"

    def test_tokens_without_tokenizer(self, state_with_attention):
        analyzer = AttentionAnalyzer(state_with_attention, tokenizer=None)
        tokens = analyzer.tokens

        assert len(tokens) == 5
        assert tokens[0] == "[1]"  # Token ID as string

    def test_get_attention_pattern(self, analyzer):
        pattern = analyzer.get_attention_pattern(layer_idx=0)

        assert pattern is not None
        assert pattern.layer_idx == 0
        assert pattern.num_heads == 4
        assert pattern.seq_len == 5

    def test_get_attention_pattern_missing_layer(self, analyzer):
        pattern = analyzer.get_attention_pattern(layer_idx=99)
        assert pattern is None

    def test_get_attention_focus(self, analyzer):
        focus = analyzer.get_attention_focus(layer_idx=0, position=-1)

        assert focus is not None
        assert focus.layer_idx == 0
        assert focus.query_position == 4  # Last position

    def test_get_attention_focus_specific_head(self, analyzer):
        focus = analyzer.get_attention_focus(layer_idx=0, position=-1, head_idx=2)

        assert focus is not None

    def test_find_high_attention_pairs(self, analyzer):
        pairs = analyzer.find_high_attention_pairs(layer_idx=0, threshold=0.3)

        # Should find (4, 1) with weight 0.5 and (4, 2) with weight 0.3
        positions = [(q, k) for q, k, w in pairs]
        assert (4, 1) in positions

    def test_get_attention_entropy(self, analyzer):
        entropy = analyzer.get_attention_entropy(layer_idx=0)

        assert entropy is not None
        assert entropy.shape == (5,)  # One entropy value per position

    def test_compare_layers(self, analyzer):
        comparisons = analyzer.compare_layers(position=-1)

        assert 0 in comparisons
        assert 4 in comparisons
        assert comparisons[0].layer_idx == 0
        assert comparisons[4].layer_idx == 4


class TestAggregationStrategy:
    """Tests for AggregationStrategy enum."""

    def test_enum_values(self):
        assert AggregationStrategy.MEAN.value == "mean"
        assert AggregationStrategy.MAX.value == "max"
        assert AggregationStrategy.MIN.value == "min"
        assert AggregationStrategy.HEAD.value == "head"
