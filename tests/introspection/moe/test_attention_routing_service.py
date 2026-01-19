"""Tests for attention_routing_service.py to improve coverage."""

import mlx.core as mx
import pytest
from pydantic import ValidationError

from chuk_lazarus.introspection.moe.attention_routing_service import (
    DEFAULT_ATTENTION_CONTEXTS,
    AttentionCaptureResult,
    AttentionRoutingAnalysis,
    AttentionRoutingService,
    AttentionSummary,
    ContextRoutingResult,
    LayerRoutingResults,
)


class TestAttentionCaptureResult:
    """Tests for AttentionCaptureResult class."""

    def test_success_true_with_weights(self):
        """Test success property returns True when attention_weights is present (line 34)."""
        weights = mx.random.normal((8, 10, 10))  # (num_heads, seq_len, seq_len)
        result = AttentionCaptureResult(
            tokens=["hello", "world"],
            attention_weights=weights,
            layer=5,
        )
        assert result.success is True

    def test_success_false_without_weights(self):
        """Test success property returns False when attention_weights is None (line 34)."""
        result = AttentionCaptureResult(
            tokens=["hello", "world"],
            attention_weights=None,
            layer=5,
        )
        assert result.success is False


class TestAttentionSummary:
    """Tests for AttentionSummary class."""

    def test_creation(self):
        """Test AttentionSummary creation."""
        summary = AttentionSummary(
            top_attended=[("hello", 0.4), ("world", 0.3), ("test", 0.2)],
            self_attention_weight=0.4,
        )
        assert len(summary.top_attended) == 3
        assert summary.self_attention_weight == 0.4


class TestContextRoutingResult:
    """Tests for ContextRoutingResult class."""

    def test_creation(self):
        """Test ContextRoutingResult creation."""
        result = ContextRoutingResult(
            context_name="test",
            context="test context",
            tokens=["test", "context"],
            target_pos=1,
            target_token="context",
            primary_expert=5,
            all_experts=[5, 10],
            weights=[0.6, 0.4],
            attention_summary=None,
        )
        assert result.context_name == "test"
        assert result.primary_expert == 5
        assert result.attention_summary is None

    def test_with_attention_summary(self):
        """Test ContextRoutingResult with attention summary."""
        summary = AttentionSummary(
            top_attended=[("a", 0.5)],
            self_attention_weight=0.3,
        )
        result = ContextRoutingResult(
            context_name="test",
            context="test",
            tokens=["test"],
            target_pos=0,
            target_token="test",
            primary_expert=0,
            all_experts=[0],
            weights=[1.0],
            attention_summary=summary,
        )
        assert result.attention_summary is not None
        assert result.attention_summary.self_attention_weight == 0.3


class TestLayerRoutingResults:
    """Tests for LayerRoutingResults class."""

    def _create_context_result(self, primary_expert: int) -> ContextRoutingResult:
        """Helper to create ContextRoutingResult."""
        return ContextRoutingResult(
            context_name="test",
            context="test",
            tokens=["test"],
            target_pos=0,
            target_token="test",
            primary_expert=primary_expert,
            all_experts=[primary_expert],
            weights=[1.0],
        )

    def test_unique_expert_count_single(self):
        """Test unique_expert_count with single expert (line 79)."""
        results = LayerRoutingResults(
            layer=5,
            label="Middle",
            results=[
                self._create_context_result(5),
                self._create_context_result(5),
                self._create_context_result(5),
            ],
        )
        assert results.unique_expert_count == 1

    def test_unique_expert_count_multiple(self):
        """Test unique_expert_count with multiple experts (line 79)."""
        results = LayerRoutingResults(
            layer=5,
            label="Middle",
            results=[
                self._create_context_result(5),
                self._create_context_result(10),
                self._create_context_result(15),
            ],
        )
        assert results.unique_expert_count == 3

    def test_unique_expert_count_empty(self):
        """Test unique_expert_count with no results."""
        results = LayerRoutingResults(layer=5, label="Middle", results=[])
        assert results.unique_expert_count == 0

    def test_is_context_sensitive_true(self):
        """Test is_context_sensitive returns True when multiple experts (line 85)."""
        results = LayerRoutingResults(
            layer=5,
            label="Middle",
            results=[
                self._create_context_result(5),
                self._create_context_result(10),
            ],
        )
        assert results.is_context_sensitive is True

    def test_is_context_sensitive_false(self):
        """Test is_context_sensitive returns False with single expert (line 85)."""
        results = LayerRoutingResults(
            layer=5,
            label="Middle",
            results=[
                self._create_context_result(5),
                self._create_context_result(5),
            ],
        )
        assert results.is_context_sensitive is False


class TestAttentionRoutingAnalysis:
    """Tests for AttentionRoutingAnalysis class."""

    def _create_layer_results(self, layer: int, label: str) -> LayerRoutingResults:
        """Helper to create LayerRoutingResults."""
        return LayerRoutingResults(layer=layer, label=label, results=[])

    def test_early_layer_with_data(self):
        """Test early_layer property with layers (line 101)."""
        analysis = AttentionRoutingAnalysis(
            model_id="test",
            target_token="+",
            layers=[
                self._create_layer_results(0, "Early"),
                self._create_layer_results(12, "Middle"),
                self._create_layer_results(24, "Late"),
            ],
        )
        assert analysis.early_layer is not None
        assert analysis.early_layer.layer == 0

    def test_early_layer_empty(self):
        """Test early_layer property without layers (line 101)."""
        analysis = AttentionRoutingAnalysis(
            model_id="test",
            target_token="+",
            layers=[],
        )
        assert analysis.early_layer is None

    def test_middle_layer_with_multiple(self):
        """Test middle_layer property with multiple layers (line 107-108)."""
        analysis = AttentionRoutingAnalysis(
            model_id="test",
            target_token="+",
            layers=[
                self._create_layer_results(0, "Early"),
                self._create_layer_results(12, "Middle"),
                self._create_layer_results(24, "Late"),
            ],
        )
        assert analysis.middle_layer is not None
        assert analysis.middle_layer.layer == 12

    def test_middle_layer_with_single(self):
        """Test middle_layer property with single layer (line 109)."""
        analysis = AttentionRoutingAnalysis(
            model_id="test",
            target_token="+",
            layers=[
                self._create_layer_results(0, "Early"),
            ],
        )
        assert analysis.middle_layer is not None
        assert analysis.middle_layer.layer == 0

    def test_middle_layer_empty(self):
        """Test middle_layer property without layers (line 109)."""
        analysis = AttentionRoutingAnalysis(
            model_id="test",
            target_token="+",
            layers=[],
        )
        assert analysis.middle_layer is None

    def test_late_layer_with_data(self):
        """Test late_layer property with layers (line 115)."""
        analysis = AttentionRoutingAnalysis(
            model_id="test",
            target_token="+",
            layers=[
                self._create_layer_results(0, "Early"),
                self._create_layer_results(12, "Middle"),
                self._create_layer_results(24, "Late"),
            ],
        )
        assert analysis.late_layer is not None
        assert analysis.late_layer.layer == 24

    def test_late_layer_empty(self):
        """Test late_layer property without layers (line 115)."""
        analysis = AttentionRoutingAnalysis(
            model_id="test",
            target_token="+",
            layers=[],
        )
        assert analysis.late_layer is None


class TestDefaultAttentionContexts:
    """Tests for DEFAULT_ATTENTION_CONTEXTS constant."""

    def test_default_contexts_exist(self):
        """Test default contexts are defined."""
        assert len(DEFAULT_ATTENTION_CONTEXTS) > 0

    def test_default_contexts_format(self):
        """Test default contexts are (name, prompt) tuples."""
        for name, prompt in DEFAULT_ATTENTION_CONTEXTS:
            assert isinstance(name, str)
            assert isinstance(prompt, str)


class TestParseLayersMethod:
    """Tests for AttentionRoutingService.parse_layers method."""

    def test_parse_layers_none_with_many(self):
        """Test parse_layers with None and many layers (line 280-281)."""
        moe_layers = (0, 4, 8, 12, 16, 20, 24)
        result = AttentionRoutingService.parse_layers(None, moe_layers)
        # Should return early, middle, late
        assert len(result) == 3
        assert result[0] == 0  # First
        assert result[1] == 12  # Middle
        assert result[2] == 24  # Last

    def test_parse_layers_none_with_few(self):
        """Test parse_layers with None and few layers (line 282)."""
        moe_layers = (0, 12)
        result = AttentionRoutingService.parse_layers(None, moe_layers)
        # Should return all
        assert result == [0, 12]

    def test_parse_layers_all(self):
        """Test parse_layers with 'all' (line 284-285)."""
        moe_layers = (0, 4, 8, 12)
        result = AttentionRoutingService.parse_layers("all", moe_layers)
        assert result == [0, 4, 8, 12]

    def test_parse_layers_all_case_insensitive(self):
        """Test parse_layers with 'ALL' is case insensitive."""
        moe_layers = (0, 4, 8)
        result = AttentionRoutingService.parse_layers("ALL", moe_layers)
        assert result == [0, 4, 8]

    def test_parse_layers_comma_separated(self):
        """Test parse_layers with comma-separated values (line 288)."""
        moe_layers = (0, 4, 8, 12, 16, 20)
        result = AttentionRoutingService.parse_layers("4,12,20", moe_layers)
        assert result == [4, 12, 20]

    def test_parse_layers_comma_with_spaces(self):
        """Test parse_layers with spaces in comma-separated."""
        moe_layers = (0, 4, 8)
        result = AttentionRoutingService.parse_layers("0, 4, 8", moe_layers)
        assert result == [0, 4, 8]

    def test_parse_layers_single_value(self):
        """Test parse_layers with single value."""
        moe_layers = (0, 4, 8)
        result = AttentionRoutingService.parse_layers("4", moe_layers)
        assert result == [4]


class TestParseContextsMethod:
    """Tests for AttentionRoutingService.parse_contexts method."""

    def test_parse_contexts_none(self):
        """Test parse_contexts with None returns defaults (line 300-301)."""
        result = AttentionRoutingService.parse_contexts(None)
        assert result == DEFAULT_ATTENTION_CONTEXTS

    def test_parse_contexts_custom(self):
        """Test parse_contexts with custom contexts (lines 303-309)."""
        result = AttentionRoutingService.parse_contexts("Hello world, Test prompt")
        assert len(result) == 2
        # First context: name from first word
        assert result[0][0] == "Hello"
        assert result[0][1] == "Hello world"
        # Second context
        assert result[1][0] == "Test"
        assert result[1][1] == "Test prompt"

    def test_parse_contexts_single(self):
        """Test parse_contexts with single context."""
        result = AttentionRoutingService.parse_contexts("Single test")
        assert len(result) == 1
        assert result[0][0] == "Single"
        assert result[0][1] == "Single test"

    def test_parse_contexts_empty_handling(self):
        """Test parse_contexts handles empty parts (line 306)."""
        result = AttentionRoutingService.parse_contexts("hello,,world")
        # Empty parts should be skipped
        assert len(result) == 2

    def test_parse_contexts_short_context(self):
        """Test parse_contexts with short context (line 308)."""
        result = AttentionRoutingService.parse_contexts("x")
        assert len(result) == 1
        assert result[0][0] == "x"
        assert result[0][1] == "x"


class TestGetLayerLabelsMethod:
    """Tests for AttentionRoutingService.get_layer_labels method."""

    def test_get_layer_labels_many(self):
        """Test get_layer_labels with many layers (lines 322-327)."""
        target_layers = [0, 12, 24]
        labels = AttentionRoutingService.get_layer_labels(target_layers)
        assert labels[0] == "Early"
        assert labels[12] == "Middle"
        assert labels[24] == "Late"

    def test_get_layer_labels_two(self):
        """Test get_layer_labels with two layers."""
        target_layers = [0, 24]
        labels = AttentionRoutingService.get_layer_labels(target_layers)
        assert labels[0] == "Early"
        assert labels[24] == "Late"
        assert 12 not in labels  # No middle

    def test_get_layer_labels_many_middle_index(self):
        """Test get_layer_labels correctly selects middle index."""
        target_layers = [0, 4, 8, 12, 16]  # 5 layers, middle is index 2 (layer 8)
        labels = AttentionRoutingService.get_layer_labels(target_layers)
        assert labels[0] == "Early"
        assert labels[8] == "Middle"  # Index 2 = 5//2 = 2
        assert labels[16] == "Late"


class TestComputeAttentionSummary:
    """Tests for AttentionRoutingService.compute_attention_summary method."""

    def test_compute_attention_summary_basic(self):
        """Test compute_attention_summary with basic input (lines 231-265)."""
        # Create attention weights: (num_heads, seq_len, seq_len)
        # Use random weights and softmax to get valid attention
        attn_weights = mx.random.normal((4, 5, 5))
        attn_weights = mx.softmax(attn_weights, axis=-1)

        tokens = ["a", "b", "c", "d", "e"]
        summary = AttentionRoutingService.compute_attention_summary(
            attn_weights, tokens, position=2, top_k=3
        )

        assert len(summary.top_attended) == 3
        # Self attention should be valid
        assert 0.0 <= summary.self_attention_weight <= 1.0

    def test_compute_attention_summary_top_k(self):
        """Test compute_attention_summary respects top_k."""
        attn_weights = mx.random.normal((4, 10, 10))
        attn_weights = mx.softmax(attn_weights, axis=-1)

        tokens = [f"t{i}" for i in range(10)]
        summary = AttentionRoutingService.compute_attention_summary(
            attn_weights, tokens, position=5, top_k=5
        )

        assert len(summary.top_attended) == 5

    def test_compute_attention_summary_position_bounds(self):
        """Test compute_attention_summary with position at boundary (line 252)."""
        attn_weights = mx.random.normal((4, 5, 5))
        attn_weights = mx.softmax(attn_weights, axis=-1)

        tokens = ["a", "b", "c", "d", "e"]
        summary = AttentionRoutingService.compute_attention_summary(
            attn_weights,
            tokens,
            position=4,
            top_k=3,  # Last position
        )

        assert len(summary.top_attended) == 3

    def test_compute_attention_summary_position_at_boundary(self):
        """Test compute_attention_summary with position at sequence boundary."""
        attn_weights = mx.random.normal((4, 5, 5))
        attn_weights = mx.softmax(attn_weights, axis=-1)

        tokens = ["a", "b", "c", "d", "e"]
        # Test last valid position
        summary = AttentionRoutingService.compute_attention_summary(
            attn_weights, tokens, position=4, top_k=3
        )

        # Should work for valid boundary position
        assert 0.0 <= summary.self_attention_weight <= 1.0

    def test_compute_attention_summary_unknown_token(self):
        """Test compute_attention_summary handles missing tokens (line 259)."""
        attn_weights = mx.random.normal((4, 5, 5))
        attn_weights = mx.softmax(attn_weights, axis=-1)

        # Fewer tokens than positions
        tokens = ["a", "b"]
        summary = AttentionRoutingService.compute_attention_summary(
            attn_weights, tokens, position=1, top_k=5
        )

        # Some top_attended should have "?" for unknown tokens
        token_names = [name for name, _ in summary.top_attended]
        assert "?" in token_names


class TestAttentionCaptureResultModel:
    """Additional tests for AttentionCaptureResult model."""

    def test_frozen_model(self):
        """Test model is frozen."""
        result = AttentionCaptureResult(tokens=["a"], attention_weights=None, layer=0)
        with pytest.raises(ValidationError):
            result.layer = 5

    def test_arbitrary_types_allowed(self):
        """Test mx.array is allowed."""
        weights = mx.array([[1.0, 0.0], [0.5, 0.5]])
        result = AttentionCaptureResult(tokens=["a", "b"], attention_weights=weights, layer=0)
        assert result.attention_weights is not None
