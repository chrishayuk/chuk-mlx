"""Tests for attention_routing handler."""

from argparse import Namespace
from unittest.mock import patch

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_routing import (
    handle_attention_routing,
)


class TestHandleAttentionRouting:
    """Tests for handle_attention_routing function."""

    def test_handle_attention_routing_calls_asyncio_run(self):
        """Test that handle_attention_routing calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_routing.asyncio"
        ) as mock_asyncio:
            handle_attention_routing(args)
            mock_asyncio.run.assert_called_once()

    def test_handle_attention_routing_with_layers(self):
        """Test handle_attention_routing with layers parameter."""
        args = Namespace(model="test/model", layers="0,12,23")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_routing.asyncio"
        ) as mock_asyncio:
            handle_attention_routing(args)
            mock_asyncio.run.assert_called_once()

    def test_handle_attention_routing_with_contexts(self):
        """Test handle_attention_routing with contexts parameter."""
        args = Namespace(model="test/model", contexts="def add,def hello")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_routing.asyncio"
        ) as mock_asyncio:
            handle_attention_routing(args)
            mock_asyncio.run.assert_called_once()

    def test_handle_attention_routing_with_token(self):
        """Test handle_attention_routing with token parameter."""
        args = Namespace(model="test/model", token="+")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_routing.asyncio"
        ) as mock_asyncio:
            handle_attention_routing(args)
            mock_asyncio.run.assert_called_once()


class TestAttentionRoutingServiceHelpers:
    """Tests for AttentionRoutingService helper methods."""

    def test_parse_contexts_default(self):
        """Test parse_contexts with None returns default contexts."""
        from chuk_lazarus.introspection.moe.attention_routing_service import (
            AttentionRoutingService,
        )

        result = AttentionRoutingService.parse_contexts(None)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(c, tuple) and len(c) == 2 for c in result)

    def test_parse_contexts_custom(self):
        """Test parse_contexts with custom contexts string."""
        from chuk_lazarus.introspection.moe.attention_routing_service import (
            AttentionRoutingService,
        )

        result = AttentionRoutingService.parse_contexts("def add,def hello")
        assert len(result) == 2
        assert result[0][1] == "def add"
        assert result[1][1] == "def hello"

    def test_parse_layers_default(self):
        """Test parse_layers with None returns three layers."""
        from chuk_lazarus.introspection.moe.attention_routing_service import (
            AttentionRoutingService,
        )

        moe_layers = list(range(24))
        result = AttentionRoutingService.parse_layers(None, moe_layers)
        assert len(result) == 3
        assert 0 in result  # Early layer
        assert moe_layers[-1] in result  # Late layer

    def test_parse_layers_custom(self):
        """Test parse_layers with custom layers string."""
        from chuk_lazarus.introspection.moe.attention_routing_service import (
            AttentionRoutingService,
        )

        moe_layers = list(range(24))
        result = AttentionRoutingService.parse_layers("0,12,23", moe_layers)
        assert result == [0, 12, 23]

    def test_get_layer_labels(self):
        """Test get_layer_labels returns correct labels."""
        from chuk_lazarus.introspection.moe.attention_routing_service import (
            AttentionRoutingService,
        )

        labels = AttentionRoutingService.get_layer_labels([0, 12, 23])
        assert len(labels) == 3
        assert labels[0] == "Early"
        assert labels[23] == "Late"


class TestAttentionRoutingModels:
    """Tests for AttentionRouting Pydantic models."""

    def test_attention_capture_result(self):
        """Test AttentionCaptureResult model."""
        from chuk_lazarus.introspection.moe.attention_routing_service import (
            AttentionCaptureResult,
        )

        result = AttentionCaptureResult(
            tokens=["hello", "world"],
            attention_weights=None,
            layer=0,
        )
        assert result.success is False  # No attention weights

    def test_attention_summary(self):
        """Test AttentionSummary model."""
        from chuk_lazarus.introspection.moe.attention_routing_service import (
            AttentionSummary,
        )

        summary = AttentionSummary(
            top_attended=[("the", 0.5), ("quick", 0.3)],
            self_attention_weight=0.2,
        )
        assert summary.self_attention_weight == 0.2
        assert len(summary.top_attended) == 2

    def test_context_routing_result(self):
        """Test ContextRoutingResult model."""
        from chuk_lazarus.introspection.moe.attention_routing_service import (
            ContextRoutingResult,
        )

        result = ContextRoutingResult(
            context_name="minimal",
            context="2 + 3",
            tokens=["2", " ", "+", " ", "3"],
            target_pos=2,
            target_token="+",
            primary_expert=6,
            all_experts=[6, 12],
            weights=[0.7, 0.3],
            attention_summary=None,
        )
        assert result.primary_expert == 6
        assert result.context_name == "minimal"

    def test_layer_routing_results(self):
        """Test LayerRoutingResults model with computed fields."""
        from chuk_lazarus.introspection.moe.attention_routing_service import (
            ContextRoutingResult,
            LayerRoutingResults,
        )

        results = [
            ContextRoutingResult(
                context_name="ctx1",
                context="2 + 3",
                tokens=["2", "+", "3"],
                target_pos=1,
                target_token="+",
                primary_expert=6,
                all_experts=[6],
                weights=[1.0],
                attention_summary=None,
            ),
            ContextRoutingResult(
                context_name="ctx2",
                context="Calculate: 2 + 3",
                tokens=["Calculate", ":", "2", "+", "3"],
                target_pos=3,
                target_token="+",
                primary_expert=12,  # Different expert
                all_experts=[12],
                weights=[1.0],
                attention_summary=None,
            ),
        ]

        layer_result = LayerRoutingResults(layer=0, label="Early", results=results)

        assert layer_result.unique_expert_count == 2
        assert layer_result.is_context_sensitive is True

    def test_attention_routing_analysis(self):
        """Test AttentionRoutingAnalysis model with computed fields."""
        from chuk_lazarus.introspection.moe.attention_routing_service import (
            AttentionRoutingAnalysis,
            LayerRoutingResults,
        )

        layers = [
            LayerRoutingResults(layer=0, label="Early", results=[]),
            LayerRoutingResults(layer=12, label="Middle", results=[]),
            LayerRoutingResults(layer=23, label="Late", results=[]),
        ]

        analysis = AttentionRoutingAnalysis(
            model_id="test/model",
            target_token="+",
            layers=layers,
        )

        assert analysis.early_layer.layer == 0
        assert analysis.middle_layer.layer == 12
        assert analysis.late_layer.layer == 23
