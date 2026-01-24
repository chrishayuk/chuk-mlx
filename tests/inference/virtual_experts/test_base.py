"""Tests for virtual_experts/base.py to improve coverage."""

from typing import Any

from chuk_lazarus.inference.virtual_experts.base import (
    InferenceResult,
    VirtualExpertAnalysis,
    VirtualExpertApproach,
    VirtualExpertPlugin,
)


class TestInferenceResultPostInit:
    """Tests for InferenceResult __post_init__ edge cases."""

    def test_is_correct_with_matching_float(self):
        """Test is_correct with float answer matching correct."""
        result = InferenceResult(
            prompt="test",
            answer="3.14159",
            correct_answer=3.14159,
            approach=VirtualExpertApproach.VIRTUAL_EXPERT,
            used_virtual_expert=True,
        )
        assert result.is_correct is True

    def test_is_correct_with_non_matching_float(self):
        """Test is_correct with float answer not matching."""
        result = InferenceResult(
            prompt="test",
            answer="3.14",
            correct_answer=2.71,
            approach=VirtualExpertApproach.VIRTUAL_EXPERT,
            used_virtual_expert=True,
        )
        assert result.is_correct is False

    def test_is_correct_with_no_number_in_answer(self):
        """Test is_correct when answer has no number."""
        result = InferenceResult(
            prompt="test",
            answer="no numbers here",
            correct_answer=42,
            approach=VirtualExpertApproach.VIRTUAL_EXPERT,
            used_virtual_expert=True,
        )
        # Should stay False since no number found
        assert result.is_correct is False

    def test_is_correct_with_invalid_answer_type(self):
        """Test is_correct when answer causes ValueError/TypeError."""
        result = InferenceResult(
            prompt="test",
            answer="",
            correct_answer=42,
            approach=VirtualExpertApproach.VIRTUAL_EXPERT,
            used_virtual_expert=True,
        )
        assert result.is_correct is False

    def test_is_correct_with_negative_number(self):
        """Test is_correct with negative numbers."""
        result = InferenceResult(
            prompt="test",
            answer="-5",
            correct_answer=-5,
            approach=VirtualExpertApproach.VIRTUAL_EXPERT,
            used_virtual_expert=True,
        )
        assert result.is_correct is True

    def test_is_correct_none_correct_answer(self):
        """Test is_correct when correct_answer is None."""
        result = InferenceResult(
            prompt="test",
            answer="42",
            correct_answer=None,
            approach=VirtualExpertApproach.VIRTUAL_EXPERT,
            used_virtual_expert=True,
        )
        # Should stay False since correct_answer is None
        assert result.is_correct is False


class TestVirtualExpertAnalysisSummary:
    """Tests for VirtualExpertAnalysis summary method."""

    def test_summary_with_plugins_used(self):
        """Test summary includes plugins_used section."""
        analysis = VirtualExpertAnalysis(
            model_name="test_model",
            total_problems=10,
            correct_with_virtual=8,
            correct_without_virtual=5,
            times_virtual_used=6,
            avg_routing_score=0.85,
            plugins_used={"math": 5, "code": 3},
        )
        summary = analysis.summary()

        assert "Plugins used:" in summary
        assert "math: 5" in summary
        assert "code: 3" in summary

    def test_summary_without_plugins_used(self):
        """Test summary without plugins_used."""
        analysis = VirtualExpertAnalysis(
            model_name="test_model",
            total_problems=10,
            correct_with_virtual=8,
            correct_without_virtual=5,
            times_virtual_used=6,
            avg_routing_score=0.85,
            plugins_used={},
        )
        summary = analysis.summary()

        # Should not have "Plugins used:" section
        assert "Plugins used:" not in summary

    def test_summary_with_single_plugin(self):
        """Test summary with single plugin."""
        analysis = VirtualExpertAnalysis(
            model_name="test_model",
            total_problems=5,
            correct_with_virtual=4,
            correct_without_virtual=2,
            times_virtual_used=4,
            avg_routing_score=0.9,
            plugins_used={"math": 4},
        )
        summary = analysis.summary()

        assert "Plugins used:" in summary
        assert "math: 4" in summary

    def test_summary_plugins_sorted_by_count(self):
        """Test that plugins are sorted by count (highest first)."""
        analysis = VirtualExpertAnalysis(
            model_name="test_model",
            total_problems=20,
            correct_with_virtual=15,
            correct_without_virtual=10,
            times_virtual_used=15,
            avg_routing_score=0.8,
            plugins_used={"alpha": 3, "beta": 10, "gamma": 2},
        )
        summary = analysis.summary()

        # beta should come before alpha and gamma
        assert "Plugins used:" in summary
        beta_idx = summary.find("beta")
        alpha_idx = summary.find("alpha")
        gamma_idx = summary.find("gamma")
        assert beta_idx < alpha_idx < gamma_idx


class TestVirtualExpertPluginInterface:
    """Tests for VirtualExpertPlugin (VirtualExpert) interface."""

    def test_plugin_can_handle(self):
        """Test can_handle abstract method."""

        class TestPlugin(VirtualExpertPlugin):
            name = "test"
            description = "Test plugin"

            def can_handle(self, prompt: str) -> bool:
                return "test" in prompt

            def get_operations(self) -> list[str]:
                return ["evaluate"]

            def execute_operation(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
                return {"result": "test_result"}

            def get_calibration_prompts(self):
                return ["test 1"], ["hello"]

        plugin = TestPlugin()
        assert plugin.can_handle("test prompt") is True
        assert plugin.can_handle("other") is False

    def test_plugin_repr(self):
        """Test plugin __repr__."""

        class TestPlugin(VirtualExpertPlugin):
            name = "mytest"
            description = "My test plugin"
            priority = 15

            def can_handle(self, prompt: str) -> bool:
                return False

            def get_operations(self) -> list[str]:
                return ["evaluate"]

            def execute_operation(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
                return {"result": None}

            def get_calibration_prompts(self):
                return [], []

        plugin = TestPlugin()
        repr_str = repr(plugin)
        assert "TestPlugin" in repr_str
        assert "mytest" in repr_str

    def test_plugin_execute_via_action(self):
        """Test execute with VirtualExpertAction."""
        from chuk_virtual_expert import VirtualExpertAction

        class TestPlugin(VirtualExpertPlugin):
            name = "test"
            description = "Test plugin"

            def can_handle(self, prompt: str) -> bool:
                return True

            def get_operations(self) -> list[str]:
                return ["evaluate"]

            def execute_operation(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
                return {"result": parameters.get("value", 42), "formatted": "42"}

            def get_calibration_prompts(self):
                return ["test"], ["other"]

        plugin = TestPlugin()
        action = VirtualExpertAction(
            expert="test",
            operation="evaluate",
            parameters={"value": 42},
        )
        result = plugin.execute(action)
        assert result.success is True
        assert result.data["result"] == 42


class TestVirtualExpertApproachEnum:
    """Tests for VirtualExpertApproach enum."""

    def test_all_values(self):
        """Test all enum values exist."""
        assert VirtualExpertApproach.VIRTUAL_EXPERT.value == "virtual_expert"
        assert VirtualExpertApproach.MODEL_DIRECT.value == "model_direct"

    def test_enum_comparison(self):
        """Test enum comparison."""
        assert VirtualExpertApproach.VIRTUAL_EXPERT == "virtual_expert"
        assert VirtualExpertApproach.MODEL_DIRECT == "model_direct"
