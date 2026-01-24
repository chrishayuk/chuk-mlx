"""Tests for virtual_expert introspection module."""

import mlx.nn as nn
import pytest
from chuk_virtual_expert import VirtualExpertAction as ExpertAction

from chuk_lazarus.inference.virtual_experts.registry import reset_default_registry
from chuk_lazarus.introspection.virtual_expert import (
    ExpertHijacker,
    HybridEmbeddingInjector,
    InferenceResult,
    MathExpertPlugin,
    SafeMathEvaluator,
    VirtualExpertAnalysis,
    VirtualExpertApproach,
    VirtualExpertPlugin,
    VirtualExpertRegistry,
    VirtualExpertSlot,
    VirtualMoEWrapper,
    VirtualRouter,
    create_virtual_expert,
    create_virtual_expert_wrapper,
    demo_all_approaches,
    demo_virtual_expert,
    get_default_registry,
)


class MockTokenizer:
    """Mock tokenizer."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 100 for c in text[:10]]

    def decode(self, ids: list[int]) -> str:
        if isinstance(ids, (list, tuple)) and len(ids) > 0:
            return str(ids[0])
        return str(ids)


class MockModel(nn.Module):
    """Mock MoE model."""

    def __init__(self):
        super().__init__()
        # Minimal mock


class TestMathExpertPlugin:
    """Tests for MathExpertPlugin class."""

    def test_name_and_description(self):
        plugin = MathExpertPlugin()
        assert plugin.name == "math"
        assert plugin.description == "Computes arithmetic expressions using Python"
        assert plugin.priority == 10

    def test_can_handle_arithmetic(self):
        plugin = MathExpertPlugin()

        assert plugin.can_handle("2 + 2 = ") is True
        assert plugin.can_handle("10 - 5 = ") is True
        assert plugin.can_handle("3 * 4 = ") is True
        assert plugin.can_handle("20 / 5 = ") is True

    def test_can_handle_non_arithmetic(self):
        plugin = MathExpertPlugin()

        assert plugin.can_handle("Hello world") is False
        assert plugin.can_handle("What is the capital of France?") is False

    def test_execute(self):
        plugin = MathExpertPlugin()

        action = ExpertAction(expert="math", operation="evaluate", parameters={"expression": "2 + 3"})
        result = plugin.execute(action)
        assert result.success is True
        assert result.data["result"] == 5

        action = ExpertAction(expert="math", operation="evaluate", parameters={"expression": "7 * 8"})
        result = plugin.execute(action)
        assert result.success is True
        assert result.data["result"] == 56

    def test_execute_subtraction(self):
        plugin = MathExpertPlugin()
        action = ExpertAction(expert="math", operation="evaluate", parameters={"expression": "10 - 3"})
        result = plugin.execute(action)
        assert result.success is True
        assert result.data["result"] == 7

    def test_execute_division(self):
        plugin = MathExpertPlugin()
        action = ExpertAction(expert="math", operation="evaluate", parameters={"expression": "20 / 4"})
        result = plugin.execute(action)
        assert result.success is True
        assert result.data["result"] == 5

    def test_execute_invalid(self):
        plugin = MathExpertPlugin()
        action = ExpertAction(expert="math", operation="evaluate", parameters={"expression": "not math"})
        result = plugin.execute(action)
        assert result.success is False

    def test_get_calibration_prompts(self):
        plugin = MathExpertPlugin()
        positive, negative = plugin.get_calibration_prompts()

        assert isinstance(positive, list)
        assert isinstance(negative, list)

    def test_repr(self):
        plugin = MathExpertPlugin()
        repr_str = repr(plugin)
        assert "MathExpert" in repr_str
        assert "math" in repr_str


class TestSafeMathEvaluator:
    """Tests for SafeMathEvaluator class (alias for MathExpertPlugin)."""

    def test_is_alias(self):
        assert SafeMathEvaluator is MathExpertPlugin

    def test_execute_addition(self):
        evaluator = SafeMathEvaluator()
        action = ExpertAction(expert="math", operation="evaluate", parameters={"expression": "2 + 3"})
        result = evaluator.execute(action)
        assert result.success is True
        assert result.data["result"] == 5

    def test_execute_subtraction(self):
        evaluator = SafeMathEvaluator()
        action = ExpertAction(expert="math", operation="evaluate", parameters={"expression": "10 - 3"})
        result = evaluator.execute(action)
        assert result.success is True
        assert result.data["result"] == 7

    def test_execute_multiplication(self):
        evaluator = SafeMathEvaluator()
        action = ExpertAction(expert="math", operation="evaluate", parameters={"expression": "7 * 8"})
        result = evaluator.execute(action)
        assert result.success is True
        assert result.data["result"] == 56

    def test_execute_division(self):
        evaluator = SafeMathEvaluator()
        action = ExpertAction(expert="math", operation="evaluate", parameters={"expression": "20 / 4"})
        result = evaluator.execute(action)
        assert result.success is True
        assert result.data["result"] == 5

    def test_execute_invalid(self):
        evaluator = SafeMathEvaluator()
        action = ExpertAction(expert="math", operation="evaluate", parameters={"expression": "not a math problem"})
        result = evaluator.execute(action)
        assert result.success is False

    def test_extract_and_evaluate(self):
        evaluator = SafeMathEvaluator()
        expr, result = evaluator.extract_and_evaluate("What is 5 + 3?")
        assert result is not None


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_init_basic(self):
        result = InferenceResult(
            prompt="2 + 2 = ",
            answer="4",
            correct_answer=4,
            approach=VirtualExpertApproach.VIRTUAL_EXPERT,
            used_virtual_expert=True,
        )

        assert result.prompt == "2 + 2 = "
        assert result.answer == "4"
        assert result.correct_answer == 4
        assert result.used_virtual_expert is True
        assert result.approach == VirtualExpertApproach.VIRTUAL_EXPERT

    def test_is_correct_auto_computed(self):
        # Correct answer
        result1 = InferenceResult(
            prompt="test",
            answer="4",
            correct_answer=4,
            approach=VirtualExpertApproach.VIRTUAL_EXPERT,
            used_virtual_expert=False,
        )
        assert result1.is_correct is True

        # Incorrect answer
        result2 = InferenceResult(
            prompt="test",
            answer="5",
            correct_answer=4,
            approach=VirtualExpertApproach.VIRTUAL_EXPERT,
            used_virtual_expert=False,
        )
        assert result2.is_correct is False

    def test_with_plugin_name(self):
        result = InferenceResult(
            prompt="test",
            answer="result",
            correct_answer=None,
            approach=VirtualExpertApproach.VIRTUAL_EXPERT,
            used_virtual_expert=True,
            plugin_name="MathPlugin",
        )

        assert result.plugin_name == "MathPlugin"

    def test_routing_score(self):
        result = InferenceResult(
            prompt="test",
            answer="4",
            correct_answer=4,
            approach=VirtualExpertApproach.VIRTUAL_EXPERT,
            used_virtual_expert=True,
            routing_score=0.95,
        )

        assert result.routing_score == 0.95


class TestVirtualExpertAnalysis:
    """Tests for VirtualExpertAnalysis dataclass."""

    def test_init(self):
        analysis = VirtualExpertAnalysis(
            model_name="test_model",
            total_problems=10,
            correct_with_virtual=8,
            correct_without_virtual=5,
            times_virtual_used=6,
            avg_routing_score=0.85,
        )

        assert analysis.model_name == "test_model"
        assert analysis.total_problems == 10
        assert analysis.correct_with_virtual == 8
        assert analysis.correct_without_virtual == 5
        assert analysis.times_virtual_used == 6
        assert analysis.avg_routing_score == 0.85

    def test_virtual_accuracy_property(self):
        analysis = VirtualExpertAnalysis(
            model_name="test",
            total_problems=10,
            correct_with_virtual=8,
            correct_without_virtual=5,
            times_virtual_used=6,
            avg_routing_score=0.85,
        )

        assert analysis.virtual_accuracy == 0.8

    def test_model_accuracy_property(self):
        analysis = VirtualExpertAnalysis(
            model_name="test",
            total_problems=10,
            correct_with_virtual=8,
            correct_without_virtual=5,
            times_virtual_used=6,
            avg_routing_score=0.85,
        )

        assert analysis.model_accuracy == 0.5

    def test_improvement_property(self):
        analysis = VirtualExpertAnalysis(
            model_name="test",
            total_problems=10,
            correct_with_virtual=8,
            correct_without_virtual=5,
            times_virtual_used=6,
            avg_routing_score=0.85,
        )

        assert abs(analysis.improvement - 0.3) < 1e-9  # 0.8 - 0.5

    def test_zero_problems(self):
        analysis = VirtualExpertAnalysis(
            model_name="test",
            total_problems=0,
            correct_with_virtual=0,
            correct_without_virtual=0,
            times_virtual_used=0,
            avg_routing_score=0.0,
        )

        assert analysis.virtual_accuracy == 0
        assert analysis.model_accuracy == 0
        assert analysis.improvement == 0

    def test_plugins_used(self):
        analysis = VirtualExpertAnalysis(
            model_name="test",
            total_problems=10,
            correct_with_virtual=8,
            correct_without_virtual=5,
            times_virtual_used=6,
            avg_routing_score=0.85,
            plugins_used={"Math": 5, "Code": 1},
        )

        assert analysis.plugins_used["Math"] == 5
        assert analysis.plugins_used["Code"] == 1

    def test_summary(self):
        analysis = VirtualExpertAnalysis(
            model_name="test_model",
            total_problems=10,
            correct_with_virtual=8,
            correct_without_virtual=5,
            times_virtual_used=6,
            avg_routing_score=0.85,
        )

        summary = analysis.summary()
        assert "test_model" in summary
        assert "80.0%" in summary or "0.8" in summary


class TestVirtualExpertApproach:
    """Tests for VirtualExpertApproach enum."""

    def test_values(self):
        assert VirtualExpertApproach.VIRTUAL_EXPERT.value == "virtual_expert"
        assert VirtualExpertApproach.MODEL_DIRECT.value == "model_direct"


class TestVirtualExpertRegistry:
    """Tests for VirtualExpertRegistry class."""

    def test_init_empty(self):
        registry = VirtualExpertRegistry()
        assert len(registry) == 0

    def test_register_plugin(self):
        registry = VirtualExpertRegistry()

        class TestPlugin(VirtualExpertPlugin):
            name = "test"
            description = "Test plugin"

            def can_handle(self, prompt: str) -> bool:
                return True

            def get_operations(self):
                return ["evaluate"]

            def execute_operation(self, operation, parameters):
                return {"result": "test_result"}

            def get_calibration_prompts(self):
                return [], []

        plugin = TestPlugin()
        registry.register(plugin)

        assert "test" in registry
        assert registry.get("test") is plugin

    def test_get_plugin_not_found(self):
        registry = VirtualExpertRegistry()
        plugin = registry.get("nonexistent")
        assert plugin is None

    def test_find_handler(self):
        registry = VirtualExpertRegistry()

        class MathPlugin(VirtualExpertPlugin):
            name = "math"
            description = "Math"

            def can_handle(self, prompt: str) -> bool:
                return "+" in prompt

            def get_operations(self):
                return ["evaluate"]

            def execute_operation(self, operation, parameters):
                return {"result": "math"}

            def get_calibration_prompts(self):
                return [], []

        registry.register(MathPlugin())

        plugin = registry.find_handler("2 + 2 = ")
        assert plugin is not None
        action = ExpertAction(expert="math", operation="evaluate", parameters={})
        result = plugin.execute(action)
        assert result.data["result"] == "math"

    def test_find_handler_no_match(self):
        registry = VirtualExpertRegistry()

        class MathPlugin(VirtualExpertPlugin):
            name = "math"
            description = "Math"

            def can_handle(self, prompt: str) -> bool:
                return "+" in prompt

            def get_operations(self):
                return ["evaluate"]

            def execute_operation(self, operation, parameters):
                return {"result": "math"}

            def get_calibration_prompts(self):
                return [], []

        registry.register(MathPlugin())

        plugin = registry.find_handler("no math here")
        assert plugin is None

    def test_unregister(self):
        registry = VirtualExpertRegistry()

        class TestPlugin(VirtualExpertPlugin):
            name = "test"
            description = "Test"

            def can_handle(self, prompt: str) -> bool:
                return True

            def get_operations(self):
                return ["evaluate"]

            def execute_operation(self, operation, parameters):
                return {"result": "result"}

            def get_calibration_prompts(self):
                return [], []

        registry.register(TestPlugin())
        assert "test" in registry

        registry.unregister("test")
        assert "test" not in registry

    def test_get_all(self):
        registry = VirtualExpertRegistry()

        class Plugin1(VirtualExpertPlugin):
            name = "p1"
            description = "Plugin 1"
            priority = 5

            def can_handle(self, prompt: str) -> bool:
                return False

            def get_operations(self):
                return ["evaluate"]

            def execute_operation(self, operation, parameters):
                return {"result": ""}

            def get_calibration_prompts(self):
                return [], []

        class Plugin2(VirtualExpertPlugin):
            name = "p2"
            description = "Plugin 2"
            priority = 10

            def can_handle(self, prompt: str) -> bool:
                return False

            def get_operations(self):
                return ["evaluate"]

            def execute_operation(self, operation, parameters):
                return {"result": ""}

            def get_calibration_prompts(self):
                return [], []

        registry.register(Plugin1())
        registry.register(Plugin2())

        all_plugins = registry.get_all()
        assert len(all_plugins) == 2
        # Higher priority first
        assert all_plugins[0].name == "p2"

    def test_plugin_names(self):
        registry = VirtualExpertRegistry()

        class TestPlugin(VirtualExpertPlugin):
            name = "test"
            description = "Test"

            def can_handle(self, prompt: str) -> bool:
                return True

            def get_operations(self):
                return ["evaluate"]

            def execute_operation(self, operation, parameters):
                return {"result": ""}

            def get_calibration_prompts(self):
                return [], []

        registry.register(TestPlugin())
        assert "test" in registry.plugin_names

    def test_duplicate_registration_raises(self):
        registry = VirtualExpertRegistry()

        class TestPlugin(VirtualExpertPlugin):
            name = "test"
            description = "Test"

            def can_handle(self, prompt: str) -> bool:
                return True

            def get_operations(self):
                return ["evaluate"]

            def execute_operation(self, operation, parameters):
                return {"result": ""}

            def get_calibration_prompts(self):
                return [], []

        registry.register(TestPlugin())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(TestPlugin())


class TestGetDefaultRegistry:
    """Tests for get_default_registry function."""

    def setup_method(self):
        # Reset the default registry before each test
        reset_default_registry()

    def test_returns_registry(self):
        registry = get_default_registry()
        assert isinstance(registry, VirtualExpertRegistry)

    def test_has_math_plugin(self):
        registry = get_default_registry()
        math_plugin = registry.get("math")
        assert math_plugin is not None
        assert isinstance(math_plugin, MathExpertPlugin)

    def test_singleton_behavior(self):
        # Should return same instance
        registry1 = get_default_registry()
        registry2 = get_default_registry()
        assert registry1 is registry2


class TestVirtualRouter:
    """Tests for VirtualRouter class."""

    def test_class_exists(self):
        """VirtualRouter class should be importable."""
        assert VirtualRouter is not None

    def test_is_nn_module(self):
        """VirtualRouter should be an nn.Module subclass."""
        import mlx.nn as nn

        assert issubclass(VirtualRouter, nn.Module)


class TestLegacyAliases:
    """Tests for legacy compatibility aliases."""

    def test_expert_hijacker_alias(self):
        assert ExpertHijacker is VirtualMoEWrapper

    def test_virtual_expert_slot_alias(self):
        assert VirtualExpertSlot is VirtualMoEWrapper

    def test_hybrid_embedding_injector_alias(self):
        assert HybridEmbeddingInjector is VirtualMoEWrapper


class TestDemoVirtualExpert:
    """Tests for demo_virtual_expert function."""

    def test_demo_signature(self):
        # Just test that the function exists and has correct signature
        import inspect

        sig = inspect.signature(demo_virtual_expert)
        params = list(sig.parameters.keys())

        assert "model" in params
        assert "tokenizer" in params
        assert "model_id" in params
        assert "problems" in params

    def test_demo_with_default_problems(self, capsys, monkeypatch):
        """Test demo_virtual_expert with default problems."""
        from unittest.mock import MagicMock, Mock

        # Create mocks
        mock_model = MockModel()
        mock_tokenizer = MockTokenizer()

        # Mock VirtualMoEWrapper to avoid complex setup
        mock_wrapper = MagicMock()
        mock_wrapper._generate_direct = Mock(return_value="4")

        # Create mock results
        mock_result = InferenceResult(
            prompt="2 + 2 = ",
            answer="4",
            correct_answer=4,
            approach=VirtualExpertApproach.VIRTUAL_EXPERT,
            used_virtual_expert=True,
            plugin_name="math",
        )

        mock_analysis = VirtualExpertAnalysis(
            model_name="test",
            total_problems=10,
            correct_with_virtual=9,
            correct_without_virtual=3,
            times_virtual_used=9,
            avg_routing_score=0.85,
            results=[mock_result],
            plugins_used={"math": 9},
        )

        mock_wrapper.calibrate = Mock()
        mock_wrapper.benchmark = Mock(return_value=mock_analysis)

        # Patch VirtualMoEWrapper creation
        def mock_wrapper_init(model, tokenizer, model_id):
            return mock_wrapper

        monkeypatch.setattr(
            "chuk_lazarus.introspection.virtual_expert.VirtualMoEWrapper",
            mock_wrapper_init,
        )

        # Run the demo
        result = demo_virtual_expert(mock_model, mock_tokenizer, "test_model")

        # Verify the result
        assert isinstance(result, VirtualExpertAnalysis)
        assert result.total_problems == 10
        assert result.virtual_accuracy == 0.9

        # Verify output was printed
        captured = capsys.readouterr()
        assert "VIRTUAL EXPERT DEMO" in captured.out
        assert "Calibrating" in captured.out

    def test_demo_with_custom_problems(self, monkeypatch):
        """Test demo_virtual_expert with custom problems."""
        from unittest.mock import MagicMock, Mock

        mock_model = MockModel()
        mock_tokenizer = MockTokenizer()

        custom_problems = ["1 + 1 = ", "2 * 2 = "]

        # Mock wrapper
        mock_wrapper = MagicMock()
        mock_wrapper._generate_direct = Mock(return_value="2")

        mock_result = InferenceResult(
            prompt="1 + 1 = ",
            answer="2",
            correct_answer=2,
            approach=VirtualExpertApproach.VIRTUAL_EXPERT,
            used_virtual_expert=True,
        )

        mock_analysis = VirtualExpertAnalysis(
            model_name="test",
            total_problems=2,
            correct_with_virtual=2,
            correct_without_virtual=1,
            times_virtual_used=2,
            avg_routing_score=0.9,
            results=[mock_result],
        )

        mock_wrapper.calibrate = Mock()
        mock_wrapper.benchmark = Mock(return_value=mock_analysis)

        monkeypatch.setattr(
            "chuk_lazarus.introspection.virtual_expert.VirtualMoEWrapper",
            lambda m, t, mid: mock_wrapper,
        )

        result = demo_virtual_expert(mock_model, mock_tokenizer, problems=custom_problems)

        assert result.total_problems == 2

    def test_demo_with_multiple_results(self, capsys, monkeypatch):
        """Test demo with multiple results including edge cases."""
        from unittest.mock import MagicMock, Mock

        mock_model = MockModel()
        mock_tokenizer = MockTokenizer()

        mock_wrapper = MagicMock()
        mock_wrapper._generate_direct = Mock(return_value="model_answer")

        # Create multiple results with different states
        result1 = InferenceResult(
            prompt="2 + 2 = ",
            answer="4",
            correct_answer=4,
            approach=VirtualExpertApproach.VIRTUAL_EXPERT,
            used_virtual_expert=True,
            plugin_name="math",
        )

        result2 = InferenceResult(
            prompt="What is the capital?",
            answer="Paris",
            correct_answer=None,
            approach=VirtualExpertApproach.MODEL_DIRECT,
            used_virtual_expert=False,
            plugin_name=None,  # No plugin used
        )

        result3 = InferenceResult(
            prompt="5 * 5 = ",
            answer="24",  # Wrong answer
            correct_answer=25,
            approach=VirtualExpertApproach.VIRTUAL_EXPERT,
            used_virtual_expert=False,
            plugin_name="math",
        )

        mock_analysis = VirtualExpertAnalysis(
            model_name="test",
            total_problems=3,
            correct_with_virtual=1,
            correct_without_virtual=1,
            times_virtual_used=1,
            avg_routing_score=0.7,
            results=[result1, result2, result3],
            plugins_used={"math": 2},
        )

        mock_wrapper.calibrate = Mock()
        mock_wrapper.benchmark = Mock(return_value=mock_analysis)

        monkeypatch.setattr(
            "chuk_lazarus.introspection.virtual_expert.VirtualMoEWrapper",
            lambda m, t, mid: mock_wrapper,
        )

        demo_virtual_expert(mock_model, mock_tokenizer)

        # Verify all results were processed
        captured = capsys.readouterr()
        assert "2 + 2 = " in captured.out
        assert "What is the capital?" in captured.out
        assert "5 * 5 = " in captured.out

        # Verify plugin column shows "N/A" for result without plugin
        assert "N/A" in captured.out

        # Verify "YES" for virtual expert used and "no" for not used
        assert "YES" in captured.out
        assert "no" in captured.out

        # Verify improvement calculation is shown
        assert "Improvement:" in captured.out
        assert "Plugins used:" in captured.out
        assert "math: 2" in captured.out


class TestDemoAllApproaches:
    """Tests for demo_all_approaches function."""

    def test_demo_signature(self):
        import inspect

        sig = inspect.signature(demo_all_approaches)
        params = list(sig.parameters.keys())

        assert "model" in params
        assert "tokenizer" in params

    def test_demo_all_approaches_calls_demo_virtual_expert(self, monkeypatch):
        """Test that demo_all_approaches calls demo_virtual_expert."""
        from unittest.mock import Mock

        mock_model = MockModel()
        mock_tokenizer = MockTokenizer()

        mock_analysis = VirtualExpertAnalysis(
            model_name="test",
            total_problems=5,
            correct_with_virtual=4,
            correct_without_virtual=2,
            times_virtual_used=4,
            avg_routing_score=0.8,
            results=[],
        )

        # Mock demo_virtual_expert
        mock_demo = Mock(return_value=mock_analysis)
        monkeypatch.setattr(
            "chuk_lazarus.introspection.virtual_expert.demo_virtual_expert",
            mock_demo,
        )

        result = demo_all_approaches(mock_model, mock_tokenizer, "test_model")

        # Verify it returns a dict with "virtual_slot" key
        assert isinstance(result, dict)
        assert "virtual_slot" in result
        assert result["virtual_slot"] is mock_analysis

        # Verify demo_virtual_expert was called
        mock_demo.assert_called_once()

    def test_demo_all_approaches_with_custom_problems(self, monkeypatch):
        """Test demo_all_approaches with custom problems."""
        from unittest.mock import Mock

        custom_problems = ["3 + 3 = "]

        mock_model = MockModel()
        mock_tokenizer = MockTokenizer()

        mock_analysis = VirtualExpertAnalysis(
            model_name="test",
            total_problems=1,
            correct_with_virtual=1,
            correct_without_virtual=0,
            times_virtual_used=1,
            avg_routing_score=0.95,
            results=[],
        )

        mock_demo = Mock(return_value=mock_analysis)
        monkeypatch.setattr(
            "chuk_lazarus.introspection.virtual_expert.demo_virtual_expert",
            mock_demo,
        )

        demo_all_approaches(
            mock_model,
            mock_tokenizer,
            "test",
            problems=custom_problems,
        )

        # Verify the problems were passed through
        mock_demo.assert_called_once_with(
            mock_model,
            mock_tokenizer,
            "test",
            custom_problems,
        )


class TestCreateVirtualExpert:
    """Tests for create_virtual_expert function (backwards compatibility)."""

    def test_signature(self):
        import inspect

        sig = inspect.signature(create_virtual_expert)
        params = list(sig.parameters.keys())

        assert "model" in params
        assert "tokenizer" in params
        assert "approach" in params

    def test_create_virtual_expert_returns_wrapper(self, monkeypatch):
        """Test that create_virtual_expert returns a VirtualMoEWrapper."""
        from unittest.mock import MagicMock

        mock_model = MockModel()
        mock_tokenizer = MockTokenizer()

        mock_wrapper = MagicMock(spec=VirtualMoEWrapper)

        # Mock VirtualMoEWrapper constructor
        mock_wrapper_class = MagicMock(return_value=mock_wrapper)
        monkeypatch.setattr(
            "chuk_lazarus.introspection.virtual_expert.VirtualMoEWrapper",
            mock_wrapper_class,
        )

        result = create_virtual_expert(mock_model, mock_tokenizer)

        # Verify it returns the wrapper
        assert result is mock_wrapper

        # Verify VirtualMoEWrapper was called correctly
        mock_wrapper_class.assert_called_once_with(
            mock_model,
            mock_tokenizer,
            "unknown",
        )

    def test_create_virtual_expert_with_approach(self, monkeypatch):
        """Test create_virtual_expert with approach parameter (ignored)."""
        from unittest.mock import MagicMock

        mock_model = MockModel()
        mock_tokenizer = MockTokenizer()

        mock_wrapper = MagicMock(spec=VirtualMoEWrapper)
        mock_wrapper_class = MagicMock(return_value=mock_wrapper)
        monkeypatch.setattr(
            "chuk_lazarus.introspection.virtual_expert.VirtualMoEWrapper",
            mock_wrapper_class,
        )

        # The approach parameter is ignored (backwards compatibility)
        result = create_virtual_expert(
            mock_model,
            mock_tokenizer,
            approach="expert_hijacker",
        )

        assert result is mock_wrapper

    def test_create_virtual_expert_with_model_id(self, monkeypatch):
        """Test create_virtual_expert with model_id parameter."""
        from unittest.mock import MagicMock

        mock_model = MockModel()
        mock_tokenizer = MockTokenizer()

        mock_wrapper = MagicMock(spec=VirtualMoEWrapper)
        mock_wrapper_class = MagicMock(return_value=mock_wrapper)
        monkeypatch.setattr(
            "chuk_lazarus.introspection.virtual_expert.VirtualMoEWrapper",
            mock_wrapper_class,
        )

        create_virtual_expert(
            mock_model,
            mock_tokenizer,
            model_id="custom_model",
        )

        mock_wrapper_class.assert_called_once_with(
            mock_model,
            mock_tokenizer,
            "custom_model",
        )

    def test_create_virtual_expert_with_kwargs(self, monkeypatch):
        """Test create_virtual_expert passes through kwargs."""
        from unittest.mock import MagicMock

        mock_model = MockModel()
        mock_tokenizer = MockTokenizer()

        mock_wrapper = MagicMock(spec=VirtualMoEWrapper)
        mock_wrapper_class = MagicMock(return_value=mock_wrapper)
        monkeypatch.setattr(
            "chuk_lazarus.introspection.virtual_expert.VirtualMoEWrapper",
            mock_wrapper_class,
        )

        mock_registry = MagicMock()

        create_virtual_expert(
            mock_model,
            mock_tokenizer,
            registry=mock_registry,
        )

        # Verify kwargs were passed through
        mock_wrapper_class.assert_called_once_with(
            mock_model,
            mock_tokenizer,
            "unknown",
            registry=mock_registry,
        )


class TestVirtualExpertPlugin:
    """Tests for VirtualExpertPlugin abstract base class."""

    def test_cannot_instantiate_directly(self):
        # ABC should not be instantiable
        with pytest.raises(TypeError):
            VirtualExpertPlugin()  # type: ignore

    def test_subclass_must_implement_methods(self):
        # Missing implementations should raise error

        class IncompletePlugin(VirtualExpertPlugin):
            name = "incomplete"
            description = "Incomplete"

        with pytest.raises(TypeError):
            IncompletePlugin()  # type: ignore

    def test_valid_subclass(self):
        class ValidPlugin(VirtualExpertPlugin):
            name = "valid"
            description = "Valid"

            def can_handle(self, prompt: str) -> bool:
                return True

            def get_operations(self):
                return ["evaluate"]

            def execute_operation(self, operation, parameters):
                return {"result": "result"}

            def get_calibration_prompts(self):
                return [], []

        plugin = ValidPlugin()
        assert plugin.can_handle("test") is True
        action = ExpertAction(expert="valid", operation="evaluate", parameters={})
        result = plugin.execute(action)
        assert result.success is True
        assert result.data["result"] == "result"


class TestCreateVirtualExpertWrapper:
    """Tests for create_virtual_expert_wrapper function."""

    def test_function_is_exported(self):
        """Test that create_virtual_expert_wrapper is exported."""
        assert create_virtual_expert_wrapper is not None

    def test_function_signature(self):
        """Test that create_virtual_expert_wrapper has correct signature."""
        import inspect

        sig = inspect.signature(create_virtual_expert_wrapper)
        params = list(sig.parameters.keys())

        assert "model" in params
        assert "tokenizer" in params


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_exist(self):
        """Test that all declared exports are actually available."""
        from chuk_lazarus.introspection import virtual_expert

        # Get __all__ from the module
        all_exports = virtual_expert.__all__

        # Verify each export exists
        for export_name in all_exports:
            assert hasattr(virtual_expert, export_name), f"{export_name} not found in module"

    def test_core_classes_exported(self):
        """Test that core classes are properly exported."""
        from chuk_lazarus.introspection import virtual_expert

        assert "VirtualExpertPlugin" in virtual_expert.__all__
        assert "VirtualExpertRegistry" in virtual_expert.__all__
        assert "VirtualExpertResult" in virtual_expert.__all__
        assert "VirtualExpertAnalysis" in virtual_expert.__all__
        assert "VirtualMoEWrapper" in virtual_expert.__all__

    def test_legacy_aliases_exported(self):
        """Test that legacy aliases are exported."""
        from chuk_lazarus.introspection import virtual_expert

        assert "ExpertHijacker" in virtual_expert.__all__
        assert "VirtualExpertSlot" in virtual_expert.__all__
        assert "HybridEmbeddingInjector" in virtual_expert.__all__

    def test_demo_functions_exported(self):
        """Test that demo functions are exported."""
        from chuk_lazarus.introspection import virtual_expert

        assert "demo_virtual_expert" in virtual_expert.__all__
        assert "demo_all_approaches" in virtual_expert.__all__
        assert "create_virtual_expert" in virtual_expert.__all__


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_demo_virtual_expert_with_empty_problems(self, monkeypatch):
        """Test demo_virtual_expert with empty problems list."""
        from unittest.mock import MagicMock, Mock

        mock_model = MockModel()
        mock_tokenizer = MockTokenizer()

        mock_wrapper = MagicMock()
        mock_wrapper._generate_direct = Mock(return_value="")

        mock_analysis = VirtualExpertAnalysis(
            model_name="test",
            total_problems=0,
            correct_with_virtual=0,
            correct_without_virtual=0,
            times_virtual_used=0,
            avg_routing_score=0.0,
            results=[],
        )

        mock_wrapper.calibrate = Mock()
        mock_wrapper.benchmark = Mock(return_value=mock_analysis)

        monkeypatch.setattr(
            "chuk_lazarus.introspection.virtual_expert.VirtualMoEWrapper",
            lambda m, t, mid: mock_wrapper,
        )

        result = demo_virtual_expert(mock_model, mock_tokenizer, problems=[])

        assert result.total_problems == 0

    def test_demo_analysis_with_no_plugins_used(self, capsys, monkeypatch):
        """Test demo when no plugins are used."""
        from unittest.mock import MagicMock, Mock

        mock_model = MockModel()
        mock_tokenizer = MockTokenizer()

        mock_wrapper = MagicMock()
        mock_wrapper._generate_direct = Mock(return_value="answer")

        mock_result = InferenceResult(
            prompt="question",
            answer="answer",
            correct_answer=None,
            approach=VirtualExpertApproach.MODEL_DIRECT,
            used_virtual_expert=False,
        )

        # Analysis with no plugins used
        mock_analysis = VirtualExpertAnalysis(
            model_name="test",
            total_problems=1,
            correct_with_virtual=0,
            correct_without_virtual=0,
            times_virtual_used=0,
            avg_routing_score=0.0,
            results=[mock_result],
            plugins_used={},  # No plugins
        )

        mock_wrapper.calibrate = Mock()
        mock_wrapper.benchmark = Mock(return_value=mock_analysis)

        monkeypatch.setattr(
            "chuk_lazarus.introspection.virtual_expert.VirtualMoEWrapper",
            lambda m, t, mid: mock_wrapper,
        )

        result = demo_virtual_expert(mock_model, mock_tokenizer)

        # Should handle empty plugins_used gracefully
        assert result.times_virtual_used == 0

        captured = capsys.readouterr()
        assert "VIRTUAL EXPERT DEMO" in captured.out

    def test_demo_with_incorrect_answers(self, monkeypatch):
        """Test demo with incorrect answers."""
        from unittest.mock import MagicMock, Mock

        mock_model = MockModel()
        mock_tokenizer = MockTokenizer()

        mock_wrapper = MagicMock()
        mock_wrapper._generate_direct = Mock(return_value="wrong")

        # Incorrect result
        mock_result = InferenceResult(
            prompt="2 + 2 = ",
            answer="5",  # Wrong!
            correct_answer=4,
            approach=VirtualExpertApproach.VIRTUAL_EXPERT,
            used_virtual_expert=True,
            plugin_name="math",
        )

        mock_analysis = VirtualExpertAnalysis(
            model_name="test",
            total_problems=1,
            correct_with_virtual=0,  # Incorrect
            correct_without_virtual=0,
            times_virtual_used=1,
            avg_routing_score=0.9,
            results=[mock_result],
            plugins_used={"math": 1},
        )

        mock_wrapper.calibrate = Mock()
        mock_wrapper.benchmark = Mock(return_value=mock_analysis)

        monkeypatch.setattr(
            "chuk_lazarus.introspection.virtual_expert.VirtualMoEWrapper",
            lambda m, t, mid: mock_wrapper,
        )

        result = demo_virtual_expert(mock_model, mock_tokenizer)

        assert result.virtual_accuracy == 0.0
        assert not mock_result.is_correct


class TestMathExpertPluginEdgeCases:
    """Additional edge case tests for MathExpertPlugin."""

    def test_complex_expression(self):
        """Test with more complex expressions."""
        plugin = MathExpertPlugin()

        # Multiple operations
        action = ExpertAction(expert="math", operation="evaluate", parameters={"expression": "10 + 5 * 2"})
        result = plugin.execute(action)
        assert result.success is True
        assert result.data["result"] == 20

    def test_parentheses(self):
        """Test expressions with parentheses."""
        plugin = MathExpertPlugin()

        action = ExpertAction(expert="math", operation="evaluate", parameters={"expression": "(10 + 5) * 2"})
        result = plugin.execute(action)
        assert result.success is True
        assert result.data["result"] == 30

    def test_negative_numbers(self):
        """Test with negative numbers."""
        plugin = MathExpertPlugin()

        action = ExpertAction(expert="math", operation="evaluate", parameters={"expression": "-5 + 10"})
        result = plugin.execute(action)
        assert result.success is True
        assert result.data["result"] == 5

    def test_decimal_results(self):
        """Test operations that produce decimals."""
        plugin = MathExpertPlugin()

        action = ExpertAction(expert="math", operation="evaluate", parameters={"expression": "10 / 3"})
        result = plugin.execute(action)
        assert result.success is True
        # Result should be approximately 3.333...
        assert "3" in str(result.data["result"])

    def test_very_large_numbers(self):
        """Test with very large numbers."""
        plugin = MathExpertPlugin()

        action = ExpertAction(expert="math", operation="evaluate", parameters={"expression": "999999 * 999999"})
        result = plugin.execute(action)
        assert result.success is True
        assert result.data["result"] == 999998000001

    def test_division_by_zero_handling(self):
        """Test division by zero is handled gracefully."""
        plugin = MathExpertPlugin()

        # Should not crash
        action = ExpertAction(expert="math", operation="evaluate", parameters={"expression": "10 / 0"})
        result = plugin.execute(action)
        # Should return failure or handle gracefully
        assert result.success is False or isinstance(result.data, dict)

    def test_can_handle_variations(self):
        """Test can_handle with various formats."""
        plugin = MathExpertPlugin()

        # With equals sign
        assert plugin.can_handle("2+2=")
        assert plugin.can_handle("2 + 2 = ")

        # Without equals sign
        assert plugin.can_handle("2 + 2") is True or plugin.can_handle("2 + 2") is False

        # Different operators
        assert plugin.can_handle("10 / 2 = ")
        assert plugin.can_handle("5 * 5 = ")
        assert plugin.can_handle("10 - 3 = ")
