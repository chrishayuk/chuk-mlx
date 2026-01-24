"""Tests for virtual_experts/dense_wrapper.py to improve coverage."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.inference.virtual_experts.base import (
    VirtualExpertApproach,
    VirtualExpertPlugin,
)
from chuk_lazarus.inference.virtual_experts.dense_wrapper import (
    VirtualDenseRouter,
    VirtualDenseWrapper,
    create_virtual_dense_wrapper,
)
from chuk_lazarus.inference.virtual_experts.plugins.math import MathExpertPlugin
from chuk_lazarus.inference.virtual_experts.registry import (
    VirtualExpertRegistry,
    reset_default_registry,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 100 for c in text[:10]]

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        return str(ids[0])


class MockLayer(nn.Module):
    """Mock transformer layer."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        return x


class MockModelConfig:
    """Mock model config."""

    def __init__(self, hidden_size: int = 64, embedding_scale: float | None = None):
        self.hidden_size = hidden_size
        self.embedding_scale = embedding_scale


class MockEmbedding(nn.Module):
    """Mock embedding layer."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = mx.random.normal((vocab_size, hidden_size))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.take(self.weight, x, axis=0)


class MockLMHead(nn.Module):
    """Mock LM head."""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.weight = mx.random.normal((vocab_size, hidden_size))

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.T


class MockBackbone(nn.Module):
    """Mock model backbone."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 4, vocab_size: int = 100):
        super().__init__()
        self.embed_tokens = MockEmbedding(vocab_size, hidden_size)
        self.layers = [MockLayer(hidden_size) for _ in range(num_layers)]
        self.norm = nn.LayerNorm(hidden_size)


class MockDenseModel(nn.Module):
    """Mock dense model for testing."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 4, vocab_size: int = 100):
        super().__init__()
        self.model = MockBackbone(hidden_size, num_layers, vocab_size)
        self.lm_head = MockLMHead(hidden_size, vocab_size)
        self.config = MockModelConfig(hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.model.embed_tokens(x)
        for layer in self.model.layers:
            h = layer(h)
        h = self.model.norm(h)
        return self.lm_head(h)


class TestVirtualDenseRouter:
    """Tests for VirtualDenseRouter class."""

    def test_init(self):
        """Test VirtualDenseRouter initialization."""
        router = VirtualDenseRouter(hidden_size=64, num_virtual_experts=2)

        assert router.hidden_size == 64
        assert router.num_virtual_experts == 2
        assert len(router.directions) == 2
        assert len(router._calibrated) == 2

    def test_calibrate_expert(self):
        """Test expert calibration."""
        router = VirtualDenseRouter(hidden_size=64, num_virtual_experts=2)

        pos = [mx.ones((64,)) * 10 for _ in range(3)]
        neg = [mx.ones((64,)) * -10 for _ in range(3)]

        router.calibrate_expert(0, pos, neg)

        assert router._calibrated[0] is True
        assert router._calibrated[1] is False

    def test_calibrate_expert_invalid_index(self):
        """Test calibration with invalid expert index."""
        router = VirtualDenseRouter(hidden_size=64, num_virtual_experts=2)

        pos = [mx.random.normal((64,))]
        neg = [mx.random.normal((64,))]

        with pytest.raises(ValueError, match="Expert index"):
            router.calibrate_expert(5, pos, neg)

    def test_calibrate_expert_scale_fallback(self):
        """Test scale falls back to 1.0 when difference is small."""
        router = VirtualDenseRouter(hidden_size=64, num_virtual_experts=1)

        # Very similar activations
        base = mx.random.normal((64,))
        pos = [base + mx.random.normal((64,)) * 0.001 for _ in range(3)]
        neg = [base + mx.random.normal((64,)) * 0.001 for _ in range(3)]

        router.calibrate_expert(0, pos, neg)
        assert router._calibrated[0] is True

    def test_get_routing_score_uncalibrated(self):
        """Test routing score for uncalibrated expert."""
        router = VirtualDenseRouter(hidden_size=64, num_virtual_experts=1)

        x = mx.random.normal((1, 1, 64))
        score = router.get_routing_score(x, 0)

        assert score == 0.0

    def test_get_routing_score_calibrated(self):
        """Test routing score for calibrated expert."""
        router = VirtualDenseRouter(hidden_size=64, num_virtual_experts=1)

        pos = [mx.ones((64,)) * 10 for _ in range(3)]
        neg = [mx.ones((64,)) * -10 for _ in range(3)]
        router.calibrate_expert(0, pos, neg)

        x = mx.ones((1, 1, 64)) * 10
        score = router.get_routing_score(x, 0)

        assert 0.0 <= score <= 1.0

    def test_get_routing_score_3d_input(self):
        """Test routing score with 3D input."""
        router = VirtualDenseRouter(hidden_size=64, num_virtual_experts=1)

        pos = [mx.ones((64,)) * 10 for _ in range(3)]
        neg = [mx.ones((64,)) * -10 for _ in range(3)]
        router.calibrate_expert(0, pos, neg)

        x = mx.ones((2, 3, 64)) * 10
        score = router.get_routing_score(x, 0)

        assert 0.0 <= score <= 1.0

    def test_should_route_to_expert(self):
        """Test should_route_to_expert method."""
        router = VirtualDenseRouter(hidden_size=64, num_virtual_experts=1)

        pos = [mx.ones((64,)) * 10 for _ in range(3)]
        neg = [mx.ones((64,)) * -10 for _ in range(3)]
        router.calibrate_expert(0, pos, neg)

        x = mx.ones((1, 1, 64)) * 10
        # Test with different thresholds
        assert isinstance(router.should_route_to_expert(x, 0, 0.1), bool)


class TestVirtualDenseWrapper:
    """Tests for VirtualDenseWrapper class."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_default_registry()

    def test_init_basic(self):
        """Test basic initialization."""
        model = MockDenseModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualDenseWrapper(model, tokenizer, "test_model")

        assert wrapper.model is model
        assert wrapper.tokenizer is tokenizer
        assert wrapper.model_id == "test_model"
        assert wrapper._calibrated is False

    def test_init_with_target_layer(self):
        """Test initialization with specific target layer."""
        model = MockDenseModel(num_layers=4)
        tokenizer = MockTokenizer()

        wrapper = VirtualDenseWrapper(model, tokenizer, target_layer=1)

        assert wrapper.target_layer == 1

    def test_init_detects_structure(self):
        """Test that init detects model structure."""
        model = MockDenseModel(hidden_size=128)
        tokenizer = MockTokenizer()

        wrapper = VirtualDenseWrapper(model, tokenizer)

        assert wrapper.hidden_size == 128
        assert wrapper.num_layers == 4

    def test_init_with_custom_registry(self):
        """Test initialization with custom registry."""
        model = MockDenseModel()
        tokenizer = MockTokenizer()
        registry = VirtualExpertRegistry()
        registry.register(MathExpertPlugin())  # Add a plugin so it's non-empty

        wrapper = VirtualDenseWrapper(model, tokenizer, registry=registry)

        # Registry should be the one we passed (has math plugin we registered)
        assert "math" in wrapper.registry

    def test_init_cannot_detect_structure(self):
        """Test error when model structure cannot be detected."""

        class BadModel(nn.Module):
            def __call__(self, x):
                return x

        with pytest.raises(ValueError, match="Cannot detect model structure"):
            VirtualDenseWrapper(BadModel(), MockTokenizer())

    def test_init_direct_layers(self):
        """Test initialization when model has layers directly."""

        class DirectLayersModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [MockLayer(64) for _ in range(4)]
                self.embed_tokens = MockEmbedding(100, 64)
                self.norm = nn.LayerNorm(64)
                self.config = MockModelConfig(64)

            def __call__(self, x):
                h = self.embed_tokens(x)
                for layer in self.layers:
                    h = layer(h)
                return self.norm(h)

        model = DirectLayersModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualDenseWrapper(model, tokenizer)
        assert wrapper.num_layers == 4

    def test_register_plugin(self):
        """Test plugin registration."""
        model = MockDenseModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualDenseWrapper(model, tokenizer)

        class TestPlugin(VirtualExpertPlugin):
            name = "test_plugin"
            description = "Test"

            def can_handle(self, prompt: str) -> bool:
                return "test" in prompt

            def get_operations(self):
                return ["evaluate"]

            def execute_operation(self, operation, parameters):
                return {"result": "test_result", "formatted": "test_result"}

            def get_calibration_prompts(self):
                return ["test 1", "test 2"], ["hello", "world"]

        wrapper.register_plugin(TestPlugin())

        assert "test_plugin" in wrapper.registry
        assert wrapper._calibrated is False

    def test_calibrate(self):
        """Test calibration."""
        model = MockDenseModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualDenseWrapper(model, tokenizer)
        wrapper.calibrate()

        assert wrapper._calibrated is True

    def test_solve_uses_virtual_expert(self):
        """Test solve uses virtual expert for math."""
        model = MockDenseModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualDenseWrapper(model, tokenizer)
        wrapper.calibrate()

        # Math expression should route to math expert
        result = wrapper.solve("127 * 89 = ")

        # Result should have expected fields
        assert result.prompt == "127 * 89 = "
        assert result.answer is not None

    def test_solve_without_calibration(self):
        """Test solve auto-calibrates if not calibrated."""
        model = MockDenseModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualDenseWrapper(model, tokenizer)
        assert wrapper._calibrated is False

        result = wrapper.solve("2 + 2 = ")

        # Should have auto-calibrated
        assert wrapper._calibrated is True
        assert result.answer is not None

    def test_solve_model_direct(self):
        """Test solve falls back to model direct."""
        model = MockDenseModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualDenseWrapper(model, tokenizer)
        wrapper.calibrate()

        # Non-math prompt
        result = wrapper.solve("Hello world")

        assert result.approach == VirtualExpertApproach.MODEL_DIRECT

    def test_compare(self, capsys):
        """Test compare method."""
        model = MockDenseModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualDenseWrapper(model, tokenizer)
        wrapper.calibrate()

        wrapper.compare("2 + 2 = ")

        captured = capsys.readouterr()
        assert "Prompt:" in captured.out

    def test_benchmark(self):
        """Test benchmark method."""
        model = MockDenseModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualDenseWrapper(model, tokenizer)

        problems = ["2 + 2 = ", "5 * 5 = "]
        analysis = wrapper.benchmark(problems)

        assert analysis.total_problems == 2
        assert analysis.model_name == "unknown"


class TestCreateVirtualDenseWrapper:
    """Tests for create_virtual_dense_wrapper factory function."""

    def setup_method(self):
        reset_default_registry()

    def test_basic_creation(self):
        """Test basic wrapper creation."""
        model = MockDenseModel()
        tokenizer = MockTokenizer()

        wrapper = create_virtual_dense_wrapper(model, tokenizer, "test")

        assert isinstance(wrapper, VirtualDenseWrapper)
        assert wrapper.model_id == "test"

    def test_with_plugins(self):
        """Test creation with additional plugins."""
        model = MockDenseModel()
        tokenizer = MockTokenizer()

        class TestPlugin(VirtualExpertPlugin):
            name = "extra"
            description = "Extra plugin"

            def can_handle(self, prompt: str) -> bool:
                return True

            def get_operations(self):
                return ["evaluate"]

            def execute_operation(self, operation, parameters):
                return {"result": "extra", "formatted": "extra"}

            def get_calibration_prompts(self):
                return [], []

        wrapper = create_virtual_dense_wrapper(model, tokenizer, "test", plugins=[TestPlugin()])

        assert "extra" in wrapper.registry

    def test_with_kwargs(self):
        """Test creation with kwargs."""
        model = MockDenseModel()
        tokenizer = MockTokenizer()

        wrapper = create_virtual_dense_wrapper(model, tokenizer, "test", routing_threshold=0.7)

        assert wrapper.routing_threshold == 0.7


class TestVirtualDenseWrapperGenerateDirect:
    """Tests for _generate_direct method."""

    def setup_method(self):
        reset_default_registry()

    def test_generate_direct_basic(self):
        """Test basic direct generation."""
        model = MockDenseModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualDenseWrapper(model, tokenizer)
        result = wrapper._generate_direct("test prompt")

        assert isinstance(result, str)

    def test_generate_direct_max_tokens(self):
        """Test generation with max tokens."""
        model = MockDenseModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualDenseWrapper(model, tokenizer)
        result = wrapper._generate_direct("test", max_tokens=5)

        assert isinstance(result, str)


class TestVirtualDenseWrapperGetHiddenState:
    """Tests for _get_hidden_state method."""

    def setup_method(self):
        reset_default_registry()

    def test_get_hidden_state(self):
        """Test getting hidden state."""
        model = MockDenseModel(hidden_size=64)
        tokenizer = MockTokenizer()

        wrapper = VirtualDenseWrapper(model, tokenizer)
        hidden = wrapper._get_hidden_state("test")

        assert hidden.shape == (64,)


class TestModelWithEmbeddingScale:
    """Tests for model with embedding scale."""

    def setup_method(self):
        reset_default_registry()

    def test_model_with_embedding_scale(self):
        """Test model that has embedding_scale in config."""

        class ModelWithScale(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = MockBackbone(64, 4, 100)
                self.lm_head = MockLMHead(64, 100)
                self.config = MockModelConfig(64, embedding_scale=2.0)

            def __call__(self, x):
                h = self.model.embed_tokens(x)
                for layer in self.model.layers:
                    h = layer(h)
                return self.lm_head(h)

        model = ModelWithScale()
        tokenizer = MockTokenizer()

        wrapper = VirtualDenseWrapper(model, tokenizer)
        # Should use embedding scale
        assert wrapper._embed_scale == 2.0


class TestModelWithoutConfig:
    """Tests for model without config."""

    def setup_method(self):
        reset_default_registry()

    def test_model_without_config(self):
        """Test model that doesn't have config attribute."""

        class ModelNoConfig(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = MockBackbone(64, 4, 100)
                self.lm_head = MockLMHead(64, 100)
                # No config attribute

            def __call__(self, x):
                h = self.model.embed_tokens(x)
                for layer in self.model.layers:
                    h = layer(h)
                return self.lm_head(h)

        model = ModelNoConfig()
        tokenizer = MockTokenizer()

        # Should infer hidden_size from embedding
        wrapper = VirtualDenseWrapper(model, tokenizer)
        assert wrapper.hidden_size == 64


class TestModelCannotDetermineHiddenSize:
    """Tests for model where hidden size cannot be determined."""

    def setup_method(self):
        reset_default_registry()

    def test_cannot_determine_hidden_size(self):
        """Test error when hidden size cannot be determined."""

        class BadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [MockLayer(64)]
                # No embed_tokens and no config

            def __call__(self, x):
                return x

        model = BadModel()
        tokenizer = MockTokenizer()

        with pytest.raises(ValueError, match="Could not determine hidden size"):
            VirtualDenseWrapper(model, tokenizer)
