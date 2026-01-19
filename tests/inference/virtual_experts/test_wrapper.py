"""Tests for virtual_experts/wrapper.py to improve coverage."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.inference.virtual_experts.base import (
    VirtualExpertApproach,
    VirtualExpertPlugin,
)
from chuk_lazarus.inference.virtual_experts.registry import (
    VirtualExpertRegistry,
    reset_default_registry,
)
from chuk_lazarus.inference.virtual_experts.wrapper import (
    VirtualMoEWrapper,
    create_virtual_expert_wrapper,
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


class MockRouter(nn.Module):
    """Mock MoE router."""

    def __init__(self, hidden_size: int = 64, num_experts: int = 8):
        super().__init__()
        self.weight = mx.random.normal((num_experts, hidden_size))
        self.bias = mx.zeros((num_experts,))
        self.num_experts = num_experts
        self.num_experts_per_tok = 2


class MockMoE(nn.Module):
    """Mock MoE layer."""

    def __init__(self, hidden_size: int = 64, num_experts: int = 8):
        super().__init__()
        self.router = MockRouter(hidden_size, num_experts)


class MockMoELayer(nn.Module):
    """Mock transformer layer with MoE."""

    def __init__(self, hidden_size: int = 64, num_experts: int = 8):
        super().__init__()
        self.mlp = MockMoE(hidden_size, num_experts)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        return x


class MockDenseLayer(nn.Module):
    """Mock dense (non-MoE) layer."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        return x


class MockEmbedding(nn.Module):
    """Mock embedding."""

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


class MockModelConfig:
    """Mock model config."""

    def __init__(self, hidden_size: int = 64, embedding_scale: float | None = None):
        self.hidden_size = hidden_size
        self.embedding_scale = embedding_scale


class MockBackbone(nn.Module):
    """Mock model backbone with mixed layers."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 4,
        vocab_size: int = 100,
        moe_layers: list[int] | None = None,
    ):
        super().__init__()
        self.embed_tokens = MockEmbedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

        moe_layers = moe_layers or [1, 3]  # Default: layers 1 and 3 are MoE
        self.layers = []
        for i in range(num_layers):
            if i in moe_layers:
                self.layers.append(MockMoELayer(hidden_size))
            else:
                self.layers.append(MockDenseLayer(hidden_size))


class MockMoEModel(nn.Module):
    """Mock MoE model for testing."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 4,
        vocab_size: int = 100,
        moe_layers: list[int] | None = None,
    ):
        super().__init__()
        self.model = MockBackbone(hidden_size, num_layers, vocab_size, moe_layers)
        self.lm_head = MockLMHead(hidden_size, vocab_size)
        self.config = MockModelConfig(hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.model.embed_tokens(x)
        for layer in self.model.layers:
            h = layer(h)
        h = self.model.norm(h)
        return self.lm_head(h)


class TestVirtualMoEWrapperInit:
    """Tests for VirtualMoEWrapper initialization."""

    def setup_method(self):
        reset_default_registry()

    def test_init_basic(self):
        """Test basic initialization."""
        model = MockMoEModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualMoEWrapper(model, tokenizer, "test_model")

        assert wrapper.model is model
        assert wrapper.tokenizer is tokenizer
        assert wrapper.model_id == "test_model"
        assert wrapper._calibrated is False

    def test_init_finds_moe_layers(self):
        """Test that init finds MoE layers."""
        model = MockMoEModel(moe_layers=[1, 3])
        tokenizer = MockTokenizer()

        wrapper = VirtualMoEWrapper(model, tokenizer)

        assert wrapper.moe_layers == [1, 3]

    def test_init_with_target_layers(self):
        """Test initialization with specific target layers."""
        model = MockMoEModel(moe_layers=[1, 3])
        tokenizer = MockTokenizer()

        wrapper = VirtualMoEWrapper(model, tokenizer, target_layers=[1])

        assert wrapper.target_layers == [1]

    def test_init_no_moe_layers(self):
        """Test error when no MoE layers found."""

        class DenseOnlyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = type(
                    "Backbone",
                    (),
                    {
                        "embed_tokens": MockEmbedding(100, 64),
                        "layers": [MockDenseLayer(64) for _ in range(4)],
                        "norm": nn.LayerNorm(64),
                    },
                )()
                self.lm_head = MockLMHead(64, 100)
                self.config = MockModelConfig(64)

            def __call__(self, x):
                return x

        model = DenseOnlyModel()
        tokenizer = MockTokenizer()

        with pytest.raises(ValueError, match="No MoE layers found"):
            VirtualMoEWrapper(model, tokenizer)

    def test_init_detects_structure(self):
        """Test that init detects model structure."""
        model = MockMoEModel(hidden_size=128)
        tokenizer = MockTokenizer()

        wrapper = VirtualMoEWrapper(model, tokenizer)

        assert wrapper.num_layers == 4

    def test_init_with_custom_registry(self):
        """Test initialization with custom registry."""
        from chuk_lazarus.inference.virtual_experts.plugins.math import MathExpertPlugin

        model = MockMoEModel()
        tokenizer = MockTokenizer()
        registry = VirtualExpertRegistry()
        registry.register(MathExpertPlugin())

        wrapper = VirtualMoEWrapper(model, tokenizer, registry=registry)

        # Registry should contain our math plugin
        assert "math" in wrapper.registry

    def test_init_cannot_detect_structure(self):
        """Test error when model structure cannot be detected."""

        class BadModel(nn.Module):
            def __call__(self, x):
                return x

        with pytest.raises(ValueError, match="Cannot detect model structure"):
            VirtualMoEWrapper(BadModel(), MockTokenizer())

    def test_init_direct_layers(self):
        """Test initialization when model has layers directly."""

        class DirectLayersModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = MockEmbedding(100, 64)
                self.layers = [MockMoELayer(64) for _ in range(4)]
                self.norm = nn.LayerNorm(64)
                self.config = MockModelConfig(64)

            def __call__(self, x):
                return x

        model = DirectLayersModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualMoEWrapper(model, tokenizer)
        assert wrapper.num_layers == 4


class TestVirtualMoEWrapperRegisterPlugin:
    """Tests for register_plugin method."""

    def setup_method(self):
        reset_default_registry()

    def test_register_plugin(self):
        """Test plugin registration."""
        model = MockMoEModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualMoEWrapper(model, tokenizer)

        class TestPlugin(VirtualExpertPlugin):
            name = "test_plugin"
            description = "Test"

            def can_handle(self, prompt: str) -> bool:
                return "test" in prompt

            def execute(self, prompt: str) -> str:
                return "test_result"

            def get_calibration_prompts(self):
                return ["test 1", "test 2"], ["hello", "world"]

        wrapper.register_plugin(TestPlugin())

        assert "test_plugin" in wrapper.registry
        assert wrapper._calibrated is False


class TestVirtualMoEWrapperCalibrate:
    """Tests for calibrate method."""

    def setup_method(self):
        reset_default_registry()

    def test_calibrate(self):
        """Test calibration."""
        model = MockMoEModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualMoEWrapper(model, tokenizer)
        wrapper.calibrate()

        assert wrapper._calibrated is True


class TestVirtualMoEWrapperSolve:
    """Tests for solve method."""

    def setup_method(self):
        reset_default_registry()

    def test_solve_basic(self):
        """Test basic solve."""
        model = MockMoEModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualMoEWrapper(model, tokenizer)
        wrapper.calibrate()

        result = wrapper.solve("127 * 89 = ")

        assert result.prompt == "127 * 89 = "
        assert result.answer is not None

    def test_solve_auto_calibrates(self):
        """Test solve auto-calibrates if not calibrated."""
        model = MockMoEModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualMoEWrapper(model, tokenizer)
        assert wrapper._calibrated is False

        result = wrapper.solve("2 + 2 = ")

        assert wrapper._calibrated is True
        assert result.answer is not None

    def test_solve_model_direct(self):
        """Test solve falls back to model direct."""
        model = MockMoEModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualMoEWrapper(model, tokenizer)
        wrapper.calibrate()

        result = wrapper.solve("Hello world")

        assert result.approach == VirtualExpertApproach.MODEL_DIRECT


class TestVirtualMoEWrapperCompare:
    """Tests for compare method."""

    def setup_method(self):
        reset_default_registry()

    def test_compare(self, capsys):
        """Test compare method."""
        model = MockMoEModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualMoEWrapper(model, tokenizer)
        wrapper.calibrate()

        wrapper.compare("2 + 2 = ")

        captured = capsys.readouterr()
        assert "Prompt:" in captured.out


class TestVirtualMoEWrapperBenchmark:
    """Tests for benchmark method."""

    def setup_method(self):
        reset_default_registry()

    def test_benchmark(self):
        """Test benchmark method."""
        model = MockMoEModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualMoEWrapper(model, tokenizer)

        problems = ["2 + 2 = ", "5 * 5 = "]
        analysis = wrapper.benchmark(problems)

        assert analysis.total_problems == 2


class TestCreateVirtualExpertWrapper:
    """Tests for create_virtual_expert_wrapper factory function."""

    def setup_method(self):
        reset_default_registry()

    def test_basic_creation(self):
        """Test basic wrapper creation."""
        model = MockMoEModel()
        tokenizer = MockTokenizer()

        wrapper = create_virtual_expert_wrapper(model, tokenizer, "test")

        assert isinstance(wrapper, VirtualMoEWrapper)
        assert wrapper.model_id == "test"

    def test_with_plugins(self):
        """Test creation with additional plugins."""
        model = MockMoEModel()
        tokenizer = MockTokenizer()

        class TestPlugin(VirtualExpertPlugin):
            name = "extra"
            description = "Extra plugin"

            def can_handle(self, prompt: str) -> bool:
                return True

            def execute(self, prompt: str) -> str:
                return "extra"

            def get_calibration_prompts(self):
                return [], []

        wrapper = create_virtual_expert_wrapper(model, tokenizer, "test", plugins=[TestPlugin()])

        assert "extra" in wrapper.registry


class TestVirtualMoEWrapperGenerateDirect:
    """Tests for _generate_direct method."""

    def setup_method(self):
        reset_default_registry()

    def test_generate_direct_basic(self):
        """Test basic direct generation."""
        model = MockMoEModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualMoEWrapper(model, tokenizer)
        result = wrapper._generate_direct("test prompt")

        assert isinstance(result, str)

    def test_generate_direct_max_tokens(self):
        """Test generation with max tokens."""
        model = MockMoEModel()
        tokenizer = MockTokenizer()

        wrapper = VirtualMoEWrapper(model, tokenizer)
        result = wrapper._generate_direct("test", max_tokens=5)

        assert isinstance(result, str)


class TestVirtualMoEWrapperGetHiddenState:
    """Tests for _get_hidden_state method."""

    def setup_method(self):
        reset_default_registry()

    def test_get_hidden_state(self):
        """Test getting hidden state."""
        model = MockMoEModel(hidden_size=64)
        tokenizer = MockTokenizer()

        wrapper = VirtualMoEWrapper(model, tokenizer)
        hidden = wrapper._get_hidden_state("test", layer_idx=1)

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
                self.model = MockBackbone(64, 4, 100, [1, 3])
                self.lm_head = MockLMHead(64, 100)
                self.config = MockModelConfig(64, embedding_scale=2.0)

            def __call__(self, x):
                h = self.model.embed_tokens(x)
                for layer in self.model.layers:
                    h = layer(h)
                return self.lm_head(x)

        model = ModelWithScale()
        tokenizer = MockTokenizer()

        wrapper = VirtualMoEWrapper(model, tokenizer)
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
                self.model = MockBackbone(64, 4, 100, [1, 3])
                self.lm_head = MockLMHead(64, 100)
                # No config attribute

            def __call__(self, x):
                h = self.model.embed_tokens(x)
                for layer in self.model.layers:
                    h = layer(h)
                return self.lm_head(x)

        model = ModelNoConfig()
        tokenizer = MockTokenizer()

        wrapper = VirtualMoEWrapper(model, tokenizer)
        # Should work without config
        assert wrapper._embed_scale is None
