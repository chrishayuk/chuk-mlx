"""Tests for unified inference pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest
from pydantic import BaseModel

from chuk_lazarus.inference.chat import ChatHistory
from chuk_lazarus.inference.generation import GenerationConfig, GenerationResult
from chuk_lazarus.inference.loader import DType
from chuk_lazarus.inference.unified import (
    IntrospectionResult,
    UnifiedPipeline,
    UnifiedPipelineConfig,
    UnifiedPipelineState,
)
from chuk_lazarus.models_v2.families import FamilyInfo, ModelFamilyType


class TestUnifiedPipelineConfig:
    """Tests for UnifiedPipelineConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = UnifiedPipelineConfig()
        assert config.dtype == DType.BFLOAT16
        assert config.cache_dir is None
        assert config.default_system_message == "You are a helpful assistant."
        assert config.default_max_tokens == 256
        assert config.default_temperature == 0.7
        assert config.enable_introspection is True
        assert config.introspection_layers is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = UnifiedPipelineConfig(
            dtype=DType.FLOAT16,
            cache_dir=Path("/tmp/cache"),
            default_system_message="Be concise.",
            default_max_tokens=50,
            default_temperature=0.5,
            enable_introspection=False,
            introspection_layers=[0, 1, 2],
        )
        assert config.dtype == DType.FLOAT16
        assert config.cache_dir == Path("/tmp/cache")
        assert config.default_system_message == "Be concise."
        assert config.default_max_tokens == 50
        assert config.default_temperature == 0.5
        assert config.enable_introspection is False
        assert config.introspection_layers == [0, 1, 2]

    def test_no_system_message(self):
        """Test with no system message."""
        config = UnifiedPipelineConfig(default_system_message=None)
        assert config.default_system_message is None


class TestUnifiedPipelineState:
    """Tests for UnifiedPipelineState model."""

    def test_create_state(self):
        """Test creating pipeline state."""
        state = UnifiedPipelineState(
            model_id="org/model",
            model_path=Path("/tmp/model"),
            family_type=ModelFamilyType.LLAMA,
            tensor_count=100,
        )
        assert state.model_id == "org/model"
        assert state.model_path == Path("/tmp/model")
        assert state.family_type == ModelFamilyType.LLAMA
        assert state.tensor_count == 100
        assert state.is_loaded is False

    def test_is_loaded(self):
        """Test is_loaded flag."""
        state = UnifiedPipelineState(
            model_id="org/model",
            model_path=Path("/tmp/model"),
            family_type=ModelFamilyType.GPT2,
            tensor_count=100,
            is_loaded=True,
        )
        assert state.is_loaded is True
        assert state.family_type == ModelFamilyType.GPT2


class TestIntrospectionResult:
    """Tests for IntrospectionResult model."""

    def test_empty_result(self):
        """Test empty introspection result."""
        result = IntrospectionResult()
        assert result.hidden_states is None
        assert result.attention_patterns is None
        assert result.pre_head_activations is None

    def test_with_data(self):
        """Test introspection result with data."""
        hidden = [mx.zeros((1, 5, 64)) for _ in range(3)]
        result = IntrospectionResult(
            hidden_states=hidden,
            pre_head_activations=mx.zeros((1, 5, 64)),
        )
        assert result.hidden_states is not None
        assert len(result.hidden_states) == 3


class MockConfig(BaseModel):
    """Mock model config for testing."""

    vocab_size: int = 1000
    hidden_size: int = 64
    num_hidden_layers: int = 4
    eos_token_id: int | list[int] | None = 50256

    @classmethod
    def from_hf_config(cls, hf_config: dict):
        """Create from HF config dict."""
        return cls(
            vocab_size=hf_config.get("vocab_size", 1000),
            hidden_size=hf_config.get("hidden_size", 64),
            num_hidden_layers=hf_config.get("num_hidden_layers", 4),
        )


class MockModel:
    """Mock model for testing."""

    def __init__(self, config=None):
        self.config = config
        self._params = {"weight": mx.zeros((10, 10))}

    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=None,
        stop_tokens=None,
    ):
        _ = input_ids.shape[0]  # batch_size unused but validates shape
        _ = input_ids.shape[1]  # input_length unused but validates shape
        new_tokens = mx.array([[10, 11, 12, 13, 14]])
        return mx.concatenate([input_ids, new_tokens[:, :max_new_tokens]], axis=1)

    def update(self, weights):
        pass

    def parameters(self):
        return self._params

    @staticmethod
    def sanitize(weights):
        """Mock sanitize method."""
        return weights


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.eos_token_id = 50256
        self.pad_token = "<pad>"
        self.chat_template = "template"

    def __len__(self):
        return 32000

    def encode(self, text, return_tensors=None):
        import numpy as np

        return np.array([[1, 2, 3, 4, 5]])

    def decode(self, tokens, skip_special_tokens=False):
        return f"decoded_{len(tokens)}_tokens"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"<formatted>{len(messages)} messages</formatted>"


class TestUnifiedPipeline:
    """Tests for UnifiedPipeline class."""

    @pytest.fixture
    def mock_family_info(self):
        """Create mock family info."""
        return FamilyInfo(
            family_type=ModelFamilyType.LLAMA,
            config_class=MockConfig,
            model_class=MockModel,
            model_types=["llama"],
            architectures=["LlamaForCausalLM"],
        )

    def test_create_pipeline(self, mock_family_info):
        """Test creating pipeline directly."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        pipeline = UnifiedPipeline(
            model=model,
            tokenizer=tokenizer,
            model_config=config,
            family_info=mock_family_info,
        )

        assert pipeline.model is model
        assert pipeline.tokenizer is tokenizer
        assert pipeline.config is config
        assert pipeline.family is mock_family_info
        assert pipeline.family_type == ModelFamilyType.LLAMA

    def test_pipeline_with_config(self, mock_family_info):
        """Test pipeline with custom config."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        pipeline_config = UnifiedPipelineConfig(
            default_max_tokens=50,
            default_temperature=0.5,
        )

        pipeline = UnifiedPipeline(
            model=model,
            tokenizer=tokenizer,
            model_config=config,
            family_info=mock_family_info,
            pipeline_config=pipeline_config,
        )

        assert pipeline._pipeline_config.default_max_tokens == 50
        assert pipeline._pipeline_config.default_temperature == 0.5

    def test_chat_basic(self, mock_family_info):
        """Test basic chat generation."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        pipeline = UnifiedPipeline(
            model=model,
            tokenizer=tokenizer,
            model_config=config,
            family_info=mock_family_info,
        )

        result = pipeline.chat("Hello!")

        assert isinstance(result, GenerationResult)
        assert result.text.startswith("decoded_")

    def test_chat_with_system(self, mock_family_info):
        """Test chat with custom system message."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        pipeline = UnifiedPipeline(
            model=model,
            tokenizer=tokenizer,
            model_config=config,
            family_info=mock_family_info,
        )

        result = pipeline.chat("Hello!", system_message="Be brief.")

        assert isinstance(result, GenerationResult)

    def test_chat_with_params(self, mock_family_info):
        """Test chat with custom parameters."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        pipeline = UnifiedPipeline(
            model=model,
            tokenizer=tokenizer,
            model_config=config,
            family_info=mock_family_info,
        )

        result = pipeline.chat(
            "Hello!",
            max_new_tokens=50,
            temperature=0.5,
        )

        assert isinstance(result, GenerationResult)

    def test_chat_with_history(self, mock_family_info):
        """Test chat with history."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        pipeline = UnifiedPipeline(
            model=model,
            tokenizer=tokenizer,
            model_config=config,
            family_info=mock_family_info,
        )

        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi!")
        history.add_user("How are you?")

        result = pipeline.chat_with_history(history)

        assert isinstance(result, GenerationResult)

    def test_generate_raw(self, mock_family_info):
        """Test raw generation without chat formatting."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        pipeline = UnifiedPipeline(
            model=model,
            tokenizer=tokenizer,
            model_config=config,
            family_info=mock_family_info,
        )

        result = pipeline.generate("Raw prompt text")

        assert isinstance(result, GenerationResult)

    def test_generate_with_config(self, mock_family_info):
        """Test generation with full config."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        pipeline = UnifiedPipeline(
            model=model,
            tokenizer=tokenizer,
            model_config=config,
            family_info=mock_family_info,
        )

        gen_config = GenerationConfig(
            max_new_tokens=20,
            temperature=0.3,
        )
        result = pipeline.generate("Test", config=gen_config)

        assert isinstance(result, GenerationResult)

    def test_list_supported_families(self, mock_family_info):
        """Test listing supported families."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        pipeline = UnifiedPipeline(
            model=model,
            tokenizer=tokenizer,
            model_config=config,
            family_info=mock_family_info,
        )

        families = pipeline.list_supported_families()

        assert isinstance(families, list)
        assert len(families) > 0
        assert "llama" in families


class TestUnifiedPipelineFromPretrained:
    """Tests for UnifiedPipeline.from_pretrained method."""

    @patch("chuk_lazarus.inference.unified.HFLoader")
    @patch("chuk_lazarus.inference.unified.detect_model_family")
    @patch("chuk_lazarus.inference.unified.get_family_info")
    def test_from_pretrained(self, mock_get_family, mock_detect, mock_loader, tmp_path):
        """Test loading from pretrained."""
        # Setup config file
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump({"model_type": "llama", "vocab_size": 1000}, f)

        # Setup mocks
        mock_loader.download.return_value = MagicMock(model_path=tmp_path)
        mock_loader.load_tokenizer.return_value = MockTokenizer()
        mock_loader.load_weights.return_value = MagicMock(
            tensor_count=100,
            weights={"weight": mx.zeros((10, 10))},
            layer_count=4,
        )
        mock_loader.build_nested_weights.return_value = {"model": {}}

        mock_detect.return_value = ModelFamilyType.LLAMA
        mock_get_family.return_value = FamilyInfo(
            family_type=ModelFamilyType.LLAMA,
            config_class=MockConfig,
            model_class=MockModel,
            model_types=["llama"],
            architectures=["LlamaForCausalLM"],
        )

        pipeline = UnifiedPipeline.from_pretrained("org/model", verbose=False)

        assert isinstance(pipeline, UnifiedPipeline)
        assert pipeline.family_type == ModelFamilyType.LLAMA
        mock_loader.download.assert_called_once()

    @patch("chuk_lazarus.inference.unified.HFLoader")
    @patch("chuk_lazarus.inference.unified.detect_model_family")
    @patch("chuk_lazarus.inference.unified.get_family_info")
    def test_from_pretrained_with_config(self, mock_get_family, mock_detect, mock_loader, tmp_path):
        """Test loading with custom pipeline config."""
        # Setup config file
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump({"model_type": "llama", "vocab_size": 1000}, f)

        mock_loader.download.return_value = MagicMock(model_path=tmp_path)
        mock_loader.load_tokenizer.return_value = MockTokenizer()
        mock_loader.load_weights.return_value = MagicMock(
            tensor_count=100,
            weights={"model.weight": mx.zeros((10, 10))},
            layer_count=4,
        )
        mock_loader.build_nested_weights.return_value = {"model": {"weight": mx.zeros((10, 10))}}

        mock_detect.return_value = ModelFamilyType.LLAMA
        mock_get_family.return_value = FamilyInfo(
            family_type=ModelFamilyType.LLAMA,
            config_class=MockConfig,
            model_class=MockModel,
            model_types=["llama"],
            architectures=["LlamaForCausalLM"],
        )

        pipeline_config = UnifiedPipelineConfig(
            dtype=DType.FLOAT16,
            default_max_tokens=50,
        )

        pipeline = UnifiedPipeline.from_pretrained(
            "org/model",
            pipeline_config=pipeline_config,
            verbose=False,
        )

        assert pipeline._pipeline_config.dtype == DType.FLOAT16
        assert pipeline._pipeline_config.default_max_tokens == 50

    @patch("chuk_lazarus.inference.unified.HFLoader")
    @patch("chuk_lazarus.inference.unified.detect_model_family")
    def test_from_pretrained_unknown_family(self, mock_detect, mock_loader, tmp_path):
        """Test error when family cannot be detected."""
        # Setup config file
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump({"model_type": "unknown_model", "vocab_size": 1000}, f)

        mock_loader.download.return_value = MagicMock(model_path=tmp_path)
        mock_detect.return_value = None

        with pytest.raises(ValueError, match="Unable to detect model family"):
            UnifiedPipeline.from_pretrained("org/model", verbose=False)


class TestUnifiedPipelineFromPretrainedAsync:
    """Tests for UnifiedPipeline.from_pretrained_async method."""

    @pytest.mark.asyncio
    @patch("chuk_lazarus.inference.unified.HFLoader")
    @patch("chuk_lazarus.inference.unified.detect_model_family")
    @patch("chuk_lazarus.inference.unified.get_family_info")
    async def test_from_pretrained_async(self, mock_get_family, mock_detect, mock_loader, tmp_path):
        """Test async loading."""
        # Setup config file
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump({"model_type": "llama", "vocab_size": 1000}, f)

        mock_loader.download.return_value = MagicMock(model_path=tmp_path)
        mock_loader.load_tokenizer.return_value = MockTokenizer()
        mock_loader.load_weights.return_value = MagicMock(
            tensor_count=100,
            weights={"model.weight": mx.zeros((10, 10))},
            layer_count=4,
        )
        mock_loader.build_nested_weights.return_value = {"model": {"weight": mx.zeros((10, 10))}}

        mock_detect.return_value = ModelFamilyType.LLAMA
        mock_get_family.return_value = FamilyInfo(
            family_type=ModelFamilyType.LLAMA,
            config_class=MockConfig,
            model_class=MockModel,
            model_types=["llama"],
            architectures=["LlamaForCausalLM"],
        )

        pipeline = await UnifiedPipeline.from_pretrained_async("org/model", verbose=False)

        assert isinstance(pipeline, UnifiedPipeline)
        assert pipeline.family_type == ModelFamilyType.LLAMA
