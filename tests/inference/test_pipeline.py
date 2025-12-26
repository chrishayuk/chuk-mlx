"""Tests for inference/pipeline.py module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest
from pydantic import BaseModel

from chuk_lazarus.inference.chat import ChatHistory
from chuk_lazarus.inference.generation import GenerationConfig, GenerationResult
from chuk_lazarus.inference.loader import DType
from chuk_lazarus.inference.pipeline import (
    InferencePipeline,
    PipelineConfig,
    PipelineState,
    _load_config,
)


class TestPipelineConfig:
    """Tests for PipelineConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PipelineConfig()
        assert config.dtype == DType.BFLOAT16
        assert config.cache_dir is None
        assert config.default_system_message == "You are a helpful assistant."
        assert config.default_max_tokens == 100
        assert config.default_temperature == 0.7

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            dtype=DType.FLOAT16,
            cache_dir=Path("/tmp/cache"),
            default_system_message="Be concise.",
            default_max_tokens=50,
            default_temperature=0.5,
        )
        assert config.dtype == DType.FLOAT16
        assert config.cache_dir == Path("/tmp/cache")
        assert config.default_system_message == "Be concise."
        assert config.default_max_tokens == 50
        assert config.default_temperature == 0.5

    def test_no_system_message(self):
        """Test with no system message."""
        config = PipelineConfig(default_system_message=None)
        assert config.default_system_message is None


class TestPipelineState:
    """Tests for PipelineState model."""

    def test_create_state(self):
        """Test creating pipeline state."""
        state = PipelineState(
            model_id="org/model",
            model_path=Path("/tmp/model"),
            tensor_count=100,
        )
        assert state.model_id == "org/model"
        assert state.model_path == Path("/tmp/model")
        assert state.tensor_count == 100
        assert state.is_loaded is False

    def test_is_loaded(self):
        """Test is_loaded flag."""
        state = PipelineState(
            model_id="org/model",
            model_path=Path("/tmp/model"),
            tensor_count=100,
            is_loaded=True,
        )
        assert state.is_loaded is True


class MockConfig(BaseModel):
    """Mock model config for testing."""

    vocab_size: int = 1000
    hidden_size: int = 64
    num_hidden_layers: int = 4
    eos_token_id: int | list[int] | None = 50256


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


class TestLoadConfig:
    """Tests for _load_config function."""

    def test_load_basic_config(self, tmp_path):
        """Test loading basic config."""
        config_data = {
            "vocab_size": 32000,
            "hidden_size": 128,
            "num_hidden_layers": 8,
        }
        with open(tmp_path / "config.json", "w") as f:
            json.dump(config_data, f)

        config = _load_config(tmp_path, MockConfig)

        assert config.vocab_size == 32000
        assert config.hidden_size == 128

    def test_load_config_with_list_token_ids(self, tmp_path):
        """Test loading config with list token IDs."""
        config_data = {
            "vocab_size": 32000,
            "hidden_size": 128,
            "num_hidden_layers": 8,
            "eos_token_id": [50256, 50257],
            "bos_token_id": [1],
            "pad_token_id": [],
        }
        with open(tmp_path / "config.json", "w") as f:
            json.dump(config_data, f)

        config = _load_config(tmp_path, MockConfig)

        assert config.eos_token_id == 50256  # First element

    def test_load_config_empty_list_token_id(self, tmp_path):
        """Test loading config with empty list token ID."""
        config_data = {
            "vocab_size": 32000,
            "hidden_size": 128,
            "num_hidden_layers": 8,
            "eos_token_id": [],
        }
        with open(tmp_path / "config.json", "w") as f:
            json.dump(config_data, f)

        config = _load_config(tmp_path, MockConfig)

        assert config.eos_token_id is None


class TestInferencePipeline:
    """Tests for InferencePipeline class."""

    def test_create_pipeline(self):
        """Test creating pipeline directly."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        pipeline = InferencePipeline(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )

        assert pipeline.model is model
        assert pipeline.tokenizer is tokenizer
        assert pipeline.config is config

    def test_pipeline_with_config(self):
        """Test pipeline with custom config."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        pipeline_config = PipelineConfig(
            default_max_tokens=50,
            default_temperature=0.5,
        )

        pipeline = InferencePipeline(
            model=model,
            tokenizer=tokenizer,
            config=config,
            pipeline_config=pipeline_config,
        )

        assert pipeline._pipeline_config.default_max_tokens == 50

    def test_chat_basic(self):
        """Test basic chat generation."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        pipeline = InferencePipeline(model=model, tokenizer=tokenizer, config=config)

        result = pipeline.chat("Hello!")

        assert isinstance(result, GenerationResult)
        assert result.text.startswith("decoded_")

    def test_chat_with_system(self):
        """Test chat with custom system message."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        pipeline = InferencePipeline(model=model, tokenizer=tokenizer, config=config)

        result = pipeline.chat("Hello!", system_message="Be brief.")

        assert isinstance(result, GenerationResult)

    def test_chat_with_params(self):
        """Test chat with custom parameters."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        pipeline = InferencePipeline(model=model, tokenizer=tokenizer, config=config)

        result = pipeline.chat(
            "Hello!",
            max_new_tokens=50,
            temperature=0.5,
        )

        assert isinstance(result, GenerationResult)

    def test_chat_with_history(self):
        """Test chat with history."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        pipeline = InferencePipeline(model=model, tokenizer=tokenizer, config=config)

        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi!")
        history.add_user("How are you?")

        result = pipeline.chat_with_history(history)

        assert isinstance(result, GenerationResult)

    def test_generate_raw(self):
        """Test raw generation without chat formatting."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        pipeline = InferencePipeline(model=model, tokenizer=tokenizer, config=config)

        result = pipeline.generate("Raw prompt text")

        assert isinstance(result, GenerationResult)

    def test_generate_with_config(self):
        """Test generation with full config."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        pipeline = InferencePipeline(model=model, tokenizer=tokenizer, config=config)

        gen_config = GenerationConfig(
            max_new_tokens=20,
            temperature=0.3,
        )
        result = pipeline.generate("Test", config=gen_config)

        assert isinstance(result, GenerationResult)


class TestInferencePipelineFromPretrained:
    """Tests for InferencePipeline.from_pretrained method."""

    @patch("chuk_lazarus.inference.pipeline.HFLoader")
    @patch("chuk_lazarus.inference.pipeline._load_config")
    def test_from_pretrained(self, mock_load_config, mock_loader):
        """Test loading from pretrained."""
        # Setup mocks
        mock_loader.download.return_value = MagicMock(model_path=Path("/tmp/model"))
        mock_loader.load_tokenizer.return_value = MockTokenizer()
        mock_loader.load_weights.return_value = MagicMock(
            tensor_count=100,
            weights={"weight": mx.zeros((10, 10))},
            layer_count=4,
        )
        mock_loader.build_nested_weights.return_value = {"model": {}}
        mock_load_config.return_value = MockConfig()

        pipeline = InferencePipeline.from_pretrained(
            "org/model",
            MockModel,
            MockConfig,
        )

        assert isinstance(pipeline, InferencePipeline)
        mock_loader.download.assert_called_once()

    @patch("chuk_lazarus.inference.pipeline.HFLoader")
    @patch("chuk_lazarus.inference.pipeline._load_config")
    def test_from_pretrained_with_config(self, mock_load_config, mock_loader):
        """Test loading with custom pipeline config."""
        mock_loader.download.return_value = MagicMock(model_path=Path("/tmp/model"))
        mock_loader.load_tokenizer.return_value = MockTokenizer()
        mock_loader.load_weights.return_value = MagicMock(
            tensor_count=100,
            weights={},
            layer_count=4,
        )
        mock_loader.build_nested_weights.return_value = {}
        mock_load_config.return_value = MockConfig()

        pipeline_config = PipelineConfig(
            dtype=DType.FLOAT16,
            default_max_tokens=50,
        )

        pipeline = InferencePipeline.from_pretrained(
            "org/model",
            MockModel,
            MockConfig,
            pipeline_config=pipeline_config,
        )

        assert pipeline._pipeline_config.dtype == DType.FLOAT16


class TestInferencePipelineFromPretrainedAsync:
    """Tests for InferencePipeline.from_pretrained_async method."""

    @pytest.mark.asyncio
    @patch("chuk_lazarus.inference.pipeline.HFLoader")
    @patch("chuk_lazarus.inference.pipeline._load_config")
    async def test_from_pretrained_async(self, mock_load_config, mock_loader):
        """Test async loading."""
        mock_loader.download.return_value = MagicMock(model_path=Path("/tmp/model"))
        mock_loader.load_tokenizer.return_value = MockTokenizer()
        mock_loader.load_weights.return_value = MagicMock(
            tensor_count=100,
            weights={},
            layer_count=4,
        )
        mock_loader.build_nested_weights.return_value = {}
        mock_load_config.return_value = MockConfig()

        pipeline = await InferencePipeline.from_pretrained_async(
            "org/model",
            MockModel,
            MockConfig,
        )

        assert isinstance(pipeline, InferencePipeline)


class TestCausalLMProtocol:
    """Tests for CausalLMProtocol."""

    def test_mock_model_implements_protocol(self):
        """Test that MockModel implements the protocol."""
        model = MockModel()

        # Check protocol methods exist
        assert hasattr(model, "generate")
        assert hasattr(model, "update")
        assert hasattr(model, "parameters")
        assert callable(model.generate)
        assert callable(model.update)
        assert callable(model.parameters)
