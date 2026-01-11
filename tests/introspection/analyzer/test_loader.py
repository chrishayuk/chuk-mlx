"""Tests for analyzer loader module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from chuk_lazarus.introspection.analyzer.loader import (
    _is_quantized_model,
    _load_model_sync,
    get_model_hidden_size,
    get_model_num_layers,
    get_model_vocab_size,
)


class TestIsQuantizedModel:
    """Tests for _is_quantized_model function."""

    def test_quantization_config_present(self):
        """Test detection via quantization_config."""
        config_data = {"quantization_config": {"bits": 4}}
        assert _is_quantized_model(config_data, "model-id") is True

    def test_4bit_in_model_id(self):
        """Test detection via -4bit in model ID."""
        config_data = {}
        assert _is_quantized_model(config_data, "my-model-4bit") is True
        assert _is_quantized_model(config_data, "model-4Bit") is True

    def test_8bit_in_model_id(self):
        """Test detection via -8bit in model ID."""
        config_data = {}
        assert _is_quantized_model(config_data, "my-model-8bit") is True

    def test_bnb_in_model_id(self):
        """Test detection via -bnb- pattern."""
        config_data = {}
        assert _is_quantized_model(config_data, "model-bnb-4bit") is True

    def test_awq_in_model_id(self):
        """Test detection via -awq pattern."""
        config_data = {}
        assert _is_quantized_model(config_data, "model-awq") is True

    def test_not_quantized(self):
        """Test regular model is not detected as quantized."""
        config_data = {}
        assert _is_quantized_model(config_data, "regular-model") is False

    def test_case_insensitive(self):
        """Test case insensitive matching."""
        config_data = {}
        assert _is_quantized_model(config_data, "Model-4BIT") is True
        assert _is_quantized_model(config_data, "MODEL-AWQ") is True


class TestLoadModelSync:
    """Tests for _load_model_sync function."""

    @patch("chuk_lazarus.inference.loader.HFLoader")
    @patch("chuk_lazarus.models_v2.families.registry.detect_model_family")
    @patch("chuk_lazarus.models_v2.families.registry.get_family_info")
    def test_load_model_success(self, mock_get_family, mock_detect, mock_loader):
        """Test successful model loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            config_data = {"model_type": "llama", "hidden_size": 64}
            with open(model_path / "config.json", "w") as f:
                json.dump(config_data, f)

            mock_result = Mock()
            mock_result.model_path = model_path
            mock_loader.download.return_value = mock_result

            mock_detect.return_value = "llama"

            mock_family_info = Mock()
            mock_config_class = Mock()
            mock_model_class = Mock()
            mock_family_info.config_class = mock_config_class
            mock_family_info.model_class = mock_model_class
            mock_get_family.return_value = mock_family_info

            mock_config = Mock()
            mock_config_class.from_hf_config.return_value = mock_config

            mock_model = Mock()
            mock_model_class.return_value = mock_model

            mock_tokenizer = Mock()
            mock_loader.load_tokenizer.return_value = mock_tokenizer

            model, tokenizer, config = _load_model_sync("test-model")

            assert model is mock_model
            assert tokenizer is mock_tokenizer
            assert config is mock_config

    @patch("chuk_lazarus.inference.loader.HFLoader")
    @patch("chuk_lazarus.models_v2.families.registry.detect_model_family")
    def test_load_unsupported_raises(self, mock_detect, mock_loader):
        """Test unsupported model family raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            config_data = {"model_type": "unknown"}
            with open(model_path / "config.json", "w") as f:
                json.dump(config_data, f)

            mock_result = Mock()
            mock_result.model_path = model_path
            mock_loader.download.return_value = mock_result

            mock_detect.return_value = None

            with pytest.raises(ValueError, match="Unsupported model family"):
                _load_model_sync("unknown-model")

    @patch("chuk_lazarus.introspection.analyzer.loader._central_load")
    def test_gemma_embedding_scale(self, mock_central_load):
        """Test Gemma models get embedding scale attached."""
        # Create mock model and config
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_config = Mock()
        mock_config.model_type = "gemma"
        mock_config.hidden_size = 256

        mock_central_load.return_value = (mock_model, mock_tokenizer, mock_config)

        model, _, _ = _load_model_sync("gemma-model")

        # Check embedding scale was set (sqrt(256) = 16)
        assert hasattr(model, "_embedding_scale_for_hooks")
        assert model._embedding_scale_for_hooks == 16.0


class TestGetModelHiddenSize:
    """Tests for get_model_hidden_size function."""

    def test_from_config_hidden_size(self):
        """Test getting hidden size from config.hidden_size."""
        config = Mock()
        config.hidden_size = 4096
        model = Mock()

        result = get_model_hidden_size(model, config)
        assert result == 4096

    def test_from_config_d_model(self):
        """Test getting hidden size from config.d_model."""
        config = Mock(spec=["d_model"])
        config.d_model = 2048
        model = Mock()

        result = get_model_hidden_size(model, config)
        assert result == 2048

    def test_from_model_attribute(self):
        """Test getting hidden size from model.model.hidden_size."""
        config = None
        model = Mock()
        model.model.hidden_size = 1024

        result = get_model_hidden_size(model, config)
        assert result == 1024

    def test_from_model_args(self):
        """Test getting hidden size from model.args.hidden_size."""
        config = Mock(spec=[])  # No hidden_size or d_model
        model = Mock(spec=["args"])
        model.args.hidden_size = 512

        result = get_model_hidden_size(model, config)
        assert result == 512

    def test_fallback(self):
        """Test fallback to default value."""
        config = Mock(spec=[])
        model = Mock(spec=[])

        result = get_model_hidden_size(model, config)
        assert result == 4096  # Default fallback


class TestGetModelNumLayers:
    """Tests for get_model_num_layers function."""

    def test_from_config_num_hidden_layers(self):
        """Test getting num layers from config.num_hidden_layers."""
        config = Mock()
        config.num_hidden_layers = 32
        model = Mock()

        result = get_model_num_layers(model, config)
        assert result == 32

    def test_from_config_num_layers(self):
        """Test getting num layers from config.num_layers."""
        config = Mock(spec=["num_layers"])
        config.num_layers = 24
        model = Mock()

        result = get_model_num_layers(model, config)
        assert result == 24

    def test_from_model_layers(self):
        """Test getting num layers from model.model.layers."""
        config = None
        model = Mock()
        model.model.layers = [Mock() for _ in range(16)]

        result = get_model_num_layers(model, config)
        assert result == 16

    def test_from_direct_layers(self):
        """Test getting num layers from model.layers."""
        config = Mock(spec=[])
        model = Mock(spec=["layers"])
        model.layers = [Mock() for _ in range(12)]

        result = get_model_num_layers(model, config)
        assert result == 12

    def test_from_transformer_h(self):
        """Test getting num layers from model.transformer.h (GPT-2 style)."""
        config = Mock(spec=[])
        model = Mock(spec=["transformer"])
        model.transformer.h = [Mock() for _ in range(8)]

        result = get_model_num_layers(model, config)
        assert result == 8

    def test_fallback(self):
        """Test fallback to default value."""
        config = Mock(spec=[])
        model = Mock(spec=[])

        result = get_model_num_layers(model, config)
        assert result == 32  # Default fallback


class TestGetModelVocabSize:
    """Tests for get_model_vocab_size function."""

    def test_from_config(self):
        """Test getting vocab size from config.vocab_size."""
        config = Mock()
        config.vocab_size = 50000
        model = Mock()
        tokenizer = Mock()

        result = get_model_vocab_size(model, tokenizer, config)
        assert result == 50000

    def test_from_tokenizer_vocab_size(self):
        """Test getting vocab size from tokenizer.vocab_size."""
        config = Mock(spec=[])
        model = Mock()
        tokenizer = Mock()
        tokenizer.vocab_size = 32000

        result = get_model_vocab_size(model, tokenizer, config)
        assert result == 32000

    def test_from_tokenizer_len(self):
        """Test getting vocab size from len(tokenizer)."""
        config = Mock(spec=[])
        model = Mock()
        tokenizer = Mock(spec=["__len__"])
        tokenizer.__len__ = Mock(return_value=48000)

        result = get_model_vocab_size(model, tokenizer, config)
        assert result == 48000

    def test_from_lm_head(self):
        """Test getting vocab size from model.lm_head.weight."""
        config = Mock(spec=[])
        model = Mock()
        model.lm_head.weight.shape = (64000, 4096)
        tokenizer = Mock(spec=[])

        result = get_model_vocab_size(model, tokenizer, config)
        assert result == 64000

    def test_fallback(self):
        """Test fallback to default value."""
        config = Mock(spec=[])
        model = Mock(spec=[])
        tokenizer = Mock(spec=[])

        result = get_model_vocab_size(model, tokenizer, config)
        assert result == 32000  # Default fallback
