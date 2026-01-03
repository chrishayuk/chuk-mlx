"""Tests for ablation loader module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from chuk_lazarus.introspection.ablation.loader import load_model_for_ablation


class TestLoadModelForAblation:
    """Tests for load_model_for_ablation function."""

    @patch("chuk_lazarus.inference.loader.HFLoader")
    @patch("chuk_lazarus.models_v2.families.registry.detect_model_family")
    @patch("chuk_lazarus.models_v2.families.registry.get_family_info")
    @patch("chuk_lazarus.introspection.ablation.adapter.ModelAdapter")
    def test_load_model_success(
        self, mock_adapter, mock_get_family, mock_detect, mock_loader
    ):
        """Test successful model loading."""
        # Setup mock model path with config
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            config_data = {"model_type": "gemma", "hidden_size": 64}
            with open(model_path / "config.json", "w") as f:
                json.dump(config_data, f)

            # Setup mocks
            mock_result = Mock()
            mock_result.model_path = model_path
            mock_loader.download.return_value = mock_result

            mock_detect.return_value = "gemma"

            mock_config_class = Mock()
            mock_model_class = Mock()
            mock_family_info = Mock()
            mock_family_info.config_class = mock_config_class
            mock_family_info.model_class = mock_model_class
            mock_get_family.return_value = mock_family_info

            mock_config = Mock()
            mock_config_class.from_hf_config.return_value = mock_config

            mock_model = Mock()
            mock_model_class.return_value = mock_model

            mock_tokenizer = Mock()
            mock_loader.load_tokenizer.return_value = mock_tokenizer

            mock_adapter_instance = Mock()
            mock_adapter.return_value = mock_adapter_instance

            # Call function
            result = load_model_for_ablation("test-model")

            # Verify
            assert result is mock_adapter_instance
            mock_loader.download.assert_called_once_with("test-model")
            mock_detect.assert_called_once()
            mock_get_family.assert_called_once_with("gemma")
            mock_adapter.assert_called_once_with(mock_model, mock_tokenizer, mock_config)

    @patch("chuk_lazarus.inference.loader.HFLoader")
    @patch("chuk_lazarus.models_v2.families.registry.detect_model_family")
    def test_load_unsupported_family_raises(self, mock_detect, mock_loader):
        """Test that unsupported model family raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            config_data = {"model_type": "unknown_model"}
            with open(model_path / "config.json", "w") as f:
                json.dump(config_data, f)

            mock_result = Mock()
            mock_result.model_path = model_path
            mock_loader.download.return_value = mock_result

            # Return None for unsupported family
            mock_detect.return_value = None

            with pytest.raises(ValueError, match="Unsupported model family"):
                load_model_for_ablation("unsupported-model")

    @patch("chuk_lazarus.inference.loader.HFLoader")
    @patch("chuk_lazarus.models_v2.families.registry.detect_model_family")
    @patch("chuk_lazarus.models_v2.families.registry.get_family_info")
    @patch("chuk_lazarus.introspection.ablation.adapter.ModelAdapter")
    def test_load_with_local_path(
        self, mock_adapter, mock_get_family, mock_detect, mock_loader
    ):
        """Test loading from local path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            config_data = {"model_type": "llama", "hidden_size": 128}
            with open(model_path / "config.json", "w") as f:
                json.dump(config_data, f)

            mock_result = Mock()
            mock_result.model_path = model_path
            mock_loader.download.return_value = mock_result

            mock_detect.return_value = "llama"

            mock_family_info = Mock()
            mock_family_info.config_class = Mock()
            mock_family_info.model_class = Mock()
            mock_get_family.return_value = mock_family_info

            mock_family_info.config_class.from_hf_config.return_value = Mock()
            mock_family_info.model_class.return_value = Mock()
            mock_loader.load_tokenizer.return_value = Mock()
            mock_adapter.return_value = Mock()

            result = load_model_for_ablation(str(model_path))

            assert result is not None
            mock_loader.download.assert_called_once_with(str(model_path))
