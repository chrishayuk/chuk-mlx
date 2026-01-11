"""Tests for training config loader."""

import tempfile
from pathlib import Path

import pytest
import yaml

from chuk_lazarus.utils.training_config_loader import load_training_config


class TestLoadTrainingConfig:
    """Tests for load_training_config function."""

    def test_load_basic_config(self):
        """Test loading a basic config."""
        config_data = {
            "model": "test-model",
            "epochs": 3,
            "batch_size": 8,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()

            result = load_training_config(f.name)

        assert result == config_data
        Path(f.name).unlink()

    def test_load_nested_config(self):
        """Test loading a config with nested structure."""
        config_data = {
            "model": {
                "name": "test-model",
                "layers": 12,
            },
            "training": {
                "epochs": 5,
                "learning_rate": 1e-4,
            },
            "optimizer": {
                "name": "AdamW",
                "betas": [0.9, 0.999],
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()

            result = load_training_config(f.name)

        assert result == config_data
        assert result["model"]["name"] == "test-model"
        assert result["training"]["learning_rate"] == 1e-4
        Path(f.name).unlink()

    def test_load_empty_config(self):
        """Test loading an empty config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()

            result = load_training_config(f.name)

        assert result is None
        Path(f.name).unlink()

    def test_load_config_file_not_found(self):
        """Test loading a config that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_training_config("/nonexistent/path/config.yaml")
