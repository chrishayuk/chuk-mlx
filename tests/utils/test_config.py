"""Tests for config utilities."""

import tempfile
from pathlib import Path

import pytest
import yaml


class TestConfigLoader:
    """Tests for config loader."""

    def test_load_yaml_config(self):
        """Test loading YAML config."""
        from chuk_lazarus.utils.config import load_config

        config_data = {
            "model": {"name": "test-model", "layers": 12},
            "training": {"epochs": 5, "lr": 1e-4},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()

            config = load_config(f.name)

        assert config["model"]["name"] == "test-model"
        assert config["training"]["epochs"] == 5

        Path(f.name).unlink()

    def test_load_json_config(self):
        """Test loading JSON config."""
        import json

        from chuk_lazarus.utils.config import load_config

        config_data = {"key": "value", "nested": {"inner": 42}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            config = load_config(f.name)

        assert config["key"] == "value"
        assert config["nested"]["inner"] == 42

        Path(f.name).unlink()

    def test_load_config_file_not_found(self):
        """Test loading non-existent config."""
        from chuk_lazarus.utils.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")
