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

    def test_load_unsupported_format(self):
        """Test loading unsupported format raises error."""
        from chuk_lazarus.utils.config import load_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            f.flush()

            with pytest.raises(ValueError, match="Unsupported config format"):
                load_config(f.name)

            Path(f.name).unlink()


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_dict_as_yaml(self):
        """Test saving dict to YAML."""
        from chuk_lazarus.utils.config import save_config

        config_data = {"name": "test", "value": 42}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"

            save_config(config_data, str(path))

            assert path.exists()
            with open(path) as f:
                loaded = yaml.safe_load(f)
            assert loaded == config_data

    def test_save_dict_as_json(self):
        """Test saving dict to JSON."""
        import json

        from chuk_lazarus.utils.config import save_config

        config_data = {"name": "test", "value": 42}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"

            save_config(config_data, str(path))

            assert path.exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == config_data

    def test_save_creates_parent_dirs(self):
        """Test that save_config creates parent directories."""
        from chuk_lazarus.utils.config import save_config

        config_data = {"test": True}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "config.yaml"

            save_config(config_data, str(path))

            assert path.exists()

    def test_save_unsupported_format(self):
        """Test saving to unsupported format raises error."""
        from chuk_lazarus.utils.config import save_config

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.txt"

            with pytest.raises(ValueError, match="Unsupported config format"):
                save_config({}, str(path))


class TestMergeConfigs:
    """Tests for merge_configs function."""

    def test_merge_simple(self):
        """Test simple config merge."""
        from chuk_lazarus.utils.config import merge_configs

        config1 = {"a": 1, "b": 2}
        config2 = {"b": 3, "c": 4}

        result = merge_configs(config1, config2)

        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested(self):
        """Test nested config merge."""
        from chuk_lazarus.utils.config import merge_configs

        config1 = {"a": {"x": 1, "y": 2}, "b": 3}
        config2 = {"a": {"y": 10, "z": 20}}

        result = merge_configs(config1, config2)

        assert result == {"a": {"x": 1, "y": 10, "z": 20}, "b": 3}

    def test_merge_multiple(self):
        """Test merging multiple configs."""
        from chuk_lazarus.utils.config import merge_configs

        config1 = {"a": 1}
        config2 = {"b": 2}
        config3 = {"c": 3}

        result = merge_configs(config1, config2, config3)

        assert result == {"a": 1, "b": 2, "c": 3}


class TestDictToDataclass:
    """Tests for _dict_to_dataclass function."""

    def test_simple_conversion(self):
        """Test simple dict to dataclass conversion."""
        from dataclasses import dataclass

        from chuk_lazarus.utils.config import _dict_to_dataclass

        @dataclass
        class SimpleConfig:
            name: str
            value: int

        data = {"name": "test", "value": 42}

        result = _dict_to_dataclass(data, SimpleConfig)

        assert isinstance(result, SimpleConfig)
        assert result.name == "test"
        assert result.value == 42

    def test_non_dataclass_raises(self):
        """Test that non-dataclass raises error."""
        from chuk_lazarus.utils.config import _dict_to_dataclass

        with pytest.raises(ValueError, match="is not a dataclass"):
            _dict_to_dataclass({}, dict)
