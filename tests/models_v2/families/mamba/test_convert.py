"""
Tests for Mamba weight conversion utilities.
"""

import json
import tempfile
from pathlib import Path

import mlx.core as mx

from chuk_lazarus.models_v2.families.mamba.convert import (
    load_hf_config,
    load_weights,
)


class TestLoadHfConfig:
    """Tests for load_hf_config function."""

    def test_load_valid_config(self):
        """Test loading a valid config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_data = {
                "hidden_size": 2560,
                "num_hidden_layers": 64,
                "vocab_size": 50280,
                "model_type": "mamba",
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            result = load_hf_config(tmpdir)

            assert result["hidden_size"] == 2560
            assert result["model_type"] == "mamba"

    def test_load_config_with_path_object(self):
        """Test loading config with Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_data = {"model_type": "mamba", "ssm_cfg": {"d_state": 16}}
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            result = load_hf_config(Path(tmpdir))

            assert result["model_type"] == "mamba"
            assert result["ssm_cfg"]["d_state"] == 16


class TestLoadWeights:
    """Tests for load_weights function."""

    def test_load_empty_directory(self):
        """Test loading from directory with no safetensor files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_weights(tmpdir)
            assert result == {}

    def test_load_single_safetensor(self):
        """Test loading from directory with single safetensor file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights = {
                "backbone.embedding.weight": mx.random.normal((100, 64)),
            }
            sf_path = Path(tmpdir) / "model.safetensors"
            mx.save_safetensors(str(sf_path), weights)

            result = load_weights(tmpdir)

            assert "backbone.embedding.weight" in result

    def test_load_multiple_safetensors(self):
        """Test loading from directory with multiple safetensor files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights1 = {"layers.0.mixer.in_proj.weight": mx.random.normal((64, 64))}
            weights2 = {"layers.1.mixer.in_proj.weight": mx.random.normal((64, 64))}

            mx.save_safetensors(str(Path(tmpdir) / "model-00001-of-00002.safetensors"), weights1)
            mx.save_safetensors(str(Path(tmpdir) / "model-00002-of-00002.safetensors"), weights2)

            result = load_weights(tmpdir)

            assert "layers.0.mixer.in_proj.weight" in result
            assert "layers.1.mixer.in_proj.weight" in result
