"""
Tests for Granite weight conversion utilities.
"""

import json
import tempfile
from pathlib import Path

import mlx.core as mx

from chuk_lazarus.models_v2.families.granite.convert import (
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
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "vocab_size": 32000,
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            result = load_hf_config(tmpdir)

            assert result["hidden_size"] == 768
            assert result["num_hidden_layers"] == 12
            assert result["vocab_size"] == 32000

    def test_load_config_with_path_object(self):
        """Test loading config with Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_data = {"model_type": "granite"}
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            result = load_hf_config(Path(tmpdir))

            assert result["model_type"] == "granite"


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
            # Create a mock safetensor file
            weights = {
                "model.embed_tokens.weight": mx.random.normal((100, 64)),
            }
            sf_path = Path(tmpdir) / "model.safetensors"
            mx.save_safetensors(str(sf_path), weights)

            result = load_weights(tmpdir)

            assert "model.embed_tokens.weight" in result
            assert result["model.embed_tokens.weight"].shape == (100, 64)

    def test_load_multiple_safetensors(self):
        """Test loading from directory with multiple safetensor files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock safetensor files
            weights1 = {"layer1.weight": mx.random.normal((64, 64))}
            weights2 = {"layer2.weight": mx.random.normal((64, 64))}

            mx.save_safetensors(str(Path(tmpdir) / "model-00001-of-00002.safetensors"), weights1)
            mx.save_safetensors(str(Path(tmpdir) / "model-00002-of-00002.safetensors"), weights2)

            result = load_weights(tmpdir)

            assert "layer1.weight" in result
            assert "layer2.weight" in result
