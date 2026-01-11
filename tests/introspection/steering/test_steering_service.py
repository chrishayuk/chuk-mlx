"""Tests for steering service."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from chuk_lazarus.introspection.steering.service import (
    DirectionExtractionResult,
    SteeringComparisonResult,
    SteeringGenerationResult,
    SteeringService,
    SteeringServiceConfig,
)


class TestSteeringServiceConfig:
    """Tests for SteeringServiceConfig."""

    def test_create_config(self):
        """Test creating steering service config."""
        config = SteeringServiceConfig(
            model="/path/to/model",
            layer=10,
            coefficient=0.5,
            max_tokens=50,
            temperature=0.7,
        )

        assert config.model == "/path/to/model"
        assert config.layer == 10
        assert config.coefficient == 0.5
        assert config.max_tokens == 50
        assert config.temperature == 0.7

    def test_default_values(self):
        """Test default values."""
        config = SteeringServiceConfig(model="/path/to/model")

        assert config.layer is None
        assert config.coefficient == 1.0
        assert config.max_tokens == 100
        assert config.temperature == 0.0

    def test_frozen_config(self):
        """Test that config is frozen (immutable)."""
        from pydantic import ValidationError

        config = SteeringServiceConfig(model="/path/to/model")

        with pytest.raises(ValidationError):
            config.model = "new_model"


class TestDirectionExtractionResult:
    """Tests for DirectionExtractionResult."""

    def test_create_result(self):
        """Test creating direction extraction result."""
        direction = np.random.randn(128).astype(np.float32)

        result = DirectionExtractionResult(
            direction=direction,
            layer=5,
            norm=1.5,
            cosine_similarity=0.3,
            separation=0.7,
            positive_prompt="Be helpful",
            negative_prompt="Be harmful",
        )

        assert result.layer == 5
        assert result.norm == 1.5
        assert result.cosine_similarity == 0.3
        assert result.separation == 0.7
        assert result.positive_prompt == "Be helpful"
        assert result.negative_prompt == "Be harmful"


class TestSteeringGenerationResult:
    """Tests for SteeringGenerationResult."""

    def test_create_result(self):
        """Test creating steering generation result."""
        result = SteeringGenerationResult(
            prompt="Hello",
            output="Hi there!",
            layer=10,
            coefficient=1.5,
        )

        assert result.prompt == "Hello"
        assert result.output == "Hi there!"
        assert result.layer == 10
        assert result.coefficient == 1.5


class TestSteeringComparisonResult:
    """Tests for SteeringComparisonResult."""

    def test_create_result(self):
        """Test creating steering comparison result."""
        result = SteeringComparisonResult(
            prompt="Hello",
            results={0.0: "Baseline", 1.0: "Steered", -1.0: "Reverse"},
        )

        assert result.prompt == "Hello"
        assert result.results[0.0] == "Baseline"
        assert result.results[1.0] == "Steered"
        assert result.results[-1.0] == "Reverse"


class TestSteeringServiceSaveDirection:
    """Tests for SteeringService.save_direction."""

    def test_save_direction(self):
        """Test saving direction to file."""
        direction = np.random.randn(128).astype(np.float32)
        result = DirectionExtractionResult(
            direction=direction,
            layer=5,
            norm=1.5,
            cosine_similarity=0.3,
            separation=0.7,
            positive_prompt="positive",
            negative_prompt="negative",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "direction.npz"

            SteeringService.save_direction(result, path, "model-id")

            assert path.exists()

            # Verify contents
            loaded = np.load(path, allow_pickle=True)
            assert "direction" in loaded
            assert "layer" in loaded
            assert "positive_prompt" in loaded
            assert "negative_prompt" in loaded


class TestSteeringServiceLoadDirection:
    """Tests for SteeringService.load_direction."""

    def test_load_direction_npz(self):
        """Test loading direction from npz file."""
        direction = np.random.randn(128).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "direction.npz"

            np.savez(
                path,
                direction=direction,
                layer=10,
                positive_prompt="pos",
                negative_prompt="neg",
                norm=1.5,
                cosine_similarity=0.3,
            )

            loaded_direction, layer, metadata = SteeringService.load_direction(path)

            assert np.allclose(loaded_direction, direction)
            assert layer == 10
            assert metadata["positive_prompt"] == "pos"
            assert metadata["negative_prompt"] == "neg"
            assert metadata["norm"] == 1.5
            assert metadata["cosine_similarity"] == 0.3

    def test_load_direction_npz_minimal(self):
        """Test loading direction from npz with minimal fields."""
        direction = np.random.randn(128).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "direction.npz"

            np.savez(path, direction=direction)

            loaded_direction, layer, metadata = SteeringService.load_direction(path)

            assert np.allclose(loaded_direction, direction)
            assert layer is None
            assert metadata == {}

    def test_load_direction_json(self):
        """Test loading direction from json file."""
        direction = np.random.randn(128).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "direction.json"

            with open(path, "w") as f:
                json.dump(
                    {
                        "direction": direction.tolist(),
                        "layer": 5,
                        "extra_field": "test",
                    },
                    f,
                )

            loaded_direction, layer, metadata = SteeringService.load_direction(path)

            assert np.allclose(loaded_direction, direction)
            assert layer == 5
            assert metadata["extra_field"] == "test"

    def test_load_direction_unsupported_format(self):
        """Test loading direction from unsupported format raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "direction.txt"
            path.write_text("test")

            with pytest.raises(ValueError, match="Unsupported direction format"):
                SteeringService.load_direction(path)


class TestSteeringServiceCreateNeuronDirection:
    """Tests for SteeringService.create_neuron_direction."""

    def test_create_neuron_direction(self):
        """Test creating one-hot neuron direction."""
        direction = SteeringService.create_neuron_direction(hidden_size=128, neuron_idx=50)

        assert direction.shape == (128,)
        assert direction.dtype == np.float32
        assert direction[50] == 1.0
        assert np.sum(direction) == 1.0  # Only one non-zero element

    def test_create_neuron_direction_first(self):
        """Test creating direction for first neuron."""
        direction = SteeringService.create_neuron_direction(hidden_size=64, neuron_idx=0)

        assert direction[0] == 1.0
        assert np.sum(direction) == 1.0

    def test_create_neuron_direction_last(self):
        """Test creating direction for last neuron."""
        direction = SteeringService.create_neuron_direction(hidden_size=64, neuron_idx=63)

        assert direction[63] == 1.0
        assert np.sum(direction) == 1.0


class TestSteeringServiceConfigAlias:
    """Tests for Config class alias."""

    def test_config_alias(self):
        """Test that SteeringService.Config is SteeringServiceConfig."""
        assert SteeringService.Config == SteeringServiceConfig
