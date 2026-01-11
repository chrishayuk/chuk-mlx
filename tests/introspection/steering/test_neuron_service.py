"""Tests for neuron analysis service."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chuk_lazarus.introspection.steering.neuron_service import (
    DiscoveredNeuron,
    NeuronActivationResult,
    NeuronAnalysisService,
    NeuronAnalysisServiceConfig,
)


class TestNeuronActivationResult:
    """Tests for NeuronActivationResult."""

    def test_basic_result(self):
        """Test basic neuron activation result."""
        result = NeuronActivationResult(
            neuron_idx=42,
            min_val=-1.0,
            max_val=1.0,
            mean_val=0.0,
            std_val=0.5,
        )
        assert result.neuron_idx == 42
        assert result.min_val == -1.0
        assert result.max_val == 1.0
        assert result.mean_val == 0.0
        assert result.std_val == 0.5
        assert result.weight is None
        assert result.separation is None

    def test_result_with_weight(self):
        """Test result with weight."""
        result = NeuronActivationResult(
            neuron_idx=10,
            min_val=-0.5,
            max_val=0.5,
            mean_val=0.1,
            std_val=0.2,
            weight=0.95,
        )
        assert result.weight == 0.95

    def test_result_with_separation(self):
        """Test result with separation score."""
        result = NeuronActivationResult(
            neuron_idx=5,
            min_val=-2.0,
            max_val=2.0,
            mean_val=0.0,
            std_val=1.0,
            separation=3.5,
        )
        assert result.separation == 3.5


class TestDiscoveredNeuron:
    """Tests for DiscoveredNeuron."""

    def test_basic_discovered_neuron(self):
        """Test basic discovered neuron."""
        neuron = DiscoveredNeuron(
            idx=100,
            separation=5.0,
            overall_std=0.8,
            mean_range=2.5,
        )
        assert neuron.idx == 100
        assert neuron.separation == 5.0
        assert neuron.overall_std == 0.8
        assert neuron.mean_range == 2.5
        assert neuron.best_pair is None
        assert neuron.group_means == {}

    def test_discovered_neuron_with_all_fields(self):
        """Test discovered neuron with all fields."""
        neuron = DiscoveredNeuron(
            idx=50,
            separation=8.0,
            best_pair=("positive", "negative"),
            overall_std=1.2,
            mean_range=4.0,
            group_means={"positive": 2.0, "negative": -2.0},
        )
        assert neuron.best_pair == ("positive", "negative")
        assert neuron.group_means["positive"] == 2.0
        assert neuron.group_means["negative"] == -2.0


class TestNeuronAnalysisServiceConfig:
    """Tests for NeuronAnalysisServiceConfig."""

    def test_basic_config(self):
        """Test basic config."""
        config = NeuronAnalysisServiceConfig(
            model="test-model",
            layers=[10, 11, 12],
        )
        assert config.model == "test-model"
        assert config.layers == [10, 11, 12]
        assert config.neurons is None
        assert config.top_k == 10

    def test_config_with_neurons(self):
        """Test config with specific neurons."""
        config = NeuronAnalysisServiceConfig(
            model="test-model",
            layers=[5],
            neurons=[10, 20, 30],
            top_k=5,
        )
        assert config.neurons == [10, 20, 30]
        assert config.top_k == 5


class TestNeuronAnalysisService:
    """Tests for NeuronAnalysisService."""

    def test_load_neurons_from_direction(self):
        """Test loading neurons from direction file."""
        # Create a mock direction file
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            direction = np.array([0.1, -0.5, 0.3, -0.8, 0.2])
            np.savez(
                f.name,
                direction=direction,
                label_positive="happy",
                label_negative="sad",
            )

            neurons, weights, metadata = NeuronAnalysisService.load_neurons_from_direction(
                f.name, top_k=3
            )

        # Should return top 3 by absolute weight
        assert len(neurons) == 3
        assert 3 in neurons  # -0.8 has highest abs weight
        assert 1 in neurons  # -0.5 has second highest
        assert 2 in neurons or 4 in neurons  # 0.3 or 0.2

        # Check weights
        assert weights[3] == -0.8
        assert weights[1] == -0.5

        # Check metadata
        assert metadata["positive_label"] == "happy"
        assert metadata["negative_label"] == "sad"

        Path(f.name).unlink()

    def test_load_neurons_without_labels(self):
        """Test loading neurons without labels in file."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            direction = np.array([0.5, -0.5, 0.3])
            np.savez(f.name, direction=direction)

            neurons, weights, metadata = NeuronAnalysisService.load_neurons_from_direction(
                f.name, top_k=2
            )

        assert len(neurons) == 2
        assert metadata == {}

        Path(f.name).unlink()
