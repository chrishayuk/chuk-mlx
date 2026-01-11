"""Tests for circuit activation collector."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from chuk_lazarus.introspection.circuit.collector import (
    ActivationCollector,
    CollectedActivations,
    CollectorConfig,
    collect_activations,
)
from chuk_lazarus.introspection.circuit.dataset import CircuitDataset, LabeledPrompt


class SimpleMockModel(nn.Module):
    """Simple mock model for testing."""

    def __init__(self, hidden_size=64, num_layers=4, vocab_size=100):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Create inner model structure
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.layers = [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        self.model.norm = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """Forward pass returning logits."""
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        h = self.model.norm(h)
        return self.lm_head(h)


class TestCollectorConfig:
    """Tests for CollectorConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CollectorConfig()
        assert config.layers == "all"
        assert config.decision_layer_range == (8, 14)
        assert config.capture_hidden_states is True
        assert config.capture_attention_weights is False
        assert config.capture_mlp_intermediate is False
        assert config.position == -1
        assert config.dtype == "float32"
        assert config.max_new_tokens == 30
        assert config.temperature == 0.0

    def test_custom_layers(self):
        """Test custom layer selection."""
        config = CollectorConfig(layers=[0, 2, 4])
        assert config.layers == [0, 2, 4]

    def test_decision_layers(self):
        """Test decision layer string."""
        config = CollectorConfig(layers="decision")
        assert config.layers == "decision"

    def test_all_layers(self):
        """Test all layers string."""
        config = CollectorConfig(layers="all")
        assert config.layers == "all"

    def test_custom_position(self):
        """Test custom position."""
        config = CollectorConfig(position=0)
        assert config.position == 0

    def test_enable_attention_capture(self):
        """Test enabling attention weight capture."""
        config = CollectorConfig(capture_attention_weights=True)
        assert config.capture_attention_weights is True

    def test_enable_mlp_intermediate(self):
        """Test enabling MLP intermediate capture."""
        config = CollectorConfig(capture_mlp_intermediate=True)
        assert config.capture_mlp_intermediate is True

    def test_custom_dtype(self):
        """Test custom dtype."""
        config = CollectorConfig(dtype="float16")
        assert config.dtype == "float16"

    def test_generation_settings(self):
        """Test generation settings."""
        config = CollectorConfig(max_new_tokens=50, temperature=0.7)
        assert config.max_new_tokens == 50
        assert config.temperature == 0.7


class TestCollectedActivations:
    """Tests for CollectedActivations."""

    def test_create_empty(self):
        """Test creating empty collected activations."""
        acts = CollectedActivations()
        assert len(acts) == 0
        assert acts.captured_layers == []

    def test_len(self):
        """Test length calculation."""
        acts = CollectedActivations()
        acts.labels = [0, 1, 0]
        assert len(acts) == 3

    def test_captured_layers(self):
        """Test getting captured layer indices."""
        acts = CollectedActivations()
        acts.hidden_states = {
            2: mx.zeros((1, 64)),
            0: mx.zeros((1, 64)),
            4: mx.zeros((1, 64)),
        }
        assert acts.captured_layers == [0, 2, 4]

    def test_get_layer_activations(self):
        """Test getting activations for a specific layer."""
        acts = CollectedActivations()
        layer_acts = mx.ones((10, 64))
        acts.hidden_states[5] = layer_acts
        retrieved = acts.get_layer_activations(5)
        assert retrieved is not None
        assert retrieved.shape == (10, 64)

    def test_get_layer_activations_nonexistent(self):
        """Test getting activations for non-existent layer."""
        acts = CollectedActivations()
        assert acts.get_layer_activations(99) is None

    def test_get_activations_numpy(self):
        """Test converting to numpy array."""
        acts = CollectedActivations()
        acts.hidden_states[0] = mx.ones((5, 64))
        np_acts = acts.get_activations_numpy(0)
        assert isinstance(np_acts, np.ndarray)
        assert np_acts.shape == (5, 64)

    def test_get_activations_numpy_handles_bfloat16(self):
        """Test converting bfloat16 to numpy."""
        acts = CollectedActivations()
        acts.hidden_states[0] = mx.ones((5, 64), dtype=mx.bfloat16)
        np_acts = acts.get_activations_numpy(0)
        assert isinstance(np_acts, np.ndarray)
        assert np_acts.dtype == np.float32

    def test_get_activations_numpy_nonexistent(self):
        """Test getting numpy array for non-existent layer."""
        acts = CollectedActivations()
        assert acts.get_activations_numpy(99) is None

    def test_get_by_label(self):
        """Test getting indices by label."""
        acts = CollectedActivations()
        acts.labels = [0, 1, 0, 1, 1]
        indices_arr, indices_list = acts.get_by_label(1)
        assert len(indices_list) == 3
        assert indices_list == [1, 3, 4]

    def test_get_label_mask(self):
        """Test getting boolean mask for label."""
        acts = CollectedActivations()
        acts.labels = [0, 1, 0, 1]
        mask = acts.get_label_mask(1)
        assert mask.tolist() == [False, True, False, True]

    def test_split_by_label(self):
        """Test splitting activations by label."""
        # Note: MLX now supports boolean indexing in newer versions
        # If this test fails with "boolean indices not supported",
        # it means the MLX version doesn't support it yet
        try:
            acts = CollectedActivations()
            # Create sample activations with 3 samples
            acts.hidden_states[0] = mx.array([[1.0] * 64, [2.0] * 64, [3.0] * 64])
            acts.labels = [0, 1, 0]

            result = acts.split_by_label(0)

            # Should have activations for both labels
            assert 0 in result
            assert 1 in result
            # Label 0 should have 2 samples
            assert result[0].shape[0] == 2
            # Label 1 should have 1 sample
            assert result[1].shape[0] == 1
        except ValueError as e:
            if "boolean indices" in str(e):
                pytest.skip("MLX version doesn't support boolean indexing yet")
            raise

    def test_get_positive_negative(self):
        """Test getting positive and negative activations."""
        # Note: MLX now supports boolean indexing in newer versions
        try:
            acts = CollectedActivations()
            # Create sample activations
            acts.hidden_states[0] = mx.array([[1.0] * 64, [2.0] * 64, [3.0] * 64])
            acts.labels = [0, 1, 0]

            pos, neg = acts.get_positive_negative(0)

            # Positive (label=1) should have 1 sample
            assert pos.shape[0] == 1
            # Negative (label=0) should have 2 samples
            assert neg.shape[0] == 2
        except ValueError as e:
            if "boolean indices" in str(e):
                pytest.skip("MLX version doesn't support boolean indexing yet")
            raise

    def test_summary(self):
        """Test generating summary statistics."""
        acts = CollectedActivations()
        acts.labels = [0, 1, 0, 1, 1]
        acts.categories = ["cat1", "cat2", "cat1", "cat2", "cat2"]
        acts.hidden_states = {0: mx.zeros((5, 64)), 2: mx.zeros((5, 64))}
        acts.hidden_size = 64
        acts.model_id = "test-model"
        acts.dataset_name = "test-dataset"
        acts.dataset_label_names = {0: "negative", 1: "positive"}
        summary = acts.summary()
        assert summary["num_samples"] == 5
        assert summary["by_label"]["negative"] == 2
        assert summary["by_label"]["positive"] == 3
        assert summary["by_category"]["cat1"] == 2
        assert summary["by_category"]["cat2"] == 3
        assert summary["captured_layers"] == [0, 2]
        assert summary["hidden_size"] == 64

    def test_save_and_load_basic(self):
        """Test saving and loading activations."""
        acts = CollectedActivations()
        acts.hidden_states = {0: mx.ones((3, 64))}
        acts.labels = [0, 1, 0]
        acts.label_names = ["neg", "pos", "neg"]
        acts.categories = ["cat1", "cat2", "cat1"]
        acts.prompts = ["p1", "p2", "p3"]
        acts.expected_outputs = ["o1", "o2", "o3"]
        acts.model_id = "test-model"
        acts.hidden_size = 64
        acts.num_layers = 4
        acts.dataset_name = "test"
        acts.dataset_label_names = {0: "negative", 1: "positive"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations"
            acts.save(path)
            loaded = CollectedActivations.load(path)
            assert len(loaded) == 3
            assert loaded.model_id == "test-model"
            assert loaded.hidden_size == 64
            assert loaded.captured_layers == [0]

    def test_save_with_outputs(self):
        """Test saving with model outputs included."""
        acts = CollectedActivations()
        acts.hidden_states = {0: mx.ones((2, 64))}
        acts.labels = [0, 1]
        acts.prompts = ["p1", "p2"]
        acts.expected_outputs = ["o1", "o2"]
        acts.model_outputs = ["generated1", "generated2"]
        acts.model_id = "test"
        acts.hidden_size = 64
        acts.num_layers = 4
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations"
            acts.save(path, include_outputs=True)
            loaded = CollectedActivations.load(path)
            assert loaded.model_outputs == ["generated1", "generated2"]

    def test_save_with_attention_weights(self):
        """Test saving with attention weights."""
        acts = CollectedActivations()
        acts.hidden_states = {0: mx.ones((2, 64))}
        acts.attention_weights = {0: mx.ones((2, 4, 10, 10))}
        acts.labels = [0, 1]
        acts.prompts = ["p1", "p2"]
        acts.model_id = "test"
        acts.hidden_size = 64
        acts.num_layers = 4
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations"
            acts.save(path)
            loaded = CollectedActivations.load(path)
            assert 0 in loaded.attention_weights

    def test_save_with_mlp_intermediates(self):
        """Test saving with MLP intermediates."""
        acts = CollectedActivations()
        acts.hidden_states = {0: mx.ones((2, 64))}
        acts.mlp_intermediates = {0: mx.ones((2, 256))}
        acts.labels = [0, 1]
        acts.prompts = ["p1", "p2"]
        acts.model_id = "test"
        acts.hidden_size = 64
        acts.num_layers = 4
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations"
            acts.save(path)
            loaded = CollectedActivations.load(path)
            assert 0 in loaded.mlp_intermediates

    def test_save_fallback_to_npz(self):
        """Test fallback to npz when safetensors fails."""
        acts = CollectedActivations()
        acts.hidden_states = {0: mx.ones((2, 64))}
        acts.labels = [0, 1]
        acts.prompts = ["p1", "p2"]
        acts.model_id = "test"
        acts.hidden_size = 64
        acts.num_layers = 4

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations"
            tensors = {"hidden_states.layer_0": np.ones((2, 64))}

            # Test the actual fallback by temporarily making safetensors unavailable
            import sys

            _ = sys.modules.get("safetensors.numpy")  # Check if module exists

            # Mock the import to fail
            with patch.dict("sys.modules", {"safetensors.numpy": None}):
                CollectedActivations._save_safetensors(tensors, path.with_suffix(".safetensors"))

                # Should create npz file instead
                npz_path = path.with_suffix(".npz")
                assert npz_path.exists()

    def test_load_fallback_to_npz(self):
        """Test loading from npz when safetensors not available."""
        acts = CollectedActivations()
        acts.hidden_states = {0: mx.ones((2, 64))}
        acts.labels = [0, 1]
        acts.prompts = ["p1", "p2"]
        acts.model_id = "test"
        acts.hidden_size = 64
        acts.num_layers = 4

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations"

            # Save as npz directly
            npz_path = path.with_suffix(".npz")
            np.savez(npz_path, **{"hidden_states.layer_0": np.ones((2, 64))})

            # Save metadata
            import json

            json_path = path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(
                    {
                        "model_id": "test",
                        "hidden_size": 64,
                        "num_layers": 4,
                        "num_samples": 2,
                        "captured_layers": [0],
                        "labels": [0, 1],
                        "prompts": ["p1", "p2"],
                        "expected_outputs": [None, None],
                        "dataset_label_names": {},
                    },
                    f,
                )

            # Test loading when safetensors.load_file fails

            with patch.dict("sys.modules", {"safetensors.numpy": None}):
                loaded = CollectedActivations.load(path)
                assert len(loaded) == 2


class TestActivationCollector:
    """Tests for ActivationCollector."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = SimpleMockModel(hidden_size=64, num_layers=4)
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=np.array([[1, 2, 3, 4, 5]]))
        tokenizer.decode = Mock(return_value="mock_output")
        return tokenizer

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = Mock()
        config.hidden_size = 64
        return config

    def test_init(self, mock_model, mock_tokenizer, mock_config):
        """Test initialization."""
        collector = ActivationCollector(mock_model, mock_tokenizer, mock_config)
        assert collector.model is mock_model
        assert collector.tokenizer is mock_tokenizer
        assert collector.model_id == "unknown"
        assert collector.num_layers == 4
        assert collector.hidden_size == 64

    def test_detect_structure(self, mock_model, mock_tokenizer, mock_config):
        """Test model structure detection."""
        collector = ActivationCollector(mock_model, mock_tokenizer, mock_config)
        assert len(collector._layers) == 4

    def test_detect_structure_missing_layers_raises(self, mock_tokenizer, mock_config):
        """Test structure detection raises on missing layers."""
        bad_model = nn.Module()
        with pytest.raises(ValueError, match="Cannot detect"):
            ActivationCollector(bad_model, mock_tokenizer, mock_config)

    def test_detect_structure_direct_layers(self, mock_tokenizer, mock_config):
        """Test detection when model has layers attribute directly."""
        # Create a model with direct layers attribute (no model.model)
        model = nn.Module()
        model.layers = [nn.Linear(64, 64) for _ in range(4)]
        model.hidden_size = 64

        collector = ActivationCollector(model, mock_tokenizer, mock_config)
        assert collector.num_layers == 4
        assert collector._backbone is model

    def test_detect_structure_fallback_hidden_size(self, mock_tokenizer):
        """Test fallback hidden size when not in config or model."""
        model = SimpleMockModel(hidden_size=64, num_layers=4)
        config = Mock(spec=[])  # Mock with no attributes

        _ = ActivationCollector(model, mock_tokenizer, config)
        # Should use fallback value of 768 when config doesn't have hidden_size
        # and backbone doesn't have it either
        # But in our SimpleMockModel, it does have hidden_size, so let's test differently
        # We need a model structure that doesn't expose hidden_size

        # Create a minimal model without hidden_size attribute
        minimal_model = nn.Module()
        minimal_model.model = nn.Module()
        minimal_model.model.layers = [nn.Linear(64, 64) for _ in range(4)]

        collector2 = ActivationCollector(minimal_model, mock_tokenizer, config)
        # Should use fallback value
        assert collector2.hidden_size == 768

    def test_get_layers_to_capture_all(self, mock_model, mock_tokenizer, mock_config):
        """Test getting all layers."""
        collector = ActivationCollector(mock_model, mock_tokenizer, mock_config)
        config = CollectorConfig(layers="all")
        layers = collector._get_layers_to_capture(config)
        assert layers == [0, 1, 2, 3]

    def test_get_layers_to_capture_specific(self, mock_model, mock_tokenizer, mock_config):
        """Test getting specific layers."""
        collector = ActivationCollector(mock_model, mock_tokenizer, mock_config)
        config = CollectorConfig(layers=[0, 2])
        layers = collector._get_layers_to_capture(config)
        assert layers == [0, 2]

    def test_get_layers_to_capture_decision(self, mock_model, mock_tokenizer, mock_config):
        """Test getting decision layers."""
        # Create model with more layers
        model = SimpleMockModel(hidden_size=64, num_layers=16)
        collector = ActivationCollector(model, mock_tokenizer, mock_config)
        config = CollectorConfig(layers="decision", decision_layer_range=(8, 14))
        layers = collector._get_layers_to_capture(config)
        assert layers == [8, 9, 10, 11, 12, 13]

    def test_get_layers_to_capture_decision_clamps(self, mock_model, mock_tokenizer, mock_config):
        """Test decision layers clamps to model size."""
        collector = ActivationCollector(mock_model, mock_tokenizer, mock_config)
        config = CollectorConfig(layers="decision", decision_layer_range=(0, 100))
        layers = collector._get_layers_to_capture(config)
        assert max(layers) < 4

    def test_get_layers_to_capture_fallback(self, mock_model, mock_tokenizer, mock_config):
        """Test fallback when layers value is unknown."""
        collector = ActivationCollector(mock_model, mock_tokenizer, mock_config)
        config = CollectorConfig(layers="unknown_value")
        layers = collector._get_layers_to_capture(config)
        # Should return middle and last layer as fallback
        assert len(layers) == 2
        assert layers[0] == collector.num_layers // 2
        assert layers[1] == collector.num_layers - 1

    @patch("chuk_lazarus.introspection.ablation.AblationStudy")
    def test_from_pretrained(self, mock_ablation_study):
        """Test loading from pretrained."""
        mock_study = Mock()
        mock_study.adapter.model = SimpleMockModel()
        mock_study.adapter.tokenizer = Mock()
        mock_study.adapter.config = Mock(hidden_size=64)
        mock_ablation_study.from_pretrained.return_value = mock_study
        collector = ActivationCollector.from_pretrained("test-model")
        assert collector.model_id == "test-model"

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_collect_single(self, mock_hooks_cls, mock_model, mock_tokenizer, mock_config):
        """Test collecting activations for a single prompt."""
        # Setup mock hooks
        mock_hooks = Mock()
        mock_state = Mock()
        mock_state.hidden_states = {
            0: mx.ones((1, 5, 64)),
            2: mx.ones((1, 5, 64)),
        }
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks
        collector = ActivationCollector(mock_model, mock_tokenizer, mock_config)
        config = CollectorConfig(layers=[0, 2])
        result = collector.collect_single("Test prompt", config)
        assert 0 in result
        assert 2 in result

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_collect_single_default_config(
        self, mock_hooks_cls, mock_model, mock_tokenizer, mock_config
    ):
        """Test collecting with default config (None)."""
        mock_hooks = Mock()
        mock_state = Mock()
        mock_state.hidden_states = {0: mx.ones((1, 5, 64))}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks

        collector = ActivationCollector(mock_model, mock_tokenizer, mock_config)
        # Call without config - should use default
        result = collector.collect_single("Test prompt", config=None)
        assert len(result) > 0

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_collect_single_2d_hidden_states(
        self, mock_hooks_cls, mock_model, mock_tokenizer, mock_config
    ):
        """Test collecting when hidden states are 2D instead of 3D."""
        mock_hooks = Mock()
        mock_state = Mock()
        # Simulate 2D hidden states (batch flattened or single sample)
        mock_state.hidden_states = {
            0: mx.ones((5, 64)),  # 2D: [seq, hidden]
        }
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks

        collector = ActivationCollector(mock_model, mock_tokenizer, mock_config)
        config = CollectorConfig(layers=[0])
        result = collector.collect_single("Test prompt", config)
        assert 0 in result
        # Should extract last position from 2D tensor
        assert result[0].shape == (64,)

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_collect_dataset(self, mock_hooks_cls, mock_model, mock_tokenizer, mock_config):
        """Test collecting activations for a dataset."""
        # Setup mock hooks
        mock_hooks = Mock()
        mock_state = Mock()
        mock_state.hidden_states = {0: mx.ones((1, 5, 64))}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks
        collector = ActivationCollector(mock_model, mock_tokenizer, mock_config)
        dataset = CircuitDataset()
        dataset.add(LabeledPrompt(text="Test 1", label=1, category="test"))
        dataset.add(LabeledPrompt(text="Test 2", label=0, category="test"))
        config = CollectorConfig(layers=[0])
        result = collector.collect(dataset, config, progress=False)
        assert len(result) == 2
        assert result.dataset_name == "circuit_dataset"

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_collect_list_of_prompts(self, mock_hooks_cls, mock_model, mock_tokenizer, mock_config):
        """Test collecting from list of prompts."""
        mock_hooks = Mock()
        mock_state = Mock()
        mock_state.hidden_states = {0: mx.ones((1, 5, 64))}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks
        collector = ActivationCollector(mock_model, mock_tokenizer, mock_config)
        prompts = [
            LabeledPrompt(text="Test 1", label=1, category="test"),
            LabeledPrompt(text="Test 2", label=0, category="test"),
        ]
        result = collector.collect(prompts, progress=False)
        assert len(result) == 2
        assert result.dataset_name == "custom"

    @patch("chuk_lazarus.introspection.ablation.ModelAdapter")
    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_collect_with_generation(
        self, mock_hooks_cls, mock_adapter_cls, mock_model, mock_tokenizer, mock_config
    ):
        """Test collecting with generation enabled."""
        # Setup mocks
        mock_hooks = Mock()
        mock_state = Mock()
        mock_state.hidden_states = {0: mx.ones((1, 5, 64))}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks
        mock_adapter = Mock()
        mock_adapter.generate.return_value = "Generated text"
        mock_adapter_cls.return_value = mock_adapter
        collector = ActivationCollector(mock_model, mock_tokenizer, mock_config)
        dataset = CircuitDataset()
        dataset.add(LabeledPrompt(text="Test", label=1, category="test"))
        config = CollectorConfig(layers=[0], max_new_tokens=10)
        result = collector.collect(dataset, config, progress=False)
        assert len(result.model_outputs) == 1

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_collect_with_progress(
        self, mock_hooks_cls, mock_model, mock_tokenizer, mock_config, capsys
    ):
        """Test collecting with progress output."""
        mock_hooks = Mock()
        mock_state = Mock()
        mock_state.hidden_states = {0: mx.ones((1, 5, 64))}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks

        collector = ActivationCollector(mock_model, mock_tokenizer, mock_config)

        # Create 10 prompts to trigger progress message
        dataset = CircuitDataset()
        for i in range(10):
            dataset.add(LabeledPrompt(text=f"Test {i}", label=i % 2, category="test"))

        config = CollectorConfig(layers=[0])
        _ = collector.collect(dataset, config, progress=True)

        # Check that progress was printed
        captured = capsys.readouterr()
        assert "Collecting 10/10" in captured.out


@patch("chuk_lazarus.introspection.circuit.collector.ActivationCollector")
@patch("chuk_lazarus.introspection.circuit.dataset.create_arithmetic_dataset")
def test_collect_activations_convenience(mock_create_dataset, mock_collector_cls):
    """Test collect_activations convenience function."""
    mock_dataset = Mock()
    mock_create_dataset.return_value = mock_dataset
    mock_collector = Mock()
    mock_activations = Mock()
    mock_collector.collect.return_value = mock_activations
    mock_collector_cls.from_pretrained.return_value = mock_collector
    result = collect_activations("test-model", layers=[0, 2])
    assert result is mock_activations
    mock_collector_cls.from_pretrained.assert_called_once_with("test-model")


@patch("chuk_lazarus.introspection.circuit.collector.ActivationCollector")
def test_collect_activations_with_save(mock_collector_cls):
    """Test collect_activations with save."""
    mock_collector = Mock()
    mock_activations = Mock()
    mock_collector.collect.return_value = mock_activations
    mock_collector_cls.from_pretrained.return_value = mock_collector
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "acts"
        collect_activations("test-model", output_path=str(path))
        mock_activations.save.assert_called_once_with(str(path))
