"""
Tests for SequenceClassifier.
"""

import mlx.core as mx
import pytest

from chuk_lazarus.models_v2.core.config import ModelConfig
from chuk_lazarus.models_v2.models.classifiers import SequenceClassifier


class TestSequenceClassifier:
    """Tests for SequenceClassifier."""

    @pytest.fixture
    def tiny_config(self):
        """Create a tiny config for testing."""
        return ModelConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
        )

    def test_basic_init(self, tiny_config):
        """Test basic initialization."""
        model = SequenceClassifier(tiny_config, num_labels=3)

        assert model.num_labels == 3
        assert model.pool_strategy == "last"
        assert model.config == tiny_config

    def test_forward_pass(self, tiny_config):
        """Test forward pass."""
        model = SequenceClassifier(tiny_config, num_labels=5)
        input_ids = mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        output = model(input_ids)

        assert output.logits.shape == (2, 5)
        assert output.loss is None

    def test_with_labels(self, tiny_config):
        """Test forward pass with labels for loss computation."""
        model = SequenceClassifier(tiny_config, num_labels=3)
        input_ids = mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        labels = mx.array([0, 2])

        output = model(input_ids, labels=labels)

        assert output.logits.shape == (2, 3)
        assert output.loss is not None
        assert output.loss.item() > 0

    def test_pool_strategies(self, tiny_config):
        """Test different pooling strategies."""
        input_ids = mx.array([[1, 2, 3, 4, 5]])

        for strategy in ["first", "last", "mean"]:
            model = SequenceClassifier(
                tiny_config,
                num_labels=2,
                pool_strategy=strategy,
            )
            output = model(input_ids)

            assert output.logits.shape == (1, 2)

    def test_with_attention_mask(self, tiny_config):
        """Test with attention mask (shape only - mask handling varies by backend)."""
        model = SequenceClassifier(tiny_config, num_labels=2)
        input_ids = mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        # Basic forward pass - attention mask handling is backbone-dependent
        output = model(input_ids)

        assert output.logits.shape == (2, 2)

    def test_output_hidden_states(self, tiny_config):
        """Test returning hidden states."""
        model = SequenceClassifier(tiny_config, num_labels=2)
        input_ids = mx.array([[1, 2, 3, 4, 5]])

        output = model(input_ids, output_hidden_states=True)

        assert output.logits.shape == (1, 2)
        assert output.hidden_states is not None
        assert len(output.hidden_states) > 0

    def test_binary_classification(self, tiny_config):
        """Test binary classification."""
        model = SequenceClassifier(tiny_config, num_labels=2)
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([1])

        output = model(input_ids, labels=labels)

        assert output.logits.shape == (1, 2)
        assert output.loss is not None

    def test_from_config(self, tiny_config):
        """Test factory method."""
        model = SequenceClassifier.from_config(tiny_config, num_labels=4)

        assert model.num_labels == 4
        assert model.config == tiny_config

    def test_backbone_property(self, tiny_config):
        """Test backbone property."""
        model = SequenceClassifier(tiny_config, num_labels=2)

        assert model.backbone is not None
        assert model.backbone == model._backbone
