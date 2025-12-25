"""
Tests for classifier factory functions.
"""

import mlx.core as mx

from chuk_lazarus.models_v2.core.enums import ClassificationTask
from chuk_lazarus.models_v2.models.classifiers import (
    SequenceClassifier,
    TokenClassifier,
    create_classifier,
)


class TestCreateClassifier:
    """Tests for create_classifier factory function."""

    def test_create_sequence_classifier(self):
        """Test creating a sequence classifier."""
        model = create_classifier(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            num_labels=3,
            task=ClassificationTask.SEQUENCE,
        )

        assert isinstance(model, SequenceClassifier)
        assert model.num_labels == 3

    def test_create_token_classifier(self):
        """Test creating a token classifier."""
        model = create_classifier(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            num_labels=9,
            task=ClassificationTask.TOKEN,
        )

        assert isinstance(model, TokenClassifier)
        assert model.num_labels == 9

    def test_create_with_string_task(self):
        """Test creating with string task type."""
        model = create_classifier(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            num_labels=5,
            task="sequence",
        )

        assert isinstance(model, SequenceClassifier)

    def test_default_task_is_sequence(self):
        """Test that default task is sequence classification."""
        model = create_classifier(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            num_labels=2,
        )

        assert isinstance(model, SequenceClassifier)

    def test_sequence_classifier_forward(self):
        """Test forward pass of created sequence classifier."""
        model = create_classifier(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            num_labels=3,
            task=ClassificationTask.SEQUENCE,
        )

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.logits.shape == (1, 3)

    def test_token_classifier_forward(self):
        """Test forward pass of created token classifier."""
        model = create_classifier(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            num_labels=9,
            task=ClassificationTask.TOKEN,
        )

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.logits.shape == (1, 5, 9)
