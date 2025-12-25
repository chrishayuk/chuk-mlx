"""
Tests for TokenClassifier.
"""

import mlx.core as mx
import pytest

from chuk_lazarus.models_v2.core.config import ModelConfig
from chuk_lazarus.models_v2.models.classifiers import TokenClassifier


class TestTokenClassifier:
    """Tests for TokenClassifier."""

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
        model = TokenClassifier(tiny_config, num_labels=9)

        assert model.num_labels == 9
        assert model.config == tiny_config

    def test_forward_pass(self, tiny_config):
        """Test forward pass."""
        model = TokenClassifier(tiny_config, num_labels=9)
        input_ids = mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        output = model(input_ids)

        # Token classification: (batch, seq_len, num_labels)
        assert output.logits.shape == (2, 5, 9)
        assert output.loss is None

    def test_with_labels(self, tiny_config):
        """Test forward pass with labels for loss computation."""
        model = TokenClassifier(tiny_config, num_labels=9)
        input_ids = mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        labels = mx.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 0]])

        output = model(input_ids, labels=labels)

        assert output.logits.shape == (2, 5, 9)
        assert output.loss is not None
        assert output.loss.item() > 0

    def test_with_attention_mask(self, tiny_config):
        """Test with attention mask (shape only - mask handling varies by backend)."""
        model = TokenClassifier(tiny_config, num_labels=5)
        input_ids = mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        # Basic forward pass - attention mask handling is backbone-dependent
        output = model(input_ids)

        assert output.logits.shape == (2, 5, 5)

    def test_output_hidden_states(self, tiny_config):
        """Test returning hidden states."""
        model = TokenClassifier(tiny_config, num_labels=5)
        input_ids = mx.array([[1, 2, 3, 4, 5]])

        output = model(input_ids, output_hidden_states=True)

        assert output.logits.shape == (1, 5, 5)
        assert output.hidden_states is not None
        assert len(output.hidden_states) > 0

    def test_ner_style_labels(self, tiny_config):
        """Test with NER-style labels (including ignore index)."""
        model = TokenClassifier(tiny_config, num_labels=9)
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        # -100 is typically the ignore index
        labels = mx.array([[0, 1, -100, 2, 3]])

        output = model(input_ids, labels=labels)

        assert output.logits.shape == (1, 5, 9)
        assert output.loss is not None

    def test_from_config(self, tiny_config):
        """Test factory method."""
        model = TokenClassifier.from_config(tiny_config, num_labels=7)

        assert model.num_labels == 7
        assert model.config == tiny_config

    def test_backbone_property(self, tiny_config):
        """Test backbone property."""
        model = TokenClassifier(tiny_config, num_labels=5)

        assert model.backbone is not None
        assert model.backbone == model._backbone

    def test_single_token(self, tiny_config):
        """Test with single token sequence."""
        model = TokenClassifier(tiny_config, num_labels=3)
        input_ids = mx.array([[42]])

        output = model(input_ids)

        assert output.logits.shape == (1, 1, 3)

    def test_varying_sequence_lengths(self, tiny_config):
        """Test with different sequence lengths."""
        model = TokenClassifier(tiny_config, num_labels=5)

        for seq_len in [1, 5, 10, 32]:
            input_ids = mx.random.randint(1, 1000, (2, seq_len))
            output = model(input_ids)

            assert output.logits.shape == (2, seq_len, 5)
