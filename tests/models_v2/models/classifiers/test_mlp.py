"""
Tests for MLPClassifier.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.core.enums import ActivationType
from chuk_lazarus.models_v2.models.classifiers import MLPClassifier


class TestMLPClassifier:
    """Tests for MLPClassifier."""

    def test_basic_init(self):
        """Test basic initialization."""
        clf = MLPClassifier(input_size=64, hidden_size=32, num_labels=2)

        assert clf.mlp is not None
        assert clf.classifier.weight.shape == (2, 64)

    def test_binary_classification(self):
        """Test binary classification (single output)."""
        clf = MLPClassifier(input_size=32, hidden_size=16, num_labels=1)
        x = mx.random.normal((8, 32))

        logits = clf(x)

        assert logits.shape == (8, 1)

    def test_multiclass_classification(self):
        """Test multi-class classification."""
        clf = MLPClassifier(input_size=128, hidden_size=64, num_labels=10)
        x = mx.random.normal((16, 128))

        logits = clf(x)

        assert logits.shape == (16, 10)

    def test_different_activations(self):
        """Test different activation functions."""
        x = mx.random.normal((4, 64))

        for activation in [
            ActivationType.GELU,
            ActivationType.RELU,
            ActivationType.SILU,
        ]:
            clf = MLPClassifier(
                input_size=64,
                hidden_size=32,
                num_labels=3,
                activation=activation,
            )
            logits = clf(x)

            assert logits.shape == (4, 3)

    def test_no_bias(self):
        """Test classifier without bias."""
        clf = MLPClassifier(
            input_size=64,
            hidden_size=32,
            num_labels=3,
            bias=False,
        )

        # When bias=False, the bias attribute doesn't exist in MLX
        assert not hasattr(clf.classifier, "bias") or clf.classifier.get("bias") is None

        x = mx.random.normal((4, 64))
        logits = clf(x)

        assert logits.shape == (4, 3)

    def test_hidden_size_variations(self):
        """Test various hidden sizes."""
        x = mx.random.normal((4, 64))

        for hidden_size in [16, 32, 64, 128, 256]:
            clf = MLPClassifier(
                input_size=64,
                hidden_size=hidden_size,
                num_labels=2,
            )
            logits = clf(x)

            assert logits.shape == (4, 2)

    def test_single_sample(self):
        """Test with single sample (batch size 1)."""
        clf = MLPClassifier(input_size=256, hidden_size=128, num_labels=5)
        x = mx.random.normal((1, 256))

        logits = clf(x)

        assert logits.shape == (1, 5)

    def test_large_input(self):
        """Test with larger input (embedding-like)."""
        clf = MLPClassifier(input_size=768, hidden_size=256, num_labels=3)
        x = mx.random.normal((32, 768))

        logits = clf(x)

        assert logits.shape == (32, 3)

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        clf = MLPClassifier(input_size=32, hidden_size=16, num_labels=2)
        x = mx.random.normal((4, 32))
        targets = mx.array([0, 1, 0, 1])

        def loss_fn(model):
            logits = model(x)
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            return -mx.mean(log_probs[mx.arange(4), targets])

        loss_and_grad_fn = nn.value_and_grad(clf, loss_fn)
        loss, grads = loss_and_grad_fn(clf)

        assert loss.item() > 0
        assert "mlp" in grads
        assert "classifier" in grads

    def test_mlp_uses_internal_component(self):
        """Test that MLP uses the chuk-mlx MLP component."""
        from chuk_lazarus.models_v2.components.ffn import MLP

        clf = MLPClassifier(input_size=64, hidden_size=32, num_labels=2)

        assert isinstance(clf.mlp, MLP)
