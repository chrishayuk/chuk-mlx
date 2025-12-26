"""
Tests for LinearClassifier.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.models.classifiers import LinearClassifier


class TestLinearClassifier:
    """Tests for LinearClassifier."""

    def test_basic_init(self):
        """Test basic initialization."""
        clf = LinearClassifier(input_size=64, num_labels=2)

        assert clf.fc.weight.shape == (2, 64)
        assert clf.fc.bias.shape == (2,)

    def test_binary_classification(self):
        """Test binary classification (single output)."""
        clf = LinearClassifier(input_size=32, num_labels=1)
        x = mx.random.normal((8, 32))

        logits = clf(x)

        assert logits.shape == (8, 1)

    def test_multiclass_classification(self):
        """Test multi-class classification."""
        clf = LinearClassifier(input_size=128, num_labels=10)
        x = mx.random.normal((16, 128))

        logits = clf(x)

        assert logits.shape == (16, 10)

    def test_no_bias(self):
        """Test classifier without bias."""
        clf = LinearClassifier(input_size=64, num_labels=3, bias=False)

        # When bias=False, the bias attribute doesn't exist in MLX
        assert not hasattr(clf.fc, "bias") or clf.fc.get("bias") is None

        x = mx.random.normal((4, 64))
        logits = clf(x)

        assert logits.shape == (4, 3)

    def test_single_sample(self):
        """Test with single sample (batch size 1)."""
        clf = LinearClassifier(input_size=256, num_labels=5)
        x = mx.random.normal((1, 256))

        logits = clf(x)

        assert logits.shape == (1, 5)

    def test_different_input_sizes(self):
        """Test various input sizes."""
        for input_size in [2, 64, 768, 4096]:
            clf = LinearClassifier(input_size=input_size, num_labels=2)
            x = mx.random.normal((4, input_size))

            logits = clf(x)

            assert logits.shape == (4, 2)

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        clf = LinearClassifier(input_size=32, num_labels=2)
        x = mx.random.normal((4, 32))
        targets = mx.array([0, 1, 0, 1])

        def loss_fn(model):
            logits = model(x)
            # Simple cross-entropy approximation
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            return -mx.mean(log_probs[mx.arange(4), targets])

        loss_and_grad_fn = nn.value_and_grad(clf, loss_fn)
        loss, grads = loss_and_grad_fn(clf)

        assert loss.item() > 0
        assert "fc" in grads
