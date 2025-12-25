"""
Tests for head components.

Tests LMHead, ClassifierHead, and RegressionHead.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.heads import (
    ClassifierHead,
    HeadOutput,
    LMHead,
    RegressionHead,
    cross_entropy_loss,
)
from chuk_lazarus.models_v2.heads.classifier import (
    PoolerHead,
    create_classifier_head,
    sequence_classification_loss,
    token_classification_loss,
)
from chuk_lazarus.models_v2.heads.regression import (
    MultiTaskRegressionHead,
    create_regression_head,
    mse_loss,
)


class TestLMHead:
    """Tests for LMHead."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        head = LMHead(
            hidden_size=512,
            vocab_size=32000,
        )

        hidden_states = mx.random.normal((2, 10, 512))
        output = head(hidden_states)

        assert isinstance(output, HeadOutput)
        assert output.logits.shape == (2, 10, 32000)
        assert output.loss is None

    def test_with_labels(self):
        """Test forward pass with labels for loss computation."""
        head = LMHead(
            hidden_size=256,
            vocab_size=1000,
        )

        hidden_states = mx.random.normal((2, 10, 256))
        labels = mx.random.randint(0, 1000, (2, 10))
        output = head(hidden_states, labels=labels)

        assert output.logits.shape == (2, 10, 1000)
        assert output.loss is not None
        assert output.loss.shape == ()  # Scalar loss

    def test_with_tied_embeddings(self):
        """Test with tied embedding weights."""
        # Create embedding layer
        embeddings = nn.Embedding(1000, 256)

        head = LMHead(
            hidden_size=256,
            vocab_size=1000,
            tied_embeddings=embeddings,
        )

        hidden_states = mx.random.normal((1, 5, 256))
        output = head(hidden_states)

        assert output.logits.shape == (1, 5, 1000)
        assert head.lm_head is None  # Should not have separate projection

    def test_tie_weights_method(self):
        """Test tie_weights method."""
        head = LMHead(
            hidden_size=256,
            vocab_size=1000,
        )

        # Initially should have separate projection
        assert head.lm_head is not None

        # Tie weights
        embeddings = nn.Embedding(1000, 256)
        head.tie_weights(embeddings)

        assert head.lm_head is None
        assert head.tied_embeddings is embeddings

    def test_output_size_property(self):
        """Test output_size property."""
        head = LMHead(
            hidden_size=512,
            vocab_size=50000,
        )

        assert head.output_size == 50000

    def test_with_bias(self):
        """Test LMHead with bias."""
        head = LMHead(
            hidden_size=256,
            vocab_size=1000,
            bias=True,
        )

        hidden_states = mx.random.normal((1, 5, 256))
        output = head(hidden_states)

        assert output.logits.shape == (1, 5, 1000)


class TestClassifierHead:
    """Tests for ClassifierHead."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        head = ClassifierHead(
            hidden_size=512,
            num_labels=10,
        )

        hidden_states = mx.random.normal((2, 10, 512))
        output = head(hidden_states)

        assert isinstance(output, HeadOutput)
        assert output.logits.shape == (2, 10)

    def test_with_labels(self):
        """Test forward pass with labels."""
        head = ClassifierHead(
            hidden_size=256,
            num_labels=5,
        )

        hidden_states = mx.random.normal((4, 10, 256))
        labels = mx.random.randint(0, 5, (4,))
        output = head(hidden_states, labels=labels)

        assert output.logits.shape == (4, 5)
        assert output.loss is not None

    def test_mean_pooling(self):
        """Test mean pooling."""
        head = ClassifierHead(
            hidden_size=256,
            num_labels=3,
            pool_strategy="mean",
        )

        hidden_states = mx.random.normal((2, 20, 256))
        output = head(hidden_states)

        assert output.logits.shape == (2, 3)

    def test_first_token_pooling(self):
        """Test first token pooling (CLS-style)."""
        head = ClassifierHead(
            hidden_size=256,
            num_labels=3,
            pool_strategy="first",
        )

        hidden_states = mx.random.normal((2, 20, 256))
        output = head(hidden_states)

        assert output.logits.shape == (2, 3)

    def test_last_token_pooling(self):
        """Test last token pooling (default)."""
        head = ClassifierHead(
            hidden_size=256,
            num_labels=3,
            pool_strategy="last",
        )

        hidden_states = mx.random.normal((2, 20, 256))
        output = head(hidden_states)

        assert output.logits.shape == (2, 3)

    def test_output_size_property(self):
        """Test output_size property."""
        head = ClassifierHead(
            hidden_size=512,
            num_labels=100,
        )

        assert head.output_size == 100

    def test_binary_classification(self):
        """Test binary classification."""
        head = ClassifierHead(
            hidden_size=256,
            num_labels=2,
        )

        hidden_states = mx.random.normal((4, 10, 256))
        output = head(hidden_states)

        assert output.logits.shape == (4, 2)


class TestRegressionHead:
    """Tests for RegressionHead."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        head = RegressionHead(
            hidden_size=512,
            output_dim=1,
        )

        hidden_states = mx.random.normal((2, 10, 512))
        output = head(hidden_states)

        assert isinstance(output, HeadOutput)
        assert output.logits.shape == (2, 1)

    def test_multi_output(self):
        """Test multiple regression outputs."""
        head = RegressionHead(
            hidden_size=256,
            output_dim=5,
        )

        hidden_states = mx.random.normal((2, 10, 256))
        output = head(hidden_states)

        assert output.logits.shape == (2, 5)

    def test_with_labels_mse(self):
        """Test forward pass with labels (MSE loss)."""
        head = RegressionHead(
            hidden_size=256,
            output_dim=1,
        )

        hidden_states = mx.random.normal((4, 10, 256))
        labels = mx.random.normal((4, 1))
        output = head(hidden_states, labels=labels)

        assert output.loss is not None
        assert output.loss.shape == ()

    def test_mean_pooling(self):
        """Test mean pooling for regression."""
        head = RegressionHead(
            hidden_size=256,
            output_dim=1,
            pool_strategy="mean",
        )

        hidden_states = mx.random.normal((2, 20, 256))
        output = head(hidden_states)

        assert output.logits.shape == (2, 1)


class TestCrossEntropyLoss:
    """Tests for cross_entropy_loss function."""

    def test_basic_loss(self):
        """Test basic loss computation."""
        batch_size, seq_len, vocab_size = 2, 5, 100
        logits = mx.random.normal((batch_size, seq_len, vocab_size))
        labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))

        loss = cross_entropy_loss(logits, labels)

        assert loss.shape == ()
        assert loss.item() > 0

    def test_ignore_index(self):
        """Test ignore_index functionality."""
        batch_size, seq_len, vocab_size = 2, 5, 100
        logits = mx.random.normal((batch_size, seq_len, vocab_size))

        # Set some labels to ignore
        labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        labels = mx.where(labels < 10, -100, labels)  # Ignore some

        loss = cross_entropy_loss(logits, labels, ignore_index=-100)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_all_ignored(self):
        """Test when all labels are ignored."""
        batch_size, seq_len, vocab_size = 2, 5, 100
        logits = mx.random.normal((batch_size, seq_len, vocab_size))
        labels = mx.full((batch_size, seq_len), -100)

        loss = cross_entropy_loss(logits, labels, ignore_index=-100)

        # Should handle gracefully (divide by max(0, 1))
        assert loss.shape == ()


class TestHeadOutput:
    """Tests for HeadOutput dataclass."""

    def test_basic_output(self):
        """Test HeadOutput creation."""
        logits = mx.random.normal((2, 10, 1000))
        output = HeadOutput(logits=logits)

        assert output.logits.shape == (2, 10, 1000)
        assert output.loss is None

    def test_with_loss(self):
        """Test HeadOutput with loss."""
        logits = mx.random.normal((2, 10, 1000))
        loss = mx.array(2.5)

        output = HeadOutput(logits=logits, loss=loss)

        assert output.logits is not None
        assert output.loss.item() == 2.5


class TestHeadGradients:
    """Tests for gradient flow through heads."""

    def test_lm_head_gradients(self):
        """Test gradients flow through LMHead."""
        head = LMHead(
            hidden_size=64,
            vocab_size=100,
        )

        hidden_states = mx.random.normal((1, 5, 64))
        labels = mx.random.randint(0, 100, (1, 5))

        def loss_fn(model, hidden_states, labels):
            out = model(hidden_states, labels=labels)
            return out.loss

        loss_and_grad_fn = nn.value_and_grad(head, loss_fn)
        loss, grads = loss_and_grad_fn(head, hidden_states, labels)

        assert loss.item() > 0
        assert any(g is not None for g in grads.values())

    def test_classifier_head_gradients(self):
        """Test gradients flow through ClassifierHead."""
        head = ClassifierHead(
            hidden_size=64,
            num_labels=5,
        )

        hidden_states = mx.random.normal((2, 10, 64))
        labels = mx.random.randint(0, 5, (2,))

        def loss_fn(model, hidden_states, labels):
            out = model(hidden_states, labels=labels)
            return out.loss

        loss_and_grad_fn = nn.value_and_grad(head, loss_fn)
        loss, grads = loss_and_grad_fn(head, hidden_states, labels)

        assert loss.item() > 0

    def test_regression_head_gradients(self):
        """Test gradients flow through RegressionHead."""
        head = RegressionHead(
            hidden_size=64,
            output_dim=1,
        )

        hidden_states = mx.random.normal((2, 10, 64))
        labels = mx.random.normal((2, 1))

        def loss_fn(model, hidden_states, labels):
            out = model(hidden_states, labels=labels)
            return out.loss

        loss_and_grad_fn = nn.value_and_grad(head, loss_fn)
        loss, grads = loss_and_grad_fn(head, hidden_states, labels)

        assert loss.item() >= 0


class TestClassifierAdvanced:
    """Advanced tests for ClassifierHead."""

    def test_token_classification(self):
        """Test token classification (no pooling)."""
        head = ClassifierHead(
            hidden_size=256,
            num_labels=9,
            pool_strategy="none",
        )

        hidden_states = mx.random.normal((2, 20, 256))
        output = head(hidden_states)

        # Should output logits for each token
        assert output.logits.shape == (2, 20, 9)

    def test_token_classification_with_labels(self):
        """Test token classification with labels."""
        head = ClassifierHead(
            hidden_size=256,
            num_labels=9,
            pool_strategy="none",
        )

        hidden_states = mx.random.normal((2, 20, 256))
        labels = mx.random.randint(0, 9, (2, 20))
        output = head(hidden_states, labels=labels)

        assert output.loss is not None
        assert output.loss.shape == ()

    def test_with_attention_mask(self):
        """Test mean pooling with attention mask."""
        head = ClassifierHead(
            hidden_size=256,
            num_labels=5,
            pool_strategy="mean",
        )

        hidden_states = mx.random.normal((2, 20, 256))
        attention_mask = mx.ones((2, 20))
        # Mask out last 5 positions
        attention_mask = mx.where(mx.arange(20) >= 15, 0.0, 1.0)
        attention_mask = mx.broadcast_to(attention_mask, (2, 20))

        output = head(hidden_states, attention_mask=attention_mask)
        assert output.logits.shape == (2, 5)

    def test_with_dropout(self):
        """Test classifier with dropout."""
        head = ClassifierHead(
            hidden_size=256,
            num_labels=5,
            dropout=0.1,
        )

        hidden_states = mx.random.normal((2, 10, 256))
        output = head(hidden_states)
        assert output.logits.shape == (2, 5)


class TestPoolerHead:
    """Tests for PoolerHead."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        head = PoolerHead(
            hidden_size=256,
            num_labels=3,
        )

        hidden_states = mx.random.normal((2, 10, 256))
        output = head(hidden_states)

        assert output.logits.shape == (2, 3)

    def test_with_labels(self):
        """Test with labels for loss."""
        head = PoolerHead(
            hidden_size=256,
            num_labels=5,
        )

        hidden_states = mx.random.normal((2, 10, 256))
        labels = mx.random.randint(0, 5, (2,))
        output = head(hidden_states, labels=labels)

        assert output.loss is not None

    def test_tanh_activation(self):
        """Test with tanh activation (default)."""
        head = PoolerHead(
            hidden_size=256,
            num_labels=2,
            activation="tanh",
        )

        hidden_states = mx.random.normal((2, 10, 256))
        output = head(hidden_states)
        assert output.logits.shape == (2, 2)

    def test_gelu_activation(self):
        """Test with gelu activation."""
        head = PoolerHead(
            hidden_size=256,
            num_labels=2,
            activation="gelu",
        )

        hidden_states = mx.random.normal((2, 10, 256))
        output = head(hidden_states)
        assert output.logits.shape == (2, 2)

    def test_no_activation(self):
        """Test with no activation."""
        head = PoolerHead(
            hidden_size=256,
            num_labels=2,
            activation="none",
        )

        hidden_states = mx.random.normal((2, 10, 256))
        output = head(hidden_states)
        assert output.logits.shape == (2, 2)

    def test_last_pool_strategy(self):
        """Test with last token pooling."""
        head = PoolerHead(
            hidden_size=256,
            num_labels=2,
            pool_strategy="last",
        )

        hidden_states = mx.random.normal((2, 10, 256))
        output = head(hidden_states)
        assert output.logits.shape == (2, 2)

    def test_output_size_property(self):
        """Test output_size property."""
        head = PoolerHead(hidden_size=256, num_labels=10)
        assert head.output_size == 10


class TestRegressionAdvanced:
    """Advanced tests for RegressionHead."""

    def test_first_token_pooling(self):
        """Test first token pooling."""
        head = RegressionHead(
            hidden_size=256,
            output_dim=1,
            pool_strategy="first",
        )

        hidden_states = mx.random.normal((2, 20, 256))
        output = head(hidden_states)
        assert output.logits.shape == (2, 1)

    def test_with_attention_mask(self):
        """Test mean pooling with attention mask."""
        head = RegressionHead(
            hidden_size=256,
            output_dim=1,
            pool_strategy="mean",
        )

        hidden_states = mx.random.normal((2, 20, 256))
        attention_mask = mx.ones((2, 20))
        output = head(hidden_states, attention_mask=attention_mask)
        assert output.logits.shape == (2, 1)

    def test_per_position_regression(self):
        """Test per-position regression (no pooling)."""
        head = RegressionHead(
            hidden_size=256,
            output_dim=1,
            pool_strategy="none",
        )

        hidden_states = mx.random.normal((2, 20, 256))
        output = head(hidden_states)
        assert output.logits.shape == (2, 20, 1)

    def test_with_hidden_layer(self):
        """Test with additional hidden layer."""
        head = RegressionHead(
            hidden_size=256,
            output_dim=1,
            use_hidden_layer=True,
        )

        hidden_states = mx.random.normal((2, 10, 256))
        output = head(hidden_states)
        assert output.logits.shape == (2, 1)

    def test_with_dropout(self):
        """Test with dropout."""
        head = RegressionHead(
            hidden_size=256,
            output_dim=1,
            dropout=0.1,
        )

        hidden_states = mx.random.normal((2, 10, 256))
        output = head(hidden_states)
        assert output.logits.shape == (2, 1)

    def test_1d_labels(self):
        """Test with 1D labels (auto-expanded)."""
        head = RegressionHead(
            hidden_size=256,
            output_dim=1,
        )

        hidden_states = mx.random.normal((4, 10, 256))
        labels = mx.random.normal((4,))  # 1D labels
        output = head(hidden_states, labels=labels)

        assert output.loss is not None
        assert output.loss.shape == ()


class TestMultiTaskRegressionHead:
    """Tests for MultiTaskRegressionHead."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        head = MultiTaskRegressionHead(
            hidden_size=256,
            task_dims={"sentiment": 1, "rating": 1},
        )

        hidden_states = mx.random.normal((2, 10, 256))
        output = head(hidden_states)

        assert output.logits.shape == (2, 2)  # Combined outputs
        assert "sentiment" in output.aux_outputs
        assert "rating" in output.aux_outputs

    def test_with_labels(self):
        """Test with task labels."""
        head = MultiTaskRegressionHead(
            hidden_size=256,
            task_dims={"sentiment": 1, "rating": 1},
        )

        hidden_states = mx.random.normal((2, 10, 256))
        labels = {
            "sentiment": mx.random.normal((2, 1)),
            "rating": mx.random.normal((2, 1)),
        }
        output = head(hidden_states, labels=labels)

        assert output.loss is not None

    def test_partial_labels(self):
        """Test with partial labels (only some tasks)."""
        head = MultiTaskRegressionHead(
            hidden_size=256,
            task_dims={"sentiment": 1, "rating": 1, "embedding": 128},
        )

        hidden_states = mx.random.normal((2, 10, 256))
        labels = {"sentiment": mx.random.normal((2, 1))}  # Only sentiment
        output = head(hidden_states, labels=labels)

        assert output.loss is not None

    def test_without_shared_hidden(self):
        """Test without shared hidden layer."""
        head = MultiTaskRegressionHead(
            hidden_size=256,
            task_dims={"a": 1, "b": 2},
            shared_hidden=False,
        )

        hidden_states = mx.random.normal((2, 10, 256))
        output = head(hidden_states)

        assert output.logits.shape == (2, 3)

    def test_first_pool_strategy(self):
        """Test with first token pooling."""
        head = MultiTaskRegressionHead(
            hidden_size=256,
            task_dims={"task1": 1},
            pool_strategy="first",
        )

        hidden_states = mx.random.normal((2, 10, 256))
        output = head(hidden_states)
        assert output.aux_outputs["task1"].shape == (2, 1)

    def test_mean_pool_strategy(self):
        """Test with mean pooling."""
        head = MultiTaskRegressionHead(
            hidden_size=256,
            task_dims={"task1": 1},
            pool_strategy="mean",
        )

        hidden_states = mx.random.normal((2, 10, 256))
        output = head(hidden_states)
        assert output.aux_outputs["task1"].shape == (2, 1)

    def test_output_size_property(self):
        """Test output_size property."""
        head = MultiTaskRegressionHead(
            hidden_size=256,
            task_dims={"a": 1, "b": 5, "c": 10},
        )
        assert head.output_size == 16


class TestLossFunctions:
    """Tests for loss functions."""

    def test_sequence_classification_loss(self):
        """Test sequence classification loss."""
        logits = mx.random.normal((4, 5))
        labels = mx.random.randint(0, 5, (4,))

        loss = sequence_classification_loss(logits, labels)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_token_classification_loss(self):
        """Test token classification loss."""
        logits = mx.random.normal((2, 10, 5))
        labels = mx.random.randint(0, 5, (2, 10))

        loss = token_classification_loss(logits, labels)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_token_classification_loss_with_ignore(self):
        """Test token classification loss with ignored labels."""
        logits = mx.random.normal((2, 10, 5))
        labels = mx.random.randint(-100, 5, (2, 10))  # Some -100 values

        loss = token_classification_loss(logits, labels, ignore_index=-100)
        assert loss.shape == ()

    def test_mse_loss(self):
        """Test MSE loss."""
        predictions = mx.random.normal((4, 1))
        targets = mx.random.normal((4, 1))

        loss = mse_loss(predictions, targets)
        assert loss.shape == ()
        assert loss.item() >= 0


class TestHeadFactoryFunctions:
    """Tests for factory functions."""

    def test_create_classifier_head(self):
        """Test classifier head factory."""
        head = create_classifier_head(
            hidden_size=256,
            num_labels=5,
            pool_strategy="mean",
        )
        assert isinstance(head, ClassifierHead)
        assert head.output_size == 5

    def test_create_regression_head(self):
        """Test regression head factory."""
        head = create_regression_head(
            hidden_size=256,
            output_dim=3,
            pool_strategy="first",
        )
        assert isinstance(head, RegressionHead)
        assert head.output_size == 3
