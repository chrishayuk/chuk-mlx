"""
Tests for LSTM.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.components.recurrent import LSTM


class TestLSTM:
    """Tests for LSTM."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        lstm = LSTM(
            input_size=256,
            hidden_size=512,
        )

        x = mx.random.normal((2, 10, 256))
        output, (h_n, c_n) = lstm(x)

        assert output.shape == (2, 10, 512)
        # States have shape (num_layers, batch, hidden_size)
        assert h_n.shape == (1, 2, 512)
        assert c_n.shape == (1, 2, 512)

    def test_with_initial_state(self):
        """Test LSTM with initial hidden state."""
        lstm = LSTM(
            input_size=128,
            hidden_size=256,
        )

        x = mx.random.normal((2, 10, 128))
        # Initial state shape: (num_layers, batch, hidden_size)
        h_0 = mx.random.normal((1, 2, 256))
        c_0 = mx.random.normal((1, 2, 256))

        output, (h_n, c_n) = lstm(x, (h_0, c_0))

        assert output.shape == (2, 10, 256)
        assert h_n.shape == (1, 2, 256)
        assert c_n.shape == (1, 2, 256)

    def test_single_token(self):
        """Test LSTM with single token."""
        lstm = LSTM(
            input_size=64,
            hidden_size=128,
        )

        x = mx.random.normal((1, 1, 64))
        output, _ = lstm(x)

        assert output.shape == (1, 1, 128)

    def test_different_hidden_size(self):
        """Test LSTM with different hidden sizes."""
        for hidden_size in [64, 128, 256, 512]:
            lstm = LSTM(
                input_size=128,
                hidden_size=hidden_size,
            )

            x = mx.random.normal((2, 5, 128))
            output, _ = lstm(x)

            assert output.shape == (2, 5, hidden_size)

    def test_multi_layer(self):
        """Test multi-layer LSTM."""
        lstm = LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=3,
        )

        x = mx.random.normal((2, 10, 128))
        output, (h_n, c_n) = lstm(x)

        assert output.shape == (2, 10, 256)
        # States have shape (num_layers, batch, hidden_size)
        assert h_n.shape == (3, 2, 256)
        assert c_n.shape == (3, 2, 256)


class TestLSTMAdvanced:
    """Advanced tests for LSTM components."""

    def test_lstm_cell_init_state(self):
        """Test LSTMCell.init_state."""
        from chuk_lazarus.models_v2.components.recurrent.lstm import LSTMCell

        cell = LSTMCell(input_size=128, hidden_size=256)
        h, c = cell.init_state(batch_size=4)
        assert h.shape == (4, 256)
        assert c.shape == (4, 256)

    def test_lstm_with_dropout(self):
        """Test LSTM with dropout between layers."""
        lstm = LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=3,
            dropout=0.1,
        )

        x = mx.random.normal((2, 10, 128))
        output, (h_n, c_n) = lstm(x)

        assert output.shape == (2, 10, 256)
        assert h_n.shape == (3, 2, 256)
        assert c_n.shape == (3, 2, 256)

    def test_lstm_init_state(self):
        """Test LSTM.init_state for all layers."""
        lstm = LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
        )

        h, c = lstm.init_state(batch_size=4)
        assert h.shape == (2, 4, 256)
        assert c.shape == (2, 4, 256)


class TestLSTMGradients:
    """Tests for gradient flow through LSTM."""

    def test_lstm_gradients(self):
        """Test gradients flow through LSTM."""
        lstm = LSTM(
            input_size=64,
            hidden_size=128,
        )

        x = mx.random.normal((1, 5, 64))

        def loss_fn(model, x):
            out, _ = model(x)
            return mx.mean(out**2)

        loss_and_grad_fn = nn.value_and_grad(lstm, loss_fn)
        loss, grads = loss_and_grad_fn(lstm, x)

        assert loss.item() > 0


class TestLSTMBatchHandling:
    """Tests for batch dimension handling."""

    def test_lstm_batch_sizes(self):
        """Test LSTM handles different batch sizes."""
        lstm = LSTM(input_size=128, hidden_size=256)

        for batch_size in [1, 2, 4, 8]:
            x = mx.random.normal((batch_size, 10, 128))
            output, _ = lstm(x)
            assert output.shape == (batch_size, 10, 256)

    def test_lstm_sequence_lengths(self):
        """Test LSTM handles different sequence lengths."""
        lstm = LSTM(input_size=128, hidden_size=256)

        for seq_len in [1, 5, 10, 50, 100]:
            x = mx.random.normal((2, seq_len, 128))
            output, _ = lstm(x)
            assert output.shape == (2, seq_len, 256)
