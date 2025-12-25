"""
Tests for GRU.
"""

import mlx.core as mx

from chuk_lazarus.models_v2.components.recurrent import GRU


class TestGRU:
    """Tests for GRU."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        gru = GRU(
            input_size=256,
            hidden_size=512,
        )

        x = mx.random.normal((2, 10, 256))
        output, h_n = gru(x)

        assert output.shape == (2, 10, 512)
        # State has shape (num_layers, batch, hidden_size)
        assert h_n.shape == (1, 2, 512)

    def test_with_initial_state(self):
        """Test GRU with initial hidden state."""
        gru = GRU(
            input_size=128,
            hidden_size=256,
        )

        x = mx.random.normal((2, 10, 128))
        # Initial state shape: (num_layers, batch, hidden_size)
        h_0 = mx.random.normal((1, 2, 256))

        output, h_n = gru(x, h_0)

        assert output.shape == (2, 10, 256)
        assert h_n.shape == (1, 2, 256)

    def test_fewer_gates_than_lstm(self):
        """Test GRU has fewer parameters than LSTM (only h, no c)."""
        gru = GRU(
            input_size=128,
            hidden_size=256,
        )

        x = mx.random.normal((1, 5, 128))
        _, state = gru(x)

        # GRU returns just h_n with shape (num_layers, batch, hidden_size)
        assert state.shape == (1, 1, 256)

    def test_single_token(self):
        """Test GRU with single token."""
        gru = GRU(
            input_size=64,
            hidden_size=128,
        )

        x = mx.random.normal((1, 1, 64))
        output, _ = gru(x)

        assert output.shape == (1, 1, 128)

    def test_multi_layer(self):
        """Test multi-layer GRU."""
        gru = GRU(
            input_size=128,
            hidden_size=256,
            num_layers=2,
        )

        x = mx.random.normal((2, 10, 128))
        output, h_n = gru(x)

        assert output.shape == (2, 10, 256)
        assert h_n.shape == (2, 2, 256)


class TestGRUAdvanced:
    """Advanced tests for GRU components."""

    def test_gru_cell_init_state(self):
        """Test GRUCell.init_state."""
        from chuk_lazarus.models_v2.components.recurrent.gru import GRUCell

        cell = GRUCell(input_size=128, hidden_size=256)
        state = cell.init_state(batch_size=4)
        assert state.shape == (4, 256)

    def test_gru_with_dropout(self):
        """Test GRU with dropout between layers."""
        gru = GRU(
            input_size=128,
            hidden_size=256,
            num_layers=3,
            dropout=0.1,
        )

        x = mx.random.normal((2, 10, 128))
        output, h_n = gru(x)

        assert output.shape == (2, 10, 256)
        assert h_n.shape == (3, 2, 256)

    def test_gru_init_state(self):
        """Test GRU.init_state for all layers."""
        gru = GRU(
            input_size=128,
            hidden_size=256,
            num_layers=2,
        )

        state = gru.init_state(batch_size=4)
        assert state.shape == (2, 4, 256)


class TestGRUGradients:
    """Tests for gradient flow through GRU."""

    def test_gru_gradients(self):
        """Test gradients flow through GRU."""
        gru = GRU(
            input_size=64,
            hidden_size=128,
        )

        x = mx.random.normal((1, 5, 64))

        def loss_fn(model, x):
            out, _ = model(x)
            return mx.mean(out**2)

        loss, grads = mx.value_and_grad(loss_fn)(gru, x)

        assert loss.item() > 0


class TestGRUBatchHandling:
    """Tests for batch dimension handling."""

    def test_gru_batch_sizes(self):
        """Test GRU handles different batch sizes."""
        gru = GRU(input_size=128, hidden_size=256)

        for batch_size in [1, 2, 4, 8]:
            x = mx.random.normal((batch_size, 10, 128))
            output, _ = gru(x)
            assert output.shape == (batch_size, 10, 256)

    def test_gru_sequence_lengths(self):
        """Test GRU handles different sequence lengths."""
        gru = GRU(input_size=128, hidden_size=256)

        for seq_len in [1, 5, 10, 50, 100]:
            x = mx.random.normal((2, seq_len, 128))
            output, _ = gru(x)
            assert output.shape == (2, seq_len, 256)
