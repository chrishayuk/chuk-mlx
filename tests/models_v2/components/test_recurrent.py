"""
Tests for recurrent components.

Tests LSTM, GRU, and MinGRU.
"""

import mlx.core as mx

from chuk_lazarus.models_v2.components.recurrent import (
    GRU,
    LSTM,
    MinGRU,
)


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


class TestMinGRU:
    """Tests for MinGRU (minimal GRU variant)."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        mingru = MinGRU(
            input_size=256,
            hidden_size=512,
        )

        x = mx.random.normal((2, 10, 256))
        output, h_n = mingru(x)

        assert output.shape == (2, 10, 512)
        # State has shape (num_layers, batch, hidden_size)
        assert h_n.shape == (1, 2, 512)

    def test_with_initial_state(self):
        """Test MinGRU with initial hidden state."""
        mingru = MinGRU(
            input_size=128,
            hidden_size=256,
        )

        x = mx.random.normal((2, 10, 128))
        # Initial state shape: (num_layers, batch, hidden_size)
        h_0 = mx.random.normal((1, 2, 256))

        output, h_n = mingru(x, h_0)

        assert output.shape == (2, 10, 256)
        assert h_n.shape == (1, 2, 256)

    def test_fewer_parameters(self):
        """Test MinGRU has fewer parameters than full GRU."""
        mingru = MinGRU(
            input_size=128,
            hidden_size=256,
        )
        gru = GRU(
            input_size=128,
            hidden_size=256,
        )

        # Count parameters using mlx tree utilities
        import mlx.utils

        def count_params(model):
            params = model.parameters()
            leaf_params = mlx.utils.tree_flatten(params)
            total = sum(p[1].size for p in leaf_params if hasattr(p[1], "size"))
            return total

        mingru_params = count_params(mingru)
        gru_params = count_params(gru)

        # MinGRU should have fewer parameters (3 matrices vs 6 matrices)
        # MinGRU: W_z, U_z, W_h (3 projections)
        # GRU: W_ir, W_iz, W_in, W_hr, W_hz, W_hn (6 projections)
        assert mingru_params < gru_params

    def test_single_token(self):
        """Test MinGRU with single token."""
        mingru = MinGRU(
            input_size=64,
            hidden_size=128,
        )

        x = mx.random.normal((1, 1, 64))
        output, _ = mingru(x)

        assert output.shape == (1, 1, 128)

    def test_parallel_scan_shape(self):
        """Test MinGRU with parallel scan produces correct shape."""
        mingru = MinGRU(
            input_size=256,
            hidden_size=512,
        )

        x = mx.random.normal((4, 100, 256))
        output, h_n = mingru(x)

        assert output.shape == (4, 100, 512)
        assert h_n.shape == (1, 4, 512)


class TestRecurrentGradients:
    """Tests for gradient flow through recurrent components."""

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

        loss, grads = mx.value_and_grad(loss_fn)(lstm, x)

        assert loss.item() > 0

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

    def test_mingru_gradients(self):
        """Test gradients flow through MinGRU."""
        mingru = MinGRU(
            input_size=64,
            hidden_size=128,
        )

        x = mx.random.normal((1, 5, 64))

        def loss_fn(model, x):
            out, _ = model(x)
            return mx.mean(out**2)

        loss, grads = mx.value_and_grad(loss_fn)(mingru, x)

        assert loss.item() > 0


class TestRecurrentBatchHandling:
    """Tests for batch dimension handling."""

    def test_lstm_batch_sizes(self):
        """Test LSTM handles different batch sizes."""
        lstm = LSTM(input_size=128, hidden_size=256)

        for batch_size in [1, 2, 4, 8]:
            x = mx.random.normal((batch_size, 10, 128))
            output, _ = lstm(x)
            assert output.shape == (batch_size, 10, 256)

    def test_gru_batch_sizes(self):
        """Test GRU handles different batch sizes."""
        gru = GRU(input_size=128, hidden_size=256)

        for batch_size in [1, 2, 4, 8]:
            x = mx.random.normal((batch_size, 10, 128))
            output, _ = gru(x)
            assert output.shape == (batch_size, 10, 256)

    def test_mingru_batch_sizes(self):
        """Test MinGRU handles different batch sizes."""
        mingru = MinGRU(input_size=128, hidden_size=256)

        for batch_size in [1, 2, 4, 8]:
            x = mx.random.normal((batch_size, 10, 128))
            output, _ = mingru(x)
            assert output.shape == (batch_size, 10, 256)


class TestRecurrentSequenceLengths:
    """Tests for different sequence lengths."""

    def test_lstm_sequence_lengths(self):
        """Test LSTM handles different sequence lengths."""
        lstm = LSTM(input_size=128, hidden_size=256)

        for seq_len in [1, 5, 10, 50, 100]:
            x = mx.random.normal((2, seq_len, 128))
            output, _ = lstm(x)
            assert output.shape == (2, seq_len, 256)

    def test_gru_sequence_lengths(self):
        """Test GRU handles different sequence lengths."""
        gru = GRU(input_size=128, hidden_size=256)

        for seq_len in [1, 5, 10, 50, 100]:
            x = mx.random.normal((2, seq_len, 128))
            output, _ = gru(x)
            assert output.shape == (2, seq_len, 256)

    def test_mingru_sequence_lengths(self):
        """Test MinGRU handles different sequence lengths."""
        mingru = MinGRU(input_size=128, hidden_size=256)

        for seq_len in [1, 5, 10, 50, 100]:
            x = mx.random.normal((2, seq_len, 128))
            output, _ = mingru(x)
            assert output.shape == (2, seq_len, 256)


class TestMinGRUAdvanced:
    """Advanced tests for MinGRU components."""

    def test_mingru_cell_init_state(self):
        """Test MinGRUCell.init_state."""
        from chuk_lazarus.models_v2.components.recurrent.mingru import MinGRUCell

        cell = MinGRUCell(input_size=128, hidden_size=256)
        state = cell.init_state(batch_size=4)
        assert state.shape == (4, 256)

    def test_mingru_with_dropout(self):
        """Test MinGRU with dropout between layers."""
        mingru = MinGRU(
            input_size=128,
            hidden_size=256,
            num_layers=3,
            dropout=0.1,
        )

        x = mx.random.normal((2, 10, 128))
        output, h_n = mingru(x)

        assert output.shape == (2, 10, 256)
        assert h_n.shape == (3, 2, 256)

    def test_mingru_parallel_forward(self):
        """Test MinGRU parallel_forward method."""
        mingru = MinGRU(
            input_size=128,
            hidden_size=256,
        )

        x = mx.random.normal((2, 10, 128))
        output, h_n = mingru.parallel_forward(x)

        assert output.shape == (2, 10, 256)
        assert h_n.shape == (1, 2, 256)

    def test_mingru_init_state(self):
        """Test MinGRU.init_state for all layers."""
        mingru = MinGRU(
            input_size=128,
            hidden_size=256,
            num_layers=3,
        )

        state = mingru.init_state(batch_size=4)
        assert state.shape == (3, 4, 256)

    def test_mingru_block(self):
        """Test MinGRUBlock with residual connection."""
        from chuk_lazarus.models_v2.components.recurrent.mingru import MinGRUBlock

        block = MinGRUBlock(d_model=256)

        x = mx.random.normal((2, 10, 256))
        output, h_n = block(x)

        # Output should have same shape as input (residual)
        assert output.shape == (2, 10, 256)


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
