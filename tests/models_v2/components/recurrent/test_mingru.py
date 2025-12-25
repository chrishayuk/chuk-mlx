"""
Tests for MinGRU (minimal GRU variant).
"""

import mlx.core as mx
import mlx.utils

from chuk_lazarus.models_v2.components.recurrent import GRU, MinGRU


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

        def count_params(model):
            params = model.parameters()
            leaf_params = mlx.utils.tree_flatten(params)
            total = sum(p[1].size for p in leaf_params if hasattr(p[1], "size"))
            return total

        mingru_params = count_params(mingru)
        gru_params = count_params(gru)

        # MinGRU should have fewer parameters (3 matrices vs 6 matrices)
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


class TestMinGRUGradients:
    """Tests for gradient flow through MinGRU."""

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


class TestMinGRUBatchHandling:
    """Tests for batch dimension handling."""

    def test_mingru_batch_sizes(self):
        """Test MinGRU handles different batch sizes."""
        mingru = MinGRU(input_size=128, hidden_size=256)

        for batch_size in [1, 2, 4, 8]:
            x = mx.random.normal((batch_size, 10, 128))
            output, _ = mingru(x)
            assert output.shape == (batch_size, 10, 256)

    def test_mingru_sequence_lengths(self):
        """Test MinGRU handles different sequence lengths."""
        mingru = MinGRU(input_size=128, hidden_size=256)

        for seq_len in [1, 5, 10, 50, 100]:
            x = mx.random.normal((2, seq_len, 128))
            output, _ = mingru(x)
            assert output.shape == (2, seq_len, 256)
