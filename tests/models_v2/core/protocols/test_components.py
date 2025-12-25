"""
Tests for component protocols.
"""

from chuk_lazarus.models_v2.core.protocols import (
    FFN,
    SSM,
    Attention,
    Embedding,
    Norm,
    PositionEmbedding,
    RecurrentCell,
)


class TestEmbeddingProtocol:
    """Tests for Embedding protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test Embedding protocol is runtime checkable."""

        assert hasattr(Embedding, "__protocol_attrs__") or hasattr(
            Embedding, "_is_runtime_protocol"
        )

    def test_mock_embedding_implements_protocol(self):
        """Test a mock class can implement Embedding."""

        class MockEmbedding:
            vocab_size: int = 32000
            hidden_size: int = 4096

            def __call__(self, input_ids):
                return input_ids

            def as_linear(self, hidden):
                return hidden

        # Should be instance-checkable
        mock = MockEmbedding()
        assert isinstance(mock, Embedding)

    def test_incomplete_embedding_not_protocol(self):
        """Test incomplete implementation is not Embedding."""

        class IncompleteEmbedding:
            vocab_size: int = 32000
            # Missing hidden_size, __call__, as_linear

        mock = IncompleteEmbedding()
        assert not isinstance(mock, Embedding)


class TestPositionEmbeddingProtocol:
    """Tests for PositionEmbedding protocol."""

    def test_mock_position_embedding_implements_protocol(self):
        """Test a mock class can implement PositionEmbedding."""

        class MockPositionEmbedding:
            def __call__(self, x, offset: int = 0):
                return x

        mock = MockPositionEmbedding()
        assert isinstance(mock, PositionEmbedding)


class TestNormProtocol:
    """Tests for Norm protocol."""

    def test_mock_norm_implements_protocol(self):
        """Test a mock class can implement Norm."""

        class MockNorm:
            def __call__(self, x):
                return x

        mock = MockNorm()
        assert isinstance(mock, Norm)


class TestFFNProtocol:
    """Tests for FFN protocol."""

    def test_mock_ffn_implements_protocol(self):
        """Test a mock class can implement FFN."""

        class MockFFN:
            hidden_size: int = 4096
            intermediate_size: int = 11008

            def __call__(self, x):
                return x

        mock = MockFFN()
        assert isinstance(mock, FFN)


class TestAttentionProtocol:
    """Tests for Attention protocol."""

    def test_mock_attention_implements_protocol(self):
        """Test a mock class can implement Attention."""

        class MockAttention:
            num_heads: int = 32
            head_dim: int = 128

            def __call__(self, x, mask=None, cache=None):
                return x, cache

        mock = MockAttention()
        assert isinstance(mock, Attention)


class TestSSMProtocol:
    """Tests for SSM protocol."""

    def test_mock_ssm_implements_protocol(self):
        """Test a mock class can implement SSM."""

        class MockSSM:
            hidden_size: int = 4096
            state_size: int = 16

            def __call__(self, x, state=None):
                return x, state

        mock = MockSSM()
        assert isinstance(mock, SSM)


class TestRecurrentCellProtocol:
    """Tests for RecurrentCell protocol."""

    def test_mock_recurrent_cell_implements_protocol(self):
        """Test a mock class can implement RecurrentCell."""

        class MockRecurrentCell:
            hidden_size: int = 256

            def __call__(self, x, state=None):
                return x, state or ()

        mock = MockRecurrentCell()
        assert isinstance(mock, RecurrentCell)
