"""
Tests for block and backbone protocols.
"""

from chuk_lazarus.models_v2.core.protocols import (
    Backbone,
    Block,
    Head,
)


class TestBlockProtocol:
    """Tests for Block protocol."""

    def test_mock_block_implements_protocol(self):
        """Test a mock class can implement Block."""

        class MockBlock:
            hidden_size: int = 256

            def __call__(self, x, mask=None, cache=None):
                return x, cache

        mock = MockBlock()
        assert isinstance(mock, Block)


class TestBackboneProtocol:
    """Tests for Backbone protocol."""

    def test_mock_backbone_implements_protocol(self):
        """Test a mock class can implement Backbone."""

        class MockBackbone:
            num_layers: int = 12
            hidden_size: int = 4096

            def __call__(self, input_ids, mask=None, cache=None):
                return input_ids, cache

        mock = MockBackbone()
        assert isinstance(mock, Backbone)


class TestHeadProtocol:
    """Tests for Head protocol."""

    def test_mock_head_implements_protocol(self):
        """Test a mock class can implement Head."""

        class MockHead:
            def __call__(self, hidden_states):
                return hidden_states

        mock = MockHead()
        assert isinstance(mock, Head)
