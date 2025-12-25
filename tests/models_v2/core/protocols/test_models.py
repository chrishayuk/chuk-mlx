"""
Tests for Model protocol.
"""

from chuk_lazarus.models_v2.core.protocols import Model


class TestModelProtocol:
    """Tests for Model protocol."""

    def test_mock_model_implements_protocol(self):
        """Test a mock class can implement Model."""

        class MockModel:
            hidden_size: int = 4096

            def __call__(self, input_ids, cache=None):
                return input_ids, cache

            def set_mode(self, mode: str) -> None:
                pass

        mock = MockModel()
        assert isinstance(mock, Model)

    def test_incomplete_model_not_protocol(self):
        """Test incomplete implementation is not Model."""

        class IncompleteModel:
            def __call__(self, input_ids):
                return input_ids

            # Missing hidden_size and set_mode

        mock = IncompleteModel()
        assert not isinstance(mock, Model)
