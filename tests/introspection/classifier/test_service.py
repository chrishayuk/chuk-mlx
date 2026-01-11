"""Tests for classifier service."""


class TestClassifierService:
    """Tests for classifier service."""

    def test_import(self):
        """Test classifier service can be imported."""
        from chuk_lazarus.introspection.classifier.service import ClassifierService

        assert ClassifierService is not None
