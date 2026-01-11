"""Tests for memory utilities."""

from unittest.mock import MagicMock, patch


class TestMemoryUtils:
    """Tests for memory utilities."""

    def test_get_memory_usage(self):
        """Test getting memory usage."""
        from chuk_lazarus.utils.memory import get_memory_usage

        usage = get_memory_usage()

        assert usage is not None
        assert "rss_mb" in usage
        assert "vms_mb" in usage

    def test_log_memory_usage(self):
        """Test logging memory usage."""
        from chuk_lazarus.utils.memory import log_memory_usage

        # Should not raise
        log_memory_usage("test_label")
        log_memory_usage()

    def test_format_memory_usage(self):
        """Test formatting memory usage."""
        from chuk_lazarus.utils.memory import format_memory_usage

        formatted = format_memory_usage()

        assert "RSS=" in formatted
        assert "VMS=" in formatted
        assert "MB" in formatted
