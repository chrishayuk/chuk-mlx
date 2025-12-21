"""Memory utilities."""

import logging
import os

logger = logging.getLogger(__name__)


def get_memory_usage() -> dict:
    """
    Get current process memory usage.

    Returns:
        Dict with memory stats in MB
    """
    try:
        import psutil

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            "rss_mb": mem_info.rss / (1024**2),
            "vms_mb": mem_info.vms / (1024**2),
        }
    except ImportError:
        return {"rss_mb": 0, "vms_mb": 0}


def log_memory_usage(label: str = ""):
    """Log current memory usage."""
    mem = get_memory_usage()
    logger.info(f"{label} - Memory: RSS={mem['rss_mb']:.2f}MB, VMS={mem['vms_mb']:.2f}MB")


def format_memory_usage() -> str:
    """Format memory usage as string."""
    mem = get_memory_usage()
    return f"RSS={mem['rss_mb']:.2f}MB, VMS={mem['vms_mb']:.2f}MB"
