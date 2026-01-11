"""Gym info command handler.

This module provides the gym info display implementation.
"""

from __future__ import annotations

import logging
from argparse import Namespace

logger = logging.getLogger(__name__)


async def gym_info() -> None:
    """Display gym stream configuration info."""
    from ....data.batching.streaming import (
        GymOutputMode,
        GymTransport,
    )

    print(f"\n{'=' * 60}")
    print("Gym Stream Configuration")
    print(f"{'=' * 60}")

    print("\nSupported Transports:")
    for transport in GymTransport:
        print(f"  - {transport.value}")

    print("\nSupported Output Modes:")
    for mode in GymOutputMode:
        print(f"  - {mode.value}")

    print("\nDefault Configuration:")
    print("  Host:             localhost")
    print("  Port:             8023")
    print("  Transport:        telnet")
    print("  Output Mode:      json")
    print("  Connect Timeout:  10.0s")
    print("  Max Retries:      3")

    print("\nExample Usage:")
    print("  # Run mock stream for testing")
    print("  lazarus gym run --tokenizer gpt2 --mock --num-episodes 10")
    print()
    print("  # Connect to puzzle arcade server")
    print("  lazarus gym run --tokenizer gpt2 --host localhost --port 8023")
    print()
    print("  # Save samples to buffer file")
    print("  lazarus gym run --tokenizer gpt2 --mock --output buffer.json")


async def gym_info_cmd(args: Namespace) -> None:
    """CLI entry point for gym info command.

    Args:
        args: Parsed command-line arguments (unused)
    """
    await gym_info()


__all__ = [
    "gym_info",
    "gym_info_cmd",
]
