"""Tests for distributed configuration."""

import pytest

from chuk_lazarus.distributed import (
    DistributedConfig,
    get_rank,
    get_world_size,
    is_main_process,
)


class TestDistributedConfig:
    """Tests for DistributedConfig."""

    def test_default_config(self):
        """Test default single-worker config."""
        config = DistributedConfig()
        assert config.rank == 0
        assert config.world_size == 1
        assert config.local_rank == 0
        assert config.local_world_size == 1

    def test_is_main(self):
        """Test is_main property."""
        assert DistributedConfig(rank=0, world_size=4).is_main
        assert not DistributedConfig(rank=1, world_size=4).is_main

    def test_is_distributed(self):
        """Test is_distributed property."""
        assert not DistributedConfig(rank=0, world_size=1).is_distributed
        assert DistributedConfig(rank=0, world_size=2).is_distributed

    def test_invalid_rank(self):
        """Test validation for invalid rank."""
        with pytest.raises(ValueError):
            DistributedConfig(rank=-1, world_size=4)

        with pytest.raises(ValueError):
            DistributedConfig(rank=4, world_size=4)

    def test_from_env(self, monkeypatch):
        """Test creating config from environment."""
        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")

        config = DistributedConfig.from_env()
        assert config.rank == 2
        assert config.world_size == 8
        assert config.local_rank == 1
        assert config.local_world_size == 4

    def test_from_env_defaults(self, monkeypatch):
        """Test from_env with no environment variables."""
        # Clear any existing env vars
        for var in [
            "RANK",
            "WORLD_RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
        ]:
            monkeypatch.delenv(var, raising=False)

        config = DistributedConfig.from_env()
        assert config.rank == 0
        assert config.world_size == 1

    def test_global_config(self):
        """Test global config management."""
        config = DistributedConfig(rank=1, world_size=4)
        config.set_global()

        assert get_rank() == 1
        assert get_world_size() == 4
        assert not is_main_process()

        # Reset to default
        DistributedConfig(rank=0, world_size=1).set_global()
