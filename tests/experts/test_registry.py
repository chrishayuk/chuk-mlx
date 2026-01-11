"""Tests for expert registry."""

from unittest.mock import MagicMock, patch

import pytest

from chuk_lazarus.experts.registry import (
    ExpertRegistry,
    create_expert,
    get_expert,
    list_experts,
    register_expert,
    register_expert_class,
    _EXPERT_CLASSES,
    _EXPERT_INSTANCES,
)
from chuk_lazarus.experts.rnn_expert_base import ExpertConfig


class TestModuleFunctions:
    """Tests for module-level functions."""

    def setup_method(self):
        """Clear expert instances before each test."""
        _EXPERT_INSTANCES.clear()

    def test_register_expert_class(self):
        """Test registering a new expert class."""
        mock_class = MagicMock()
        register_expert_class("test_expert", mock_class)
        assert "test_expert" in _EXPERT_CLASSES
        assert _EXPERT_CLASSES["test_expert"] == mock_class
        # Cleanup
        del _EXPERT_CLASSES["test_expert"]

    def test_register_expert(self):
        """Test registering an expert instance."""
        mock_expert = MagicMock()
        register_expert("test_instance", mock_expert)
        assert "test_instance" in _EXPERT_INSTANCES
        assert _EXPERT_INSTANCES["test_instance"] == mock_expert

    def test_get_expert_exists(self):
        """Test getting an existing expert."""
        mock_expert = MagicMock()
        _EXPERT_INSTANCES["existing"] = mock_expert
        result = get_expert("existing")
        assert result == mock_expert

    def test_get_expert_not_exists(self):
        """Test getting a non-existent expert."""
        result = get_expert("nonexistent")
        assert result is None

    def test_list_experts(self):
        """Test listing experts."""
        _EXPERT_INSTANCES["expert1"] = MagicMock()
        _EXPERT_INSTANCES["expert2"] = MagicMock()
        result = list_experts()
        assert "expert1" in result
        assert "expert2" in result

    def test_create_expert_gru(self):
        """Test creating a GRU expert."""
        config = ExpertConfig(
            name="test_gru",
            obs_dim=10,
            action_dim=5,
            hidden_dim=64,
        )
        result = create_expert("gru", config)

        assert result is not None
        assert result.config.name == "test_gru"
        # Verify it was auto-registered
        assert get_expert("test_gru") == result

    def test_create_expert_unknown_type(self):
        """Test creating an expert with unknown type."""
        config = ExpertConfig(
            name="test",
            obs_dim=10,
            action_dim=5,
            hidden_dim=64,
        )
        with pytest.raises(ValueError, match="Unknown expert type"):
            create_expert("unknown", config)


class TestExpertRegistry:
    """Tests for ExpertRegistry class."""

    def test_init(self):
        """Test registry initialization."""
        registry = ExpertRegistry()
        assert registry.experts == {}
        assert registry.configs == {}

    def test_register(self):
        """Test registering an expert."""
        registry = ExpertRegistry()
        mock_expert = MagicMock()
        mock_expert.config = ExpertConfig(
            name="test",
            obs_dim=10,
            action_dim=5,
            hidden_dim=64,
        )

        registry.register(mock_expert)

        assert "test" in registry.experts
        assert registry.experts["test"] == mock_expert

    def test_get_existing(self):
        """Test getting an existing expert."""
        registry = ExpertRegistry()
        mock_expert = MagicMock()
        mock_expert.config = ExpertConfig(
            name="test",
            obs_dim=10,
            action_dim=5,
            hidden_dim=64,
        )
        registry.register(mock_expert)

        result = registry.get("test")
        assert result == mock_expert

    def test_get_nonexistent(self):
        """Test getting a non-existent expert."""
        registry = ExpertRegistry()
        result = registry.get("nonexistent")
        assert result is None

    def test_getitem_existing(self):
        """Test __getitem__ for existing expert."""
        registry = ExpertRegistry()
        mock_expert = MagicMock()
        mock_expert.config = ExpertConfig(
            name="test",
            obs_dim=10,
            action_dim=5,
            hidden_dim=64,
        )
        registry.register(mock_expert)

        result = registry["test"]
        assert result == mock_expert

    def test_getitem_nonexistent(self):
        """Test __getitem__ for non-existent expert."""
        registry = ExpertRegistry()
        with pytest.raises(KeyError, match="Expert not found"):
            _ = registry["nonexistent"]

    def test_contains(self):
        """Test __contains__."""
        registry = ExpertRegistry()
        mock_expert = MagicMock()
        mock_expert.config = ExpertConfig(
            name="test",
            obs_dim=10,
            action_dim=5,
            hidden_dim=64,
        )
        registry.register(mock_expert)

        assert "test" in registry
        assert "other" not in registry

    def test_list_names(self):
        """Test listing expert names."""
        registry = ExpertRegistry()
        mock_expert1 = MagicMock()
        mock_expert1.config = ExpertConfig(
            name="expert1",
            obs_dim=10,
            action_dim=5,
            hidden_dim=64,
        )
        mock_expert2 = MagicMock()
        mock_expert2.config = ExpertConfig(
            name="expert2",
            obs_dim=10,
            action_dim=5,
            hidden_dim=64,
        )
        registry.register(mock_expert1)
        registry.register(mock_expert2)

        names = registry.list_names()
        assert "expert1" in names
        assert "expert2" in names

    def test_step(self):
        """Test stepping an expert."""
        registry = ExpertRegistry()
        mock_expert = MagicMock()
        mock_expert.config = ExpertConfig(
            name="test",
            obs_dim=10,
            action_dim=5,
            hidden_dim=64,
        )
        mock_expert.return_value = {"action": [1, 2, 3]}
        registry.register(mock_expert)

        obs = MagicMock()
        result = registry.step("test", obs, deterministic=True)

        mock_expert.assert_called_once_with(obs, deterministic=True)
        assert result == {"action": [1, 2, 3]}

    def test_reset_expert(self):
        """Test resetting a single expert."""
        registry = ExpertRegistry()
        mock_expert = MagicMock()
        mock_expert.config = ExpertConfig(
            name="test",
            obs_dim=10,
            action_dim=5,
            hidden_dim=64,
        )
        registry.register(mock_expert)

        registry.reset_expert("test", batch_size=2)
        mock_expert.reset_hidden.assert_called_once_with(2)

    def test_reset_all(self):
        """Test resetting all experts."""
        registry = ExpertRegistry()
        mock_expert1 = MagicMock()
        mock_expert1.config = ExpertConfig(
            name="expert1",
            obs_dim=10,
            action_dim=5,
            hidden_dim=64,
        )
        mock_expert2 = MagicMock()
        mock_expert2.config = ExpertConfig(
            name="expert2",
            obs_dim=10,
            action_dim=5,
            hidden_dim=64,
        )
        registry.register(mock_expert1)
        registry.register(mock_expert2)

        registry.reset_all(batch_size=4)

        mock_expert1.reset_hidden.assert_called_once_with(4)
        mock_expert2.reset_hidden.assert_called_once_with(4)

    def test_get_all_parameters(self):
        """Test getting all parameters."""
        registry = ExpertRegistry()
        mock_expert = MagicMock()
        mock_expert.config = ExpertConfig(
            name="test",
            obs_dim=10,
            action_dim=5,
            hidden_dim=64,
        )
        mock_expert.parameters.return_value = {
            "weight": MagicMock(),
            "bias": MagicMock(),
        }
        registry.register(mock_expert)

        params = registry.get_all_parameters()

        assert "test.weight" in params
        assert "test.bias" in params
