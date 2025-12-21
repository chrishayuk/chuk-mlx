"""
Expert Registry for managing multiple tiny RNN experts.

This allows Mistral to select experts by name and provides
a clean interface for the orchestrator.
"""

import logging
from typing import Any

import mlx.core as mx

from .gru_expert import GRUExpert
from .lstm_expert import LSTMExpert
from .rnn_expert_base import ExpertConfig, RNNExpertBase

logger = logging.getLogger(__name__)


# Global registry
_EXPERT_CLASSES: dict[str, type[RNNExpertBase]] = {
    "gru": GRUExpert,
    "lstm": LSTMExpert,
}

_EXPERT_INSTANCES: dict[str, RNNExpertBase] = {}


def register_expert_class(name: str, expert_class: type[RNNExpertBase]):
    """Register a new expert class type."""
    _EXPERT_CLASSES[name] = expert_class
    logger.info(f"Registered expert class: {name}")


def register_expert(name: str, expert: RNNExpertBase):
    """Register an expert instance by name."""
    _EXPERT_INSTANCES[name] = expert
    logger.info(f"Registered expert instance: {name}")


def get_expert(name: str) -> RNNExpertBase | None:
    """Get an expert instance by name."""
    return _EXPERT_INSTANCES.get(name)


def list_experts() -> list[str]:
    """List all registered expert names."""
    return list(_EXPERT_INSTANCES.keys())


def create_expert(expert_type: str, config: ExpertConfig) -> RNNExpertBase:
    """
    Create and register an expert.

    Args:
        expert_type: "gru" or "lstm"
        config: Expert configuration

    Returns:
        The created expert instance
    """
    if expert_type not in _EXPERT_CLASSES:
        raise ValueError(
            f"Unknown expert type: {expert_type}. Available: {list(_EXPERT_CLASSES.keys())}"
        )

    expert_class = _EXPERT_CLASSES[expert_type]
    expert = expert_class(config)

    # Auto-register
    register_expert(config.name, expert)

    return expert


class ExpertRegistry:
    """
    Registry for managing multiple experts.

    Provides a clean interface for:
    - Creating experts from configs
    - Loading/saving expert checkpoints
    - Routing calls from Mistral to appropriate expert
    """

    def __init__(self):
        self.experts: dict[str, RNNExpertBase] = {}
        self.configs: dict[str, ExpertConfig] = {}

    def register(self, expert: RNNExpertBase):
        """Register an expert instance."""
        name = expert.config.name
        self.experts[name] = expert
        self.configs[name] = expert.config
        logger.info(f"Registered expert: {name}")

    def create_and_register(self, expert_type: str, config: ExpertConfig) -> RNNExpertBase:
        """Create and register an expert."""
        expert = create_expert(expert_type, config)
        self.register(expert)
        return expert

    def get(self, name: str) -> RNNExpertBase | None:
        """Get expert by name."""
        return self.experts.get(name)

    def __getitem__(self, name: str) -> RNNExpertBase:
        """Get expert by name (raises if not found)."""
        if name not in self.experts:
            raise KeyError(f"Expert not found: {name}. Available: {list(self.experts.keys())}")
        return self.experts[name]

    def __contains__(self, name: str) -> bool:
        return name in self.experts

    def list_names(self) -> list[str]:
        """List all expert names."""
        return list(self.experts.keys())

    def step(self, expert_name: str, obs: mx.array, deterministic: bool = False) -> dict:
        """
        Step an expert with an observation.

        This is the main interface for the orchestrator to call experts.

        Args:
            expert_name: Name of the expert to call
            obs: Observation tensor
            deterministic: Whether to use deterministic action

        Returns:
            Expert output dict (action, log_prob, value, hidden, entropy)
        """
        expert = self[expert_name]
        return expert(obs, deterministic=deterministic)

    def reset_expert(self, expert_name: str, batch_size: int = 1):
        """Reset an expert's hidden state."""
        expert = self[expert_name]
        expert.reset_hidden(batch_size)

    def reset_all(self, batch_size: int = 1):
        """Reset all experts' hidden states."""
        for expert in self.experts.values():
            expert.reset_hidden(batch_size)

    def save_checkpoint(self, path: str):
        """Save all expert weights to a checkpoint."""
        checkpoint = {}
        for name, expert in self.experts.items():
            checkpoint[name] = {
                "weights": expert.parameters(),
                "config": {
                    "name": expert.config.name,
                    "obs_dim": expert.config.obs_dim,
                    "action_dim": expert.config.action_dim,
                    "hidden_dim": expert.config.hidden_dim,
                    "num_layers": expert.config.num_layers,
                    "discrete_actions": expert.config.discrete_actions,
                    "num_actions": expert.config.num_actions,
                    "use_value_head": expert.config.use_value_head,
                },
            }

        mx.save(path, checkpoint)
        logger.info(f"Saved {len(self.experts)} experts to {path}")

    def load_checkpoint(self, path: str, expert_type: str = "gru"):
        """Load expert weights from a checkpoint."""
        checkpoint = mx.load(path)

        for name, data in checkpoint.items():
            config = ExpertConfig(**data["config"])
            expert = create_expert(expert_type, config)
            expert.load_weights(list(data["weights"].items()))
            self.register(expert)

        logger.info(f"Loaded {len(checkpoint)} experts from {path}")

    def get_all_parameters(self) -> dict[str, Any]:
        """Get parameters from all experts (for joint training)."""
        all_params = {}
        for name, expert in self.experts.items():
            for param_name, param in expert.parameters().items():
                all_params[f"{name}.{param_name}"] = param
        return all_params
