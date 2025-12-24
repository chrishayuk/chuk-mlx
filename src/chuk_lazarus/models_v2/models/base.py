"""
Base model abstractions.

Defines the common interface for complete models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..core.config import ModelConfig


@dataclass
class ModelOutput:
    """
    Output from a model forward pass.

    Attributes:
        loss: Training loss (if labels provided)
        logits: Model predictions
        hidden_states: Optional tuple of all layer hidden states
        cache: Optional cache for inference
        aux_loss: Optional auxiliary loss (e.g., MoE load balancing)
    """

    loss: mx.array | None = None
    logits: mx.array | None = None
    hidden_states: tuple[mx.array, ...] | None = None
    cache: list[Any] | None = None
    aux_loss: mx.array | None = None

    # Additional fields for specific model types
    aux_outputs: dict[str, Any] = field(default_factory=dict)


class Model(nn.Module, ABC):
    """
    Abstract base class for complete models.

    A model combines:
    - A backbone (embeddings + blocks)
    - One or more heads (task-specific outputs)

    Models handle the full forward pass from input IDs to outputs.
    """

    @property
    @abstractmethod
    def config(self) -> ModelConfig:
        """Return the model configuration."""
        pass

    @property
    @abstractmethod
    def backbone(self) -> nn.Module:
        """Return the backbone module."""
        pass

    @abstractmethod
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        labels: mx.array | None = None,
        cache: list[Any] | None = None,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        """
        Forward pass.

        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
            cache: Optional cache for inference
            output_hidden_states: Whether to return all hidden states

        Returns:
            ModelOutput with loss, logits, and optional extras
        """
        pass

    def init_cache(
        self,
        batch_size: int,
        max_seq_len: int,
    ) -> list[Any]:
        """Initialize cache for inference."""
        return self.backbone.init_cache(batch_size, max_seq_len)

    def get_input_embeddings(self) -> nn.Module:
        """Get input embedding layer."""
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        """Set input embedding layer."""
        self.backbone.set_input_embeddings(embeddings)

    def num_parameters(self, only_trainable: bool = True) -> int:
        """
        Count model parameters.

        Args:
            only_trainable: Only count trainable parameters

        Returns:
            Number of parameters
        """
        import mlx.utils

        total = 0
        for name, param in mlx.utils.tree_flatten(self.parameters()):
            if isinstance(param, mx.array):
                total += param.size
        return total

    def save_weights(self, path: str) -> None:
        """Save model weights to file."""
        import numpy as np

        weights = {}
        for name, param in self.parameters().items():
            if isinstance(param, mx.array):
                weights[name] = np.array(param)

        np.savez(path, **weights)

    def load_weights(self, path: str) -> None:
        """Load model weights from file."""
        import numpy as np

        data = np.load(path)
        weights = {k: mx.array(v) for k, v in data.items()}
        self.update(weights)

    @classmethod
    @abstractmethod
    def from_config(cls, config: ModelConfig) -> Model:
        """Create model from configuration."""
        pass

    @classmethod
    async def from_pretrained_async(
        cls,
        model_path: str,
        config: ModelConfig | None = None,
    ) -> Model:
        """
        Load pretrained model asynchronously.

        Args:
            model_path: Path to model directory or HuggingFace model ID
            config: Optional override config

        Returns:
            Loaded model
        """
        import json
        from pathlib import Path

        import aiofiles

        path = Path(model_path)

        # Load config if not provided
        if config is None:
            config_path = path / "config.json"
            async with aiofiles.open(config_path) as f:
                config_data = json.loads(await f.read())
            config = ModelConfig(**config_data)

        # Create model
        model = cls.from_config(config)

        # Load weights
        weights_path = path / "model.safetensors"
        if weights_path.exists():
            # Use safetensors
            try:
                import safetensors.numpy as st

                weights = st.load_file(str(weights_path))
                weights = {k: mx.array(v) for k, v in weights.items()}
                model.update(weights)
            except ImportError:
                pass

        return model
