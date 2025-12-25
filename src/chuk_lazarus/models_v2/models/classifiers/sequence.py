"""
Sequence classifier.

Classifies entire sequences into categories using a backbone.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...backbones import MambaBackbone, TransformerBackbone
from ...core.config import ModelConfig
from ...core.enums import BackboneType
from ...heads import ClassifierHead, PoolerHead
from ..base import Model, ModelOutput


class SequenceClassifier(Model):
    """
    Sequence classification model.

    Classifies entire sequences into categories.
    Uses pooled representation (first/last/mean token).

    Args:
        config: Model configuration
        num_labels: Number of output classes
        pool_strategy: Pooling strategy ("first", "last", "mean")
        use_pooler: Use BERT-style pooler layer
        backbone_type: Type of backbone

    Example:
        >>> config = ModelConfig(vocab_size=32000, hidden_size=768)
        >>> model = SequenceClassifier(config, num_labels=3)
        >>> input_ids = mx.array([[1, 2, 3, 4, 5]])
        >>> output = model(input_ids)
        >>> output.logits.shape
        (1, 3)
    """

    def __init__(
        self,
        config: ModelConfig,
        num_labels: int,
        pool_strategy: str = "last",
        use_pooler: bool = False,
        backbone_type: BackboneType = BackboneType.TRANSFORMER,
    ):
        super().__init__()

        self._config = config
        self.num_labels = num_labels
        self.pool_strategy = pool_strategy

        # Create backbone
        if backbone_type == BackboneType.TRANSFORMER:
            self._backbone = TransformerBackbone.from_config(config)
        elif backbone_type == BackboneType.MAMBA:
            self._backbone = MambaBackbone.from_config(config)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

        # Classification head
        if use_pooler:
            self.classifier = PoolerHead(
                hidden_size=config.hidden_size,
                num_labels=num_labels,
                pool_strategy=pool_strategy,
            )
        else:
            self.classifier = ClassifierHead(
                hidden_size=config.hidden_size,
                num_labels=num_labels,
                pool_strategy=pool_strategy,
            )

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

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
            labels: Optional labels, shape (batch,)
            cache: Optional cache (not typically used for classification)
            output_hidden_states: Return all hidden states

        Returns:
            ModelOutput with logits and optional loss
        """
        # Backbone forward
        backbone_output = self._backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache=cache,
            output_hidden_states=output_hidden_states,
        )

        # Classification
        head_output = self.classifier(
            hidden_states=backbone_output.last_hidden_state,
            labels=labels,
            attention_mask=attention_mask,
        )

        return ModelOutput(
            loss=head_output.loss,
            logits=head_output.logits,
            hidden_states=backbone_output.hidden_states,
        )

    @classmethod
    def from_config(cls, config: ModelConfig, num_labels: int = 2) -> SequenceClassifier:
        """Create from config."""
        return cls(config=config, num_labels=num_labels)
