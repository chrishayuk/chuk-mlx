"""Mistral model architecture."""

import mlx.nn as nn

from ..config import ModelConfig
from .attention import Attention
from .base import BaseModel, TransformerBlock, TransformerModel
from .mlp import MLP


class MistralBlock(TransformerBlock):
    """Mistral transformer block."""

    def __init__(self, config: ModelConfig):
        super().__init__(
            config=config, attention_class=Attention, mlp_class=MLP, norm_class=nn.RMSNorm
        )


class MistralTransformer(TransformerModel):
    """Mistral transformer (without LM head)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config=config, layer_class=MistralBlock, norm_class=nn.RMSNorm)


class MistralModel(BaseModel):
    """
    Mistral language model.

    Supports:
    - Mistral 7B
    - Mixtral (MoE variant needs separate implementation)
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = MistralTransformer(config)
