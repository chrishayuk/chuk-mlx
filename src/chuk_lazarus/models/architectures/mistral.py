"""Mistral model architecture."""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..config import ModelConfig
from .base import BaseModel, TransformerModel, TransformerBlock
from .attention import Attention
from .mlp import MLP


class MistralBlock(TransformerBlock):
    """Mistral transformer block."""

    def __init__(self, config: ModelConfig):
        super().__init__(
            config=config,
            attention_class=Attention,
            mlp_class=MLP,
            norm_class=nn.RMSNorm
        )


class MistralTransformer(TransformerModel):
    """Mistral transformer (without LM head)."""

    def __init__(self, config: ModelConfig):
        super().__init__(
            config=config,
            layer_class=MistralBlock,
            norm_class=nn.RMSNorm
        )


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
