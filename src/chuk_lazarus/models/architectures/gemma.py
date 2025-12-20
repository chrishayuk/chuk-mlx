"""Gemma model architecture."""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..config import ModelConfig
from .base import BaseModel, TransformerModel, TransformerBlock
from .attention import Attention
from .mlp import MLP


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm with +1 offset."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # Gemma adds 1 to the weight
        return mx.fast.rms_norm(x, self.weight + 1, self.eps)


class GemmaBlock(TransformerBlock):
    """Gemma transformer block."""

    def __init__(self, config: ModelConfig):
        super().__init__(
            config=config,
            attention_class=Attention,
            mlp_class=MLP,
            norm_class=GemmaRMSNorm
        )


class GemmaTransformer(TransformerModel):
    """Gemma transformer (without LM head)."""

    def __init__(self, config: ModelConfig):
        super().__init__(
            config=config,
            layer_class=GemmaBlock,
            norm_class=GemmaRMSNorm
        )
        # Gemma scales embeddings
        self._embed_scale = math.sqrt(config.hidden_size)

    def _scale_embeddings(self, embeddings: mx.array) -> mx.array:
        """Gemma scales embeddings by sqrt(hidden_size)."""
        return embeddings * self._embed_scale


class GemmaModel(BaseModel):
    """
    Gemma language model.

    Supports:
    - Gemma 2B, 7B
    - Gemma 2
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = GemmaTransformer(config)
