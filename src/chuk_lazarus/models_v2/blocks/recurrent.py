"""
Recurrent block wrappers.

Wraps recurrent components (LSTM, GRU, MinGRU) into the Block interface.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..components.normalization import RMSNorm
from ..components.recurrent import GRU, LSTM, MinGRU
from ..core.enums import BlockType, RecurrentType
from .base import Block, BlockOutput


class RecurrentBlockWrapper(Block):
    """
    Recurrent block with Block interface.

    Wraps LSTM, GRU, or MinGRU to provide the standard Block interface.
    Includes normalization and residual connection.

    Architecture:
        x -> norm -> RNN -> + residual -> output

    Args:
        d_model: Model dimension
        rnn_type: Type of RNN (RecurrentType.LSTM, GRU, or MINGRU)
        num_layers: Number of stacked RNN layers
        bidirectional: Use bidirectional RNN
        norm_eps: Normalization epsilon

    Example:
        >>> block = RecurrentBlockWrapper(d_model=768, rnn_type=RecurrentType.GRU)
        >>> x = mx.random.normal((2, 100, 768))
        >>> output = block(x)
    """

    def __init__(
        self,
        d_model: int,
        rnn_type: RecurrentType | str = RecurrentType.MINGRU,
        num_layers: int = 1,
        bidirectional: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self._hidden_size = d_model
        # Normalize rnn_type to enum
        self.rnn_type = RecurrentType(rnn_type) if isinstance(rnn_type, str) else rnn_type
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Pre-norm
        self.norm = RMSNorm(d_model, eps=norm_eps)

        # RNN
        # Note: bidirectional doubles output size, so we halve hidden
        if bidirectional:
            rnn_hidden = d_model // 2
        else:
            rnn_hidden = d_model

        if self.rnn_type == RecurrentType.LSTM:
            self.rnn = LSTM(
                input_size=d_model,
                hidden_size=rnn_hidden,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )
        elif self.rnn_type == RecurrentType.GRU:
            self.rnn = GRU(
                input_size=d_model,
                hidden_size=rnn_hidden,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )
        else:  # mingru
            if bidirectional:
                raise ValueError("MinGRU doesn't support bidirectional")
            self.rnn = MinGRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
            )

        # Output projection if dimensions don't match
        if bidirectional and rnn_hidden * 2 != d_model:
            self.out_proj = nn.Linear(rnn_hidden * 2, d_model)
        else:
            self.out_proj = None

    @property
    def block_type(self) -> BlockType:
        if self.rnn_type == RecurrentType.LSTM:
            return BlockType.LSTM
        elif self.rnn_type == RecurrentType.GRU:
            return BlockType.GRU
        else:
            return BlockType.MINGRU

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,  # Not used for RNNs
        cache: Any | None = None,
    ) -> BlockOutput:
        """
        Forward pass.

        Args:
            x: Input, shape (batch, seq_len, d_model)
            mask: Ignored (RNNs process sequentially)
            cache: Optional hidden state for inference

        Returns:
            BlockOutput with hidden states and updated cache
        """
        residual = x
        x = self.norm(x)

        # Run RNN
        if self.rnn_type == RecurrentType.LSTM:
            x, (h, c) = self.rnn(x, cache)
            new_cache = (h, c)
        else:
            x, h = self.rnn(x, cache)
            new_cache = h

        # Project if needed
        if self.out_proj is not None:
            x = self.out_proj(x)

        # Residual connection
        x = residual + x

        return BlockOutput(hidden_states=x, cache=new_cache)

    def init_cache(
        self,
        batch_size: int,
        max_seq_len: int,  # Not used for RNNs
    ) -> Any:
        """Initialize RNN hidden state."""
        if self.rnn_type == RecurrentType.LSTM:
            return self.rnn.init_state(batch_size)
        else:
            return self.rnn.init_state(batch_size)


class RecurrentWithFFN(Block):
    """
    Recurrent block with FFN sublayer.

    Similar to a transformer block but with RNN instead of attention.

    Architecture:
        x -> norm1 -> RNN -> + residual -> norm2 -> FFN -> + residual -> output

    Args:
        d_model: Model dimension
        rnn_type: Type of RNN
        intermediate_size: FFN intermediate dimension
        norm_eps: Normalization epsilon

    Example:
        >>> block = RecurrentWithFFN(d_model=768, rnn_type=RecurrentType.GRU)
        >>> x = mx.random.normal((2, 100, 768))
        >>> output = block(x)
    """

    def __init__(
        self,
        d_model: int,
        rnn_type: RecurrentType | str = RecurrentType.MINGRU,
        num_layers: int = 1,
        intermediate_size: int | None = None,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self._hidden_size = d_model
        # Normalize rnn_type to enum
        self.rnn_type = RecurrentType(rnn_type) if isinstance(rnn_type, str) else rnn_type

        intermediate_size = intermediate_size or d_model * 4

        # RNN sublayer
        self.rnn_norm = RMSNorm(d_model, eps=norm_eps)
        if self.rnn_type == RecurrentType.LSTM:
            self.rnn = LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
            )
        elif self.rnn_type == RecurrentType.GRU:
            self.rnn = GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
            )
        else:
            self.rnn = MinGRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
            )

        # FFN sublayer
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, d_model),
        )

    @property
    def block_type(self) -> BlockType:
        if self.rnn_type == RecurrentType.LSTM:
            return BlockType.LSTM
        elif self.rnn_type == RecurrentType.GRU:
            return BlockType.GRU
        else:
            return BlockType.MINGRU

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> BlockOutput:
        """Forward pass."""
        # RNN with residual
        residual = x
        x = self.rnn_norm(x)
        if self.rnn_type == RecurrentType.LSTM:
            x, (h, c) = self.rnn(x, cache)
            new_cache = (h, c)
        else:
            x, h = self.rnn(x, cache)
            new_cache = h
        x = residual + x

        # FFN with residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return BlockOutput(hidden_states=x, cache=new_cache)

    def init_cache(
        self,
        batch_size: int,
        max_seq_len: int,
    ) -> Any:
        """Initialize cache."""
        if self.rnn_type == RecurrentType.LSTM:
            return self.rnn.init_state(batch_size)
        else:
            return self.rnn.init_state(batch_size)


def create_recurrent_block(
    d_model: int,
    rnn_type: RecurrentType | str = RecurrentType.MINGRU,
    num_layers: int = 1,
    norm_eps: float = 1e-5,
) -> RecurrentBlockWrapper:
    """Factory function for RecurrentBlockWrapper."""
    return RecurrentBlockWrapper(
        d_model=d_model,
        rnn_type=rnn_type,
        num_layers=num_layers,
        norm_eps=norm_eps,
    )


def create_recurrent_with_ffn(
    d_model: int,
    rnn_type: RecurrentType | str = RecurrentType.MINGRU,
    intermediate_size: int | None = None,
) -> RecurrentWithFFN:
    """Factory function for RecurrentWithFFN."""
    return RecurrentWithFFN(
        d_model=d_model,
        rnn_type=rnn_type,
        intermediate_size=intermediate_size,
    )
