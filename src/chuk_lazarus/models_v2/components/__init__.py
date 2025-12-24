"""
Reusable model components.

Components are the building blocks that get composed into blocks,
which get stacked into backbones, which get wrapped with heads to form models.

Submodules:
- embeddings: Token and position embeddings (RoPE, ALiBi, learned)
- attention: Attention mechanisms (MHA, GQA, MQA, sliding window)
- ffn: Feed-forward networks (MLP, SwiGLU, GEGLU, MoE)
- normalization: Normalization layers (RMSNorm, LayerNorm)
- ssm: State space models (Mamba, Mamba2)
- recurrent: Recurrent cells (LSTM, GRU, minGRU)
"""

from . import attention, embeddings, ffn, normalization, recurrent, ssm

__all__ = [
    "embeddings",
    "attention",
    "ffn",
    "normalization",
    "ssm",
    "recurrent",
]
