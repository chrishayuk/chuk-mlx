"""
StarCoder and StarCoder2 model family.

StarCoder (original, GPT-BigCode):
- StarCoder 15.5B
- StarCoderBase 15.5B
- SantaCoder 1.1B

StarCoder2:
- StarCoder2 3B
- StarCoder2 7B
- StarCoder2 15B

StarCoder (original) uses:
- LayerNorm (not RMSNorm)
- GELU activation (not SiLU/SwiGLU)
- Standard MLP (not gated)
- Multi-Query Attention (MQA)
- Learned positional embeddings (like GPT-2)
- 8K context window

StarCoder2 uses:
- LayerNorm (not RMSNorm)
- GELU activation (not SiLU/SwiGLU)
- Standard MLP (not gated)
- Grouped Query Attention (GQA)
- Sliding window attention
- RoPE positional embeddings
- 16K context window

References:
- StarCoder: https://huggingface.co/bigcode/starcoder
- StarCoder2: https://huggingface.co/bigcode/starcoder2-3b
"""

from .config import StarCoder2Config, StarCoderConfig
from .convert import (
    STARCODER2_WEIGHT_MAP,
    convert_hf_weights,
    load_hf_config,
    load_weights,
)
from .model import (
    StarCoder2Block,
    StarCoder2ForCausalLM,
    StarCoder2Model,
    StarCoderBlock,
    StarCoderForCausalLM,
    StarCoderModel,
)

__all__ = [
    # StarCoder (original)
    "StarCoderConfig",
    "StarCoderForCausalLM",
    "StarCoderModel",
    "StarCoderBlock",
    # StarCoder2
    "StarCoder2Config",
    "StarCoder2ForCausalLM",
    "StarCoder2Model",
    "StarCoder2Block",
    # Loading utilities
    "load_hf_config",
    "load_weights",
    "convert_hf_weights",
    "STARCODER2_WEIGHT_MAP",
]
