"""
Jamba model family.

Jamba is a hybrid Mamba-Transformer MoE model from AI21 Labs.

Supports:
- Jamba v0.1 (52B total, ~12B active)
- Jamba 1.5 Mini (52B total, 12B active)
- Jamba 1.5 Large (398B total, 94B active)

Key architectural innovations:
- Hybrid: 1 attention layer every 8 layers (rest are Mamba SSM)
- MoE: Every 2nd layer uses MoE (16 experts, 2 active per token)
- 256K context window
- O(n) complexity from Mamba layers with precise recall from sparse attention

Reference: https://huggingface.co/ai21labs/Jamba-v0.1
"""

from .config import JambaConfig
from .convert import JAMBA_WEIGHT_MAP, convert_hf_weights
from .model import JambaAttentionBlock, JambaForCausalLM, JambaMambaBlock, JambaModel

__all__ = [
    "JambaConfig",
    "JambaForCausalLM",
    "JambaModel",
    "JambaMambaBlock",
    "JambaAttentionBlock",
    "convert_hf_weights",
    "JAMBA_WEIGHT_MAP",
]
