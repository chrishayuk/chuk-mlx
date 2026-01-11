"""OLMoE model family.

Allen AI's Open Language Model with Mixture of Experts.
Based on Llama architecture with MoE FFN layers.

Features:
- 64 experts per layer
- Top-8 routing (8 experts active per token)
- Standard softmax routing with top-k selection
- No shared expert
"""

from .config import OLMoEConfig
from .model import OLMoEForCausalLM, OLMoEModel

__all__ = [
    "OLMoEConfig",
    "OLMoEModel",
    "OLMoEForCausalLM",
]
