"""
Llama model family.

Supports:
- Llama 1 (original Meta models)
- Llama 2 (7B, 13B, 70B)
- Llama 3 (8B, 70B)
- Mistral (7B, with sliding window)
- Mixtral (8x7B MoE)
- Code Llama
- And other compatible architectures

This is the reference implementation demonstrating how to use the
models_v2 composable architecture.
"""

from .config import LlamaConfig
from .convert import LLAMA_WEIGHT_MAP, convert_hf_weights
from .model import LlamaBlock, LlamaForCausalLM, LlamaModel

__all__ = [
    "LlamaConfig",
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaBlock",
    "convert_hf_weights",
    "LLAMA_WEIGHT_MAP",
]
