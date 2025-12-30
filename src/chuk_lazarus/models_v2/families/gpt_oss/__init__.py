"""
GPT-OSS model family.

GPT-OSS is OpenAI's open-source Mixture-of-Experts model with:
- 24 layers with alternating sliding window and full attention
- 32 experts, 4 active per token
- SwiGLU activation
- YaRN RoPE scaling for extended context (131K)

Supports both:
- Dense (unquantized) models
- 8-bit quantized models (lmstudio-community/gpt-oss-20b-MLX-8bit)
"""

from .config import GptOssConfig
from .model import GptOssForCausalLM
from .quantized import QuantizedGptOssForCausalLM

__all__ = [
    "GptOssConfig",
    "GptOssForCausalLM",
    "QuantizedGptOssForCausalLM",
]
