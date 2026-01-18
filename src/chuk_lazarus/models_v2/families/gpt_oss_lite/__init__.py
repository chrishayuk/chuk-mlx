"""
GPT-OSS-Lite: Reduced expert model for efficient inference.

This model uses variable experts per layer based on activation analysis:
- Early layers (0-7):   6 experts
- Middle layers (8-17): 12 experts
- Late layers (18-23):  8 experts

Total: 216 experts vs 768 in original (71.9% reduction)
Parameters: 2.64B vs 4.79B (44.9% reduction)
"""

from .config import GptOssLiteConfig
from .model import GptOssLiteForCausalLM

__all__ = ["GptOssLiteConfig", "GptOssLiteForCausalLM"]
