"""
Model family implementations.

Each family provides:
- Configuration class
- Model implementation
- Weight conversion utilities
- Registry integration

Available families:
- gemma: Gemma 3, FunctionGemma
- granite: IBM Granite 3.x/4.x with hybrid Mamba-2/Transformer
- llama: Llama 1/2/3, Mistral, and compatible models
- llama4: Llama 4 with MoE and multimodal support
- mamba: Mamba SSM models
"""

from . import gemma, granite, llama, llama4, mamba

__all__ = ["gemma", "granite", "llama", "llama4", "mamba"]
