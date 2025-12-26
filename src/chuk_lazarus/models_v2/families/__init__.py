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
- jamba: Jamba hybrid Mamba-Transformer MoE from AI21 Labs
- llama: Llama 1/2/3, Mistral, and compatible models
- llama4: Llama 4 with MoE and multimodal support
- mamba: Mamba SSM models
- starcoder2: StarCoder2 code generation models
"""

from . import gemma, granite, jamba, llama, llama4, mamba, starcoder2

__all__ = ["gemma", "granite", "jamba", "llama", "llama4", "mamba", "starcoder2"]
